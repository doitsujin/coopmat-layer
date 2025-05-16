#include "vkroots.h"

#include "util/object_map.h"
#include "util/vulkan_util.h"

#include "lower_coopmat.h"

#include <array>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>

namespace CoopmatLayer {

using namespace util;

/** Definitions of the extensions we add in this layer */
constexpr VkExtensionProperties makeExtension(const char* name, uint32_t version) {
  VkExtensionProperties result = { };
  for (size_t i = 0; i < VK_MAX_EXTENSION_NAME_SIZE && name[i]; i++)
    result.extensionName[i] = name[i];
  result.specVersion = version;
  return result;
}

static const std::array<VkExtensionProperties, 2> g_extensions = {{
  makeExtension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_KHR_COOPERATIVE_MATRIX_SPEC_VERSION),
}};


/** Helper to order extension names */
struct OrderExtensionByName {
  bool operator () (const VkExtensionProperties& a, const VkExtensionProperties& b) const {
    for (size_t i = 0; i < VK_MAX_EXTENSION_NAME_SIZE; i++) {
      if (a.extensionName[i] > b.extensionName[i]) return false;
      if (a.extensionName[i] < b.extensionName[i]) return true;
      if (!a.extensionName[i]) break;
    }
    return false;
  }
};


/** Helper struct for device properties */
struct DeviceProperties {
  VkPhysicalDeviceVulkan13Properties    vk13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES };
  VkPhysicalDeviceVulkan12Properties    vk12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES, &vk13 };
  VkPhysicalDeviceVulkan11Properties    vk11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &vk12 };
  VkPhysicalDeviceProperties2           core = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &vk11 };
};


/** Helper struct for device features */
struct DeviceFeatures {
  VkPhysicalDeviceVulkan13Features      vk13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
  VkPhysicalDeviceVulkan12Features      vk12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &vk13 };
  VkPhysicalDeviceVulkan11Features      vk11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &vk12 };
  VkPhysicalDeviceFeatures2             core = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &vk11 };
};


/** Cooperative matrix dimensions */
struct CoopmatDimension {
  uint32_t mSize;
  uint32_t nSize;
  uint32_t kSize;
};

static const std::vector<CoopmatDimension> g_coopmatDimensions = {{
  {  4u,  4u,  4u },
  {  8u,  8u,  8u },
  { 16u, 16u, 16u },
}};



/** Physical device */
class PhysicalDeviceInfo {

public:

  PhysicalDeviceInfo(
    const vkroots::VkInstanceDispatch*        pDispatch,
          VkPhysicalDevice                    physicalDevice)
  : m_dispatch                (pDispatch)
  , m_physicalDeviceDispatch  (vkroots::tables::LookupPhysicalDeviceDispatch(physicalDevice))
  , m_handle                  (physicalDevice) {
    /* Check adapter info, don't bother if we don't have Vulkan 1.3 available */
    VkPhysicalDeviceProperties properties = { };
    m_dispatch->GetPhysicalDeviceProperties(m_handle, &properties);

    if (properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 3, 0)) {
      std::cerr << "Ignoring adapter " << properties.deviceName << " with unsupported Vulkan version " <<
        VK_API_VERSION_MAJOR(properties.apiVersion) << "." <<
        VK_API_VERSION_MINOR(properties.apiVersion) << std::endl;
      return;
    }

    /* Query device properties and features */
    m_dispatch->GetPhysicalDeviceFeatures2(m_handle, &m_features.core);
    m_dispatch->GetPhysicalDeviceProperties2(m_handle, &m_properties.core);

    /* If we can't meaningfully use subgroup ops, skip */
    constexpr static VkSubgroupFeatureFlags subgroupFlags =
      VK_SUBGROUP_FEATURE_BASIC_BIT |
      VK_SUBGROUP_FEATURE_VOTE_BIT |
      VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
      VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
      VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
      VK_SUBGROUP_FEATURE_CLUSTERED_BIT;

    if (!(m_properties.vk11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT)
     || (m_properties.vk11.subgroupSupportedOperations & subgroupFlags) != subgroupFlags
     || (m_properties.vk13.maxSubgroupSize < 4)) {
      std::cerr << "Ignoring adapter " << properties.deviceName << " with spotty subgroup support" << std::endl;
      return;
    }

    /* Query extensions */
    uint32_t extensionCount = 0u;
    m_dispatch->EnumerateDeviceExtensionProperties(m_handle, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions(extensionCount);
    m_dispatch->EnumerateDeviceExtensionProperties(m_handle, nullptr, &extensionCount, extensions.data());

    for (const auto& e : extensions)
      m_extensions.insert(e);

    /* Add our fake extensions */
    for (const auto& e : g_extensions)
      m_extensions.insert(e);

    /* Hide the NV extensions, we don't want to deal with that */
    m_extensions.erase(makeExtension(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME, 0));
    m_extensions.erase(makeExtension(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME, 0));

    /* We can emulate c: */
    m_enable = true;
  }

  /**
   * \brief Checks whether to use the extension layer
   * \returns \c true if the extension layer is to be used
   */
  bool isLayerEnabled() const {
    return m_enable;
  }

  /**
   * \brief Queries device features
   * \returns Device features
   */
  const DeviceFeatures& getFeatures() const {
    return m_features;
  }

  /**
   * \brief Queries device properties
   * \returns Device properties
   */
  const DeviceProperties& getProperties() const {
    return m_properties;
  }

  /**
   * \brief Queries device extensions
   *
   * \param [in] pLayerName Layer to query extensions for
   * \param [in,out] pExtensionCount Number of extensions
   * \param [out] pExtensionProperties Extension infos
   * \returns Status of the operation
   */
  VkResult enumerateExtensionProperties(
    const char*                               pLayerName,
          uint32_t*                           pExtensionCount,
          VkExtensionProperties*              pExtensionProperties) const {
    if (!pExtensionProperties) {
      *pExtensionCount = m_extensions.size();
      return VK_SUCCESS;
    }

    uint32_t count = 0u;

    for (const auto& e : m_extensions) {
      if (count == *pExtensionCount)
        return VK_INCOMPLETE;

      pExtensionProperties[count++] = e;
    }

    *pExtensionCount = count;
    return VK_SUCCESS;
  }

  /**
   * \brief Queries device features
   *
   * Overwrites coopmat features as necessary.
   * \param [out] pProperties Device properties
   */
  void getPhysicalDeviceFeatures(
          VkPhysicalDeviceFeatures2*          pFeatures) const {
    auto [coopmatKhr, headerKhr] = vkroots::RemoveFromChain<VkPhysicalDeviceCooperativeMatrixFeaturesKHR>(pFeatures);
    m_dispatch->GetPhysicalDeviceFeatures2(m_handle, pFeatures);

    if (coopmatKhr) {
      coopmatKhr->cooperativeMatrix = true;
      coopmatKhr->cooperativeMatrixRobustBufferAccess = true;

      headerKhr->pNext = reinterpret_cast<VkBaseOutStructure*>(coopmatKhr);
    }
  }

  /**
   * \brief Queries device properties
   *
   * Overwrites coopmat properties as necessary.
   * \param [out] pProperties Device properties
   */
  void getPhysicalDeviceProperties(
          VkPhysicalDeviceProperties2*        pProperties) const {
    auto [coopmatKhr, headerKhr] = vkroots::RemoveFromChain<VkPhysicalDeviceCooperativeMatrixPropertiesKHR>(pProperties);
    m_dispatch->GetPhysicalDeviceProperties2(m_handle, pProperties);

    if (coopmatKhr) {
      coopmatKhr->cooperativeMatrixSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT;

      headerKhr->pNext = reinterpret_cast<VkBaseOutStructure*>(coopmatKhr);
    }
  }


  /**
   * \brief Queries coopmat properties
   *
   * Overwrites coopmat properties as necessary.
   * \param [out] pProperties Device properties
   */
  VkResult getCooperativeMatrixProperties(
          uint32_t*                           pPropertyCount,
          VkCooperativeMatrixPropertiesKHR*   pProperties) {
    uint32_t count = 0u;

    for (const auto& dim : g_coopmatDimensions) {
      /* Order float types from small to large and only support
       * operations where a and b are of the same type, and the
       * result type matches the accumulator. */
      static const std::array<VkComponentTypeKHR, 2> s_floatFormats = {
        VK_COMPONENT_TYPE_FLOAT16_KHR,
        VK_COMPONENT_TYPE_FLOAT32_KHR,
      };

      for (size_t i = 0u; i < s_floatFormats.size(); i++) {
        if (!supportsComponentType(s_floatFormats[i]))
          continue;

        if (!supportsDimForABType(dim, s_floatFormats[i]))
          continue;

        for (size_t j = i; j < s_floatFormats.size(); j++) {
          if (pProperties) {
            auto& e = pProperties[count];
            e.MSize = dim.mSize;
            e.NSize = dim.nSize;
            e.KSize = dim.kSize;
            e.AType = s_floatFormats[i];
            e.BType = s_floatFormats[i];
            e.CType = s_floatFormats[j];
            e.ResultType = s_floatFormats[j];
            e.saturatingAccumulation = VK_FALSE;
            e.scope = VK_SCOPE_SUBGROUP_KHR;
          }

          count += 1u;
        }
      }

      /* Only support 8-bit integers with 32-bit returns. This is a mess where each
       * operand can either be signed or unsigned and saturation can be on or off. */
      for (uint32_t i = 0u; i < 0x40u; i++) {
        VkComponentTypeKHR aType = (i & 0x01u) ? VK_COMPONENT_TYPE_SINT8_KHR : VK_COMPONENT_TYPE_UINT8_KHR;
        VkComponentTypeKHR bType = (i & 0x02u) ? VK_COMPONENT_TYPE_SINT8_KHR : VK_COMPONENT_TYPE_UINT8_KHR;
        VkComponentTypeKHR cType = (i & 0x08u)
          ? ((i & 0x04u) ? VK_COMPONENT_TYPE_SINT8_KHR : VK_COMPONENT_TYPE_UINT8_KHR)
          : ((i & 0x04u) ? VK_COMPONENT_TYPE_SINT32_KHR : VK_COMPONENT_TYPE_UINT32_KHR);
        VkComponentTypeKHR resultType = (i & 0x10u) ? VK_COMPONENT_TYPE_SINT32_KHR : VK_COMPONENT_TYPE_UINT32_KHR;

        bool sat = bool(i & 0x20u);

        if (!supportsComponentType(aType)
         || !supportsComponentType(bType)
         || !supportsComponentType(cType)
         || !supportsComponentType(resultType))
          continue;

        if (!supportsDimForABType(dim, aType)
         || !supportsDimForABType(dim, bType))
          continue;

        /* If saturation is enabled and the product is signed,
         * require a signed accumulator as well. */
        if (sat && (i & 0x3u) && !(i & 0x4u))
          continue;

        if (pProperties) {
          auto& e = pProperties[count];
          e.MSize = dim.mSize;
          e.NSize = dim.nSize;
          e.KSize = dim.kSize;
          e.AType = aType;
          e.BType = bType;
          e.CType = cType;
          e.ResultType = resultType;
          e.saturatingAccumulation = sat;
          e.scope = VK_SCOPE_SUBGROUP_KHR;
        }

        count += 1u;
      }
    }

    if (pProperties && count > *pPropertyCount)
      return VK_INCOMPLETE;

    *pPropertyCount = count;
    return VK_SUCCESS;
  }

private:

  const vkroots::VkInstanceDispatch*        m_dispatch                = nullptr;
  const vkroots::VkPhysicalDeviceDispatch*  m_physicalDeviceDispatch  = nullptr;

  VkPhysicalDevice                    m_handle    = VK_NULL_HANDLE;

  bool                                m_enable    = false;

  std::set<VkExtensionProperties,
    OrderExtensionByName>             m_extensions;

  DeviceProperties                    m_properties;
  DeviceFeatures                      m_features;

  bool hasExtension(const char* name) const {
    return m_extensions.find(makeExtension(name, 0)) != m_extensions.end();
  }

  bool supportsComponentType(VkComponentTypeKHR type) const {
    switch (type) {
      case VK_COMPONENT_TYPE_FLOAT32_KHR:
      case VK_COMPONENT_TYPE_UINT32_KHR:
      case VK_COMPONENT_TYPE_SINT32_KHR:
        return true;

      case VK_COMPONENT_TYPE_FLOAT16_KHR:
        return m_features.vk12.shaderFloat16;

      case VK_COMPONENT_TYPE_UINT8_KHR:
      case VK_COMPONENT_TYPE_SINT8_KHR:
        return m_features.vk12.shaderInt8
            && m_features.vk13.shaderIntegerDotProduct;

      default:
        return false;
    }
  }

  bool supportsDimForABType(const CoopmatDimension& dim, VkComponentTypeKHR type) const {
    /* Things get awkward if a single column or row spans
     * multiple registers, so ignore any of that */
    uint32_t colSize = dim.mSize * getComponentSize(type);
    uint32_t rowSize = dim.nSize * getComponentSize(type);

    /* We pack small component types into one dword if possible */
    uint32_t minGuaranteedSize = std::max(4u, m_properties.vk13.minSubgroupSize) * sizeof(uint32_t);

    return std::max(colSize, rowSize) > minGuaranteedSize;
  }

  static bool componentTypeIsInteger(VkComponentTypeKHR type) {
    return type == VK_COMPONENT_TYPE_UINT32_KHR
        || type == VK_COMPONENT_TYPE_SINT32_KHR
        || type == VK_COMPONENT_TYPE_UINT16_KHR
        || type == VK_COMPONENT_TYPE_SINT16_KHR
        || type == VK_COMPONENT_TYPE_UINT8_KHR
        || type == VK_COMPONENT_TYPE_SINT8_KHR;
  }

};


static ObjectMap<VkPhysicalDevice, PhysicalDeviceInfo> g_physicalDevices;

static std::shared_ptr<PhysicalDeviceInfo> findPhysicalDevice(
        VkPhysicalDevice                    physicalDevice) {
  return g_physicalDevices.find(physicalDevice);
}

static std::shared_ptr<PhysicalDeviceInfo> getOrCreatePhysicalDevice(
        VkPhysicalDevice                    physicalDevice,
  const vkroots::VkInstanceDispatch*        pDispatch) {
  auto device = findPhysicalDevice(physicalDevice);

  if (device)
    return device;

  return g_physicalDevices.create(physicalDevice, pDispatch, physicalDevice);
}


/** Shader module */
class ShaderModuleInfo {

public:

  ShaderModuleInfo(const VkShaderModuleCreateInfo* pCreateInfo)
  : m_code(pCreateInfo->codeSize / sizeof(uint32_t)) {
    std::memcpy(m_code.data(), pCreateInfo->pCode, pCreateInfo->codeSize);
  }

  /**
   * \brief Retrieves create info
   *
   * This can be passed in directly for pipeline creation.
   * \returns Shader module create info
   */
  VkShaderModuleCreateInfo getCreateInfo() const {
    VkShaderModuleCreateInfo info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    info.codeSize = m_code.size() * sizeof(uint32_t);
    info.pCode = m_code.data();
    return info;
  }

private:

  std::vector<uint32_t> m_code;

};


/** Device info */
class DeviceInfo {

public:

  DeviceInfo(
          VkDevice                          handle,
          VkPhysicalDevice                  physicalDevice,
    const VkDeviceCreateInfo*               pCreateInfo)
  : m_handle              (handle)
  , m_physicalDeviceInfo  (findPhysicalDevice(physicalDevice)) {

  }

  /**
   * \brief Creates shader module
   *
   * If the module contains a compute shader that uses cooperative
   * matrix features, a local copy of the code will be created so
   * it can be patched at pipeline compile time.
   * \param [in] pCreateInfo Shader module create info
   * \param [in] pAllocator Allocation callbacks
   * \param [out] pShaderModule Shader module handle
   * \returns Return value from the host
   */
  VkResult createShaderModule(
    const VkShaderModuleCreateInfo*         pCreateInfo,
    const VkAllocationCallbacks*            pAllocator,
          VkShaderModule*                   pShaderModule) {
    VkResult vr = m_dispatch->CreateShaderModule(m_handle, pCreateInfo, pAllocator, pShaderModule);

    if (vr)
      return vr;

    if (shaderUsesCoopmat(*pCreateInfo))
      m_shaderModules.create(*pShaderModule, pCreateInfo);

    return vr;
  }

  /**
   * \brief Destroys shader module
   *
   * \param [in] shaderModule Module handle
   * \param [in] pAllocator Allocation callbacks
   */
  void destroyShaderModule(
          VkShaderModule                    shaderModule,
    const VkAllocationCallbacks*            pAllocator) {
    m_shaderModules.erase(shaderModule);

    m_dispatch->DestroyShaderModule(m_handle, shaderModule, pAllocator);
  }


  /**
   * \brief Creates a single compute pipeline
   *
   * Patches shader code as necessary.
   * \param [in] pipelineCache Pipeline cache
   * \param [in] pCreateInfo Pipeline create info
   * \param [in] pAllocator Allocation callbacks
   * \param [out] pPipeline The pipeline
   * \param [out] pResult Per-pipeline result
   */
  bool createComputePipeline(
          VkPipelineCache                     pipelineCache,
    const VkComputePipelineCreateInfo*        pCreateInfo,
    const VkAllocationCallbacks*              pAllocator,
          VkPipeline*                         pPipeline,
          VkResult*                           pResult) {
    VkComputePipelineCreateInfo createInfo = *pCreateInfo;
    createInfo.basePipelineHandle = VK_NULL_HANDLE;
    createInfo.basePipelineIndex = -1;

    /* Find actual pipeline create flags */
    VkPipelineCreateFlags2 createFlags2 = createInfo.flags;

    if (auto* chainedFlags = vkroots::FindInChain<VkPipelineCreateFlags2CreateInfo>(&createInfo))
      createFlags2 = chainedFlags->flags;

    /* Check if a specific subgroup size is required. If not, use the
     * smallest available subgroup size that is at least 4. We need to
     * know the actual subgroup size at compile time. */
    uint32_t subgroupSize = std::max(4u, m_physicalDeviceInfo->getProperties().vk13.minSubgroupSize);

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo requiredSubgroupSizeOut = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO };
    requiredSubgroupSizeOut.requiredSubgroupSize = subgroupSize;

    if (auto requiredSubgroupSizeIn = vkroots::FindInChain<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo>(&createInfo)) {
      subgroupSize = requiredSubgroupSizeIn->requiredSubgroupSize;
    } else {
      /* More api-level constness memes */
      requiredSubgroupSizeOut.pNext = const_cast<void*>(createInfo.stage.pNext);
      createInfo.stage.pNext = &requiredSubgroupSizeOut;
    }

    if (createInfo.stage.module) {
      VkShaderModule tmpModule = VK_NULL_HANDLE;

      auto moduleInfo = m_shaderModules.find(createInfo.stage.module);

      if (moduleInfo) {
        auto patchedCode = lowerCoopmat(moduleInfo->getCreateInfo(),
          createInfo.stage.pSpecializationInfo, subgroupSize);

        VkShaderModuleCreateInfo moduleCreateInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        moduleCreateInfo.codeSize = patchedCode.size() * sizeof(uint32_t);
        moduleCreateInfo.pCode = patchedCode.data();

        if (auto vr = m_dispatch->CreateShaderModule(m_handle, &moduleCreateInfo, pAllocator, &tmpModule))
          return vr;

        createInfo.stage.module = tmpModule;
      }

      *pResult = m_dispatch->CreateComputePipelines(m_handle,
        pipelineCache, 1, &createInfo, pAllocator, pPipeline);

      if (tmpModule)
        m_dispatch->DestroyShaderModule(m_handle, tmpModule, pAllocator);
    } else if (auto* chainedCode = const_cast<VkShaderModuleCreateInfo*>(vkroots::FindInChain<VkShaderModuleCreateInfo>(&createInfo.stage))) {
      VkShaderModuleCreateInfo origCode = *chainedCode;

      auto patchedCode = lowerCoopmat(origCode,
        createInfo.stage.pSpecializationInfo, subgroupSize);
      chainedCode->codeSize = patchedCode.size() * sizeof(uint32_t);
      chainedCode->pCode = patchedCode.data();

      *pResult = m_dispatch->CreateComputePipelines(m_handle,
        pipelineCache, 1, &createInfo, pAllocator, pPipeline);

      /* Restore original structure */
      *chainedCode = origCode;
    } else {
      /* No code, probably uses shader module identifier? */
      *pResult = VK_PIPELINE_COMPILE_REQUIRED;
    }

    return !pResult || !(createFlags2 & VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT);
  }


  /**
   * \brief Assigns device dispatch function
   * \param [in] dispatch Dispatcher
   */
  void setDispatch(const vkroots::VkDeviceDispatch* dispatch) {
    m_dispatch = dispatch;
  }

private:

  const vkroots::VkDeviceDispatch*  m_dispatch  = nullptr;
  VkDevice                          m_handle    = VK_NULL_HANDLE;

  std::shared_ptr<PhysicalDeviceInfo> m_physicalDeviceInfo;

  ObjectMap<VkShaderModule, ShaderModuleInfo> m_shaderModules;

};

static ObjectMap<VkDevice, DeviceInfo> g_devices;

auto findDevice(const vkroots::VkDeviceDispatch* dispatch, VkDevice device) {
  auto dev = g_devices.find(device);

  if (dev)
    dev->setDispatch(dispatch);

  return dev;
}


class VkInstanceOverrides {

public:

  static void DestroyInstance(
    const vkroots::VkInstanceDispatch*        pDispatch,
          VkInstance                          instance,
    const VkAllocationCallbacks*              pAllocator) {
    /* Destroy adapter infos before the handles can be reused */
    uint32_t adapterCount = 0u;

    if (!pDispatch->EnumeratePhysicalDevices(instance, &adapterCount, nullptr)) {
      std::vector<VkPhysicalDevice> adapters(adapterCount);

      if (!pDispatch->EnumeratePhysicalDevices(instance, &adapterCount, adapters.data())) {
        for (auto adapter : adapters)
          g_physicalDevices.erase(adapter);
      }
    }

    pDispatch->DestroyInstance(instance, pAllocator);
  }


  static VkResult EnumerateDeviceExtensionProperties(
    const vkroots::VkInstanceDispatch*        pDispatch,
          VkPhysicalDevice                    physicalDevice,
    const char*                               pLayerName,
          uint32_t*                           pPropertyCount,
          VkExtensionProperties*              pProperties) {
    auto device = getOrCreatePhysicalDevice(physicalDevice, pDispatch);

    if (!device->isLayerEnabled())
      return pDispatch->EnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties);

    return device->enumerateExtensionProperties(pLayerName, pPropertyCount, pProperties);
  }


  static void GetPhysicalDeviceFeatures2(
    const vkroots::VkInstanceDispatch*        pDispatch,
          VkPhysicalDevice                    physicalDevice,
          VkPhysicalDeviceFeatures2*          pFeatures) {
    auto device = getOrCreatePhysicalDevice(physicalDevice, pDispatch);

    if (!device->isLayerEnabled()) {
      pDispatch->GetPhysicalDeviceFeatures2(physicalDevice, pFeatures);
      return;
    }

    device->getPhysicalDeviceFeatures(pFeatures);
  }


  static void GetPhysicalDeviceFeatures2KHR(
    const vkroots::VkInstanceDispatch*        pDispatch,
          VkPhysicalDevice                    physicalDevice,
          VkPhysicalDeviceFeatures2KHR*       pFeatures) {
    auto device = getOrCreatePhysicalDevice(physicalDevice, pDispatch);

    if (!device->isLayerEnabled()) {
      pDispatch->GetPhysicalDeviceFeatures2KHR(physicalDevice, pFeatures);
      return;
    }

    device->getPhysicalDeviceFeatures(pFeatures);
  }


  static void GetPhysicalDeviceProperties2(
    const vkroots::VkInstanceDispatch*        pDispatch,
          VkPhysicalDevice                    physicalDevice,
          VkPhysicalDeviceProperties2*        pProperties) {
    auto device = getOrCreatePhysicalDevice(physicalDevice, pDispatch);

    if (!device->isLayerEnabled()) {
      pDispatch->GetPhysicalDeviceProperties2(physicalDevice, pProperties);
      return;
    }

    device->getPhysicalDeviceProperties(pProperties);
  }


  static void GetPhysicalDeviceProperties2KHR(
    const vkroots::VkInstanceDispatch*        pDispatch,
          VkPhysicalDevice                    physicalDevice,
          VkPhysicalDeviceProperties2*        pProperties) {
    auto device = getOrCreatePhysicalDevice(physicalDevice, pDispatch);

    if (!device->isLayerEnabled()) {
      pDispatch->GetPhysicalDeviceProperties2KHR(physicalDevice, pProperties);
      return;
    }

    device->getPhysicalDeviceProperties(pProperties);
  }


  static VkResult CreateDevice(
    const vkroots::VkInstanceDispatch*        pDispatch,
          VkPhysicalDevice                    physicalDevice,
    const VkDeviceCreateInfo*                 pCreateInfo,
    const VkAllocationCallbacks*              pAllocator,
          VkDevice*                           pDevice) {
    auto device = getOrCreatePhysicalDevice(physicalDevice, pDispatch);

    if (!device->isLayerEnabled())
      return pDispatch->CreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);

    /* Ignore if the device does not enable cooperative matrix features */
    auto coopmatFeatures = vkroots::FindInChain<VkPhysicalDeviceCooperativeMatrixFeaturesKHR>(pCreateInfo);

    if (!coopmatFeatures || !coopmatFeatures->cooperativeMatrix)
      return pDispatch->CreateDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);

    const auto& supported = device->getFeatures();

    /* Need to be able to enable some extra features */
    VkDeviceCreateInfo createInfo = *pCreateInfo;

    auto [vk12_in, vk12_header] = vkroots::RemoveFromChain<VkPhysicalDeviceVulkan12Features>(&createInfo);
    VkPhysicalDeviceVulkan12Features vk12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };

    if (vk12_in) {
      vk12 = *vk12_in;
      vk12.pNext = nullptr;
    }

    vk12.subgroupBroadcastDynamicId = VK_TRUE;

    auto [vk13_in, vk13_header] = vkroots::RemoveFromChain<VkPhysicalDeviceVulkan13Features>(&createInfo);
    VkPhysicalDeviceVulkan13Features vk13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };

    if (vk13_in) {
      vk13 = *vk13_in;
      vk13.pNext = nullptr;
    }

    vk13.subgroupSizeControl = VK_TRUE;
    vk13.computeFullSubgroups = VK_TRUE;
    vk13.shaderIntegerDotProduct = supported.vk13.shaderIntegerDotProduct;

    /* api-side constness issues make this annoying */
    vk12.pNext = &vk13;
    vk13.pNext = const_cast<void*>(createInfo.pNext);
    createInfo.pNext = &vk12;

    VkResult vr = pDispatch->CreateDevice(physicalDevice, &createInfo, pAllocator, pDevice);

    /* Restore original pNext chain */
    if (vk13_in) vk13_header->pNext = reinterpret_cast<VkBaseOutStructure*>(vk13_in);
    if (vk12_in) vk12_header->pNext = reinterpret_cast<VkBaseOutStructure*>(vk12_in);

    if (vr)
      return vr;

    g_devices.create(*pDevice, *pDevice, physicalDevice, &createInfo);
    return VK_SUCCESS;
  }

};

class VkPhysicalDeviceOverrides {

public:

  static VkResult GetPhysicalDeviceCooperativeMatrixPropertiesKHR(
    const vkroots::VkPhysicalDeviceDispatch*  pDispatch,
          VkPhysicalDevice                    physicalDevice,
          uint32_t*                           pPropertyCount,
          VkCooperativeMatrixPropertiesKHR*   pProperties) {
    auto device = findPhysicalDevice(physicalDevice);

    if (!device || !device->isLayerEnabled())
      return pDispatch->GetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, pPropertyCount, pProperties);

    return device->getCooperativeMatrixProperties(pPropertyCount, pProperties);
  }

};


class VkDeviceOverrides {

public:

  static void DestroyDevice(
    const vkroots::VkDeviceDispatch*          pDispatch,
          VkDevice                            device,
    const VkAllocationCallbacks*              pAllocator) {
    g_devices.erase(device);

    pDispatch->DestroyDevice(device, pAllocator);
  }


  static VkResult CreateShaderModule(
    const vkroots::VkDeviceDispatch*          pDispatch,
          VkDevice                            device,
    const VkShaderModuleCreateInfo*           pCreateInfo,
    const VkAllocationCallbacks*              pAllocator,
          VkShaderModule*                     pShaderModule) {
    auto layered = findDevice(pDispatch, device);

    if (!layered)
      return pDispatch->CreateShaderModule(device, pCreateInfo, pAllocator, pShaderModule);

    return layered->createShaderModule(pCreateInfo, pAllocator, pShaderModule);
  }


  static void DestroyShaderModule(
    const vkroots::VkDeviceDispatch*          pDispatch,
          VkDevice                            device,
          VkShaderModule                      shaderModule,
    const VkAllocationCallbacks*              pAllocator) {
    auto layered = findDevice(pDispatch, device);

    if (!layered) {
      pDispatch->DestroyShaderModule(device, shaderModule, pAllocator);
      return;
    }

    layered->destroyShaderModule(shaderModule, pAllocator);
  }


  static VkResult CreateComputePipelines(
    const vkroots::VkDeviceDispatch*          pDispatch,
          VkDevice                            device,
          VkPipelineCache                     pipelineCache,
          uint32_t                            createInfoCount,
    const VkComputePipelineCreateInfo*        pCreateInfos,
    const VkAllocationCallbacks*              pAllocator,
          VkPipeline*                         pPipelines) {
    auto layered = findDevice(pDispatch, device);

    if (!layered) {
      return pDispatch->CreateComputePipelines(device, pipelineCache,
        createInfoCount, pCreateInfos, pAllocator, pPipelines);
    }

    VkResult vr = VK_SUCCESS;

    for (uint32_t i = 0; i < createInfoCount; i++) {
      pPipelines[i] = VK_NULL_HANDLE;

      VkResult pipelineVr = VK_SUCCESS;

      if (!layered->createComputePipeline(pipelineCache, &pCreateInfos[i], pAllocator, &pPipelines[i], &pipelineVr))
        return pipelineVr;

      if (pipelineVr)
        vr = pipelineVr;
    }

    return vr;
  }

};

}

VKROOTS_DEFINE_LAYER_INTERFACES(CoopmatLayer::VkInstanceOverrides,
                                CoopmatLayer::VkPhysicalDeviceOverrides,
                                CoopmatLayer::VkDeviceOverrides);
