#pragma once

#include "vkroots.h"

namespace CoopmatLayer {

/**
 * \brief Lowers cooperative matrix code to basic SPIR-V functionality
 *
 * \param [in] pCreateInfo Module create info pointing to shader code
 * \param [in] subgroupSize Selected subgroup size
 * \returns Patched SPIR-V code, or empty vector on error
 */
std::vector<uint32_t> lowerCoopmat(
  const VkShaderModuleCreateInfo&               pCreateInfo,
  const VkSpecializationInfo*                   pSpecInfo,
        uint32_t                                subgroupSize);

/**
 * \brief Checks whether shader uses coopmat features
 *
 * \param [in] pCreateInfo Module create info pointing to shader code
 * \returns \c true if the shader uses cooperative matrix code
 */
bool shaderUsesCoopmat(
  const VkShaderModuleCreateInfo&               pCreateInfo);

}
