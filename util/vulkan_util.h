#pragma once

#include "../vkroots.h"

namespace util {

/* Computes byte size of a scalar type */
inline size_t getComponentSize(VkComponentTypeKHR type) {
  switch (type) {
    case VK_COMPONENT_TYPE_SINT8_KHR:
    case VK_COMPONENT_TYPE_UINT8_KHR:
    case VK_COMPONENT_TYPE_SINT8_PACKED_NV:
    case VK_COMPONENT_TYPE_UINT8_PACKED_NV:
    case VK_COMPONENT_TYPE_FLOAT_E4M3_NV:
    case VK_COMPONENT_TYPE_FLOAT_E5M2_NV:
      return 1u;

    case VK_COMPONENT_TYPE_FLOAT16_KHR:
    case VK_COMPONENT_TYPE_SINT16_KHR:
    case VK_COMPONENT_TYPE_UINT16_KHR:
      return 2u;

    case VK_COMPONENT_TYPE_FLOAT32_KHR:
    case VK_COMPONENT_TYPE_SINT32_KHR:
    case VK_COMPONENT_TYPE_UINT32_KHR:
      return 4u;

    case VK_COMPONENT_TYPE_FLOAT64_KHR:
    case VK_COMPONENT_TYPE_SINT64_KHR:
    case VK_COMPONENT_TYPE_UINT64_KHR:
      return 8u;

    default:
      return 0u;
  }
}

/* Checks whether a component type is a floating point type */
inline bool isFloatType(VkComponentTypeKHR type) {
  return type == VK_COMPONENT_TYPE_FLOAT16_KHR
      || type == VK_COMPONENT_TYPE_FLOAT32_KHR
      || type == VK_COMPONENT_TYPE_FLOAT64_KHR
      || type == VK_COMPONENT_TYPE_FLOAT_E4M3_NV
      || type == VK_COMPONENT_TYPE_FLOAT_E5M2_NV;
}

/* Checks whether a component type is signed */
inline bool isSignedType(VkComponentTypeKHR type) {
  return type == VK_COMPONENT_TYPE_SINT8_KHR
      || type == VK_COMPONENT_TYPE_SINT16_KHR
      || type == VK_COMPONENT_TYPE_SINT32_KHR
      || type == VK_COMPONENT_TYPE_SINT64_KHR
      || isFloatType(type);
}

/* Returns signed type for a given unsigned integer type */
inline VkComponentTypeKHR getSignedType(VkComponentTypeKHR type) {
  switch (type) {
    case VK_COMPONENT_TYPE_UINT8_KHR:       return VK_COMPONENT_TYPE_SINT8_KHR;
    case VK_COMPONENT_TYPE_UINT16_KHR:      return VK_COMPONENT_TYPE_SINT16_KHR;
    case VK_COMPONENT_TYPE_UINT32_KHR:      return VK_COMPONENT_TYPE_SINT32_KHR;
    case VK_COMPONENT_TYPE_UINT64_KHR:      return VK_COMPONENT_TYPE_SINT64_KHR;
    case VK_COMPONENT_TYPE_UINT8_PACKED_NV: return VK_COMPONENT_TYPE_SINT8_PACKED_NV;
    default: return type;
  }
}

/* Returns unsigned type for a given signed integer type */
inline VkComponentTypeKHR getUnsignedType(VkComponentTypeKHR type) {
  switch (type) {
    case VK_COMPONENT_TYPE_SINT8_KHR:       return VK_COMPONENT_TYPE_UINT8_KHR;
    case VK_COMPONENT_TYPE_SINT16_KHR:      return VK_COMPONENT_TYPE_UINT16_KHR;
    case VK_COMPONENT_TYPE_SINT32_KHR:      return VK_COMPONENT_TYPE_UINT32_KHR;
    case VK_COMPONENT_TYPE_SINT64_KHR:      return VK_COMPONENT_TYPE_UINT64_KHR;
    case VK_COMPONENT_TYPE_SINT8_PACKED_NV: return VK_COMPONENT_TYPE_UINT8_PACKED_NV;
    default: return type;
  }
}

/* Returns corresponding 32-bit type for a given type */
inline VkComponentTypeKHR get32BitType(VkComponentTypeKHR type) {
  switch (type) {
    case VK_COMPONENT_TYPE_UINT8_KHR:
    case VK_COMPONENT_TYPE_UINT16_KHR:
    case VK_COMPONENT_TYPE_UINT32_KHR:
    case VK_COMPONENT_TYPE_UINT64_KHR:
    case VK_COMPONENT_TYPE_UINT8_PACKED_NV:
      return VK_COMPONENT_TYPE_UINT32_KHR;

    case VK_COMPONENT_TYPE_SINT8_KHR:
    case VK_COMPONENT_TYPE_SINT16_KHR:
    case VK_COMPONENT_TYPE_SINT32_KHR:
    case VK_COMPONENT_TYPE_SINT64_KHR:
    case VK_COMPONENT_TYPE_SINT8_PACKED_NV:
      return VK_COMPONENT_TYPE_SINT32_KHR;

    case VK_COMPONENT_TYPE_FLOAT16_KHR:
    case VK_COMPONENT_TYPE_FLOAT32_KHR:
    case VK_COMPONENT_TYPE_FLOAT64_KHR:
    case VK_COMPONENT_TYPE_FLOAT_E4M3_NV:
    case VK_COMPONENT_TYPE_FLOAT_E5M2_NV:
      return VK_COMPONENT_TYPE_FLOAT32_KHR;

    default:
      return type;
  }
}

/* Gets human-readable name for component type */
inline const char* getComponentTypeName(VkComponentTypeKHR type) {
  switch (type) {
    case VK_COMPONENT_TYPE_SINT8_KHR:       return "i8";
    case VK_COMPONENT_TYPE_SINT16_KHR:      return "i16";
    case VK_COMPONENT_TYPE_SINT32_KHR:      return "i32";
    case VK_COMPONENT_TYPE_SINT64_KHR:      return "i64";
    case VK_COMPONENT_TYPE_UINT8_KHR:       return "u8";
    case VK_COMPONENT_TYPE_UINT16_KHR:      return "u16";
    case VK_COMPONENT_TYPE_UINT32_KHR:      return "u32";
    case VK_COMPONENT_TYPE_UINT64_KHR:      return "u64";
    case VK_COMPONENT_TYPE_FLOAT16_KHR:     return "f16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR:     return "f32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR:     return "f64";
    case VK_COMPONENT_TYPE_FLOAT_E4M3_NV:   return "f8";
    case VK_COMPONENT_TYPE_FLOAT_E5M2_NV:   return "bf8";
    case VK_COMPONENT_TYPE_SINT8_PACKED_NV: return "i8_packed";
    case VK_COMPONENT_TYPE_UINT8_PACKED_NV: return "u8_packed";

    default:
      return "Unknown";
  }
}

}
