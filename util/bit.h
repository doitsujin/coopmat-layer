#include <cstddef>
#include <cstdint>

namespace util {

inline uint32_t tzcnt(uint32_t value) {
  for (uint32_t i = 0u; i < 32u; i++) {
    if (value & (1u << i))
      return i;
  }

  return 32u;
}

}
