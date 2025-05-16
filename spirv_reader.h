#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#define SPV_ENABLE_UTILITY_CODE 1
#include <spirv/unified1/spirv.hpp>

namespace CoopmatLayer {

/* SPIR-V module header */
struct SpirvHeader {
  uint32_t magicNumber  = 0u;
  uint32_t version      = 0u;
  uint32_t generator    = 0u;
  uint32_t boundIds     = 0u;
  uint32_t schema       = 0u;
};


/** SPIR-V instruction reader */
class SpirvInstructionReader {

public:

  SpirvInstructionReader() = default;

  SpirvInstructionReader(const uint32_t* pCode)
  : m_code(pCode) { }

  /** Decodes opcode */
  spv::Op op() const {
    return spv::Op(m_code[0] & spv::OpCodeMask);
  }

  /** Decodes instruction length */
  uint32_t len() const {
    return m_code[0] >> spv::WordCountShift;
  }

  /** Queries argument. The opcode token is index 0. */
  uint32_t arg(uint32_t index) const {
    return index < len() ? m_code[index] : 0u;
  }

  /** Checks whether object is valid */
  explicit operator bool () const {
    return m_code != nullptr;
  }

private:

  const uint32_t* m_code = nullptr;

};


/** SPIR-V module reader */
class SpirvReader {

public:

  SpirvReader() = default;

  SpirvReader(size_t size, const uint32_t* code)
  : m_code(code), m_size(size / sizeof(uint32_t)) { }

  /* Reads SPIR-V header */
  SpirvHeader getHeader() const {
    if (m_size < sizeof(SpirvHeader) / sizeof(uint32_t))
      return SpirvHeader();

    SpirvHeader header  = { };
    header.magicNumber  = m_code[0];
    header.version      = m_code[1];
    header.generator    = m_code[2];
    header.boundIds     = m_code[3];
    header.schema       = m_code[4];
    return header;
  }

  /* Reads next instruction */
  SpirvInstructionReader readInstruction() {
    if (m_offset >= m_size)
      return SpirvInstructionReader();

    SpirvInstructionReader ins(m_code + m_offset);
    m_offset += ins.len();

    if (m_offset > m_size)
      return SpirvInstructionReader();

    return ins;
  }

private:

  const uint32_t* m_code    = 0u;

  uint32_t        m_offset  = sizeof(SpirvHeader) / sizeof(uint32_t);
  uint32_t        m_size    = 0u;

};

}
