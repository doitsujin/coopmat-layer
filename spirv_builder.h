#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "spirv_reader.h"
#include "vkroots.h"

namespace CoopmatLayer {

/** SPIR-V type info */
struct SpirvTypeInfo {
  spv::Op             op            = spv::OpNop;
  uint32_t            baseTypeId    = 0u;
  VkComponentTypeKHR  scalarType    = VK_COMPONENT_TYPE_MAX_ENUM_KHR;
  uint32_t            vectorSize    = 0u; /* > 1 for vector type */
  uint32_t            matrixCols    = 0u; /* > 1 for matrix type */
  uint32_t            arraySize     = 0u; /* > 0 for arrays */
  spv::StorageClass   storageClass  = spv::StorageClassMax;

  auto operator <=> (const SpirvTypeInfo&) const = default;
};

/** SPIR-V instruction builder */
class SpirvInstructionBuilder {

public:

  SpirvInstructionBuilder() = default;

  SpirvInstructionBuilder(spv::Op op, uint32_t resultType, uint32_t resultId)
  : m_op(op), m_len(1u) {
    /* Manually set up opcode token */
    m_args[0u] = uint32_t(op) | (1u << spv::WordCountShift);

    bool hasResult = false;
    bool hasResultType = false;

    spv::HasResultAndType(op, &hasResult, &hasResultType);

    if (hasResultType)
      m_typeIdx = 1u;
    if (hasResult)
      m_idIdx = m_typeIdx + 1u;

    if (resultType)
      add(resultType);
    if (resultId)
      add(resultId);
  }

  SpirvInstructionBuilder(spv::Op op)
  : SpirvInstructionBuilder(op, 0u, 0u) { }

  SpirvInstructionBuilder(SpirvInstructionReader ins)
  : SpirvInstructionBuilder(ins.op()) {
    for (uint32_t i = 1u; i < ins.len(); i++)
      add(ins.arg(i));
  }

  /** Opcode */
  spv::Op op() const {
    return m_op;
  }

  /** Length */
  uint32_t len() const {
    return m_len;
  }

  /** Result ID */
  uint32_t id() const {
    return m_idIdx ? arg(m_idIdx) : 0u;
  }

  /** Result type */
  uint32_t typeId() const {
    return m_typeIdx ? arg(m_typeIdx) : 0u;
  }

  /** Adds literal dword argument */
  void add(uint32_t arg) {
    uint32_t n = m_len++;

    if (n < m_args.size()) {
      m_args[n] = arg;
      m_args[0] += 1u << spv::WordCountShift;
    } else if (n > m_args.size()) {
      m_long.push_back(arg);
      m_long.front() += 1u << spv::WordCountShift;
    } else {
      m_long.reserve(2u * m_args.size());

      for (size_t i = 0u; i < m_args.size(); i++)
        m_long.push_back(m_args[i]);

      m_long.push_back(arg);
      m_long.front() += 1u << spv::WordCountShift;
    }
  }

  /** Adds multiple arguments */
  template<typename T1, typename T2, typename... Tn>
  void add(T1 arg, T2 next, Tn... pack) {
    add(arg);
    add(next, pack...);
  }

  void add() { }

  /** Adds float argument as dword */
  void add(float arg) {
    uint32_t dword = 0u;

    std::memcpy(&dword, &arg, sizeof(dword));
    add(dword);
  }

  /** Adds quad-word argument */
  void add(uint64_t arg) {
    add(uint32_t(arg));
    add(uint32_t(arg >> 32u));
  }

  /** Adds double argument as wword */
  void add(double arg) {
    uint32_t qword = 0u;

    std::memcpy(&qword, &arg, sizeof(qword));
    add(qword);
  }

  /** Adds string argument */
  void add(const char* str) {
    size_t len = std::strlen(str);
    addString(str, len + 1u);
  }

  /** Adds string argument */
  void add(const std::string& str) {
    addString(str.c_str(), str.size() + 1u);
  }

  /** Retrieves argument, 0 is the opcode token */
  uint32_t arg(uint32_t index) const {
    if (index < m_long.size())
      return m_long[index];
    else if (index < m_args.size())
      return m_args[index];
    else
      return 0u;
  }

  /** Sets argument */
  void set(uint32_t index, uint32_t arg) {
    if (index < m_long.size())
      m_long[index] = arg;
    else if (index < m_args.size())
      m_args[index] = arg;
  }

  /** Queries string argument at given location */
  std::string str(uint32_t index) const {
    std::string result;

    /* Strings are little endian */
    for (uint32_t i = index; i < len(); i++) {
      uint32_t dword = arg(i);

      std::array<unsigned char, sizeof(uint32_t)> chars;
      std::memcpy(chars.data(), &dword, chars.size());

      for (uint32_t j = 0u; j < chars.size(); j++) {
        if (!chars[j])
          return result;

        result.push_back(chars[j]);
      }
    }

    return result;
  }

  /**
   * \brief Adds instruction to stream
   * \param [out] code Instruction stream
   */
  void push(std::vector<uint32_t>& code) const {
    if (m_long.empty()) {
      for (uint32_t i = 0; i < m_len; i++)
        code.push_back(m_args[i]);
    } else {
      for (uint32_t i = 0; i < m_len; i++)
        code.push_back(m_long[i]);
    }
  }

private:

  spv::Op                   m_op        = { };
  uint8_t                   m_typeIdx   = 0u;
  uint8_t                   m_idIdx     = 0u;
  uint32_t                  m_len       = 0u;

  std::array<uint32_t, 8>   m_args      = { };
  std::vector<uint32_t>     m_long;

  void addString(const char* str, size_t len) {
    uint32_t dword = 0u;

    for (size_t i = 0u; i + sizeof(dword) <= len; i += sizeof(dword)) {
      std::memcpy(&dword, &str[i], sizeof(dword));
      add(dword);
    }

    uint32_t rem = len % sizeof(dword);

    if (rem) {
      dword = 0u;

      std::memcpy(&dword, &str[len - rem], rem);
      add(dword);
    }
  }

};


/** SPIR-V builder */
class SpirvBuilder {

public:

  SpirvBuilder(
          SpirvHeader                   header,
    const VkSpecializationInfo*         spec);

  ~SpirvBuilder();

  /**
   * \brief Allocates an ID
   * \returns New ID
   */
  uint32_t allocId() {
    return m_header.boundIds++;
  }

  /**
   * \brief Adds existing instruction
   *
   * Some processing may be performed in order to
   * assign the instruction to the correct block
   * and to capture metadata.
   * \param [in] ins Instruction
   * \returns Instruction ID
   */
  uint32_t addIns(const SpirvInstructionBuilder& ins);

  /**
   * \brief Helper to build basic ops
   *
   * \param [in] op Opcode
   * \param [in] typeId Result type ID
   * \param [in] args Operands
   * \returns Result ID
   */
  template<typename... Args>
  uint32_t op(spv::Op op, uint32_t typeId, Args... args) {
    bool hasResult = false;
    bool hasResultType = false;

    spv::HasResultAndType(op, &hasResult, &hasResultType);

    uint32_t id = hasResult ? allocId() : 0u;
    SpirvInstructionBuilder builder(op, typeId, id);
    builder.add(args...);
    return addIns(builder);
  }

  /**
   * \brief Extracts code
   * \returns Code
   */
  std::vector<uint32_t> finalize() const;

  /**
   * \brief Retrieves source name
   * \returns Source name, if any
   */
  std::string getSourceName() const;

  /**
   * \brief Evaluates constant expression
   *
   * \param [in] id ID to evaluate
   * \returns Constant value as 32-bit integer, or
   *    \c nullopt if the expression isn't constant.
   */
  std::optional<uint32_t> evaluateConstant(
          uint32_t                      id) const;

  /**
   * \brief Queries type info
   *
   * \param [in] typeId Type ID
   * \returns Type info
   */
  SpirvTypeInfo getTypeInfo(
          uint32_t                      id) const;

  /**
   * \brief Gets or declares basic scalar or vector type
   *
   * \param [in] scalarType Component type
   * \param [in] vectorSize Component count
   * \returns ID of basic type
   */
  uint32_t defVectorType(
          VkComponentTypeKHR            scalarType,
          uint32_t                      vectorSize);

  /**
   * \brief Gets or declares boolean scalar or vector type
   *
   * \param [in] vectorSize Component count
   * \returns ID of bool type
   */
  uint32_t defBoolType(
          uint32_t                      vectorSize);

  /**
   * \brief Gets or declares a pointer type
   *
   * \param [in] baseType Base type
   * \param [in] storageClass Storage class
   * \returns Pointer type ID
   */
  uint32_t defPointerType(
          uint32_t                      baseType,
          spv::StorageClass             storageClass);

  /**
   * \brief Gets or declares a type
   *
   * \param [in] instruction Type definition
   * \returns ID of basic type
   */
  uint32_t defType(
    const SpirvInstructionBuilder&      ins);

  /**
   * \brief Defines unique uint32 constant
   *
   * \param [in] value Constant value
   * \returns Constant ID
   */
  uint32_t defConstUint32(
          uint32_t                      value);

  /**
   * \brief Defines unique null constant for a given type
   *
   * \param [in] typeId Type ID
   * \returns Constant ID
   */
  uint32_t defNullConst(
          uint32_t                      typeId);

  /**
   * \brief Defines unique constant
   *
   * \param [in] ins Instruction
   * \returns Constant ID
   */
  uint32_t defConst(
    const SpirvInstructionBuilder&      ins);

  /**
   * \brief Removes names and decorations associated with an ID
   * \param [in] id ID for which to remove metadata
   */
  void removeMetadataForId(
          uint32_t                      id);

  /**
   * \brief Sets debug name for a given ID
   *
   * \param [in] id The ID
   * \param [in] name Debug name
   */
  void setDebugName(
          uint32_t                      id,
          std::string                   name);

  /**
   * \brief Queries type ID of an operand
   *
   * \param [in] id Operand ID
   * \returns Type ID
   */
  uint32_t getOperandTypeId(
          uint32_t                      id) const;

  /**
   * \brief Queries definition of an operand
   *
   * \param [in] id Operand ID
   * \returns Type ID
   */
  SpirvInstructionBuilder getOperandDefinition(
          uint32_t                      id) const;

  /**
   * \brief Traverses access chain to find subtype
   *
   * \param [in] typeId Aggregate type ID
   * \param [in] indexId Index ID
   */
  uint32_t getMemberTypeId(
          uint32_t                      typeId,
          uint32_t                      indexId);

  /**
   * \brief Adds variable to entry point interfaces
   * \param [in] varId Variable Id
   */
  void registerVariable(
          uint32_t                      varId);

  /**
   * \brief Wraps object in temporary function-scope variable
   *
   * Useful to create function parameters
   * \param [in] typeId Parameter type
   * \param [in] operandId Value type
   * \returns Variable ID
   */
  uint32_t wrap(
          uint32_t                      typeId,
          uint32_t                      operandId);

  /**
   * \brief Emits arbitrary type conversion
   *
   * Will emit either OpI/S/FConvert as necessary.
   * \param [in] dstTypeId Type to convert to
   * \param [in] srcTypeId Type to convert from
   * \param [in] operandId Operand to convert
   * \returns Converted operand
   */
  uint32_t convert(
          uint32_t                      dstTypeId,
          uint32_t                      operandId);

  /**
   * \brief Emits arbitrary type bitcasting
   *
   * Omits the instruction if both types are the same.
   * \param [in] dstTypeId Type to convert to
   * \param [in] operandId Operand to convert
   * \returns Converted operand
   */
  uint32_t bitcast(
          uint32_t                      dstTypeId,
          uint32_t                      operandId);

  /**
   * \brief Finds applicable decoration
   *
   * \param [in] id Object ID
   * \param [in] member Member index, pass -1 to ignore
   * \param [in] decoration Decoration type
   * \returns Decoration value if applicable, or 0 if
   *    the decoration exists for the given ID but has
   *    no value, or \c nullopt if the decoration does
   *    not exist for the given ID.
   */
  std::optional<uint32_t> getDecoration(
          uint32_t                      id,
          int32_t                       member,
          spv::Decoration               decoration) const;

  /**
   * \brief Queries debug name for an object
   *
   * \param [in] id Object ID
   * \returns Debug name, if any
   */
  std::string getName(
          uint32_t                      id) const;

private:

  SpirvHeader                           m_header;
  const VkSpecializationInfo*           m_spec = nullptr;

  uint32_t                              m_functionDepth = 0u;
  std::vector<uint32_t>                 m_functionIds;

  std::set<spv::Capability>             m_capabilities;

  std::set<std::string>                 m_extensions;
  std::map<uint32_t, std::string>       m_extImports;

  SpirvInstructionBuilder               m_memoryModel;

  std::map<uint32_t, SpirvInstructionBuilder> m_entryPoints;

  std::multimap<uint32_t, SpirvInstructionBuilder> m_executionModes;

  std::map<uint32_t, std::string>             m_strings;
  std::map<uint32_t, SpirvInstructionBuilder> m_names;

  SpirvInstructionBuilder               m_source;

  std::multimap<uint32_t, SpirvInstructionBuilder> m_decorations;

  std::unordered_map<uint32_t, SpirvInstructionBuilder> m_instructionsForId;
  std::unordered_map<uint32_t, uint32_t> m_typesForId;

  std::unordered_map<uint32_t, uint32_t> m_uintConstantIds;
  std::unordered_map<uint64_t, uint32_t> m_vectorTypeIds;

  std::unordered_multimap<uint32_t, SpirvInstructionBuilder> m_functionVars;
  std::unordered_map<uint32_t, SpirvInstructionBuilder>  m_declarations;

  std::vector<SpirvInstructionBuilder>  m_codeNewFunctions;
  std::vector<SpirvInstructionBuilder>  m_codeOldFunctions;

  void pushFunctionCode(
          std::vector<uint32_t>&        code,
    const std::vector<SpirvInstructionBuilder>& instructions) const;

  SpirvInstructionBuilder evaluateConstantExpression(
    const SpirvInstructionBuilder&      expr,
    const VkSpecializationInfo*         spec) const;

  static bool canEmitDeclaration(
    const SpirvInstructionBuilder&      ins,
    const std::unordered_set<uint32_t>& emitted);

  static bool hasDeclaration(
    const std::unordered_set<uint32_t>& emitted,
          uint32_t                      id);

};

}
