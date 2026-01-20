#include <iostream>

#include "./util/vulkan_util.h"

#include "spirv_builder.h"

namespace CoopmatLayer {

SpirvBuilder::SpirvBuilder(
        SpirvHeader                   header,
  const VkSpecializationInfo*         spec)
: m_header(header), m_spec(spec) {

}


SpirvBuilder::~SpirvBuilder() {


}


uint32_t SpirvBuilder::addIns(const SpirvInstructionBuilder& ins) {
  bool hasResult = false;
  bool hasResultType = false;

  spv::HasResultAndType(ins.op(), &hasResult, &hasResultType);

  if (hasResult && hasResultType)
    m_typesForId.insert({ ins.id(), ins.typeId() });

  if (hasResult)
    m_instructionsForId.insert({ ins.id(), ins });

  auto& functions = m_functionIds.size() > 1u
    ? m_codeNewFunctions
    : m_codeOldFunctions;

  switch (ins.op()) {
    case spv::OpCapability:
      m_capabilities.insert(spv::Capability(ins.arg(1u)));
      break;

    case spv::OpExtension:
      m_extensions.insert(ins.str(1u));
      break;

    case spv::OpExtInstImport:
      m_extImports.insert({ ins.arg(1u), ins.str(2u) });
      break;

    case spv::OpMemoryModel:
      m_memoryModel = ins;
      break;

    case spv::OpEntryPoint:
      m_entryPoints.insert({ ins.arg(2u), ins });
      break;

    case spv::OpExecutionMode:
    case spv::OpExecutionModeId:
      m_executionModes.insert({ ins.arg(1u), ins });
      break;

    case spv::OpSource:
      m_source = ins;
      break;

    case spv::OpString:
      m_strings.insert({ ins.arg(1u), ins.str(2u) });
      break;

    case spv::OpName:
    case spv::OpMemberName:
      m_names.insert({ ins.arg(1u), ins });
      break;

    case spv::OpDecorate:
    case spv::OpDecorateId:
    case spv::OpDecorateString:
    case spv::OpMemberDecorate:
    case spv::OpMemberDecorateString:
      m_decorations.insert({ ins.arg(1u), ins });
      break;

    case spv::OpDecorationGroup:
      /* no op */
      break;

    case spv::OpGroupDecorate: {
      auto range = m_decorations.equal_range(ins.arg(1u));
      std::vector<SpirvInstructionBuilder> decorations;

      for (auto i = range.first; i != range.second; i++)
        decorations.push_back(i->second);

      for (uint32_t i = 2u; i < ins.len(); i++) {
        for (const auto& d : decorations)
          m_decorations.insert({ ins.arg(i), d });
      }
    } break;

    case spv::OpGroupMemberDecorate: {
      auto range = m_decorations.equal_range(ins.arg(1u));
      std::vector<SpirvInstructionBuilder> decorations;

      for (auto i = range.first; i != range.second; i++)
        decorations.push_back(i->second);

      for (uint32_t i = 2u; i < ins.len(); i += 2u) {
        for (const auto& d : decorations) {
          SpirvInstructionBuilder builder(spv::OpMemberDecorate, 0u, 0u);
          builder.add(ins.arg(i));
          builder.add(ins.arg(i + 1u));

          for (uint32_t j = 2u; j < d.len(); j++)
            builder.add(d.arg(j));

          m_decorations.insert({ ins.arg(i), builder });
        }
      }
    } break;

    case spv::OpTypeBool:
    case spv::OpTypeInt:
    case spv::OpTypeFloat:
    case spv::OpTypeVector: {
      m_declarations.insert({ ins.id(), ins });

      SpirvTypeInfo typeInfo = getTypeInfo(ins.id());
      uint64_t code = uint64_t(typeInfo.scalarType) | (uint64_t(typeInfo.vectorSize) << 32u);
      m_vectorTypeIds.insert({ code, ins.id() });
    } break;

    case spv::OpConstant: {
      auto typeInfo = getTypeInfo(ins.typeId());

      if (typeInfo.op == spv::OpTypeInt && typeInfo.scalarType == VK_COMPONENT_TYPE_UINT32_KHR)
        m_uintConstantIds.insert({ ins.arg(3u), ins.id() });

      m_declarations.insert({ ins.id(), ins });
    } break;

    case spv::OpSpecConstantTrue:
    case spv::OpSpecConstantFalse:
    case spv::OpSpecConstant:
    case spv::OpSpecConstantComposite:
      /* TODO evaluate */
      m_declarations.insert({ ins.id(), ins });
      break;

    case spv::OpTypeVoid:
    case spv::OpTypeSampler:
    case spv::OpTypeOpaque:
    case spv::OpTypeRayQueryKHR:
    case spv::OpTypeAccelerationStructureKHR:
    case spv::OpTypeMatrix:
    case spv::OpTypeRuntimeArray:
    case spv::OpTypeImage:
    case spv::OpTypeSampledImage:
    case spv::OpTypeArray:
    case spv::OpTypeStruct:
    case spv::OpTypeFunction:
    case spv::OpTypeCooperativeMatrixKHR:
    case spv::OpTypeCooperativeMatrixNV:
    case spv::OpTypePointer:
    case spv::OpTypeForwardPointer:
    case spv::OpConstantNull:
    case spv::OpConstantTrue:
    case spv::OpConstantFalse:
    case spv::OpConstantComposite:
    case spv::OpSpecConstantOp:
    case spv::OpConstantCompositeReplicateEXT:
    case spv::OpSpecConstantCompositeReplicateEXT:
    case spv::OpUndef:
      m_declarations.insert({ ins.id(), ins });
      break;

    case spv::OpVariable: {
      if (spv::StorageClass(ins.arg(3u)) == spv::StorageClassFunction)
        m_functionVars.insert({ m_functionIds.back(), ins });
      else
        m_declarations.insert({ ins.id(), ins });
    } break;

    case spv::OpFunction:
      /* No more declarations after this */
      if (m_functionIds.empty())
        m_codeOldFunctions.push_back(ins);
      else
        m_codeNewFunctions.push_back(ins);

      m_functionIds.push_back(ins.id());
      break;

    case spv::OpFunctionEnd:
      functions.push_back(ins);

      m_functionIds.pop_back();
      break;

    case spv::OpSourceExtension:
    case spv::OpSourceContinued:
    case spv::OpModuleProcessed:
    case spv::OpLine:
    case spv::OpNoLine:
    case spv::OpNop:
      /* ignore these */
      break;

    default:
      if (m_functionIds.empty() && ins.id())
        m_declarations.insert({ ins.id(), ins });
      else
        functions.push_back(ins);
  }

  return ins.id();
}


std::vector<uint32_t> SpirvBuilder::finalize() const {
  std::vector<uint32_t> code = { };
  code.push_back(m_header.magicNumber);
  code.push_back(m_header.version);
  code.push_back(m_header.generator);
  code.push_back(m_header.boundIds);
  code.push_back(m_header.schema);

  for (auto cap : m_capabilities) {
    SpirvInstructionBuilder ins(spv::OpCapability);
    ins.add(uint32_t(cap));
    ins.push(code);
  }

  for (const auto& ext : m_extensions) {
    SpirvInstructionBuilder ins(spv::OpExtension);
    ins.add(ext);
    ins.push(code);
  }

  for (const auto& ext : m_extImports) {
    SpirvInstructionBuilder ins(spv::OpExtInstImport);
    ins.add(ext.first);
    ins.add(ext.second);
    ins.push(code);
  }

  m_memoryModel.push(code);

  for (const auto& ins : m_entryPoints)
    ins.second.push(code);

  for (const auto& ins : m_executionModes)
    ins.second.push(code);

  for (const auto& str : m_strings) {
    SpirvInstructionBuilder ins(spv::OpString);
    ins.add(str.first);
    ins.add(str.second);
    ins.push(code);
  }

  if (m_source.len())
    m_source.push(code);

  for (const auto& name : m_names)
    name.second.push(code);

  /* We un-gropup decoration groups already */
  for (const auto& ins : m_decorations)
    ins.second.push(code);

  /* Type and constant declarations are tricky since they may depend on
   * one another. Ensure that we emit everything in the correct order. */
  std::unordered_set<uint32_t> declarationsEmitted;

  bool progress;

  do {
    progress = false;

    for (const auto& ins : m_declarations) {
      if (declarationsEmitted.find(ins.first) == declarationsEmitted.end()
       && canEmitDeclaration(ins.second, declarationsEmitted)) {
        ins.second.push(code);
        declarationsEmitted.insert(ins.first);
        progress = true;
      }
    }
  } while (progress);

  /* Declarations that we don't understand go last in original order */
  for (const auto& ins : m_declarations) {
    if (declarationsEmitted.find(ins.first) == declarationsEmitted.end())
      ins.second.push(code);
  }

  pushFunctionCode(code, m_codeNewFunctions);
  pushFunctionCode(code, m_codeOldFunctions);
  return code;
}


std::string SpirvBuilder::getSourceName() const {
  uint32_t stringId = 0u;

  if (m_source.len() > 3u)
    stringId = m_source.arg(3u);

  auto entry = m_strings.find(stringId);

  if (entry == m_strings.end() && m_strings.size() == 1u)
    entry = m_strings.begin();

  if (entry == m_strings.end())
    return std::string();

  return entry->second;
}


void SpirvBuilder::pushFunctionCode(
        std::vector<uint32_t>&        code,
  const std::vector<SpirvInstructionBuilder>& instructions) const {
  uint32_t functionId = 0u;

  for (const auto& ins : instructions) {
    ins.push(code);

    switch (ins.op()) {
      case spv::OpFunction: {
        functionId = ins.id();
      } break;

      case spv::OpLabel: {
        if (functionId) {
          auto vars = m_functionVars.equal_range(functionId);

          for (auto v = vars.first; v != vars.second; v++)
            v->second.push(code);

          functionId = 0u;
        }
      } break;

      default:
        /* no op */;
    }
  }
}

std::optional<uint32_t> SpirvBuilder::evaluateConstant(
        uint32_t                      id) const {
  auto entry = m_declarations.find(id);

  if (entry == m_declarations.end())
    return std::nullopt;

  const auto& expr = entry->second;

  switch (expr.op()) {
    case spv::OpConstantNull:   return uint32_t(0u);
    case spv::OpConstantFalse:  return uint32_t(0u);
    case spv::OpConstantTrue:   return uint32_t(1u);
    case spv::OpConstant:       return uint32_t(expr.arg(3u));

    default:
      std::cerr << "Unhandled constant instruction " << uint32_t(expr.op()) << ": " << spv::OpToString(expr.op()) << std::endl;
      return std::nullopt;
  }
}


SpirvTypeInfo SpirvBuilder::getTypeInfo(
        uint32_t                      id) const {
  auto entry = m_declarations.find(id);

  if (entry == m_declarations.end()) {
    std::cerr << "Unknown type ID " << id << std::endl;
    return SpirvTypeInfo();
  }

  const auto& ins = entry->second;

  SpirvTypeInfo result = { };
  result.op = ins.op();

  switch (ins.op()) {
    case spv::OpTypeBool:
    case spv::OpTypeVoid:
      return result;

    case spv::OpTypeInt: {
      uint32_t width = ins.arg(2u);
      uint32_t sign = ins.arg(3u);

      switch (width) {
        case  8u: result.scalarType = sign ? VK_COMPONENT_TYPE_SINT8_KHR  : VK_COMPONENT_TYPE_UINT8_KHR;  break;
        case 16u: result.scalarType = sign ? VK_COMPONENT_TYPE_SINT16_KHR : VK_COMPONENT_TYPE_UINT16_KHR; break;
        case 32u: result.scalarType = sign ? VK_COMPONENT_TYPE_SINT32_KHR : VK_COMPONENT_TYPE_UINT32_KHR; break;
        case 64u: result.scalarType = sign ? VK_COMPONENT_TYPE_SINT64_KHR : VK_COMPONENT_TYPE_UINT64_KHR; break;
        default: ;
      }

      return result;
    }

    case spv::OpTypeFloat: {
      uint32_t width = ins.arg(2u);

      switch (width) {
        case  8u: result.scalarType = VK_COMPONENT_TYPE_FLOAT_E4M3_NV; break;
        case 16u: result.scalarType = VK_COMPONENT_TYPE_FLOAT16_KHR; break;
        case 32u: result.scalarType = VK_COMPONENT_TYPE_FLOAT32_KHR; break;
        case 64u: result.scalarType = VK_COMPONENT_TYPE_FLOAT64_KHR; break;
        default: ;
      }

      return result;
    }

    case spv::OpTypeVector: {
      result.baseTypeId = ins.arg(2u);

      auto base = getTypeInfo(result.baseTypeId);
      result.scalarType = base.scalarType;
      result.vectorSize = ins.arg(3u);
      return result;
    }

    case spv::OpTypeMatrix: {
      result.baseTypeId = ins.arg(2u);

      auto base = getTypeInfo(result.baseTypeId);
      result.scalarType = base.scalarType;
      result.vectorSize = base.vectorSize;
      result.matrixCols = ins.arg(3u);
      return result;
    }

    case spv::OpTypeArray:
    case spv::OpTypeRuntimeArray: {
      result.baseTypeId = ins.arg(2u);

      auto base = getTypeInfo(result.baseTypeId);

      if (base.op == spv::OpNop)
        return SpirvTypeInfo();

      if (!base.baseTypeId || base.op == spv::OpTypeVector || base.op == spv::OpTypeMatrix) {
        result.scalarType = base.scalarType;
        result.vectorSize = base.vectorSize;
        result.matrixCols = base.matrixCols;
      }

      if (ins.op() == spv::OpTypeArray)
        result.arraySize = evaluateConstant(ins.arg(3u)).value_or(0u);
      return result;
    }

    case spv::OpTypePointer: {
      result.baseTypeId = ins.arg(3u);

      auto base = getTypeInfo(result.baseTypeId);
      result.scalarType = base.scalarType;
      result.vectorSize = base.vectorSize;
      result.matrixCols = base.matrixCols;
      result.arraySize = base.arraySize;
      result.storageClass = spv::StorageClass(ins.arg(2u));
      return result;
    }

    default:
      return SpirvTypeInfo();
  }
}


uint32_t SpirvBuilder::defVectorType(
        VkComponentTypeKHR            scalarType,
        uint32_t                      vectorSize) {
  uint64_t scalarCode = uint64_t(scalarType);
  auto scalarEntry = m_vectorTypeIds.find(scalarCode);

  if (scalarEntry == m_vectorTypeIds.end()) {
    SpirvInstructionBuilder ins;
    if (util::isFloatType(scalarType)) {
      ins = SpirvInstructionBuilder(spv::OpTypeFloat, 0u, allocId());
      ins.add(8u * util::getComponentSize(scalarType));
    } else {
      ins = SpirvInstructionBuilder(spv::OpTypeInt, 0u, allocId());
      ins.add(8u * util::getComponentSize(scalarType));
      ins.add(util::isSignedType(scalarType) ? 1u : 0u);
    }

    scalarEntry = m_vectorTypeIds.insert({ scalarCode, ins.id() }).first;
    m_declarations.insert({ ins.id(), ins });
  }

  if (vectorSize <= 1u)
    return scalarEntry->second;

  uint64_t vectorCode = scalarCode | (uint64_t(vectorSize) << 32u);
  auto vectorEntry = m_vectorTypeIds.find(vectorCode);

  if (vectorEntry == m_vectorTypeIds.end()) {
    SpirvInstructionBuilder ins(spv::OpTypeVector, 0u, allocId());
    ins.add(scalarEntry->second);
    ins.add(vectorSize);

    vectorEntry = m_vectorTypeIds.insert({ vectorCode, ins.id() }).first;
    m_declarations.insert({ ins.id(), ins });
  }

  return vectorEntry->second;
}


uint32_t SpirvBuilder::defBoolType(
        uint32_t                      vectorSize) {
  SpirvInstructionBuilder builder(spv::OpTypeBool);
  builder.add(0u);

  uint32_t result = defType(builder);

  if (vectorSize > 1u) {
    SpirvInstructionBuilder builder(spv::OpTypeVector);
    builder.add(0u);
    builder.add(result);

    result = defType(builder);
  }

  return result;
}


uint32_t SpirvBuilder::defPointerType(
        uint32_t                      baseType,
        spv::StorageClass             storageClass) {
  SpirvInstructionBuilder ins(spv::OpTypePointer);
  ins.add(0u);
  ins.add(uint32_t(storageClass));
  ins.add(baseType);

  return defType(ins);
}


uint32_t SpirvBuilder::defType(
  const SpirvInstructionBuilder&      ins) {
  for (const auto& d : m_declarations) {
    bool match = ins.arg(0u) == d.second.arg(0u);

    for (uint32_t i = 2u; i < ins.len() && match; i++)
      match = ins.arg(i) == d.second.arg(i);

    uint32_t id = d.second.id();

    if (match && m_decorations.find(id) == m_decorations.end())
      return id;
  }

  uint32_t id = allocId();

  SpirvInstructionBuilder typeIns = ins;
  typeIns.set(1u, id);

  m_declarations.insert({ id, typeIns });
  return id;
}


uint32_t SpirvBuilder::defConstUint32(
        uint32_t                      value) {
  auto entry = m_uintConstantIds.find(value);

  if (entry != m_uintConstantIds.end())
    return entry->second;

  uint32_t typeId = defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);

  SpirvInstructionBuilder ins(spv::OpConstant, typeId, allocId());
  ins.add(value);

  m_uintConstantIds.insert({ value, ins.id() });

  m_declarations.insert({ ins.id(), ins });
  return ins.id();
}


uint32_t SpirvBuilder::defNullConst(
        uint32_t                      typeId) {
  SpirvInstructionBuilder ins(spv::OpConstantNull);
  ins.add(typeId);
  ins.add(0u);

  return defConst(ins);
}


uint32_t SpirvBuilder::defConst(
  const SpirvInstructionBuilder&      ins) {
  for (const auto& d : m_declarations) {
    bool match = ins.arg(0u) == d.second.arg(0u)
              && ins.arg(1u) == d.second.arg(1u);

    for (uint32_t i = 3u; i < ins.len() && match; i++)
      match = ins.arg(i) == d.second.arg(i);

    uint32_t id = d.second.id();

    if (match && m_decorations.find(id) == m_decorations.end())
      return id;
  }

  uint32_t id = allocId();

  SpirvInstructionBuilder typeIns = ins;
  typeIns.set(2u, id);

  m_declarations.insert({ id, typeIns });
  return id;
}


void SpirvBuilder::removeMetadataForId(
        uint32_t                      id) {
  m_decorations.erase(id);

  m_names.erase(id);
}


void SpirvBuilder::setDebugName(
        uint32_t                      id,
        std::string                   name) {
  SpirvInstructionBuilder ins(spv::OpName);
  ins.add(id);
  ins.add(name);

  m_names.insert({ id, ins });
}


uint32_t SpirvBuilder::getOperandTypeId(
        uint32_t                      id) const {
  auto entry = m_typesForId.find(id);

  if (entry == m_typesForId.end())
    return 0u;

  return entry->second;
}


SpirvInstructionBuilder SpirvBuilder::getOperandDefinition(
        uint32_t                      id) const {
  auto entry = m_instructionsForId.find(id);

  if (entry == m_instructionsForId.end())
    return SpirvInstructionBuilder();

  return entry->second;
}


uint32_t SpirvBuilder::getMemberTypeId(
        uint32_t                      typeId,
        uint32_t                      indexId) {
  /* Eliminate pointer types */
  auto e = m_declarations.find(typeId);

  if (e == m_declarations.end())
    return 0u;

  const auto& ins = e->second;

  switch (ins.op()) {
    case spv::OpTypeVector:
    case spv::OpTypeMatrix:
    case spv::OpTypeArray:
    case spv::OpTypeRuntimeArray:
      return ins.arg(2u);

    case spv::OpTypePointer:
      return getMemberTypeId(ins.arg(3u), indexId);

    case spv::OpTypeStruct:
      return ins.arg(2u + evaluateConstant(indexId).value());

    default:
      /* Unknown / scalar type */
      std::cerr << "Cannot subdivide type " << typeId << ": " << spv::OpToString(ins.op()) << std::endl;
      return 0u;
  }
}


void SpirvBuilder::registerVariable(
        uint32_t                      varId) {
  for (auto& e : m_entryPoints)
    e.second.add(varId);
}


uint32_t SpirvBuilder::wrap(
        uint32_t                      typeId,
        uint32_t                      operandId) {
  uint32_t ptrType = defPointerType(typeId, spv::StorageClassFunction);
  uint32_t varId = op(spv::OpVariable, ptrType, uint32_t(spv::StorageClassFunction));
  op(spv::OpStore, 0u, varId, operandId);
  return varId;
}


uint32_t SpirvBuilder::convert(
        uint32_t                      dstTypeId,
        uint32_t                      operandId) {
  auto srcTypeId = getOperandTypeId(operandId);

  if (dstTypeId == srcTypeId)
    return operandId;

  auto dstType = getTypeInfo(dstTypeId);
  auto srcType = getTypeInfo(srcTypeId);

  if (util::isFloatType(dstType.scalarType)
   && util::isFloatType(srcType.scalarType)) {
    /* Trivial conversion between different float types */
    return op(spv::OpFConvert, dstTypeId, operandId);
  } else if (util::isFloatType(dstType.scalarType)) {
    /* Convert integer to float, use signedness of operand
     * to determine the exact opcode to use. */
    spv::Op opcode = util::isSignedType(srcType.scalarType)
      ? spv::OpConvertSToF : spv::OpConvertUToF;
    return op(opcode, dstTypeId, operandId);
  } else if (util::isFloatType(srcType.scalarType)) {
    /* Convert flaot float integer, use signedness of result
     * to determine the exact opcode to use. */
    spv::Op opcode = util::isSignedType(dstType.scalarType)
      ? spv::OpConvertFToS : spv::OpConvertFToU;
    return op(opcode, dstTypeId, operandId);
  } else {
    /* Cast between integer types, use signedness of source
     * operand to determine which opcode to use. */
    spv::Op opcode = spv::OpBitcast;

    if (util::getComponentSize(dstType.scalarType)
     != util::getComponentSize(srcType.scalarType)) {
      opcode = util::isSignedType(srcType.scalarType)
        ? spv::OpSConvert : spv::OpUConvert;
    }

    return op(opcode, dstTypeId, operandId);
  }
}


uint32_t SpirvBuilder::bitcast(
        uint32_t                      dstTypeId,
        uint32_t                      operandId) {
  auto srcTypeId = getOperandTypeId(operandId);

  if (dstTypeId == srcTypeId)
    return operandId;

  return op(spv::OpBitcast, dstTypeId, operandId);
}

std::optional<uint32_t> SpirvBuilder::getDecoration(
        uint32_t                      id,
        int32_t                       member,
        spv::Decoration               decoration) const {
  auto d = m_decorations.equal_range(id);

  for (auto i = d.first; i != d.second; i++) {
    const auto& def = i->second;

    if (def.op() == spv::OpDecorate && member < 0
     && spv::Decoration(def.arg(2u)) == decoration)
      return def.arg(3u);
    else if (def.op() == spv::OpMemberDecorate && def.arg(2u) == uint32_t(member)
     && spv::Decoration(def.arg(3u)) == decoration)
      return def.arg(4u);
  }

  return std::nullopt;
}


std::string SpirvBuilder::getName(
        uint32_t                      id) const {
  auto names = m_names.equal_range(id);

  for (auto i = names.first; i != names.second; i++) {
    const auto& def = i->second;

    if (def.op() == spv::OpName)
      return def.str(2u);
  }

  return std::string();
}


bool SpirvBuilder::canEmitDeclaration(
  const SpirvInstructionBuilder&      ins,
  const std::unordered_set<uint32_t>& emitted) {
  switch (ins.op()) {
    case spv::OpTypeVoid:
    case spv::OpTypeBool:
    case spv::OpTypeInt:
    case spv::OpTypeFloat:
    case spv::OpTypeSampler:
    case spv::OpTypeOpaque:
    case spv::OpTypeRayQueryKHR:
    case spv::OpTypeAccelerationStructureKHR:
      return true;

    case spv::OpTypeVector:
    case spv::OpTypeMatrix:
    case spv::OpTypeRuntimeArray:
    case spv::OpTypeImage:
    case spv::OpTypeSampledImage:
      return hasDeclaration(emitted, ins.arg(2u));

    case spv::OpTypeArray:
      return hasDeclaration(emitted, ins.arg(2u))
          && hasDeclaration(emitted, ins.arg(3u));

    case spv::OpTypeStruct:
    case spv::OpTypeFunction:
    case spv::OpTypeCooperativeMatrixKHR:
    case spv::OpTypeCooperativeMatrixNV: {
      bool result = true;
      for (uint32_t i = 2u; i < ins.len() && result; i++)
        result = hasDeclaration(emitted, ins.arg(i));
      return result;
    }

    case spv::OpTypePointer:
      return hasDeclaration(emitted, ins.arg(3u));

    case spv::OpTypeForwardPointer:
      return true;

    case spv::OpConstantNull:
    case spv::OpConstantTrue:
    case spv::OpConstantFalse:
    case spv::OpSpecConstantTrue:
    case spv::OpSpecConstantFalse:
      return hasDeclaration(emitted, ins.typeId());

    case spv::OpConstant:
    case spv::OpSpecConstant:
      return hasDeclaration(emitted, ins.typeId());

    case spv::OpConstantComposite:
    case spv::OpSpecConstantComposite: {
      bool result = hasDeclaration(emitted, ins.typeId());
      for (uint32_t i = 3u; i < ins.len() && result; i++)
        result = hasDeclaration(emitted, ins.arg(i));
      return result;
    }

    case spv::OpSpecConstantOp: {
      bool result = hasDeclaration(emitted, ins.typeId());
      for (uint32_t i = 4u; i < ins.len() && result; i++)
        result = hasDeclaration(emitted, ins.arg(i));
      return result;
    }

    case spv::OpConstantCompositeReplicateEXT:
    case spv::OpSpecConstantCompositeReplicateEXT:
      return hasDeclaration(emitted, ins.typeId())
          && hasDeclaration(emitted, ins.arg(3u));

    case spv::OpUndef:
      return hasDeclaration(emitted, ins.typeId());

    case spv::OpVariable: {
      bool result = hasDeclaration(emitted, ins.typeId());

      if (ins.len() > 4u && result)
        result = hasDeclaration(emitted, ins.arg(4u));

      return result;
    }

    default:
      std::cerr << "Unhandled declaration " << uint32_t(ins.op()) << ": " << spv::OpToString(ins.op()) << std::endl;
      return false;
  }
}


bool SpirvBuilder::hasDeclaration(
  const std::unordered_set<uint32_t>& emitted,
        uint32_t                      id) {
  return emitted.find(id) != emitted.end();
}

}
