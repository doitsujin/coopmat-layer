#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_set>

#include "lower_coopmat.h"

#include "spirv_builder.h"
#include "spirv_reader.h"

#include "./util/bit.h"
#include "./util/vulkan_util.h"

namespace CoopmatLayer {

/* Matrix packing packing is as follows:
 *
 * At its base, A is column-major, so the lane index corresponds to the row
 * index within the given column. B and C are row-major, so the lane index
 * corresponds to the column index within the given row.
 *
 * If the subgroup size can hold `n > 1` rows or columns per register,
 * packing is such that adjacent rows or colums are stored in adjacent
 * array elements or vector components until elements wrap around into
 * the higher lanes.
 *
 * Components are packed in vectors of 32 bits each. This is done to ease
 * vectorization in the driver, but also to ensure that loads and stores
 * can be adequately performed. For all matrices, the layout is as if
 * each vector component was its own dedicated array element.
 *
 * As an example, given a subgroup size of 32, a 16x16 A matrix with 16-bit
 * components would use four array elements using two components each. For
 * the first register, lane 0-15 would hold row 0-15 of columns (0,1), while
 * lane 16-31 would hold row 0-15 of columns (8,9), etc.
 *
 * This layout trivially allows the efficient use of packed dot products for
 * supported types, trivially allows conversion between matrices of different
 * component types, and allows for a relatively simple memory layout.
 */
constexpr uint32_t MaxVectorSize = 4u;
constexpr uint32_t MaxScalarSize = sizeof(uint32_t);


/* Cooperative matrix type metadata. Used to compute the exact matrix layout,
 * and stores function IDs to perform various operations on the matrix type. */
struct CoopmatType {
  /* Scalar component type */
  VkComponentTypeKHR scalarType = VK_COMPONENT_TYPE_MAX_ENUM_KHR;
  /* Scalar component type ID */
  uint32_t scalarTypeId = 0u;
  /* Vector size for packed components */
  uint32_t vectorSize = 0u;
  /* Vector type. May be identical to the scalar type. */
  uint32_t vectorTypeId = 0u;
  /* Number of entries in the matrix array. Will be at least 1. */
  uint32_t arraySize = 0u;
  /* SPIR-V array type ID. */
  uint32_t arrayTypeId = 0u;
  /* SPIR-V type ID. Always a atruct containing an array. */
  uint32_t typeId = 0u;
  /* Number of components accessible per invocation */
  uint32_t length = 0u;
  /* Number of rows in the matrix */
  uint32_t rows = 0u;
  /* Number of columns in the matrix */
  uint32_t cols = 0u;
  /* Matrix usage */
  spv::CooperativeMatrixUse use = spv::CooperativeMatrixUseMax;
  /* Matrix layout. Depends on matrix use. */
  spv::CooperativeMatrixLayout layout = spv::CooperativeMatrixLayoutMax;
  /* Unit length per register. Equivalent to either rows
   * or columns depending on the native matrix layout. */
  uint32_t unitLength = 0u;
  /* Number of units per register. Depends on subgroup size. The
   * stride between units is equivalent to the matrix length. */
  uint32_t unitCountPerRegister = 0u;
  /* Human-readable type name */
  std::string name;
};


/**
 * \brief Generic function builder map
 *
 * Used to build dedicated SPIR-V functions for any given cooperative
 * matrix operation involving one or more unique matrix types. If a
 * function is a no-op, an ID of 0 will be returned and no function
 * call must be made.
 * \tparam K Key
 * \tparam Builder Function builder
 */
template<typename K>
class CoopmatFunctionMap {

public:

  using KeyType = K;

  template<typename... Args>
  uint32_t get(const K& key, Args&&... args) {
    auto entry = m_map.find(key);

    if (entry != m_map.end())
      return entry->second;

    uint32_t funcId = key.build(std::forward<Args>(args)...);
    m_map.insert({ key, funcId });
    return funcId;
  }

private:

  std::map<K, uint32_t> m_map;

};


template<typename... K>
class CoopmatFunctionSet;

/**
 * \brief Set of coopmat function maps
 *
 * Convenience template to bundle multiple maps.
 */
template<typename K, typename... Kx>
class CoopmatFunctionSet<K, Kx...> {

public:

  template<typename Key, typename... Args>
  uint32_t get(const Key& key, Args&&... args) {
    if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<Key>>, K>)
      return m_map.get(key, std::forward<Args>(args)...);
    else
      return m_next.get(key, std::forward<Args>(args)...);
  }

private:

  CoopmatFunctionMap<K>     m_map;
  CoopmatFunctionSet<Kx...> m_next;

};


template<>
class CoopmatFunctionSet<> { };


/**
 * \brief Matrix length function builder
 *
 * Returns a constant integer. Implemented as a function
 * for clarity of the resulting code.
 */
struct CoopmatLengthFn {
  const CoopmatType* type = nullptr;

  auto operator <=> (const CoopmatLengthFn&) const = default;

  uint32_t build(SpirvBuilder& mod) const {
    uint32_t uintType = mod.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);

    SpirvInstructionBuilder typeIns(spv::OpTypeFunction);
    typeIns.add(0u);
    typeIns.add(uintType);

    uint32_t typeId = mod.defType(typeIns);
    uint32_t funcId = mod.op(spv::OpFunction, uintType, 0u, typeId);

    mod.op(spv::OpLabel, 0u);
    mod.op(spv::OpReturnValue, mod.defConstUint32(type->length));
    mod.op(spv::OpFunctionEnd, 0u);

    mod.setDebugName(funcId, type->name + "_Length");

    return funcId;
  }
};



/**
 * \brief Scalar initialization function builder
 *
 * Takes the scalar as a parameter and fills the matrix with it.
 */
struct CoopmatFillFn {
  const CoopmatType* type = nullptr;

  auto operator <=> (const CoopmatFillFn&) const = default;

  uint32_t build(SpirvBuilder& mod) const {
    uint32_t paramType = mod.defPointerType(type->scalarTypeId, spv::StorageClassFunction);

    SpirvInstructionBuilder typeIns(spv::OpTypeFunction);
    typeIns.add(0u);
    typeIns.add(type->typeId);
    typeIns.add(paramType);

    /* Declare function */
    uint32_t funcId = mod.op(spv::OpFunction, type->typeId, 0u, mod.defType(typeIns));
    uint32_t paramId = mod.op(spv::OpFunctionParameter, paramType);
    mod.setDebugName(paramId, "value");

    mod.op(spv::OpLabel, 0u);

    /* Load scalar and put it into a vector as necessary */
    uint32_t elementId = mod.op(spv::OpLoad, type->scalarTypeId, paramId);

    if (type->vectorSize > 1u) {
      SpirvInstructionBuilder vectorOp(spv::OpCompositeConstruct, type->vectorTypeId, mod.allocId());

      for (uint32_t i = 0u; i < type->vectorSize; i++)
        vectorOp.add(elementId);

      elementId = mod.addIns(vectorOp);
    }

    /* Create the actual array */
    SpirvInstructionBuilder arrayOp(spv::OpCompositeConstruct, type->arrayTypeId, mod.allocId());

    for (uint32_t i = 0u; i < type->arraySize; i++)
      arrayOp.add(elementId);

    uint32_t resultId = mod.op(spv::OpCompositeConstruct, type->typeId, mod.addIns(arrayOp));

    mod.op(spv::OpReturnValue, resultId);
    mod.op(spv::OpFunctionEnd, 0u);

    std::stringstream name;
    name << type->name << "_Fill";

    mod.setDebugName(funcId, name.str());
    return funcId;
  }

};


/**
 * \brief Scalar scaling function builder
 *
 * Multiplies every component of the matrix with a scalar.
 */
struct CoopmatScaleFn {
  const CoopmatType* type = nullptr;

  auto operator <=> (const CoopmatScaleFn&) const = default;

  uint32_t build(SpirvBuilder& mod) const {
    uint32_t matrixParamType = mod.defPointerType(type->typeId, spv::StorageClassFunction);
    uint32_t scalarParamType = mod.defPointerType(type->scalarTypeId, spv::StorageClassFunction);

    SpirvInstructionBuilder typeIns(spv::OpTypeFunction);
    typeIns.add(0u);
    typeIns.add(type->typeId);
    typeIns.add(matrixParamType);
    typeIns.add(scalarParamType);

    /* Declare function */
    uint32_t funcId = mod.op(spv::OpFunction, type->typeId, 0u, mod.defType(typeIns));

    uint32_t matrixId = mod.op(spv::OpFunctionParameter, matrixParamType);
    uint32_t scalarId = mod.op(spv::OpFunctionParameter, scalarParamType);

    mod.setDebugName(matrixId, "self");
    mod.setDebugName(scalarId, "scale");

    mod.op(spv::OpLabel, 0u);

    /* Load factor and extend to an integer vector as necessary */
    uint32_t factorId = mod.op(spv::OpLoad, type->scalarTypeId, scalarId);

    if (type->vectorSize > 1u) {
      SpirvInstructionBuilder vectorOp(spv::OpCompositeConstruct, type->vectorTypeId, mod.allocId());

      for (uint32_t i = 0u; i < type->vectorSize; i++)
        vectorOp.add(factorId);

      factorId = mod.addIns(vectorOp);
    }

    /* Instruction to assemble the final array */
    SpirvInstructionBuilder arrayOp(spv::OpCompositeConstruct, type->arrayTypeId, mod.allocId());

    for (uint32_t i = 0u; i < type->arraySize; i++) {
      uint32_t accessChain = mod.op(spv::OpAccessChain,
        mod.defPointerType(type->vectorTypeId, spv::StorageClassFunction),
        matrixId, mod.defConstUint32(0u), mod.defConstUint32(i));

      uint32_t elementId = mod.op(spv::OpLoad, type->vectorTypeId, accessChain);

      spv::Op opcode = util::isFloatType(type->scalarType) ? spv::OpFMul : spv::OpIMul;
      elementId = mod.op(opcode, type->vectorTypeId, elementId, factorId);

      arrayOp.add(elementId);
    }

    uint32_t resultId = mod.op(spv::OpCompositeConstruct, type->typeId, mod.addIns(arrayOp));

    mod.op(spv::OpReturnValue, resultId);
    mod.op(spv::OpFunctionEnd, 0u);

    std::stringstream name;
    name << type->name << "_Scale";

    mod.setDebugName(funcId, name.str());
    return funcId;
  }

};


/**
 * \brief Scalar conversion function builder
 *
 * The resulting function takes the source matrix as an
 * argument and returns the resulting converted matrix.
 */
struct CoopmatConvertFn {
  const CoopmatType* dstType = nullptr;
  const CoopmatType* srcType = nullptr;
  spv::Op opcode = spv::OpNop;

  auto operator <=> (const CoopmatConvertFn&) const = default;

  uint32_t build(SpirvBuilder& mod) const {
    uint32_t paramType = mod.defPointerType(srcType->typeId, spv::StorageClassFunction);

    SpirvInstructionBuilder typeIns(spv::OpTypeFunction);
    typeIns.add(0u);
    typeIns.add(dstType->typeId);
    typeIns.add(paramType);

    /* Declare function */
    uint32_t funcId = mod.op(spv::OpFunction, dstType->typeId, 0u, mod.defType(typeIns));
    uint32_t paramId = mod.op(spv::OpFunctionParameter, paramType);
    mod.setDebugName(paramId, "self");

    mod.op(spv::OpLabel, 0u);

    /* Instructions to assemble appropriate array type */
    SpirvInstructionBuilder arrayOp(spv::OpCompositeConstruct, dstType->arrayTypeId, mod.allocId());
    SpirvInstructionBuilder vectorOp;

    for (uint32_t i = 0u; i < dstType->length; i++) {
      /* Load and convert one scalar at a time */
      SpirvInstructionBuilder accessChainOp(spv::OpAccessChain,
        mod.defPointerType(srcType->scalarTypeId, spv::StorageClassFunction),
        mod.allocId());
      accessChainOp.add(paramId);
      accessChainOp.add(mod.defConstUint32(0u));
      accessChainOp.add(mod.defConstUint32(i / srcType->vectorSize));

      if (srcType->vectorSize > 1u)
        accessChainOp.add(mod.defConstUint32(i % srcType->vectorSize));

      uint32_t scalarId = mod.op(spv::OpLoad, srcType->scalarTypeId, mod.addIns(accessChainOp));
      scalarId = mod.op(opcode, dstType->scalarTypeId, scalarId);

      /* Build destination vectors */
      if (dstType->vectorSize > 1u) {
        if (!(i % dstType->vectorSize))
          vectorOp = SpirvInstructionBuilder(spv::OpCompositeConstruct, dstType->vectorTypeId, mod.allocId());

        vectorOp.add(scalarId);

        if (!((i + 1u) % dstType->vectorSize))
          arrayOp.add(mod.addIns(vectorOp));
      } else {
        arrayOp.add(scalarId);
      }
    }

    uint32_t resultId = mod.op(spv::OpCompositeConstruct, dstType->typeId, mod.addIns(arrayOp));

    mod.op(spv::OpReturnValue, resultId);
    mod.op(spv::OpFunctionEnd, 0u);

    std::stringstream name;
    name << srcType->name << "_Convert";

    mod.setDebugName(funcId, name.str());
    return funcId;
  }
};


/**
 * \brief Block-transpose function builder
 *
 * Transposes quads of the element vector size in order to get the
 * correct memory layout for memory loads or stores. May be a no-op
 * for matrix types that do not use vector elements.
 */
struct CoopmatTransposeBlockFn {
  const CoopmatType* type = nullptr;

  auto operator <=> (const CoopmatTransposeBlockFn&) const = default;

  uint32_t build(SpirvBuilder& mod, uint32_t laneIdVar) const {
    /* Nothing to fix up here */
    if (type->vectorSize <= 1u)
      return 0u;

    uint32_t uintType = mod.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);
    uint32_t paramType = mod.defPointerType(type->typeId, spv::StorageClassFunction);

    SpirvInstructionBuilder typeIns(spv::OpTypeFunction);
    typeIns.add(0u);
    typeIns.add(type->typeId);
    typeIns.add(paramType);

    /* Declare function */
    uint32_t funcId = mod.op(spv::OpFunction, type->typeId, 0u, mod.defType(typeIns));
    uint32_t paramId = mod.op(spv::OpFunctionParameter, paramType);
    mod.setDebugName(paramId, "self");
    mod.op(spv::OpLabel, 0u);

    uint32_t laneId = mod.op(spv::OpLoad, uintType, laneIdVar);

    /* Instruction to build output array */
    SpirvInstructionBuilder arrayOp(spv::OpCompositeConstruct, type->arrayTypeId, mod.allocId());

    for (uint32_t i = 0u; i < type->arraySize; i++) {
      uint32_t accessChainId = mod.op(spv::OpAccessChain,
        mod.defPointerType(type->vectorTypeId, spv::StorageClassFunction),
        paramId, mod.defConstUint32(0u), mod.defConstUint32(i));

      /* Transpose 2x2 blocks first, then 4x4 */
      uint32_t elementId = mod.op(spv::OpLoad, type->vectorTypeId, accessChainId);

      for (uint32_t j = 1u; j < type->vectorSize; j <<= 1u) {
        uint32_t shuffleId = mod.op(spv::OpGroupNonUniformShuffleXor, type->vectorTypeId,
          mod.defConstUint32(uint32_t(spv::ScopeSubgroup)), elementId,
          mod.defConstUint32(j));

        SpirvInstructionBuilder loOp(spv::OpVectorShuffle, type->vectorTypeId, mod.allocId());
        SpirvInstructionBuilder hiOp(spv::OpVectorShuffle, type->vectorTypeId, mod.allocId());

        loOp.add(elementId, shuffleId);
        hiOp.add(shuffleId, elementId);

        for (uint32_t k = 0u; k < type->vectorSize; k++) {
          uint32_t index = (k & ~j) + ((k & j) ? type->vectorSize : 0u);
          loOp.add(index);
          hiOp.add(index + j);
        }

        uint32_t isHiId = mod.op(spv::OpINotEqual, mod.defBoolType(0u),
          mod.op(spv::OpBitwiseAnd, uintType, laneId, mod.defConstUint32(j)),
          mod.defConstUint32(0u));

        elementId = mod.op(spv::OpSelect, type->vectorTypeId,
          isHiId, mod.addIns(hiOp), mod.addIns(loOp));
      }

      arrayOp.add(elementId);
    }

    uint32_t resultId = mod.op(spv::OpCompositeConstruct, type->typeId, mod.addIns(arrayOp));

    mod.op(spv::OpReturnValue, resultId);
    mod.op(spv::OpFunctionEnd, 0u);

    std::stringstream name;
    name << type->name << "_Transpose" << type->vectorSize << "x" << type->vectorSize;

    mod.setDebugName(funcId, name.str());
    return funcId;
  }
};


/**
 * \brief Layout info function builder
 *
 * Takes the array element index as its only parameter, and returns
 * a (col,row) vector that describes where the first component of
 * the element vector owned by the current invocation is stored in
 * memory.
 *
 * This will take any potential block-transpose into account, so that
 * these values can be fed directly into load/store operations.
 */
struct CoopmatLayoutFn {
  const CoopmatType* type = nullptr;
  spv::CooperativeMatrixLayout layout = spv::CooperativeMatrixLayoutMax;

  auto operator <=> (const CoopmatLayoutFn&) const = default;

  uint32_t build(SpirvBuilder& mod, uint32_t laneIdVar) const {
    uint32_t uintType = mod.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);
    uint32_t uvecType = mod.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 2u);

    uint32_t paramType = mod.defPointerType(uintType, spv::StorageClassFunction);

    SpirvInstructionBuilder typeIns(spv::OpTypeFunction);
    typeIns.add(0u);
    typeIns.add(uvecType);
    typeIns.add(paramType);

    /* Declare function */
    uint32_t funcId = mod.op(spv::OpFunction, uvecType, 0u, mod.defType(typeIns));
    uint32_t paramId = mod.op(spv::OpFunctionParameter, paramType);
    mod.setDebugName(paramId, "idx");

    mod.op(spv::OpLabel, 0u);

    uint32_t laneId = mod.op(spv::OpLoad, uintType, laneIdVar);

    /* Scale array index with vector size to get the actual base 'row'.
     * Note that the wording used here assumes a row-major layout, so
     * the 'row' is the dimension that will be scaled by the stride in
     * memory load/store operations. */
    uint32_t rowId = mod.op(spv::OpLoad, uintType, paramId);

    if (type->vectorSize > 1u) {
      rowId = mod.op(spv::OpShiftLeftLogical, uintType, rowId,
        mod.defConstUint32(util::tzcnt(type->vectorSize)));
    }

    /* If there are multiple rows per register, the stride between them
     * is equivalent to the matrix length, i.e. the number of scalar
     * components per lane. */
    if (type->unitCountPerRegister > 1u) {
      int32_t shift = int32_t(util::tzcnt(type->length))
                    - int32_t(util::tzcnt(type->unitLength));

      uint32_t localRow = laneId;

      if (shift > 0) {
        localRow = mod.op(spv::OpShiftLeftLogical, uintType,
          laneId, mod.defConstUint32(shift));
      } else {
        localRow = mod.op(spv::OpShiftRightLogical, uintType,
          laneId, mod.defConstUint32(-shift));
      }

      rowId = mod.op(spv::OpBitFieldInsert, uintType, localRow, rowId,
        mod.defConstUint32(0u), mod.defConstUint32(util::tzcnt(type->length)));
    }

    /* Compute column index based on the lane ID */
    uint32_t colId = mod.op(spv::OpBitwiseAnd, uintType, laneId,
      mod.defConstUint32(type->unitLength - 1u));

    /* Transpose by flipping indices if the desired memory layout does not
     * match the native register layout */
    if (type->layout != layout )
      std::swap(colId, rowId);

    if (type->vectorSize > 1u) {
      /* If we are accessing a vectorized matrix in its native layout, we have
       * to transpose the vector-sized blocks first in order to get things in
       * the correct order. Account for that by swapping the lower bits. */
      if (type->layout == layout) {
        rowId = mod.op(spv::OpBitFieldInsert, uintType, rowId, colId,
          mod.defConstUint32(0u),
          mod.defConstUint32(util::tzcnt(type->vectorSize)));
      }

      /* Divide the new 'column' index by the vector size since memory
       * load/store operations expect the index to be given in vectors. */
      colId = mod.op(spv::OpShiftRightLogical, uintType, colId,
        mod.defConstUint32(util::tzcnt(type->vectorSize)));
    }

    uint32_t resultId = mod.op(spv::OpCompositeConstruct, uvecType, colId, rowId);

    mod.op(spv::OpReturnValue, resultId);
    mod.op(spv::OpFunctionEnd, 0u);

    std::stringstream name;
    name << type->name << "_Layout";

    if (layout == spv::CooperativeMatrixLayoutRowMajorKHR)
      name << "_RowMajor";
    else
      name << "_ColMajor";

    mod.setDebugName(funcId, name.str());
    return funcId;
  }
};


/**
 * \brief Multiply-accumulate function builder
 *
 * The function takes 3 arguments, namely the a, b and c
 * matrices in that order, and returns the resulting matrix.
 */
struct CoopmatMulAddFn {
  const CoopmatType* aType = nullptr;
  const CoopmatType* bType = nullptr;
  const CoopmatType* cType = nullptr;
  const CoopmatType* dType = nullptr;
  uint32_t operands = 0u;

  auto operator <=> (const CoopmatMulAddFn&) const = default;

  uint32_t build(SpirvBuilder& mod, uint32_t laneIdVar, uint32_t subgroupSize) const {
    uint32_t uintType = mod.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);

    bool saturateAccum = operands & spv::CooperativeMatrixOperandsSaturatingAccumulationKHRMask;

    uint32_t aParamType = mod.defPointerType(aType->typeId, spv::StorageClassFunction);
    uint32_t bParamType = mod.defPointerType(bType->typeId, spv::StorageClassFunction);
    uint32_t cParamType = mod.defPointerType(cType->typeId, spv::StorageClassFunction);

    SpirvInstructionBuilder typeIns(spv::OpTypeFunction);
    typeIns.add(0u);
    typeIns.add(dType->typeId);
    typeIns.add(aParamType);
    typeIns.add(bParamType);
    typeIns.add(cParamType);

    /* Declare function */
    uint32_t funcId = mod.op(spv::OpFunction, dType->typeId, 0u, mod.defType(typeIns));

    uint32_t aId = mod.op(spv::OpFunctionParameter, aParamType);
    uint32_t bId = mod.op(spv::OpFunctionParameter, bParamType);
    uint32_t cId = mod.op(spv::OpFunctionParameter, cParamType);

    mod.setDebugName(aId, "a");
    mod.setDebugName(bId, "b");
    mod.setDebugName(cId, "c");

    mod.op(spv::OpLabel, 0u);

    /* Lane ID, we need this in various places */
    uint32_t laneId = mod.op(spv::OpLoad, uintType, laneIdVar);

    /* For integer types, check which matrices are signed and adjust types as necessary. */
    bool aAsSigned = (util::isSignedType(aType->scalarType))
                  || (operands & spv::CooperativeMatrixOperandsMatrixASignedComponentsKHRMask);
    bool bAsSigned = (util::isSignedType(bType->scalarType))
                  || (operands & spv::CooperativeMatrixOperandsMatrixBSignedComponentsKHRMask);
    bool cAsSigned = (util::isSignedType(cType->scalarType))
                  || (operands & spv::CooperativeMatrixOperandsMatrixCSignedComponentsKHRMask);

    /* The intermediate type is the type at which accumulation is performed.
     * For floats, convert up to result type before accumulation in order to
     * maintain precision. If we're using integer dot products, we need to
     * use a 32-bit intermediate result based on operand signedness. */
    VkComponentTypeKHR intermediateType = dType->scalarType;

    if (!util::isFloatType(aType->scalarType)) {
      intermediateType = util::get32BitType(aType->scalarType);

      if (cAsSigned)
        intermediateType = util::getSignedType(intermediateType);
    }

    uint32_t intermediateTypeId = mod.defVectorType(intermediateType, 0u);;

    /* Row vectors computed in each iteration */
    std::vector<uint32_t> registerRows(dType->unitCountPerRegister);

    /* Instructions to create resulting vector array */
    SpirvInstructionBuilder arrayOp(spv::OpCompositeConstruct, dType->arrayTypeId, mod.allocId());
    SpirvInstructionBuilder vectorOp;

    for (uint32_t baseRow = 0u; baseRow < dType->length; baseRow++) {
      /* The following code largely assumes that all matrices are square, and thus of the
       * same size. The matrix layout for rectangular matrices differs substantially, and
       * broadcasts and shuffles like these would not select the correct lanes or rows. */
      for (uint32_t localRow = 0u; localRow < dType->unitCountPerRegister; localRow++) {
        uint32_t rowIndex = baseRow + localRow * dType->length;

        /* Iterate over input registers to perform and accumulate dot products */
        uint32_t rowResultId = 0u;

        for (uint32_t iter = 0u; iter < aType->arraySize; iter++) {
          uint32_t aElements = emitLoadAMatrix(mod, aId, iter, rowIndex, laneId, aAsSigned);
          uint32_t bElements = emitLoadBMatrix(mod, bId, iter, bAsSigned);

          if (util::isFloatType(aType->scalarType)) {
            /* Multiply as-is, then accumulate using the destination type. If applicable,
             * smart compilers may give us packed dot products here. */
            uint32_t intermediateVectorTypeId = mod.defVectorType(intermediateType, aType->vectorSize);

            uint32_t dotProductId = mod.op(spv::OpFMul, aType->vectorTypeId, aElements, bElements);
            dotProductId = mod.convert(intermediateVectorTypeId, dotProductId);

            if (aType->vectorSize > 1u) {
              uint32_t localSumId = mod.op(spv::OpCompositeExtract, intermediateTypeId, dotProductId, 0u);

              for (uint32_t i = 1u; i < aType->vectorSize; i++) {
                localSumId = mod.op(spv::OpFAdd, intermediateTypeId, localSumId,
                  mod.op(spv::OpCompositeExtract, intermediateTypeId, dotProductId, i));
              }

              dotProductId = localSumId;
            }

            /* Accumulate row result */
            rowResultId = rowResultId
              ? mod.op(spv::OpFAdd, intermediateTypeId, rowResultId, dotProductId)
              : dotProductId;
          } else {
            /* TODO implement */
          }
        }

        registerRows.at(localRow) = rowResultId;
      }

      /* Each intermediate register may still hold multiple different parts of the
       * same row, which we need to horizontally add. Merge the different output
       * rows at the same time. */
      for (uint32_t regs = dType->unitCountPerRegister; regs > 1u; regs >>= 1u) {
        uint32_t strideInUnits = regs / 2u;
        uint32_t strideInLanes = strideInUnits * dType->unitLength;

        for (uint32_t i = 0u; i + i < regs; i++) {
          uint32_t loId = registerRows.at(i);
          uint32_t hiId = registerRows.at(i + strideInUnits);

          registerRows.at(i) = emitMergeLanes(mod,
            intermediateTypeId, loId, hiId, laneId, strideInLanes);
        }
      }

      /* Load accumulator and accumulate */
      uint32_t accumId = emitLoadCMatrix(mod, cId, baseRow);
      accumId = mod.convert(intermediateTypeId, accumId);

      spv::Op addOp = util::isFloatType(dType->scalarType) ? spv::OpFAdd : spv::OpIAdd;
      accumId = mod.op(addOp, intermediateTypeId, accumId, registerRows.at(0));
      accumId = mod.convert(dType->scalarTypeId, accumId);

      if (dType->vectorSize > 1u) {
        if (!(baseRow & dType->vectorSize))
          vectorOp = SpirvInstructionBuilder(spv::OpCompositeConstruct, dType->vectorTypeId, mod.allocId());

        vectorOp.add(accumId);

        if (!((baseRow + 1u) & dType->vectorSize))
          arrayOp.add(mod.addIns(vectorOp));
      } else {
        arrayOp.add(accumId);
      }
    }

    uint32_t resultId = mod.op(spv::OpCompositeConstruct, dType->typeId, mod.addIns(arrayOp));

    mod.op(spv::OpReturnValue, resultId);
    mod.op(spv::OpFunctionEnd, 0u);

    std::stringstream name;
    name << "CoopMatMulAdd_" << util::getComponentTypeName(dType->scalarType) << "_" <<
      aType->rows << "x" << aType->cols << "x" << bType->cols << "_" <<
      util::getComponentTypeName(aType->scalarType);

    if (dType->scalarType != cType->scalarType)
      name << util::getComponentTypeName(cType->scalarType);

    if (saturateAccum)
      name << "_sat";

    mod.setDebugName(funcId, name.str());
    return funcId;
  }



  uint32_t emitLoadAMatrix(SpirvBuilder& mod, uint32_t varId, uint32_t index, uint32_t row, uint32_t laneId, bool isSigned) const {
    uint32_t uintType = mod.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);

    uint32_t accessChain = mod.op(spv::OpAccessChain,
      mod.defPointerType(aType->vectorTypeId, spv::StorageClassFunction), varId,
      mod.defConstUint32(0u),
      mod.defConstUint32(index));

    uint32_t resultId = mod.op(spv::OpLoad, aType->vectorTypeId, accessChain);

    /* Clustered broadcast */
    if (aType->unitCountPerRegister > 1u) {
      uint32_t laneMask = aType->unitLength * (aType->unitCountPerRegister - 1u);

      laneId = mod.op(spv::OpBitwiseAnd, uintType, laneId, mod.defConstUint32(laneMask));
      laneId = mod.op(spv::OpBitwiseOr, uintType, laneId, mod.defConstUint32(row));

      resultId = mod.op(spv::OpGroupNonUniformShuffle, aType->vectorTypeId,
        mod.defConstUint32(uint32_t(spv::ScopeSubgroup)), resultId, laneId);
    } else {
      resultId = mod.op(spv::OpGroupNonUniformBroadcast, aType->vectorTypeId,
        mod.defConstUint32(uint32_t(spv::ScopeSubgroup)), resultId,
        mod.defConstUint32(row));
    }

    return emitCastInput(mod, resultId, aType, isSigned);
  }


  uint32_t emitLoadBMatrix(SpirvBuilder& mod, uint32_t varId, uint32_t index, bool isSigned) const {
    uint32_t accessChain = mod.op(spv::OpAccessChain,
      mod.defPointerType(bType->vectorTypeId, spv::StorageClassFunction), varId,
      mod.defConstUint32(0u),
      mod.defConstUint32(index));

    /* B is already in the correct layout */
    uint32_t resultId = mod.op(spv::OpLoad, aType->vectorTypeId, accessChain);
    return emitCastInput(mod, resultId, bType, isSigned);
  }


  uint32_t emitLoadCMatrix(SpirvBuilder& mod, uint32_t varId, uint32_t row) const {
    /* Load a scalar */
    uint32_t ptrTypeId = mod.defPointerType(cType->scalarTypeId, spv::StorageClassFunction);
    SpirvInstructionBuilder accessChainOp(spv::OpAccessChain, ptrTypeId, mod.allocId());
    accessChainOp.add(varId);
    accessChainOp.add(mod.defConstUint32(0u));
    accessChainOp.add(mod.defConstUint32(row / cType->vectorSize));

    if (cType->vectorSize > 1u)
      accessChainOp.add(mod.defConstUint32(row % cType->vectorSize));

    /* Return accumulator as-is, we will cast it as necessary */
    return mod.op(spv::OpLoad, cType->scalarTypeId, mod.addIns(accessChainOp));
  }


  uint32_t emitCastInput(SpirvBuilder& mod, uint32_t inputId, const CoopmatType* type, bool isSigned) const {
    /* Make sure inputs have the correct type w.r.t. signedness */
    if (!util::isFloatType(type->scalarType)) {
      uint32_t inputType = mod.defVectorType(isSigned
        ? util::getSignedType(type->scalarType)
        : util::getUnsignedType(type->scalarType), type->vectorSize);

      inputId = mod.convert(inputType, inputId);
    }

    return inputId;
  }


  uint32_t emitMergeLanes(SpirvBuilder& mod, uint32_t typeId, uint32_t loId, uint32_t hiId, uint32_t laneId, uint32_t laneMask) const {
    auto resultType = mod.getTypeInfo(typeId).scalarType;

    /* Pack into vector for potentially faster shuffle in case of small result type */
    uint32_t uintType = mod.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);

    uint32_t vectorTypeId = mod.defVectorType(resultType, 2u);
    uint32_t vectorId = mod.op(spv::OpCompositeConstruct, vectorTypeId, loId, hiId);

    /* Make each half of the row visible to the other one */
    uint32_t shuffleId = mod.op(spv::OpGroupNonUniformShuffleXor, vectorTypeId,
      mod.defConstUint32(uint32_t(spv::ScopeSubgroup)), vectorId,
      mod.defConstUint32(laneMask));

    /* Add both halves together, potentially taking advantage of packed math */
    spv::Op opcode = util::isFloatType(resultType) ? spv::OpFAdd : spv::OpIAdd;
    vectorId = mod.op(opcode, vectorTypeId, vectorId, shuffleId);

    /* Check lane ID on whether to pick the high or low compnents */
    uint32_t selectHiId = mod.op(spv::OpINotEqual, mod.defBoolType(0u),
      mod.op(spv::OpBitwiseAnd, uintType, laneId, mod.defConstUint32(laneMask)),
      mod.defConstUint32(0u));

    uint32_t resultId = mod.op(spv::OpSelect, typeId, selectHiId,
      mod.op(spv::OpCompositeExtract, typeId, vectorId, 1u),
      mod.op(spv::OpCompositeExtract, typeId, vectorId, 0u));

    return resultId;
  }

};


using CoopmatFunctions = CoopmatFunctionSet<
  CoopmatLengthFn,
  CoopmatFillFn,
  CoopmatScaleFn,
  CoopmatConvertFn,
  CoopmatTransposeBlockFn,
  CoopmatLayoutFn,
  CoopmatMulAddFn>;


/* Cooperative matmul lookup */
struct CoopmatMulAccumFunc {
  uint32_t resultTypeId = 0u;
  uint32_t aTypeId = 0u;
  uint32_t bTypeId = 0u;
  uint32_t accumTypeId = 0u;
  uint32_t operands = 0u;

  auto operator <=> (const CoopmatMulAccumFunc&) const = default;
};


/* Implementation of the SPIR-V pass for cooperative matrix emulation */
class CoopmatPass {

public:

  CoopmatPass(
    const VkShaderModuleCreateInfo*               pCreateInfo,
    const VkSpecializationInfo*                   pSpecInfo,
          uint32_t                                subgroupSize)
  : m_reader        (pCreateInfo->codeSize, pCreateInfo->pCode)
  , m_builder       (m_reader.getHeader(), pSpecInfo)
  , m_subgroupSize  (subgroupSize) {

  }

  std::string getSourceName() const {
    return m_builder.getSourceName();
  }

  std::vector<uint32_t> run() {
    while (auto ins = m_reader.readInstruction()) {
      switch (ins.op()) {
        case spv::OpCapability: {
          if (spv::Capability(ins.arg(1u)) != spv::CapabilityCooperativeMatrixKHR)
            m_builder.addIns(SpirvInstructionBuilder(ins));
        } break;

        case spv::OpExtension: {
          SpirvInstructionBuilder ext(ins);

          if (ext.str(1u) != "SPV_KHR_cooperative_matrix")
            m_builder.addIns(SpirvInstructionBuilder(ext));
        } break;

        case spv::OpTypeCooperativeMatrixKHR: {
          CoopmatType type = { };
          type.scalarType = m_builder.getTypeInfo(ins.arg(2u)).scalarType;
          type.scalarTypeId = ins.arg(2u);
          type.vectorSize = sizeof(uint32_t) / util::getComponentSize(type.scalarType);
          type.vectorTypeId = 0u; /* allocated later */
          type.rows = m_builder.evaluateConstant(ins.arg(4u));
          type.cols = m_builder.evaluateConstant(ins.arg(5u));
          type.length = std::max(1u, (type.rows * type.cols) / m_subgroupSize);
          type.use = spv::CooperativeMatrixUse(m_builder.evaluateConstant(ins.arg(6u)));
          type.layout = type.use == spv::CooperativeMatrixUseMatrixAKHR
            ? spv::CooperativeMatrixLayoutColumnMajorKHR
            : spv::CooperativeMatrixLayoutRowMajorKHR;
          type.unitLength = type.layout == spv::CooperativeMatrixLayoutRowMajorKHR
            ? type.cols
            : type.rows;
          type.unitCountPerRegister = m_subgroupSize / type.unitLength;
          type.arraySize = std::max(1u, type.length / type.vectorSize);
          type.arrayTypeId = 0u; /* allocated later */
          type.typeId = ins.arg(1u);

          m_coopmatTypes.insert({ ins.arg(1u), type });

          m_builder.removeMetadataForId(ins.arg(1u));
        } break;

        case spv::OpAccessChain:
        case spv::OpInBoundsAccessChain: {
          SpirvInstructionBuilder accessChain(ins.op(), ins.arg(1u), ins.arg(2u));
          accessChain.add(ins.arg(3));

          /* Resolve pointer type of base type */
          auto typeId = m_builder.getOperandTypeId(ins.arg(3u));
          typeId = m_builder.getTypeInfo(typeId).baseTypeId;

          for (uint32_t i = 4u; i < ins.len(); i++) {
            auto type = findCoopmatType(typeId);

            if (type) {
              uint32_t uintType = m_builder.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);

              /* Resolve struct member, then index into array and vector */
              accessChain.add(m_builder.defConstUint32(0u));

              if (type->vectorSize > 1u) {
                accessChain.add(m_builder.op(spv::OpUDiv, uintType, ins.arg(i), m_builder.defConstUint32(type->vectorSize)));
                accessChain.add(m_builder.op(spv::OpUMod, uintType, ins.arg(i), m_builder.defConstUint32(type->vectorSize)));
              } else {
                accessChain.add(ins.arg(i));
              }

              /* This must be the last operand */
              break;
            } else {
              /* Haven't reached coopmat yet */
              accessChain.add(ins.arg(i));
            }

            typeId = m_builder.getMemberTypeId(typeId, ins.arg(i));
          }

          m_builder.addIns(accessChain);
        } break;

        case spv::OpConstantComposite: {
          auto* type = findCoopmatType(ins.arg(1u));

          if (!type) {
            m_builder.addIns(ins);
            break;
          }

          m_coopmatConstants.push_back(ins);
        } break;

        case spv::OpFConvert:
        case spv::OpSConvert:
        case spv::OpUConvert:
        case spv::OpConvertFToU:
        case spv::OpConvertFToS:
        case spv::OpConvertSToF:
        case spv::OpConvertUToF: {
          auto* dstType = findCoopmatType(ins.arg(1u));
          auto* srcType = findCoopmatType(m_builder.getOperandTypeId(ins.arg(3u)));

          if (!dstType || !srcType) {
            m_builder.addIns(ins);
            break;
          }

          CoopmatConvertFn fnInfo = { };
          fnInfo.dstType = dstType;
          fnInfo.srcType = srcType;
          fnInfo.opcode = ins.op();

          SpirvInstructionBuilder callOp(spv::OpFunctionCall, ins.arg(1u), ins.arg(2u));
          callOp.add(m_functions.get(fnInfo, m_builder));
          callOp.add(m_builder.wrap(fnInfo.srcType->typeId, ins.arg(3u)));
          m_builder.addIns(callOp);
        } break;

        case spv::OpCompositeConstruct: {
          auto* type = findCoopmatType(ins.arg(1u));

          if (!type) {
            m_builder.addIns(ins);
            break;
          }

          CoopmatFillFn fnInfo = { };
          fnInfo.type = type;

          SpirvInstructionBuilder callOp(spv::OpFunctionCall, ins.arg(1u), ins.arg(2u));
          callOp.add(m_functions.get(fnInfo, m_builder));
          callOp.add(m_builder.wrap(type->scalarTypeId, ins.arg(3u)));
          m_builder.addIns(callOp);
        } break;

        case spv::OpCompositeExtract: {
          auto* type = findCoopmatType(m_builder.getOperandTypeId(ins.arg(3u)));

          if (!type) {
            m_builder.addIns(ins);
            break;
          }

          uint32_t index = ins.arg(ins.len() - 1u);
          SpirvInstructionBuilder extractOp(spv::OpCompositeExtract, ins.arg(1u), ins.arg(2u));

          for (uint32_t i = 3u; i < ins.len() - 1u; i++)
            extractOp.add(ins.arg(i));

          extractOp.add(0u); /* struct */
          extractOp.add(index / type->vectorSize);

          if (type->vectorSize > 1u)
            extractOp.add(index % type->vectorSize);

          m_builder.addIns(extractOp);
        } break;

        case spv::OpCooperativeMatrixLengthKHR: {
          uint32_t uintType = m_builder.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);

          CoopmatLengthFn fnInfo = { };
          fnInfo.type = findCoopmatType(ins.arg(3u));

          SpirvInstructionBuilder callOp(spv::OpFunctionCall, uintType, ins.arg(2u));
          callOp.add(m_functions.get(fnInfo, m_builder));
          m_builder.addIns(callOp);
        } break;

        case spv::OpCooperativeMatrixLoadKHR: {
          auto* matrixType = findCoopmatType(ins.arg(1u));

          auto [srcArray, srcOffset, srcStride] = rewriteCoopmatLoadStoreAccessChain(ins.arg(3u), ins.arg(5u));
          auto layout = spv::CooperativeMatrixLayout(m_builder.evaluateConstant(ins.arg(4u)));

          SpirvInstructionBuilder arrayOp(spv::OpCompositeConstruct, matrixType->arrayTypeId, m_builder.allocId());

          std::vector<uint32_t> operands;

          for (uint32_t i = 6u; i < ins.len(); i++)
            operands.push_back(ins.arg(i));

          for (uint32_t i = 0u; i < matrixType->arraySize; i++) {
            uint32_t coord = computeMemoryLocation(matrixType, i, layout);

            uint32_t id = loadMatrixElement(matrixType, layout,
              srcArray, srcOffset, coord, srcStride, operands.size(), operands.data());

            arrayOp.add(id);
          }

          /* If no repacking is needed, construct directly into provided result ID */
          uint32_t repackFn = 0u;

          if (layout == matrixType->layout) {
            CoopmatTransposeBlockFn fnInfo = { };
            fnInfo.type = matrixType;

            repackFn = m_functions.get(fnInfo, m_builder, defLaneId());
          }

          SpirvInstructionBuilder constructOp(spv::OpCompositeConstruct,
            matrixType->typeId, repackFn ? m_builder.allocId() : ins.arg(2u));
          constructOp.add(m_builder.addIns(arrayOp));
          m_builder.addIns(constructOp);

          if (repackFn) {
            uint32_t paramId = m_builder.wrap(matrixType->typeId, constructOp.id());

            SpirvInstructionBuilder packIns(spv::OpFunctionCall, matrixType->typeId, ins.arg(2u));
            packIns.add(repackFn);
            packIns.add(paramId);

            m_builder.addIns(packIns);
          }
        } break;

        case spv::OpCooperativeMatrixMulAddKHR: {
          CoopmatMulAddFn fnInfo = { };
          fnInfo.aType = findCoopmatType(m_builder.getOperandTypeId(ins.arg(3u)));
          fnInfo.bType = findCoopmatType(m_builder.getOperandTypeId(ins.arg(4u)));
          fnInfo.cType = findCoopmatType(m_builder.getOperandTypeId(ins.arg(5u)));
          fnInfo.dType = findCoopmatType(ins.arg(1u));
          fnInfo.operands = ins.arg(6u);

          SpirvInstructionBuilder frog(spv::OpFunctionCall, fnInfo.dType->typeId, ins.arg(2u));
          frog.add(m_functions.get(fnInfo, m_builder, defLaneId(), m_subgroupSize));
          frog.add(m_builder.wrap(fnInfo.aType->typeId, ins.arg(3u)));
          frog.add(m_builder.wrap(fnInfo.bType->typeId, ins.arg(4u)));
          frog.add(m_builder.wrap(fnInfo.cType->typeId, ins.arg(5u)));
          m_builder.addIns(frog);
        } break;

        case spv::OpCooperativeMatrixStoreKHR: {
          auto* matrixType = findCoopmatType(m_builder.getOperandTypeId(ins.arg(2u)));

          auto [dstArray, dstOffset, dstStride] = rewriteCoopmatLoadStoreAccessChain(ins.arg(1u), ins.arg(4u));
          auto layout = spv::CooperativeMatrixLayout(m_builder.evaluateConstant(ins.arg(3u)));

          std::vector<uint32_t> operands;

          for (uint32_t i = 5u; i < ins.len(); i++)
            operands.push_back(ins.arg(i));

          uint32_t repackFn = 0u;

          if (layout == matrixType->layout) {
            CoopmatTransposeBlockFn fnInfo = { };
            fnInfo.type = matrixType;

            repackFn = m_functions.get(fnInfo, m_builder, defLaneId());
          }

          uint32_t id = ins.arg(2u);

          if (repackFn) {
            uint32_t paramId = m_builder.wrap(matrixType->typeId, ins.arg(2u));
            id = m_builder.op(spv::OpFunctionCall, matrixType->typeId, repackFn, paramId);
          }

          for (uint32_t i = 0u; i < matrixType->arraySize; i++) {
            uint32_t coord = computeMemoryLocation(matrixType, i, layout);
            uint32_t elementId = m_builder.op(spv::OpCompositeExtract, matrixType->vectorTypeId, id, 0u, i);

            storeMatrixElement(matrixType, layout, elementId,
              dstArray, dstOffset, coord, dstStride, operands.size(), operands.data());
          }
        } break;

        case spv::OpMatrixTimesScalar: {
          auto* type = findCoopmatType(ins.arg(1u));

          if (!type) {
            m_coopmatConstants.push_back(SpirvInstructionBuilder(ins));
            break;
          }

          CoopmatScaleFn fnInfo = { };
          fnInfo.type = type;

          SpirvInstructionBuilder callOp(spv::OpFunctionCall, ins.arg(1u), ins.arg(2u));
          callOp.add(m_functions.get(fnInfo, m_builder));
          callOp.add(m_builder.wrap(type->typeId, ins.arg(3u)));
          callOp.add(m_builder.wrap(type->scalarTypeId, ins.arg(4u)));
          m_builder.addIns(callOp);
        } break;

        case spv::OpFunction: {
          emitCooperativeMatrixTypes();
          m_builder.addIns(SpirvInstructionBuilder(ins));
        } break;

        default:
          m_builder.addIns(SpirvInstructionBuilder(ins));
      }
    }

    emitCapabilities();

    return m_builder.finalize();
  }

private:

  SpirvReader   m_reader;
  SpirvBuilder  m_builder;

  uint32_t      m_subgroupSize = 0u;

  std::unordered_map<uint32_t, CoopmatType> m_coopmatTypes;
  std::vector<SpirvInstructionBuilder>      m_coopmatConstants;

  std::unordered_map<uint32_t, uint32_t>    m_bufferResourceMap;

  CoopmatFunctions        m_functions;

  SpirvInstructionBuilder m_laneId = { };

  bool          m_emittedTypes = false;

  std::tuple<SpirvInstructionBuilder, uint32_t, uint32_t> rewriteCoopmatLoadStoreAccessChain(uint32_t operandId, uint32_t strideId) {
    uint32_t uintType = m_builder.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);
    SpirvInstructionBuilder def = m_builder.getOperandDefinition(operandId);

    if (def.op() != spv::OpAccessChain && def.op() != spv::OpInBoundsAccessChain) {
      std::cerr << "Load/Store memory operand not an access chain: " << def.op() << ": " << OpToString(def.op()) << std::endl;
      return {};
    }

    /* Get info about the array type we're indexing into */
    auto typeInfo = m_builder.getTypeInfo(def.typeId());
    auto typeSize = util::getComponentSize(typeInfo.scalarType) * std::max(1u, typeInfo.vectorSize);

    /* Get new access chain and offset ID */
    auto [accessChain, offsetId] = rewriteAccessChain(def.id());

    /* If the base type uses sub-dword scalars, try to promote to dwords */
    if (util::getComponentSize(typeInfo.scalarType) < sizeof(uint32_t)) {
      uint32_t newBaseId = rewriteBufferResource(accessChain.arg(3u));

      if (newBaseId) {
        /* Adjust offset and stride to match the new type */
        if (typeSize != sizeof(uint32_t)) {
          if (typeSize > 1u) {
            uint32_t oldSizeId = m_builder.defConstUint32(typeSize);
            offsetId = m_builder.op(spv::OpIMul, uintType, offsetId, oldSizeId);
            strideId = m_builder.op(spv::OpIMul, uintType, strideId, oldSizeId);
          }

          uint32_t newSizeId = m_builder.defConstUint32(sizeof(uint32_t));
          offsetId = m_builder.op(spv::OpUDiv, uintType, offsetId, newSizeId);
          strideId = m_builder.op(spv::OpUDiv, uintType, strideId, newSizeId);
        }

        accessChain.set(3u, newBaseId);
        accessChain.set(1u, getAccessChainPointerTypeId(accessChain));
      }
    }

    return std::tuple(accessChain, offsetId, strideId);
  }


  std::pair<SpirvInstructionBuilder, uint32_t> rewriteAccessChain(uint32_t operandId) {
    SpirvInstructionBuilder def = m_builder.getOperandDefinition(operandId);
    SpirvInstructionBuilder op = chainAccessChainsRecursive(def.arg(3u));

    for (uint32_t i = 4u; i < def.len() - 1u; i++)
      op.add(def.arg(i));

    uint32_t finalId = def.arg(def.len() - 1u);
    op.set(1u, getAccessChainPointerTypeId(op));

    return std::make_pair(op, finalId);
  }


  SpirvInstructionBuilder chainAccessChainsRecursive(uint32_t baseId) {
    SpirvInstructionBuilder def = m_builder.getOperandDefinition(baseId);

    if (def.op() == spv::OpAccessChain || def.op() == spv::OpInBoundsAccessChain) {
      auto op = chainAccessChainsRecursive(def.arg(3u));

      for (uint32_t i = 4u; i < def.len(); i++)
        op.add(def.arg(i));

      return op;
    } else {
      /* Don't define a type ID yet because we don't know
       * how long the access chain is going to be */
      SpirvInstructionBuilder op(spv::OpAccessChain);
      op.add(0u);
      op.add(m_builder.allocId());
      op.add(baseId);

      return op;
    }
  }


  uint32_t getAccessChainPointerTypeId(const SpirvInstructionBuilder& op) {
    auto baseDef = m_builder.getOperandDefinition(op.arg(3u));
    auto typeId = baseDef.arg(1u);

    if (op.len() <= 4u)
      return typeId;

    auto storageClass = m_builder.getTypeInfo(typeId).storageClass;

    for (uint32_t i = 4u; i < op.len(); i++)
      typeId = m_builder.getMemberTypeId(typeId, op.arg(i));

    return m_builder.defPointerType(typeId, storageClass);
  }


  uint32_t rewriteBufferResource(uint32_t baseId) {
    auto e = m_bufferResourceMap.find(baseId);

    if (e != m_bufferResourceMap.end())
      return e->second;

    auto def = m_builder.getOperandDefinition(baseId);
    auto typeId = m_builder.getOperandTypeId(baseId);
    auto type = m_builder.getTypeInfo(typeId);

    if (type.storageClass == spv::StorageClassPhysicalStorageBuffer) {
      /* BDA types are convenient since we can trivially bitcast
       * to another pointer type and call it a day */
      uint32_t newTypeId = rewriteBufferType(typeId);

      if (!newTypeId)
        return 0u;

      return m_builder.bitcast(newTypeId, baseId);
    } else if (type.storageClass == spv::StorageClassStorageBuffer) {
      /* The base must be some sort of variable with optional set / binding decorations */
      if (def.op() != spv::OpVariable)
        return 0u;

      auto descriptorSet = m_builder.getDecoration(baseId, -1, spv::DecorationDescriptorSet);
      auto descriptorBinding = m_builder.getDecoration(baseId, -1, spv::DecorationBinding);

      uint32_t newTypeId = rewriteBufferType(typeId);

      if (!newTypeId)
        return 0u;

      uint32_t newVarId = m_builder.op(spv::OpVariable, newTypeId, def.arg(3u));

      if (descriptorSet)
        m_builder.op(spv::OpDecorate, 0u, newVarId, uint32_t(spv::DecorationDescriptorSet), *descriptorSet);
      if (descriptorBinding)
        m_builder.op(spv::OpDecorate, 0u, newVarId, uint32_t(spv::DecorationBinding), *descriptorBinding);

      m_builder.registerVariable(newVarId);
      m_bufferResourceMap.insert({ baseId, newVarId });
      return newVarId;;
    } else {
      /* Unsupported thing, ignore */
      return 0u;
    }
  }


  uint32_t rewriteBufferType(uint32_t baseTypeId) {
    auto e = m_bufferResourceMap.find(baseTypeId);

    if (e != m_bufferResourceMap.end())
      return e->second;

    auto def = m_builder.getOperandDefinition(baseTypeId);

    uint32_t rewriteId = 0u;

    switch (def.op()) {
      case spv::OpTypeStruct: {
        rewriteId = rewriteBufferStructType(def);
      } break;

      case spv::OpTypePointer: {
        auto storageClass = spv::StorageClass(def.arg(2u));

        if (storageClass != spv::StorageClassStorageBuffer
         && storageClass != spv::StorageClassPhysicalStorageBuffer)
          return 0u;

        uint32_t baseId = rewriteBufferType(def.arg(3u));

        if (!baseId)
          return 0u;

        rewriteId = m_builder.defPointerType(baseId, storageClass);
      } break;

      case spv::OpTypeArray:
      case spv::OpTypeRuntimeArray: {
        /* Might be a value array */
        if ((rewriteId = rewriteBufferArrayType(def)))
          break;

        /* Might be a descriptor array */
        auto baseDef = m_builder.getOperandDefinition(def.arg(2u));

        if (auto baseId = rewriteBufferStructType(baseDef)) {
          rewriteId = def.op() == spv::OpTypeArray
            ? m_builder.op(spv::OpTypeArray, 0u, baseId, def.arg(3u))
            : m_builder.op(spv::OpTypeRuntimeArray, 0u, baseId);
        }
      } break;

      default:
        return 0u;
    }

    if (rewriteId)
      m_bufferResourceMap.insert({ def.id(), rewriteId });

    return rewriteId;
  }


  uint32_t rewriteBufferStructType(const SpirvInstructionBuilder& def) {
    /* Only support struct types that contain a single scalar or vector array
     * for now. Also require a block decoration since that is used for any
     * memory interface declaration. */
    if (def.len() != 3u)
      return 0u;

    auto memberOffset = m_builder.getDecoration(def.id(), 0, spv::DecorationOffset);

    if (!m_builder.getDecoration(def.id(), -1, spv::DecorationBlock) || !memberOffset)
      return 0u;

    auto arrayDef = m_builder.getOperandDefinition(def.arg(2u));

    if (arrayDef.op() != spv::OpTypeArray
     && arrayDef.op() != spv::OpTypeRuntimeArray)
      return 0u;

    uint32_t arrayId = rewriteBufferArrayType(arrayDef);

    /* Declare struct type */
    SpirvInstructionBuilder structOp(spv::OpTypeStruct, 0u, m_builder.allocId());
    structOp.add(arrayId);

    uint32_t structId = m_builder.addIns(structOp);

    m_builder.op(spv::OpDecorate, 0u, structId, uint32_t(spv::DecorationBlock));
    m_builder.op(spv::OpMemberDecorate, 0u, structId, 0u, uint32_t(spv::DecorationOffset), *memberOffset);
    m_builder.setDebugName(structId, "CoopMat_Memory");
    return structId;
  }


  uint32_t rewriteBufferArrayType(const SpirvInstructionBuilder& def) {
    uint32_t arrayStride = m_builder.getDecoration(def.id(), -1, spv::DecorationArrayStride).value_or(0u);
    uint32_t vectorSize = 1u;

    auto vectorDef = m_builder.getOperandDefinition(def.arg(2u));
    auto scalarDef = vectorDef;

    if (vectorDef.op() == spv::OpTypeVector) {
      scalarDef = m_builder.getOperandDefinition(vectorDef.arg(2u));
      vectorSize = vectorDef.arg(3u);
    }

    if (scalarDef.op() != spv::OpTypeFloat
     && scalarDef.op() != spv::OpTypeInt)
      return 0u;

    /* Ensure that the array stride matches the vector size, otherwise we
     * cannot meaningfully change the struct type */
    if (arrayStride != vectorSize * (scalarDef.arg(2u) / 8u))
      return 0u;

    /* Declare array type using dwords as a base unit */
    SpirvInstructionBuilder arrayOp(def.op(), 0u, m_builder.allocId());
    arrayOp.add(m_builder.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u));

    if (def.op() == spv::OpTypeArray) {
      uint32_t arraySize = m_builder.evaluateConstant(def.arg(3u));

      if (!arraySize)
        return 0u;

      arrayOp.add(m_builder.defConstUint32(arraySize * arrayStride / sizeof(uint32_t)));
    }

    uint32_t arrayId = m_builder.addIns(arrayOp);

    m_builder.op(spv::OpDecorate, 0u, arrayId, uint32_t(spv::DecorationArrayStride), uint32_t(sizeof(uint32_t)));
    return arrayId;
  }


  std::pair<uint32_t, SpirvTypeInfo> getMemoryLocationTypeInfo(uint32_t typeId) {
    auto typeInfo = m_builder.getTypeInfo(typeId);

    /* Resovle the pointer type */
    typeInfo = m_builder.getTypeInfo(typeId = typeInfo.baseTypeId);
    /* Resovle the array type */
    typeInfo = m_builder.getTypeInfo(typeId = typeInfo.baseTypeId);

    return std::make_pair(typeId, typeInfo);
  }


  uint32_t getDefaultStrideId(const CoopmatType* matType, const SpirvTypeInfo& memType, spv::CooperativeMatrixLayout layout) {
    uint32_t matDimension = layout == spv::CooperativeMatrixLayoutRowMajorKHR ? matType->cols : matType->rows;
    uint32_t matByteCount = matDimension * util::getComponentSize(matType->scalarType);

    uint32_t dstByteCount = util::getComponentSize(memType.scalarType);

    if (memType.vectorSize)
      dstByteCount *= memType.vectorSize;

    return m_builder.defConstUint32(std::max(1u, matByteCount / dstByteCount));
  }


  uint32_t loadMatrixElement(const CoopmatType* type, spv::CooperativeMatrixLayout layout,
      SpirvInstructionBuilder base, uint32_t offsetId, uint32_t coordId, uint32_t strideId,
      uint32_t operandCount, const uint32_t* operands) {
    auto [memTypeId, memTypeInfo] = getMemoryLocationTypeInfo(base.typeId());

    if (!strideId)
      strideId = getDefaultStrideId(type, memTypeInfo, layout);

    uint32_t uintType = m_builder.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);

    /* Determine underlying store type */
    uint32_t memScalarSize = util::getComponentSize(memTypeInfo.scalarType);
    uint32_t memScalarType = memTypeInfo.vectorSize ? memTypeInfo.baseTypeId : memTypeId;

    /* Declare pointer type */
    uint32_t ptrTypeId = m_builder.defPointerType(memScalarType, m_builder.getTypeInfo(base.typeId()).storageClass);

    /* Compute number of memory scalars per packed vector. Assume that the
     * underlying memory scalar type is not larger than a packed vector. */
    uint32_t matScalarSize = util::getComponentSize(type->scalarType);
    uint32_t matVectorSize = matScalarSize * type->vectorSize;

    uint32_t memScalarsPerVector = matVectorSize / memScalarSize;

    /* Multiply high index by stride and add base offset */
    uint32_t baseIndex = m_builder.op(spv::OpCompositeExtract, uintType, coordId, 1u);
    baseIndex = m_builder.op(spv::OpIMul, uintType, baseIndex, strideId);
    baseIndex = m_builder.op(spv::OpIAdd, uintType, baseIndex, offsetId);

    if (memTypeInfo.vectorSize) {
      baseIndex = m_builder.op(spv::OpIMul, uintType, baseIndex,
        m_builder.defConstUint32(memTypeInfo.vectorSize));
    }

    /* Final element index, in terms of memory scalars (not vectors) */
    uint32_t elementIndex = m_builder.op(spv::OpCompositeExtract, uintType, coordId, 0u);

    if (memScalarsPerVector > 1u) {
      elementIndex = m_builder.op(spv::OpIMul, uintType, elementIndex,
        m_builder.defConstUint32(memScalarsPerVector));
    }

    elementIndex = m_builder.op(spv::OpIAdd, uintType, elementIndex, baseIndex);

    /* Load individual scalars, deal with them later */
    std::array<uint32_t, 16u> memScalars = { };

    for (uint32_t i = 0u; i < memScalarsPerVector; i++) {
      uint32_t loadIndex = m_builder.op(spv::OpIAdd, uintType, elementIndex, m_builder.defConstUint32(i));
      SpirvInstructionBuilder srcAddress(spv::OpAccessChain, ptrTypeId, m_builder.allocId());

      for (uint32_t i = 3u; i < base.len(); i++)
        srcAddress.add(base.arg(i));

      if (memTypeInfo.vectorSize) {
        srcAddress.add(m_builder.op(spv::OpUDiv, uintType, loadIndex, m_builder.defConstUint32(memTypeInfo.vectorSize)));
        srcAddress.add(m_builder.op(spv::OpUMod, uintType, loadIndex, m_builder.defConstUint32(memTypeInfo.vectorSize)));
      } else {
        srcAddress.add(loadIndex);
      }

      SpirvInstructionBuilder loadOp(spv::OpLoad, memScalarType, m_builder.allocId());
      loadOp.add(m_builder.addIns(srcAddress));

      if (operandCount) {
        loadOp.add(operands[0u]);

        if (operands[0u] & spv::MemoryAccessAlignedMask)
          loadOp.add(std::min(operands[1u], memScalarSize));
      }

      memScalars.at(i) = m_builder.addIns(loadOp);
    }

    if (matScalarSize > memScalarSize) {
      /* Annoying case where we need to assemble scalars from multiple loads */
      uint32_t loadsPerMatScalar = matScalarSize / memScalarSize;
      uint32_t vectorType = m_builder.defVectorType(memTypeInfo.scalarType, loadsPerMatScalar);

      SpirvInstructionBuilder resultOp;

      if (type->vectorSize > 1u)
        resultOp = SpirvInstructionBuilder(spv::OpCompositeConstruct, type->vectorTypeId, m_builder.allocId());

      uint32_t resultId = 0u;

      for (uint32_t i = 0u; i < type->vectorSize; i++) {
        SpirvInstructionBuilder vectorOp(spv::OpCompositeConstruct, vectorType, m_builder.allocId());

        for (uint32_t j = 0u; j < loadsPerMatScalar; j++)
          vectorOp.add(memScalars.at(i * loadsPerMatScalar + j));

        resultId = m_builder.bitcast(type->scalarTypeId, m_builder.addIns(vectorOp));

        if (type->vectorSize > 1u)
          resultOp.add(resultId);
      }

      if (type->vectorSize > 1u)
        resultId = m_builder.addIns(resultOp);

      return resultId;
    } else if (matVectorSize > memScalarSize) {
      /* Assemble vector from loaded scalars, then bitcast to result type */
      uint32_t vectorType = m_builder.defVectorType(memTypeInfo.scalarType, memScalarsPerVector);
      SpirvInstructionBuilder vectorOp(spv::OpCompositeConstruct, vectorType, m_builder.allocId());

      for (uint32_t i = 0u; i < memScalarsPerVector; i++)
        vectorOp.add(memScalars.at(i));

      uint32_t resultId = m_builder.addIns(vectorOp);
      resultId = m_builder.bitcast(type->vectorTypeId, resultId);

      return resultId;
    } else {
      /* One memory scalar covers entire matrix vector, bitcast directly. */
      return m_builder.bitcast(type->vectorTypeId, memScalars.at(0u));
    }
  }


  void storeMatrixElement(const CoopmatType* type, spv::CooperativeMatrixLayout layout, uint32_t id,
      const SpirvInstructionBuilder& base, uint32_t offsetId, uint32_t coordId, uint32_t strideId,
      uint32_t operandCount, const uint32_t* operands) {
    auto [memTypeId, memTypeInfo] = getMemoryLocationTypeInfo(base.typeId());

    if (!strideId)
      strideId = getDefaultStrideId(type, memTypeInfo, layout);

    uint32_t uintType = m_builder.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);

    /* Determine underlying store type */
    uint32_t memScalarSize = util::getComponentSize(memTypeInfo.scalarType);
    uint32_t memScalarType = memTypeInfo.vectorSize ? memTypeInfo.baseTypeId : memTypeId;

    /* Declare pointer type */
    uint32_t ptrTypeId = m_builder.defPointerType(memScalarType, m_builder.getTypeInfo(base.typeId()).storageClass);

    /* Compute number of memory scalars per packed vector. Assume that the
     * underlying memory scalar type is not larger than a packed vector. */
    uint32_t matScalarSize = util::getComponentSize(type->scalarType);
    uint32_t matVectorSize = matScalarSize * type->vectorSize;

    uint32_t memScalarsPerVector = matVectorSize / memScalarSize;

    /* Multiply high index by stride and add base offset */
    uint32_t baseIndex = m_builder.op(spv::OpCompositeExtract, uintType, coordId, 1u);
    baseIndex = m_builder.op(spv::OpIMul, uintType, baseIndex, strideId);
    baseIndex = m_builder.op(spv::OpIAdd, uintType, baseIndex, offsetId);

    if (memTypeInfo.vectorSize) {
      baseIndex = m_builder.op(spv::OpIMul, uintType, baseIndex,
        m_builder.defConstUint32(memTypeInfo.vectorSize));
    }

    /* Final element index, in terms of memory scalars (not vectors) */
    uint32_t elementIndex = m_builder.op(spv::OpCompositeExtract, uintType, coordId, 0u);

    if (memScalarsPerVector > 1u) {
      elementIndex = m_builder.op(spv::OpIMul, uintType, elementIndex,
        m_builder.defConstUint32(memScalarsPerVector));
    }

    elementIndex = m_builder.op(spv::OpIAdd, uintType, elementIndex, baseIndex);

    /* Iterate over scalars  */
    for (uint32_t i = 0u; i < memScalarsPerVector; i++) {
      uint32_t storeIndex = m_builder.op(spv::OpIAdd, uintType, elementIndex, m_builder.defConstUint32(i));

      SpirvInstructionBuilder dstAddress(spv::OpAccessChain, ptrTypeId, m_builder.allocId());

      for (uint32_t i = 3u; i < base.len(); i++)
        dstAddress.add(base.arg(i));

      if (memTypeInfo.vectorSize > 1u) {
        dstAddress.add(m_builder.op(spv::OpUDiv, uintType, storeIndex, m_builder.defConstUint32(memTypeInfo.vectorSize)));
        dstAddress.add(m_builder.op(spv::OpUMod, uintType, storeIndex, m_builder.defConstUint32(memTypeInfo.vectorSize)));
      } else {
        dstAddress.add(storeIndex);
      }

      uint32_t valueId = id;

      if (matScalarSize > memScalarSize) {
        /* Bitcast one scalar to a vector of memory scalars and pick one component */
        uint32_t storesPerMatScalar = matScalarSize / memScalarSize;
        uint32_t vectorType = m_builder.defVectorType(memTypeInfo.scalarType, storesPerMatScalar);

        uint32_t matScalar = id;

        if (type->vectorSize > 1u)
          matScalar = m_builder.op(spv::OpCompositeExtract, type->scalarTypeId, id, i / storesPerMatScalar);

        uint32_t matVector = m_builder.bitcast(vectorType, matScalar);
        valueId = m_builder.op(spv::OpCompositeExtract, memScalarType, matVector, i % storesPerMatScalar);
      } else if (matVectorSize > memScalarSize) {
        /* Pick one or more scalars from the source and bitcast to memory scalar */
        uint32_t matScalarsPerStore = memScalarSize / matScalarSize;
        uint32_t elementType = m_builder.defVectorType(type->scalarType, matScalarsPerStore);

        if (matScalarsPerStore == 1u) {
          /* Extract one scalar and store it right away, trivial case */
          valueId = id;

          if (type->vectorSize > 1u)
            valueId = m_builder.op(spv::OpCompositeExtract, elementType, id, i);
        } else {
          /* Guaranteed to have a vectorized matrix if we reach this */
          SpirvInstructionBuilder swizzle(spv::OpVectorShuffle, elementType, m_builder.allocId());
          swizzle.add(id, id);

          for (uint32_t j = 0u; j < matScalarsPerStore; j++)
            swizzle.add(matScalarsPerStore * i + j);

          valueId = m_builder.addIns(swizzle);
        }

        valueId = m_builder.bitcast(memScalarType, valueId);
      } else {
        /* Bitcast source vector to memory scalar directly */
        valueId = m_builder.bitcast(memScalarType, id);
      }

      SpirvInstructionBuilder storeOp(spv::OpStore);
      storeOp.add(m_builder.addIns(dstAddress));
      storeOp.add(valueId);

      if (operandCount) {
        storeOp.add(operands[0u]);

        if (operands[0u] & spv::MemoryAccessAlignedMask)
          storeOp.add(std::min(operands[1u], memScalarSize));
      }

      m_builder.addIns(storeOp);
    }
  }


  uint32_t computeMemoryLocation(const CoopmatType* type, uint32_t element, spv::CooperativeMatrixLayout layout) {
    CoopmatLayoutFn fnInfo = { };
    fnInfo.type = type;
    fnInfo.layout = layout;

    uint32_t funcId = m_functions.get(fnInfo, m_builder, defLaneId());

    uint32_t uintTypeId = m_builder.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);
    uint32_t uvecTypeId = m_builder.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 2u);

    uint32_t paramId = m_builder.wrap(uintTypeId, m_builder.defConstUint32(element));
    return m_builder.op(spv::OpFunctionCall, uvecTypeId, funcId, paramId);
  }


  uint32_t defLaneId() {
    if (!m_laneId.id()) {
      uint32_t uintType = m_builder.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);
      uint32_t ptrTypeId = m_builder.defPointerType(uintType, spv::StorageClassInput);

      m_laneId = SpirvInstructionBuilder(spv::OpVariable, ptrTypeId, m_builder.allocId());
      m_laneId.add(uint32_t(spv::StorageClassInput));

      m_builder.addIns(m_laneId);

      SpirvInstructionBuilder builtin(spv::OpDecorate, 0u, 0u);
      builtin.add(m_laneId.id());
      builtin.add(uint32_t(spv::DecorationBuiltIn));
      builtin.add(uint32_t(spv::BuiltInSubgroupLocalInvocationId));

      m_builder.addIns(builtin);

      m_builder.registerVariable(m_laneId.id());
    }

    return m_laneId.id();
  }


  uint32_t getLaneId() {
    uint32_t uintType = m_builder.defVectorType(VK_COMPONENT_TYPE_UINT32_KHR, 0u);
    return m_builder.op(spv::OpLoad, uintType, defLaneId());
  }


  void emitCooperativeMatrixTypes() {
    if (std::exchange(m_emittedTypes, true))
      return;

    for (auto& e : m_coopmatTypes) {
      e.second.vectorTypeId = m_builder.defVectorType(e.second.scalarType, e.second.vectorSize);

      /* Declare array type */
      SpirvInstructionBuilder arr(spv::OpTypeArray);
      arr.add(0u);
      arr.add(e.second.vectorTypeId);
      arr.add(m_builder.defConstUint32(e.second.arraySize));

      e.second.arrayTypeId = m_builder.defType(arr);

      /* Declare struct type */
      SpirvInstructionBuilder typeIns(spv::OpTypeStruct, 0u, e.first);
      typeIns.add(e.second.arrayTypeId);

      m_builder.addIns(typeIns);

      /* Add debug name */
      std::stringstream name;
      name << "CoopMat_" << e.second.rows << "x" << e.second.cols << "_"
           << util::getComponentTypeName(e.second.scalarType) << "_";

      switch (e.second.use) {
        case spv::CooperativeMatrixUseMatrixAKHR: name << "A"; break;
        case spv::CooperativeMatrixUseMatrixBKHR: name << "B"; break;
        case spv::CooperativeMatrixUseMatrixAccumulatorKHR: name << "C"; break;
        default: name << "Unknown";
      }

      e.second.name = name.str();
      e.second.typeId = e.first;

      m_builder.setDebugName(e.first, e.second.name);
    }

    for (const auto& c : m_coopmatConstants) {
      auto type = findCoopmatType(c.typeId());

      uint32_t elementId = c.arg(3u);

      if (type->vectorSize > 1u) {
        SpirvInstructionBuilder vec(spv::OpConstantComposite, type->vectorTypeId, m_builder.allocId());

        for (uint32_t i = 0u; i < type->vectorSize; i++)
          vec.add(elementId);

        elementId = m_builder.addIns(vec);
      }

      SpirvInstructionBuilder arr(spv::OpConstantComposite, type->arrayTypeId, m_builder.allocId());

      for (uint32_t i = 0u; i < type->arraySize; i++)
        arr.add(elementId);

      m_builder.addIns(arr);

      SpirvInstructionBuilder mat(spv::OpConstantComposite, type->typeId, c.id());
      mat.add(arr.id());

      m_builder.addIns(mat);
    }
  }


  void emitCapabilities() {
    static const std::array<spv::Capability, 4> caps = {
      spv::CapabilityGroupNonUniform,
      spv::CapabilityGroupNonUniformVote,
      spv::CapabilityGroupNonUniformShuffle,
      spv::CapabilityGroupNonUniformShuffleRelative,
    };

    for (auto cap : caps)
      m_builder.op(spv::OpCapability, 0u, uint32_t(cap));
  }


  CoopmatType* findCoopmatType(uint32_t id) {
    auto entry = m_coopmatTypes.find(id);

    if (entry == m_coopmatTypes.end())
      return nullptr;

    return &entry->second;
  }

};


/* Dump patched code to a file */
static void dumpShader(
  const std::string&                            name,
  const std::vector<uint32_t>&                  patched) {
  const char* env = ::getenv("FROG_COOPMAT_SHADER_DUMP_PATH");

  if (!env || !env[0])
    return;

  /* Write to file */
  std::string sourceName = std::string(env) + std::string("/coopmat_") + name + std::string(".spv");

  std::ofstream file(sourceName, std::ios_base::binary | std::ios_base::trunc);
  file.write(reinterpret_cast<const char*>(patched.data()), patched.size() * sizeof(uint32_t));
}


std::vector<uint32_t> lowerCoopmat(
  const VkShaderModuleCreateInfo&               pCreateInfo,
  const VkSpecializationInfo*                   pSpecInfo,
        uint32_t                                subgroupSize) {
  CoopmatPass pass(&pCreateInfo, pSpecInfo, subgroupSize);
  std::vector<uint32_t> code = pass.run();

  dumpShader(pass.getSourceName(), code);
  return code;
}


bool shaderUsesCoopmat(
  const VkShaderModuleCreateInfo&               pCreateInfo) {
  SpirvReader reader(pCreateInfo.codeSize, pCreateInfo.pCode);

  while (auto ins = reader.readInstruction()) {
    if (ins.op() == spv::OpEntryPoint)
      return false;

    if (ins.op() == spv::OpCapability && ins.arg(1) == spv::CapabilityCooperativeMatrixKHR)
      return true;
  }

  return false;
}

}
