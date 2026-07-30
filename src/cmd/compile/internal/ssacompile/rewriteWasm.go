// Code generated from _gen/Wasm.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "math"
import "cmd/compile/internal/types"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func rewriteValueWasm(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpAbs:
		v.Op = ssaop.OpWasmF64Abs
		return true
	case ssaop.OpAbsFloat32x4:
		v.Op = ssaop.OpWasmF32x4Abs
		return true
	case ssaop.OpAbsFloat64x2:
		v.Op = ssaop.OpWasmF64x2Abs
		return true
	case ssaop.OpAbsInt16x8:
		v.Op = ssaop.OpWasmI16x8Abs
		return true
	case ssaop.OpAbsInt32x4:
		v.Op = ssaop.OpWasmI32x4Abs
		return true
	case ssaop.OpAbsInt64x2:
		v.Op = ssaop.OpWasmI64x2Abs
		return true
	case ssaop.OpAbsInt8x16:
		v.Op = ssaop.OpWasmI8x16Abs
		return true
	case ssaop.OpAdd16:
		v.Op = ssaop.OpWasmI64Add
		return true
	case ssaop.OpAdd32:
		v.Op = ssaop.OpWasmI64Add
		return true
	case ssaop.OpAdd32F:
		v.Op = ssaop.OpWasmF32Add
		return true
	case ssaop.OpAdd64:
		v.Op = ssaop.OpWasmI64Add
		return true
	case ssaop.OpAdd64F:
		v.Op = ssaop.OpWasmF64Add
		return true
	case ssaop.OpAdd8:
		v.Op = ssaop.OpWasmI64Add
		return true
	case ssaop.OpAddFloat32x4:
		v.Op = ssaop.OpWasmF32x4Add
		return true
	case ssaop.OpAddFloat64x2:
		v.Op = ssaop.OpWasmF64x2Add
		return true
	case ssaop.OpAddInt16x8:
		v.Op = ssaop.OpWasmI16x8Add
		return true
	case ssaop.OpAddInt32x4:
		v.Op = ssaop.OpWasmI32x4Add
		return true
	case ssaop.OpAddInt64x2:
		v.Op = ssaop.OpWasmI64x2Add
		return true
	case ssaop.OpAddInt8x16:
		v.Op = ssaop.OpWasmI8x16Add
		return true
	case ssaop.OpAddPtr:
		v.Op = ssaop.OpWasmI64Add
		return true
	case ssaop.OpAddSaturatedInt16x8:
		v.Op = ssaop.OpWasmI16x8AddSatS
		return true
	case ssaop.OpAddSaturatedInt8x16:
		v.Op = ssaop.OpWasmI8x16AddSatS
		return true
	case ssaop.OpAddSaturatedUint16x8:
		v.Op = ssaop.OpWasmI16x8AddSatU
		return true
	case ssaop.OpAddSaturatedUint8x16:
		v.Op = ssaop.OpWasmI8x16AddSatU
		return true
	case ssaop.OpAddr:
		return rewriteValueWasm_OpAddr(v)
	case ssaop.OpAnd16:
		v.Op = ssaop.OpWasmI64And
		return true
	case ssaop.OpAnd32:
		v.Op = ssaop.OpWasmI64And
		return true
	case ssaop.OpAnd64:
		v.Op = ssaop.OpWasmI64And
		return true
	case ssaop.OpAnd8:
		v.Op = ssaop.OpWasmI64And
		return true
	case ssaop.OpAndB:
		v.Op = ssaop.OpWasmI64And
		return true
	case ssaop.OpAndInt16x8:
		v.Op = ssaop.OpWasmV128And
		return true
	case ssaop.OpAndInt32x4:
		v.Op = ssaop.OpWasmV128And
		return true
	case ssaop.OpAndInt64x2:
		v.Op = ssaop.OpWasmV128And
		return true
	case ssaop.OpAndInt8x16:
		v.Op = ssaop.OpWasmV128And
		return true
	case ssaop.OpAndNotInt16x8:
		v.Op = ssaop.OpWasmV128Andnot
		return true
	case ssaop.OpAndNotInt32x4:
		v.Op = ssaop.OpWasmV128Andnot
		return true
	case ssaop.OpAndNotInt64x2:
		v.Op = ssaop.OpWasmV128Andnot
		return true
	case ssaop.OpAndNotInt8x16:
		v.Op = ssaop.OpWasmV128Andnot
		return true
	case ssaop.OpAndNotUint16x8:
		v.Op = ssaop.OpWasmV128Andnot
		return true
	case ssaop.OpAndNotUint32x4:
		v.Op = ssaop.OpWasmV128Andnot
		return true
	case ssaop.OpAndNotUint64x2:
		v.Op = ssaop.OpWasmV128Andnot
		return true
	case ssaop.OpAndNotUint8x16:
		v.Op = ssaop.OpWasmV128Andnot
		return true
	case ssaop.OpAndUint16x8:
		v.Op = ssaop.OpWasmV128And
		return true
	case ssaop.OpAndUint32x4:
		v.Op = ssaop.OpWasmV128And
		return true
	case ssaop.OpAndUint64x2:
		v.Op = ssaop.OpWasmV128And
		return true
	case ssaop.OpAndUint8x16:
		v.Op = ssaop.OpWasmV128And
		return true
	case ssaop.OpAverageUint16x8:
		v.Op = ssaop.OpWasmI16x8AvgrU
		return true
	case ssaop.OpAverageUint8x16:
		v.Op = ssaop.OpWasmI8x16AvgrU
		return true
	case ssaop.OpAvg64u:
		return rewriteValueWasm_OpAvg64u(v)
	case ssaop.OpBitLen16:
		return rewriteValueWasm_OpBitLen16(v)
	case ssaop.OpBitLen32:
		return rewriteValueWasm_OpBitLen32(v)
	case ssaop.OpBitLen64:
		return rewriteValueWasm_OpBitLen64(v)
	case ssaop.OpBitLen8:
		return rewriteValueWasm_OpBitLen8(v)
	case ssaop.OpBitSelectInt16x8:
		v.Op = ssaop.OpWasmV128Bitselect
		return true
	case ssaop.OpBitSelectInt32x4:
		v.Op = ssaop.OpWasmV128Bitselect
		return true
	case ssaop.OpBitSelectInt64x2:
		v.Op = ssaop.OpWasmV128Bitselect
		return true
	case ssaop.OpBitSelectInt8x16:
		v.Op = ssaop.OpWasmV128Bitselect
		return true
	case ssaop.OpBitSelectUint16x8:
		v.Op = ssaop.OpWasmV128Bitselect
		return true
	case ssaop.OpBitSelectUint32x4:
		v.Op = ssaop.OpWasmV128Bitselect
		return true
	case ssaop.OpBitSelectUint64x2:
		v.Op = ssaop.OpWasmV128Bitselect
		return true
	case ssaop.OpBitSelectUint8x16:
		v.Op = ssaop.OpWasmV128Bitselect
		return true
	case ssaop.OpBroadcastFloat32x4:
		v.Op = ssaop.OpWasmF32x4Splat
		return true
	case ssaop.OpBroadcastFloat64x2:
		v.Op = ssaop.OpWasmF64x2Splat
		return true
	case ssaop.OpBroadcastInt16x8:
		v.Op = ssaop.OpWasmI16x8Splat
		return true
	case ssaop.OpBroadcastInt32x4:
		v.Op = ssaop.OpWasmI32x4Splat
		return true
	case ssaop.OpBroadcastInt64x2:
		v.Op = ssaop.OpWasmI64x2Splat
		return true
	case ssaop.OpBroadcastInt8x16:
		v.Op = ssaop.OpWasmI8x16Splat
		return true
	case ssaop.OpCeil:
		v.Op = ssaop.OpWasmF64Ceil
		return true
	case ssaop.OpCeilFloat32x4:
		v.Op = ssaop.OpWasmF32x4Ceil
		return true
	case ssaop.OpCeilFloat64x2:
		v.Op = ssaop.OpWasmF64x2Ceil
		return true
	case ssaop.OpClosureCall:
		v.Op = ssaop.OpWasmLoweredClosureCall
		return true
	case ssaop.OpCom16:
		return rewriteValueWasm_OpCom16(v)
	case ssaop.OpCom32:
		return rewriteValueWasm_OpCom32(v)
	case ssaop.OpCom64:
		return rewriteValueWasm_OpCom64(v)
	case ssaop.OpCom8:
		return rewriteValueWasm_OpCom8(v)
	case ssaop.OpCondSelect:
		v.Op = ssaop.OpWasmSelect
		return true
	case ssaop.OpConst16:
		return rewriteValueWasm_OpConst16(v)
	case ssaop.OpConst32:
		return rewriteValueWasm_OpConst32(v)
	case ssaop.OpConst32F:
		v.Op = ssaop.OpWasmF32Const
		return true
	case ssaop.OpConst64:
		v.Op = ssaop.OpWasmI64Const
		return true
	case ssaop.OpConst64F:
		v.Op = ssaop.OpWasmF64Const
		return true
	case ssaop.OpConst8:
		return rewriteValueWasm_OpConst8(v)
	case ssaop.OpConstBool:
		return rewriteValueWasm_OpConstBool(v)
	case ssaop.OpConstNil:
		return rewriteValueWasm_OpConstNil(v)
	case ssaop.OpConvert:
		v.Op = ssaop.OpWasmLoweredConvert
		return true
	case ssaop.OpConvertLo2ToFloat64Int32x4:
		v.Op = ssaop.OpWasmF64x2ConvertLowI32x4S
		return true
	case ssaop.OpConvertLo2ToFloat64Uint32x4:
		v.Op = ssaop.OpWasmF64x2ConvertLowI32x4U
		return true
	case ssaop.OpConvertToFloat32Int32x4:
		v.Op = ssaop.OpWasmF32x4ConvertI32x4S
		return true
	case ssaop.OpConvertToFloat32Uint32x4:
		v.Op = ssaop.OpWasmF32x4ConvertI32x4U
		return true
	case ssaop.OpConvertToInt32Float32x4:
		v.Op = ssaop.OpWasmI32x4TruncSatF32x4S
		return true
	case ssaop.OpConvertToUint32Float32x4:
		v.Op = ssaop.OpWasmI32x4TruncSatF32x4U
		return true
	case ssaop.OpCopysign:
		v.Op = ssaop.OpWasmF64Copysign
		return true
	case ssaop.OpCtz16:
		return rewriteValueWasm_OpCtz16(v)
	case ssaop.OpCtz16NonZero:
		v.Op = ssaop.OpWasmI64Ctz
		return true
	case ssaop.OpCtz32:
		return rewriteValueWasm_OpCtz32(v)
	case ssaop.OpCtz32NonZero:
		v.Op = ssaop.OpWasmI64Ctz
		return true
	case ssaop.OpCtz64:
		v.Op = ssaop.OpWasmI64Ctz
		return true
	case ssaop.OpCtz64NonZero:
		v.Op = ssaop.OpWasmI64Ctz
		return true
	case ssaop.OpCtz8:
		return rewriteValueWasm_OpCtz8(v)
	case ssaop.OpCtz8NonZero:
		v.Op = ssaop.OpWasmI64Ctz
		return true
	case ssaop.OpCvt32Fto32:
		v.Op = ssaop.OpWasmI64TruncSatF32S
		return true
	case ssaop.OpCvt32Fto32U:
		v.Op = ssaop.OpWasmI64TruncSatF32U
		return true
	case ssaop.OpCvt32Fto64:
		v.Op = ssaop.OpWasmI64TruncSatF32S
		return true
	case ssaop.OpCvt32Fto64F:
		v.Op = ssaop.OpWasmF64PromoteF32
		return true
	case ssaop.OpCvt32Fto64U:
		v.Op = ssaop.OpWasmI64TruncSatF32U
		return true
	case ssaop.OpCvt32Uto32F:
		return rewriteValueWasm_OpCvt32Uto32F(v)
	case ssaop.OpCvt32Uto64F:
		return rewriteValueWasm_OpCvt32Uto64F(v)
	case ssaop.OpCvt32to32F:
		return rewriteValueWasm_OpCvt32to32F(v)
	case ssaop.OpCvt32to64F:
		return rewriteValueWasm_OpCvt32to64F(v)
	case ssaop.OpCvt64Fto32:
		v.Op = ssaop.OpWasmI64TruncSatF64S
		return true
	case ssaop.OpCvt64Fto32F:
		v.Op = ssaop.OpWasmF32DemoteF64
		return true
	case ssaop.OpCvt64Fto32U:
		v.Op = ssaop.OpWasmI64TruncSatF64U
		return true
	case ssaop.OpCvt64Fto64:
		v.Op = ssaop.OpWasmI64TruncSatF64S
		return true
	case ssaop.OpCvt64Fto64U:
		v.Op = ssaop.OpWasmI64TruncSatF64U
		return true
	case ssaop.OpCvt64Uto32F:
		v.Op = ssaop.OpWasmF32ConvertI64U
		return true
	case ssaop.OpCvt64Uto64F:
		v.Op = ssaop.OpWasmF64ConvertI64U
		return true
	case ssaop.OpCvt64to32F:
		v.Op = ssaop.OpWasmF32ConvertI64S
		return true
	case ssaop.OpCvt64to64F:
		v.Op = ssaop.OpWasmF64ConvertI64S
		return true
	case ssaop.OpCvtBoolToUint8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpDiv16:
		return rewriteValueWasm_OpDiv16(v)
	case ssaop.OpDiv16u:
		return rewriteValueWasm_OpDiv16u(v)
	case ssaop.OpDiv32:
		return rewriteValueWasm_OpDiv32(v)
	case ssaop.OpDiv32F:
		v.Op = ssaop.OpWasmF32Div
		return true
	case ssaop.OpDiv32u:
		return rewriteValueWasm_OpDiv32u(v)
	case ssaop.OpDiv64:
		return rewriteValueWasm_OpDiv64(v)
	case ssaop.OpDiv64F:
		v.Op = ssaop.OpWasmF64Div
		return true
	case ssaop.OpDiv64u:
		v.Op = ssaop.OpWasmI64DivU
		return true
	case ssaop.OpDiv8:
		return rewriteValueWasm_OpDiv8(v)
	case ssaop.OpDiv8u:
		return rewriteValueWasm_OpDiv8u(v)
	case ssaop.OpDivFloat32x4:
		v.Op = ssaop.OpWasmF32x4Div
		return true
	case ssaop.OpDivFloat64x2:
		v.Op = ssaop.OpWasmF64x2Div
		return true
	case ssaop.OpEq16:
		return rewriteValueWasm_OpEq16(v)
	case ssaop.OpEq32:
		return rewriteValueWasm_OpEq32(v)
	case ssaop.OpEq32F:
		v.Op = ssaop.OpWasmF32Eq
		return true
	case ssaop.OpEq64:
		v.Op = ssaop.OpWasmI64Eq
		return true
	case ssaop.OpEq64F:
		v.Op = ssaop.OpWasmF64Eq
		return true
	case ssaop.OpEq8:
		return rewriteValueWasm_OpEq8(v)
	case ssaop.OpEqB:
		v.Op = ssaop.OpWasmI64Eq
		return true
	case ssaop.OpEqPtr:
		v.Op = ssaop.OpWasmI64Eq
		return true
	case ssaop.OpEqualFloat32x4:
		v.Op = ssaop.OpWasmF32x4Eq
		return true
	case ssaop.OpEqualFloat64x2:
		v.Op = ssaop.OpWasmF64x2Eq
		return true
	case ssaop.OpEqualInt16x8:
		v.Op = ssaop.OpWasmI16x8Eq
		return true
	case ssaop.OpEqualInt32x4:
		v.Op = ssaop.OpWasmI32x4Eq
		return true
	case ssaop.OpEqualInt64x2:
		v.Op = ssaop.OpWasmI64x2Eq
		return true
	case ssaop.OpEqualInt8x16:
		v.Op = ssaop.OpWasmI8x16Eq
		return true
	case ssaop.OpEqualUint16x8:
		v.Op = ssaop.OpWasmI16x8Eq
		return true
	case ssaop.OpEqualUint32x4:
		v.Op = ssaop.OpWasmI32x4Eq
		return true
	case ssaop.OpEqualUint64x2:
		v.Op = ssaop.OpWasmI64x2Eq
		return true
	case ssaop.OpEqualUint8x16:
		v.Op = ssaop.OpWasmI8x16Eq
		return true
	case ssaop.OpExtendHi2ToInt64Int32x4:
		v.Op = ssaop.OpWasmI64x2ExtendHighI32x4S
		return true
	case ssaop.OpExtendHi2ToUint64Uint32x4:
		v.Op = ssaop.OpWasmI64x2ExtendHighI32x4U
		return true
	case ssaop.OpExtendHi4ToInt32Int16x8:
		v.Op = ssaop.OpWasmI32x4ExtendHighI16x8S
		return true
	case ssaop.OpExtendHi4ToUint32Uint16x8:
		v.Op = ssaop.OpWasmI32x4ExtendHighI16x8U
		return true
	case ssaop.OpExtendHi8ToInt16Int8x16:
		v.Op = ssaop.OpWasmI16x8ExtendHighI8x16S
		return true
	case ssaop.OpExtendHi8ToUint16Uint8x16:
		v.Op = ssaop.OpWasmI16x8ExtendHighI8x16U
		return true
	case ssaop.OpExtendLo2ToInt64Int32x4:
		v.Op = ssaop.OpWasmI64x2ExtendLowI32x4S
		return true
	case ssaop.OpExtendLo2ToUint64Uint32x4:
		v.Op = ssaop.OpWasmI64x2ExtendLowI32x4U
		return true
	case ssaop.OpExtendLo4ToInt32Int16x8:
		v.Op = ssaop.OpWasmI32x4ExtendLowI16x8S
		return true
	case ssaop.OpExtendLo4ToUint32Uint16x8:
		v.Op = ssaop.OpWasmI32x4ExtendLowI16x8U
		return true
	case ssaop.OpExtendLo8ToInt16Int8x16:
		v.Op = ssaop.OpWasmI16x8ExtendLowI8x16S
		return true
	case ssaop.OpExtendLo8ToUint16Uint8x16:
		v.Op = ssaop.OpWasmI16x8ExtendLowI8x16U
		return true
	case ssaop.OpFloor:
		v.Op = ssaop.OpWasmF64Floor
		return true
	case ssaop.OpFloorFloat32x4:
		v.Op = ssaop.OpWasmF32x4Floor
		return true
	case ssaop.OpFloorFloat64x2:
		v.Op = ssaop.OpWasmF64x2Floor
		return true
	case ssaop.OpGetCallerPC:
		v.Op = ssaop.OpWasmLoweredGetCallerPC
		return true
	case ssaop.OpGetCallerSP:
		v.Op = ssaop.OpWasmLoweredGetCallerSP
		return true
	case ssaop.OpGetClosurePtr:
		v.Op = ssaop.OpWasmLoweredGetClosurePtr
		return true
	case ssaop.OpGetElemFloat32x4:
		v.Op = ssaop.OpWasmF32x4ExtractLane
		return true
	case ssaop.OpGetElemFloat64x2:
		v.Op = ssaop.OpWasmF64x2ExtractLane
		return true
	case ssaop.OpGetElemInt16x8:
		v.Op = ssaop.OpWasmI16x8ExtractLaneS
		return true
	case ssaop.OpGetElemInt32x4:
		v.Op = ssaop.OpWasmI32x4ExtractLane
		return true
	case ssaop.OpGetElemInt64x2:
		v.Op = ssaop.OpWasmI64x2ExtractLane
		return true
	case ssaop.OpGetElemInt8x16:
		v.Op = ssaop.OpWasmI8x16ExtractLaneS
		return true
	case ssaop.OpGetElemUint16x8:
		v.Op = ssaop.OpWasmI16x8ExtractLaneU
		return true
	case ssaop.OpGetElemUint32x4:
		v.Op = ssaop.OpWasmI32x4ExtractLane
		return true
	case ssaop.OpGetElemUint64x2:
		v.Op = ssaop.OpWasmI64x2ExtractLane
		return true
	case ssaop.OpGetElemUint8x16:
		v.Op = ssaop.OpWasmI8x16ExtractLaneU
		return true
	case ssaop.OpGreaterEqualFloat32x4:
		v.Op = ssaop.OpWasmF32x4Ge
		return true
	case ssaop.OpGreaterEqualFloat64x2:
		v.Op = ssaop.OpWasmF64x2Ge
		return true
	case ssaop.OpGreaterEqualInt16x8:
		v.Op = ssaop.OpWasmI16x8GeS
		return true
	case ssaop.OpGreaterEqualInt32x4:
		v.Op = ssaop.OpWasmI32x4GeS
		return true
	case ssaop.OpGreaterEqualInt64x2:
		v.Op = ssaop.OpWasmI64x2GeS
		return true
	case ssaop.OpGreaterEqualInt8x16:
		v.Op = ssaop.OpWasmI8x16GeS
		return true
	case ssaop.OpGreaterEqualUint16x8:
		v.Op = ssaop.OpWasmI16x8GeU
		return true
	case ssaop.OpGreaterEqualUint32x4:
		v.Op = ssaop.OpWasmI32x4GeU
		return true
	case ssaop.OpGreaterEqualUint8x16:
		v.Op = ssaop.OpWasmI8x16GeU
		return true
	case ssaop.OpGreaterFloat32x4:
		v.Op = ssaop.OpWasmF32x4Gt
		return true
	case ssaop.OpGreaterFloat64x2:
		v.Op = ssaop.OpWasmF64x2Gt
		return true
	case ssaop.OpGreaterInt16x8:
		v.Op = ssaop.OpWasmI16x8GtS
		return true
	case ssaop.OpGreaterInt32x4:
		v.Op = ssaop.OpWasmI32x4GtS
		return true
	case ssaop.OpGreaterInt64x2:
		v.Op = ssaop.OpWasmI64x2GtS
		return true
	case ssaop.OpGreaterInt8x16:
		v.Op = ssaop.OpWasmI8x16GtS
		return true
	case ssaop.OpGreaterUint16x8:
		v.Op = ssaop.OpWasmI16x8GtU
		return true
	case ssaop.OpGreaterUint32x4:
		v.Op = ssaop.OpWasmI32x4GtU
		return true
	case ssaop.OpGreaterUint8x16:
		v.Op = ssaop.OpWasmI8x16GtU
		return true
	case ssaop.OpHmul64:
		return rewriteValueWasm_OpHmul64(v)
	case ssaop.OpHmul64u:
		return rewriteValueWasm_OpHmul64u(v)
	case ssaop.OpInterCall:
		v.Op = ssaop.OpWasmLoweredInterCall
		return true
	case ssaop.OpIsInBounds:
		v.Op = ssaop.OpWasmI64LtU
		return true
	case ssaop.OpIsNonNil:
		return rewriteValueWasm_OpIsNonNil(v)
	case ssaop.OpIsSliceInBounds:
		v.Op = ssaop.OpWasmI64LeU
		return true
	case ssaop.OpLast:
		return rewriteValueWasm_OpLast(v)
	case ssaop.OpLeq16:
		return rewriteValueWasm_OpLeq16(v)
	case ssaop.OpLeq16U:
		return rewriteValueWasm_OpLeq16U(v)
	case ssaop.OpLeq32:
		return rewriteValueWasm_OpLeq32(v)
	case ssaop.OpLeq32F:
		v.Op = ssaop.OpWasmF32Le
		return true
	case ssaop.OpLeq32U:
		return rewriteValueWasm_OpLeq32U(v)
	case ssaop.OpLeq64:
		v.Op = ssaop.OpWasmI64LeS
		return true
	case ssaop.OpLeq64F:
		v.Op = ssaop.OpWasmF64Le
		return true
	case ssaop.OpLeq64U:
		v.Op = ssaop.OpWasmI64LeU
		return true
	case ssaop.OpLeq8:
		return rewriteValueWasm_OpLeq8(v)
	case ssaop.OpLeq8U:
		return rewriteValueWasm_OpLeq8U(v)
	case ssaop.OpLess16:
		return rewriteValueWasm_OpLess16(v)
	case ssaop.OpLess16U:
		return rewriteValueWasm_OpLess16U(v)
	case ssaop.OpLess32:
		return rewriteValueWasm_OpLess32(v)
	case ssaop.OpLess32F:
		v.Op = ssaop.OpWasmF32Lt
		return true
	case ssaop.OpLess32U:
		return rewriteValueWasm_OpLess32U(v)
	case ssaop.OpLess64:
		v.Op = ssaop.OpWasmI64LtS
		return true
	case ssaop.OpLess64F:
		v.Op = ssaop.OpWasmF64Lt
		return true
	case ssaop.OpLess64U:
		v.Op = ssaop.OpWasmI64LtU
		return true
	case ssaop.OpLess8:
		return rewriteValueWasm_OpLess8(v)
	case ssaop.OpLess8U:
		return rewriteValueWasm_OpLess8U(v)
	case ssaop.OpLessEqualFloat32x4:
		v.Op = ssaop.OpWasmF32x4Le
		return true
	case ssaop.OpLessEqualFloat64x2:
		v.Op = ssaop.OpWasmF64x2Le
		return true
	case ssaop.OpLessEqualInt16x8:
		v.Op = ssaop.OpWasmI16x8LeS
		return true
	case ssaop.OpLessEqualInt32x4:
		v.Op = ssaop.OpWasmI32x4LeS
		return true
	case ssaop.OpLessEqualInt64x2:
		v.Op = ssaop.OpWasmI64x2LeS
		return true
	case ssaop.OpLessEqualInt8x16:
		v.Op = ssaop.OpWasmI8x16LeS
		return true
	case ssaop.OpLessEqualUint16x8:
		v.Op = ssaop.OpWasmI16x8LeU
		return true
	case ssaop.OpLessEqualUint32x4:
		v.Op = ssaop.OpWasmI32x4LeU
		return true
	case ssaop.OpLessEqualUint8x16:
		v.Op = ssaop.OpWasmI8x16LeU
		return true
	case ssaop.OpLessFloat32x4:
		v.Op = ssaop.OpWasmF32x4Lt
		return true
	case ssaop.OpLessFloat64x2:
		v.Op = ssaop.OpWasmF64x2Lt
		return true
	case ssaop.OpLessInt16x8:
		v.Op = ssaop.OpWasmI16x8LtS
		return true
	case ssaop.OpLessInt32x4:
		v.Op = ssaop.OpWasmI32x4LtS
		return true
	case ssaop.OpLessInt64x2:
		v.Op = ssaop.OpWasmI64x2LtS
		return true
	case ssaop.OpLessInt8x16:
		v.Op = ssaop.OpWasmI8x16LtS
		return true
	case ssaop.OpLessUint16x8:
		v.Op = ssaop.OpWasmI16x8LtU
		return true
	case ssaop.OpLessUint32x4:
		v.Op = ssaop.OpWasmI32x4LtU
		return true
	case ssaop.OpLessUint8x16:
		v.Op = ssaop.OpWasmI8x16LtU
		return true
	case ssaop.OpLoad:
		return rewriteValueWasm_OpLoad(v)
	case ssaop.OpLocalAddr:
		return rewriteValueWasm_OpLocalAddr(v)
	case ssaop.OpLookupOrZeroInt8x16:
		v.Op = ssaop.OpWasmI8x16Swizzle
		return true
	case ssaop.OpLsh16x16:
		return rewriteValueWasm_OpLsh16x16(v)
	case ssaop.OpLsh16x32:
		return rewriteValueWasm_OpLsh16x32(v)
	case ssaop.OpLsh16x64:
		v.Op = ssaop.OpLsh64x64
		return true
	case ssaop.OpLsh16x8:
		return rewriteValueWasm_OpLsh16x8(v)
	case ssaop.OpLsh32x16:
		return rewriteValueWasm_OpLsh32x16(v)
	case ssaop.OpLsh32x32:
		return rewriteValueWasm_OpLsh32x32(v)
	case ssaop.OpLsh32x64:
		v.Op = ssaop.OpLsh64x64
		return true
	case ssaop.OpLsh32x8:
		return rewriteValueWasm_OpLsh32x8(v)
	case ssaop.OpLsh64x16:
		return rewriteValueWasm_OpLsh64x16(v)
	case ssaop.OpLsh64x32:
		return rewriteValueWasm_OpLsh64x32(v)
	case ssaop.OpLsh64x64:
		return rewriteValueWasm_OpLsh64x64(v)
	case ssaop.OpLsh64x8:
		return rewriteValueWasm_OpLsh64x8(v)
	case ssaop.OpLsh8x16:
		return rewriteValueWasm_OpLsh8x16(v)
	case ssaop.OpLsh8x32:
		return rewriteValueWasm_OpLsh8x32(v)
	case ssaop.OpLsh8x64:
		v.Op = ssaop.OpLsh64x64
		return true
	case ssaop.OpLsh8x8:
		return rewriteValueWasm_OpLsh8x8(v)
	case ssaop.OpMaxFloat32x4:
		v.Op = ssaop.OpWasmF32x4Max
		return true
	case ssaop.OpMaxFloat64x2:
		v.Op = ssaop.OpWasmF64x2Max
		return true
	case ssaop.OpMaxInt16x8:
		v.Op = ssaop.OpWasmI16x8MaxS
		return true
	case ssaop.OpMaxInt32x4:
		v.Op = ssaop.OpWasmI32x4MaxS
		return true
	case ssaop.OpMaxInt8x16:
		v.Op = ssaop.OpWasmI8x16MaxS
		return true
	case ssaop.OpMaxUint16x8:
		v.Op = ssaop.OpWasmI16x8MaxU
		return true
	case ssaop.OpMaxUint32x4:
		v.Op = ssaop.OpWasmI32x4MaxU
		return true
	case ssaop.OpMaxUint8x16:
		v.Op = ssaop.OpWasmI8x16MaxU
		return true
	case ssaop.OpMinFloat32x4:
		v.Op = ssaop.OpWasmF32x4Min
		return true
	case ssaop.OpMinFloat64x2:
		v.Op = ssaop.OpWasmF64x2Min
		return true
	case ssaop.OpMinInt16x8:
		v.Op = ssaop.OpWasmI16x8MinS
		return true
	case ssaop.OpMinInt32x4:
		v.Op = ssaop.OpWasmI32x4MinS
		return true
	case ssaop.OpMinInt8x16:
		v.Op = ssaop.OpWasmI8x16MinS
		return true
	case ssaop.OpMinUint16x8:
		v.Op = ssaop.OpWasmI16x8MinU
		return true
	case ssaop.OpMinUint32x4:
		v.Op = ssaop.OpWasmI32x4MinU
		return true
	case ssaop.OpMinUint8x16:
		v.Op = ssaop.OpWasmI8x16MinU
		return true
	case ssaop.OpMod16:
		return rewriteValueWasm_OpMod16(v)
	case ssaop.OpMod16u:
		return rewriteValueWasm_OpMod16u(v)
	case ssaop.OpMod32:
		return rewriteValueWasm_OpMod32(v)
	case ssaop.OpMod32u:
		return rewriteValueWasm_OpMod32u(v)
	case ssaop.OpMod64:
		return rewriteValueWasm_OpMod64(v)
	case ssaop.OpMod64u:
		v.Op = ssaop.OpWasmI64RemU
		return true
	case ssaop.OpMod8:
		return rewriteValueWasm_OpMod8(v)
	case ssaop.OpMod8u:
		return rewriteValueWasm_OpMod8u(v)
	case ssaop.OpMove:
		return rewriteValueWasm_OpMove(v)
	case ssaop.OpMul16:
		v.Op = ssaop.OpWasmI64Mul
		return true
	case ssaop.OpMul32:
		v.Op = ssaop.OpWasmI64Mul
		return true
	case ssaop.OpMul32F:
		v.Op = ssaop.OpWasmF32Mul
		return true
	case ssaop.OpMul64:
		v.Op = ssaop.OpWasmI64Mul
		return true
	case ssaop.OpMul64F:
		v.Op = ssaop.OpWasmF64Mul
		return true
	case ssaop.OpMul8:
		v.Op = ssaop.OpWasmI64Mul
		return true
	case ssaop.OpMulFloat32x4:
		v.Op = ssaop.OpWasmF32x4Mul
		return true
	case ssaop.OpMulFloat64x2:
		v.Op = ssaop.OpWasmF64x2Mul
		return true
	case ssaop.OpMulInt16x8:
		v.Op = ssaop.OpWasmI16x8Mul
		return true
	case ssaop.OpMulInt32x4:
		v.Op = ssaop.OpWasmI32x4Mul
		return true
	case ssaop.OpMulInt64x2:
		v.Op = ssaop.OpWasmI64x2Mul
		return true
	case ssaop.OpMulUint16x8:
		v.Op = ssaop.OpWasmI16x8Mul
		return true
	case ssaop.OpMulUint32x4:
		v.Op = ssaop.OpWasmI32x4Mul
		return true
	case ssaop.OpMulUint64x2:
		v.Op = ssaop.OpWasmI64x2Mul
		return true
	case ssaop.OpMulWidenHiInt16x8:
		v.Op = ssaop.OpWasmI32x4ExtmulHighI16x8S
		return true
	case ssaop.OpMulWidenHiInt32x4:
		v.Op = ssaop.OpWasmI64x2ExtmulHighI32x4S
		return true
	case ssaop.OpMulWidenHiInt8x16:
		v.Op = ssaop.OpWasmI16x8ExtmulHighI8x16S
		return true
	case ssaop.OpMulWidenHiUint16x8:
		v.Op = ssaop.OpWasmI32x4ExtmulHighI16x8U
		return true
	case ssaop.OpMulWidenHiUint32x4:
		v.Op = ssaop.OpWasmI64x2ExtmulHighI32x4U
		return true
	case ssaop.OpMulWidenHiUint8x16:
		v.Op = ssaop.OpWasmI16x8ExtmulHighI8x16U
		return true
	case ssaop.OpMulWidenLoInt16x8:
		v.Op = ssaop.OpWasmI32x4ExtmulLowI16x8S
		return true
	case ssaop.OpMulWidenLoInt32x4:
		v.Op = ssaop.OpWasmI64x2ExtmulLowI32x4S
		return true
	case ssaop.OpMulWidenLoInt8x16:
		v.Op = ssaop.OpWasmI16x8ExtmulLowI8x16S
		return true
	case ssaop.OpMulWidenLoUint16x8:
		v.Op = ssaop.OpWasmI32x4ExtmulLowI16x8U
		return true
	case ssaop.OpMulWidenLoUint32x4:
		v.Op = ssaop.OpWasmI64x2ExtmulLowI32x4U
		return true
	case ssaop.OpMulWidenLoUint8x16:
		v.Op = ssaop.OpWasmI16x8ExtmulLowI8x16U
		return true
	case ssaop.OpNeg16:
		return rewriteValueWasm_OpNeg16(v)
	case ssaop.OpNeg32:
		return rewriteValueWasm_OpNeg32(v)
	case ssaop.OpNeg32F:
		v.Op = ssaop.OpWasmF32Neg
		return true
	case ssaop.OpNeg64:
		return rewriteValueWasm_OpNeg64(v)
	case ssaop.OpNeg64F:
		v.Op = ssaop.OpWasmF64Neg
		return true
	case ssaop.OpNeg8:
		return rewriteValueWasm_OpNeg8(v)
	case ssaop.OpNegFloat32x4:
		v.Op = ssaop.OpWasmF32x4Neg
		return true
	case ssaop.OpNegFloat64x2:
		v.Op = ssaop.OpWasmF64x2Neg
		return true
	case ssaop.OpNegInt16x8:
		v.Op = ssaop.OpWasmI16x8Neg
		return true
	case ssaop.OpNegInt32x4:
		v.Op = ssaop.OpWasmI32x4Neg
		return true
	case ssaop.OpNegInt64x2:
		v.Op = ssaop.OpWasmI64x2Neg
		return true
	case ssaop.OpNegInt8x16:
		v.Op = ssaop.OpWasmI8x16Neg
		return true
	case ssaop.OpNeq16:
		return rewriteValueWasm_OpNeq16(v)
	case ssaop.OpNeq32:
		return rewriteValueWasm_OpNeq32(v)
	case ssaop.OpNeq32F:
		v.Op = ssaop.OpWasmF32Ne
		return true
	case ssaop.OpNeq64:
		v.Op = ssaop.OpWasmI64Ne
		return true
	case ssaop.OpNeq64F:
		v.Op = ssaop.OpWasmF64Ne
		return true
	case ssaop.OpNeq8:
		return rewriteValueWasm_OpNeq8(v)
	case ssaop.OpNeqB:
		v.Op = ssaop.OpWasmI64Ne
		return true
	case ssaop.OpNeqPtr:
		v.Op = ssaop.OpWasmI64Ne
		return true
	case ssaop.OpNilCheck:
		v.Op = ssaop.OpWasmLoweredNilCheck
		return true
	case ssaop.OpNot:
		v.Op = ssaop.OpWasmI64Eqz
		return true
	case ssaop.OpNotEqualFloat32x4:
		v.Op = ssaop.OpWasmF32x4Ne
		return true
	case ssaop.OpNotEqualFloat64x2:
		v.Op = ssaop.OpWasmF64x2Ne
		return true
	case ssaop.OpNotEqualInt16x8:
		v.Op = ssaop.OpWasmI16x8Ne
		return true
	case ssaop.OpNotEqualInt32x4:
		v.Op = ssaop.OpWasmI32x4Ne
		return true
	case ssaop.OpNotEqualInt64x2:
		v.Op = ssaop.OpWasmI64x2Ne
		return true
	case ssaop.OpNotEqualInt8x16:
		v.Op = ssaop.OpWasmI8x16Ne
		return true
	case ssaop.OpNotEqualUint16x8:
		v.Op = ssaop.OpWasmI16x8Ne
		return true
	case ssaop.OpNotEqualUint32x4:
		v.Op = ssaop.OpWasmI32x4Ne
		return true
	case ssaop.OpNotEqualUint64x2:
		v.Op = ssaop.OpWasmI64x2Ne
		return true
	case ssaop.OpNotEqualUint8x16:
		v.Op = ssaop.OpWasmI8x16Ne
		return true
	case ssaop.OpNotInt16x8:
		v.Op = ssaop.OpWasmV128Not
		return true
	case ssaop.OpNotInt32x4:
		v.Op = ssaop.OpWasmV128Not
		return true
	case ssaop.OpNotInt64x2:
		v.Op = ssaop.OpWasmV128Not
		return true
	case ssaop.OpNotInt8x16:
		v.Op = ssaop.OpWasmV128Not
		return true
	case ssaop.OpNotUint16x8:
		v.Op = ssaop.OpWasmV128Not
		return true
	case ssaop.OpNotUint32x4:
		v.Op = ssaop.OpWasmV128Not
		return true
	case ssaop.OpNotUint64x2:
		v.Op = ssaop.OpWasmV128Not
		return true
	case ssaop.OpNotUint8x16:
		v.Op = ssaop.OpWasmV128Not
		return true
	case ssaop.OpOffPtr:
		v.Op = ssaop.OpWasmI64AddConst
		return true
	case ssaop.OpOnesCountInt8x16:
		v.Op = ssaop.OpWasmI8x16Popcnt
		return true
	case ssaop.OpOr16:
		v.Op = ssaop.OpWasmI64Or
		return true
	case ssaop.OpOr32:
		v.Op = ssaop.OpWasmI64Or
		return true
	case ssaop.OpOr64:
		v.Op = ssaop.OpWasmI64Or
		return true
	case ssaop.OpOr8:
		v.Op = ssaop.OpWasmI64Or
		return true
	case ssaop.OpOrB:
		v.Op = ssaop.OpWasmI64Or
		return true
	case ssaop.OpOrInt16x8:
		v.Op = ssaop.OpWasmV128Or
		return true
	case ssaop.OpOrInt32x4:
		v.Op = ssaop.OpWasmV128Or
		return true
	case ssaop.OpOrInt64x2:
		v.Op = ssaop.OpWasmV128Or
		return true
	case ssaop.OpOrInt8x16:
		v.Op = ssaop.OpWasmV128Or
		return true
	case ssaop.OpOrUint16x8:
		v.Op = ssaop.OpWasmV128Or
		return true
	case ssaop.OpOrUint32x4:
		v.Op = ssaop.OpWasmV128Or
		return true
	case ssaop.OpOrUint64x2:
		v.Op = ssaop.OpWasmV128Or
		return true
	case ssaop.OpOrUint8x16:
		v.Op = ssaop.OpWasmV128Or
		return true
	case ssaop.OpPopCount16:
		return rewriteValueWasm_OpPopCount16(v)
	case ssaop.OpPopCount32:
		return rewriteValueWasm_OpPopCount32(v)
	case ssaop.OpPopCount64:
		v.Op = ssaop.OpWasmI64Popcnt
		return true
	case ssaop.OpPopCount8:
		return rewriteValueWasm_OpPopCount8(v)
	case ssaop.OpRotateAllLeftVarInt16x8:
		return rewriteValueWasm_OpRotateAllLeftVarInt16x8(v)
	case ssaop.OpRotateAllLeftVarInt32x4:
		return rewriteValueWasm_OpRotateAllLeftVarInt32x4(v)
	case ssaop.OpRotateAllLeftVarInt64x2:
		return rewriteValueWasm_OpRotateAllLeftVarInt64x2(v)
	case ssaop.OpRotateAllLeftVarInt8x16:
		return rewriteValueWasm_OpRotateAllLeftVarInt8x16(v)
	case ssaop.OpRotateAllLeftVarUint16x8:
		return rewriteValueWasm_OpRotateAllLeftVarUint16x8(v)
	case ssaop.OpRotateAllLeftVarUint32x4:
		return rewriteValueWasm_OpRotateAllLeftVarUint32x4(v)
	case ssaop.OpRotateAllLeftVarUint64x2:
		return rewriteValueWasm_OpRotateAllLeftVarUint64x2(v)
	case ssaop.OpRotateAllLeftVarUint8x16:
		return rewriteValueWasm_OpRotateAllLeftVarUint8x16(v)
	case ssaop.OpRotateAllRightVarInt16x8:
		return rewriteValueWasm_OpRotateAllRightVarInt16x8(v)
	case ssaop.OpRotateAllRightVarInt32x4:
		return rewriteValueWasm_OpRotateAllRightVarInt32x4(v)
	case ssaop.OpRotateAllRightVarInt64x2:
		return rewriteValueWasm_OpRotateAllRightVarInt64x2(v)
	case ssaop.OpRotateAllRightVarInt8x16:
		return rewriteValueWasm_OpRotateAllRightVarInt8x16(v)
	case ssaop.OpRotateAllRightVarUint16x8:
		return rewriteValueWasm_OpRotateAllRightVarUint16x8(v)
	case ssaop.OpRotateAllRightVarUint32x4:
		return rewriteValueWasm_OpRotateAllRightVarUint32x4(v)
	case ssaop.OpRotateAllRightVarUint64x2:
		return rewriteValueWasm_OpRotateAllRightVarUint64x2(v)
	case ssaop.OpRotateAllRightVarUint8x16:
		return rewriteValueWasm_OpRotateAllRightVarUint8x16(v)
	case ssaop.OpRotateLeft16:
		return rewriteValueWasm_OpRotateLeft16(v)
	case ssaop.OpRotateLeft32:
		v.Op = ssaop.OpWasmI32Rotl
		return true
	case ssaop.OpRotateLeft64:
		v.Op = ssaop.OpWasmI64Rotl
		return true
	case ssaop.OpRotateLeft8:
		return rewriteValueWasm_OpRotateLeft8(v)
	case ssaop.OpRound32F:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpRound64F:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpRoundFloat32x4:
		v.Op = ssaop.OpWasmF32x4Nearest
		return true
	case ssaop.OpRoundFloat64x2:
		v.Op = ssaop.OpWasmF64x2Nearest
		return true
	case ssaop.OpRoundToEven:
		v.Op = ssaop.OpWasmF64Nearest
		return true
	case ssaop.OpRsh16Ux16:
		return rewriteValueWasm_OpRsh16Ux16(v)
	case ssaop.OpRsh16Ux32:
		return rewriteValueWasm_OpRsh16Ux32(v)
	case ssaop.OpRsh16Ux64:
		return rewriteValueWasm_OpRsh16Ux64(v)
	case ssaop.OpRsh16Ux8:
		return rewriteValueWasm_OpRsh16Ux8(v)
	case ssaop.OpRsh16x16:
		return rewriteValueWasm_OpRsh16x16(v)
	case ssaop.OpRsh16x32:
		return rewriteValueWasm_OpRsh16x32(v)
	case ssaop.OpRsh16x64:
		return rewriteValueWasm_OpRsh16x64(v)
	case ssaop.OpRsh16x8:
		return rewriteValueWasm_OpRsh16x8(v)
	case ssaop.OpRsh32Ux16:
		return rewriteValueWasm_OpRsh32Ux16(v)
	case ssaop.OpRsh32Ux32:
		return rewriteValueWasm_OpRsh32Ux32(v)
	case ssaop.OpRsh32Ux64:
		return rewriteValueWasm_OpRsh32Ux64(v)
	case ssaop.OpRsh32Ux8:
		return rewriteValueWasm_OpRsh32Ux8(v)
	case ssaop.OpRsh32x16:
		return rewriteValueWasm_OpRsh32x16(v)
	case ssaop.OpRsh32x32:
		return rewriteValueWasm_OpRsh32x32(v)
	case ssaop.OpRsh32x64:
		return rewriteValueWasm_OpRsh32x64(v)
	case ssaop.OpRsh32x8:
		return rewriteValueWasm_OpRsh32x8(v)
	case ssaop.OpRsh64Ux16:
		return rewriteValueWasm_OpRsh64Ux16(v)
	case ssaop.OpRsh64Ux32:
		return rewriteValueWasm_OpRsh64Ux32(v)
	case ssaop.OpRsh64Ux64:
		return rewriteValueWasm_OpRsh64Ux64(v)
	case ssaop.OpRsh64Ux8:
		return rewriteValueWasm_OpRsh64Ux8(v)
	case ssaop.OpRsh64x16:
		return rewriteValueWasm_OpRsh64x16(v)
	case ssaop.OpRsh64x32:
		return rewriteValueWasm_OpRsh64x32(v)
	case ssaop.OpRsh64x64:
		return rewriteValueWasm_OpRsh64x64(v)
	case ssaop.OpRsh64x8:
		return rewriteValueWasm_OpRsh64x8(v)
	case ssaop.OpRsh8Ux16:
		return rewriteValueWasm_OpRsh8Ux16(v)
	case ssaop.OpRsh8Ux32:
		return rewriteValueWasm_OpRsh8Ux32(v)
	case ssaop.OpRsh8Ux64:
		return rewriteValueWasm_OpRsh8Ux64(v)
	case ssaop.OpRsh8Ux8:
		return rewriteValueWasm_OpRsh8Ux8(v)
	case ssaop.OpRsh8x16:
		return rewriteValueWasm_OpRsh8x16(v)
	case ssaop.OpRsh8x32:
		return rewriteValueWasm_OpRsh8x32(v)
	case ssaop.OpRsh8x64:
		return rewriteValueWasm_OpRsh8x64(v)
	case ssaop.OpRsh8x8:
		return rewriteValueWasm_OpRsh8x8(v)
	case ssaop.OpSelect0:
		return rewriteValueWasm_OpSelect0(v)
	case ssaop.OpSelect1:
		return rewriteValueWasm_OpSelect1(v)
	case ssaop.OpSetElemFloat32x4:
		v.Op = ssaop.OpWasmF32x4ReplaceLane
		return true
	case ssaop.OpSetElemFloat64x2:
		v.Op = ssaop.OpWasmF64x2ReplaceLane
		return true
	case ssaop.OpSetElemInt16x8:
		v.Op = ssaop.OpWasmI16x8ReplaceLane
		return true
	case ssaop.OpSetElemInt32x4:
		v.Op = ssaop.OpWasmI32x4ReplaceLane
		return true
	case ssaop.OpSetElemInt64x2:
		v.Op = ssaop.OpWasmI64x2ReplaceLane
		return true
	case ssaop.OpSetElemInt8x16:
		v.Op = ssaop.OpWasmI8x16ReplaceLane
		return true
	case ssaop.OpSetElemUint16x8:
		v.Op = ssaop.OpWasmI16x8ReplaceLane
		return true
	case ssaop.OpSetElemUint32x4:
		v.Op = ssaop.OpWasmI32x4ReplaceLane
		return true
	case ssaop.OpSetElemUint64x2:
		v.Op = ssaop.OpWasmI64x2ReplaceLane
		return true
	case ssaop.OpSetElemUint8x16:
		v.Op = ssaop.OpWasmI8x16ReplaceLane
		return true
	case ssaop.OpShiftAllLeftInt16x8:
		return rewriteValueWasm_OpShiftAllLeftInt16x8(v)
	case ssaop.OpShiftAllLeftInt32x4:
		return rewriteValueWasm_OpShiftAllLeftInt32x4(v)
	case ssaop.OpShiftAllLeftInt64x2:
		return rewriteValueWasm_OpShiftAllLeftInt64x2(v)
	case ssaop.OpShiftAllLeftInt8x16:
		return rewriteValueWasm_OpShiftAllLeftInt8x16(v)
	case ssaop.OpShiftAllLeftUint16x8:
		return rewriteValueWasm_OpShiftAllLeftUint16x8(v)
	case ssaop.OpShiftAllLeftUint32x4:
		return rewriteValueWasm_OpShiftAllLeftUint32x4(v)
	case ssaop.OpShiftAllLeftUint64x2:
		return rewriteValueWasm_OpShiftAllLeftUint64x2(v)
	case ssaop.OpShiftAllLeftUint8x16:
		return rewriteValueWasm_OpShiftAllLeftUint8x16(v)
	case ssaop.OpShiftAllRightInt16x8:
		return rewriteValueWasm_OpShiftAllRightInt16x8(v)
	case ssaop.OpShiftAllRightInt32x4:
		return rewriteValueWasm_OpShiftAllRightInt32x4(v)
	case ssaop.OpShiftAllRightInt64x2:
		return rewriteValueWasm_OpShiftAllRightInt64x2(v)
	case ssaop.OpShiftAllRightInt8x16:
		return rewriteValueWasm_OpShiftAllRightInt8x16(v)
	case ssaop.OpShiftAllRightUint16x8:
		return rewriteValueWasm_OpShiftAllRightUint16x8(v)
	case ssaop.OpShiftAllRightUint32x4:
		return rewriteValueWasm_OpShiftAllRightUint32x4(v)
	case ssaop.OpShiftAllRightUint64x2:
		return rewriteValueWasm_OpShiftAllRightUint64x2(v)
	case ssaop.OpShiftAllRightUint8x16:
		return rewriteValueWasm_OpShiftAllRightUint8x16(v)
	case ssaop.OpSignExt16to32:
		return rewriteValueWasm_OpSignExt16to32(v)
	case ssaop.OpSignExt16to64:
		return rewriteValueWasm_OpSignExt16to64(v)
	case ssaop.OpSignExt32to64:
		return rewriteValueWasm_OpSignExt32to64(v)
	case ssaop.OpSignExt8to16:
		return rewriteValueWasm_OpSignExt8to16(v)
	case ssaop.OpSignExt8to32:
		return rewriteValueWasm_OpSignExt8to32(v)
	case ssaop.OpSignExt8to64:
		return rewriteValueWasm_OpSignExt8to64(v)
	case ssaop.OpSlicemask:
		return rewriteValueWasm_OpSlicemask(v)
	case ssaop.OpSqrt:
		v.Op = ssaop.OpWasmF64Sqrt
		return true
	case ssaop.OpSqrt32:
		v.Op = ssaop.OpWasmF32Sqrt
		return true
	case ssaop.OpSqrtFloat32x4:
		v.Op = ssaop.OpWasmF32x4Sqrt
		return true
	case ssaop.OpSqrtFloat64x2:
		v.Op = ssaop.OpWasmF64x2Sqrt
		return true
	case ssaop.OpStaticCall:
		v.Op = ssaop.OpWasmLoweredStaticCall
		return true
	case ssaop.OpStore:
		return rewriteValueWasm_OpStore(v)
	case ssaop.OpSub16:
		v.Op = ssaop.OpWasmI64Sub
		return true
	case ssaop.OpSub32:
		v.Op = ssaop.OpWasmI64Sub
		return true
	case ssaop.OpSub32F:
		v.Op = ssaop.OpWasmF32Sub
		return true
	case ssaop.OpSub64:
		v.Op = ssaop.OpWasmI64Sub
		return true
	case ssaop.OpSub64F:
		v.Op = ssaop.OpWasmF64Sub
		return true
	case ssaop.OpSub8:
		v.Op = ssaop.OpWasmI64Sub
		return true
	case ssaop.OpSubFloat32x4:
		v.Op = ssaop.OpWasmF32x4Sub
		return true
	case ssaop.OpSubFloat64x2:
		v.Op = ssaop.OpWasmF64x2Sub
		return true
	case ssaop.OpSubInt16x8:
		v.Op = ssaop.OpWasmI16x8Sub
		return true
	case ssaop.OpSubInt32x4:
		v.Op = ssaop.OpWasmI32x4Sub
		return true
	case ssaop.OpSubInt64x2:
		v.Op = ssaop.OpWasmI64x2Sub
		return true
	case ssaop.OpSubInt8x16:
		v.Op = ssaop.OpWasmI8x16Sub
		return true
	case ssaop.OpSubPtr:
		v.Op = ssaop.OpWasmI64Sub
		return true
	case ssaop.OpSubSaturatedInt16x8:
		v.Op = ssaop.OpWasmI16x8SubSatS
		return true
	case ssaop.OpSubSaturatedInt8x16:
		v.Op = ssaop.OpWasmI8x16SubSatS
		return true
	case ssaop.OpSubSaturatedUint16x8:
		v.Op = ssaop.OpWasmI16x8SubSatU
		return true
	case ssaop.OpSubSaturatedUint8x16:
		v.Op = ssaop.OpWasmI8x16SubSatU
		return true
	case ssaop.OpTailCall:
		v.Op = ssaop.OpWasmLoweredTailCall
		return true
	case ssaop.OpTailCallInter:
		v.Op = ssaop.OpWasmLoweredTailCallInter
		return true
	case ssaop.OpTrunc:
		v.Op = ssaop.OpWasmF64Trunc
		return true
	case ssaop.OpTrunc16to8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpTrunc32to16:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpTrunc32to8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpTrunc64to16:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpTrunc64to32:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpTrunc64to8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpTruncFloat32x4:
		v.Op = ssaop.OpWasmF32x4Trunc
		return true
	case ssaop.OpTruncFloat64x2:
		v.Op = ssaop.OpWasmF64x2Trunc
		return true
	case ssaop.OpWB:
		v.Op = ssaop.OpWasmLoweredWB
		return true
	case ssaop.OpWasmF32DemoteF64:
		return rewriteValueWasm_OpWasmF32DemoteF64(v)
	case ssaop.OpWasmF64Add:
		return rewriteValueWasm_OpWasmF64Add(v)
	case ssaop.OpWasmF64Mul:
		return rewriteValueWasm_OpWasmF64Mul(v)
	case ssaop.OpWasmI64Add:
		return rewriteValueWasm_OpWasmI64Add(v)
	case ssaop.OpWasmI64AddConst:
		return rewriteValueWasm_OpWasmI64AddConst(v)
	case ssaop.OpWasmI64And:
		return rewriteValueWasm_OpWasmI64And(v)
	case ssaop.OpWasmI64Eq:
		return rewriteValueWasm_OpWasmI64Eq(v)
	case ssaop.OpWasmI64Eqz:
		return rewriteValueWasm_OpWasmI64Eqz(v)
	case ssaop.OpWasmI64Extend16S:
		return rewriteValueWasm_OpWasmI64Extend16S(v)
	case ssaop.OpWasmI64Extend32S:
		return rewriteValueWasm_OpWasmI64Extend32S(v)
	case ssaop.OpWasmI64Extend8S:
		return rewriteValueWasm_OpWasmI64Extend8S(v)
	case ssaop.OpWasmI64LeU:
		return rewriteValueWasm_OpWasmI64LeU(v)
	case ssaop.OpWasmI64Load:
		return rewriteValueWasm_OpWasmI64Load(v)
	case ssaop.OpWasmI64Load16S:
		return rewriteValueWasm_OpWasmI64Load16S(v)
	case ssaop.OpWasmI64Load16U:
		return rewriteValueWasm_OpWasmI64Load16U(v)
	case ssaop.OpWasmI64Load32S:
		return rewriteValueWasm_OpWasmI64Load32S(v)
	case ssaop.OpWasmI64Load32U:
		return rewriteValueWasm_OpWasmI64Load32U(v)
	case ssaop.OpWasmI64Load8S:
		return rewriteValueWasm_OpWasmI64Load8S(v)
	case ssaop.OpWasmI64Load8U:
		return rewriteValueWasm_OpWasmI64Load8U(v)
	case ssaop.OpWasmI64LtU:
		return rewriteValueWasm_OpWasmI64LtU(v)
	case ssaop.OpWasmI64Mul:
		return rewriteValueWasm_OpWasmI64Mul(v)
	case ssaop.OpWasmI64Ne:
		return rewriteValueWasm_OpWasmI64Ne(v)
	case ssaop.OpWasmI64Or:
		return rewriteValueWasm_OpWasmI64Or(v)
	case ssaop.OpWasmI64Shl:
		return rewriteValueWasm_OpWasmI64Shl(v)
	case ssaop.OpWasmI64ShrS:
		return rewriteValueWasm_OpWasmI64ShrS(v)
	case ssaop.OpWasmI64ShrU:
		return rewriteValueWasm_OpWasmI64ShrU(v)
	case ssaop.OpWasmI64Store:
		return rewriteValueWasm_OpWasmI64Store(v)
	case ssaop.OpWasmI64Store16:
		return rewriteValueWasm_OpWasmI64Store16(v)
	case ssaop.OpWasmI64Store32:
		return rewriteValueWasm_OpWasmI64Store32(v)
	case ssaop.OpWasmI64Store8:
		return rewriteValueWasm_OpWasmI64Store8(v)
	case ssaop.OpWasmI64Sub:
		return rewriteValueWasm_OpWasmI64Sub(v)
	case ssaop.OpWasmI64Xor:
		return rewriteValueWasm_OpWasmI64Xor(v)
	case ssaop.OpXor16:
		v.Op = ssaop.OpWasmI64Xor
		return true
	case ssaop.OpXor32:
		v.Op = ssaop.OpWasmI64Xor
		return true
	case ssaop.OpXor64:
		v.Op = ssaop.OpWasmI64Xor
		return true
	case ssaop.OpXor8:
		v.Op = ssaop.OpWasmI64Xor
		return true
	case ssaop.OpXorInt16x8:
		v.Op = ssaop.OpWasmV128Xor
		return true
	case ssaop.OpXorInt32x4:
		v.Op = ssaop.OpWasmV128Xor
		return true
	case ssaop.OpXorInt64x2:
		v.Op = ssaop.OpWasmV128Xor
		return true
	case ssaop.OpXorInt8x16:
		v.Op = ssaop.OpWasmV128Xor
		return true
	case ssaop.OpXorUint16x8:
		v.Op = ssaop.OpWasmV128Xor
		return true
	case ssaop.OpXorUint32x4:
		v.Op = ssaop.OpWasmV128Xor
		return true
	case ssaop.OpXorUint64x2:
		v.Op = ssaop.OpWasmV128Xor
		return true
	case ssaop.OpXorUint8x16:
		v.Op = ssaop.OpWasmV128Xor
		return true
	case ssaop.OpZero:
		return rewriteValueWasm_OpZero(v)
	case ssaop.OpZeroExt16to32:
		return rewriteValueWasm_OpZeroExt16to32(v)
	case ssaop.OpZeroExt16to64:
		return rewriteValueWasm_OpZeroExt16to64(v)
	case ssaop.OpZeroExt32to64:
		return rewriteValueWasm_OpZeroExt32to64(v)
	case ssaop.OpZeroExt8to16:
		return rewriteValueWasm_OpZeroExt8to16(v)
	case ssaop.OpZeroExt8to32:
		return rewriteValueWasm_OpZeroExt8to32(v)
	case ssaop.OpZeroExt8to64:
		return rewriteValueWasm_OpZeroExt8to64(v)
	case ssaop.OpZeroSIMD:
		v.Op = ssaop.OpWasmV128Zero
		return true
	}
	return false
}
func rewriteValueWasm_OpAddr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (LoweredAddr {sym} [0] base)
	for {
		sym := AuxToSym(v.Aux)
		base := v_0
		v.Reset(ssaop.OpWasmLoweredAddr)
		v.AuxInt = Int32ToAuxInt(0)
		v.Aux = SymToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValueWasm_OpAvg64u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Avg64u x y)
	// result: (I64Add (I64ShrU (I64Sub x y) (I64Const [1])) y)
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64Add)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64ShrU, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v1.AddArg2(x, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v2.AuxInt = Int64ToAuxInt(1)
		v0.AddArg2(v1, v2)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValueWasm_OpBitLen16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen16 x)
	// result: (BitLen64 (ZeroExt16to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpBitLen64)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpBitLen32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen32 x)
	// result: (BitLen64 (ZeroExt32to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpBitLen64)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpBitLen64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen64 x)
	// result: (I64Sub (I64Const [64]) (I64Clz x))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Sub)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(64)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Clz, typ.Int64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpBitLen8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen8 x)
	// result: (BitLen64 (ZeroExt8to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpBitLen64)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCom16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com16 x)
	// result: (I64Xor x (I64Const [-1]))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Xor)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(-1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpCom32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com32 x)
	// result: (I64Xor x (I64Const [-1]))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Xor)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(-1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpCom64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com64 x)
	// result: (I64Xor x (I64Const [-1]))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Xor)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(-1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpCom8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com8 x)
	// result: (I64Xor x (I64Const [-1]))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Xor)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(-1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpConst16(v *ssa.Value) bool {
	// match: (Const16 [c])
	// result: (I64Const [int64(c)])
	for {
		c := AuxIntToInt16(v.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(int64(c))
		return true
	}
}
func rewriteValueWasm_OpConst32(v *ssa.Value) bool {
	// match: (Const32 [c])
	// result: (I64Const [int64(c)])
	for {
		c := AuxIntToInt32(v.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(int64(c))
		return true
	}
}
func rewriteValueWasm_OpConst8(v *ssa.Value) bool {
	// match: (Const8 [c])
	// result: (I64Const [int64(c)])
	for {
		c := AuxIntToInt8(v.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(int64(c))
		return true
	}
}
func rewriteValueWasm_OpConstBool(v *ssa.Value) bool {
	// match: (ConstBool [c])
	// result: (I64Const [B2i(c)])
	for {
		c := AuxIntToBool(v.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(B2i(c))
		return true
	}
}
func rewriteValueWasm_OpConstNil(v *ssa.Value) bool {
	// match: (ConstNil)
	// result: (I64Const [0])
	for {
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
}
func rewriteValueWasm_OpCtz16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 x)
	// result: (I64Ctz (I64Or x (I64Const [0x10000])))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Ctz)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Or, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v1.AuxInt = Int64ToAuxInt(0x10000)
		v0.AddArg2(x, v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCtz32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz32 x)
	// result: (I64Ctz (I64Or x (I64Const [0x100000000])))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Ctz)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Or, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v1.AuxInt = Int64ToAuxInt(0x100000000)
		v0.AddArg2(x, v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCtz8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 x)
	// result: (I64Ctz (I64Or x (I64Const [0x100])))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Ctz)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Or, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v1.AuxInt = Int64ToAuxInt(0x100)
		v0.AddArg2(x, v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCvt32Uto32F(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32Uto32F x)
	// result: (F32ConvertI64U (ZeroExt32to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmF32ConvertI64U)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCvt32Uto64F(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32Uto64F x)
	// result: (F64ConvertI64U (ZeroExt32to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmF64ConvertI64U)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCvt32to32F(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32to32F x)
	// result: (F32ConvertI64S (SignExt32to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmF32ConvertI64S)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpCvt32to64F(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32to64F x)
	// result: (F64ConvertI64S (SignExt32to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmF64ConvertI64S)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpDiv16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 [false] x y)
	// result: (I64DivS (SignExt16to64 x) (SignExt16to64 y))
	for {
		if AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64DivS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueWasm_OpDiv16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (I64DivU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64DivU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpDiv32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32 [false] x y)
	// result: (I64DivS (SignExt32to64 x) (SignExt32to64 y))
	for {
		if AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64DivS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueWasm_OpDiv32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32u x y)
	// result: (I64DivU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64DivU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpDiv64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64 [false] x y)
	// result: (I64DivS x y)
	for {
		if AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64DivS)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueWasm_OpDiv8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (I64DivS (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64DivS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpDiv8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (I64DivU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64DivU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpEq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// result: (I64Eq (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64Eq)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpEq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq32 x y)
	// result: (I64Eq (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64Eq)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpEq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// result: (I64Eq (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64Eq)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpHmul64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul64 <t> x y)
	// result: (Last <t> x0: (ZeroExt32to64 x) x1: (I64ShrS x (I64Const [32])) y0: (ZeroExt32to64 y) y1: (I64ShrS y (I64Const [32])) x0y0: (I64Mul x0 y0) tt: (I64Add (I64Mul x1 y0) (I64ShrU x0y0 (I64Const [32]))) w1: (I64Add (I64Mul x0 y1) (ZeroExt32to64 tt)) w2: (I64ShrS tt (I64Const [32])) (I64Add (I64Add (I64Mul x1 y1) w2) (I64ShrS w1 (I64Const [32]))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLast)
		v.Type = t
		x0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		x0.AddArg(x)
		x1 := b.NewValue0(v.Pos, ssaop.OpWasmI64ShrS, typ.Int64)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v2.AuxInt = Int64ToAuxInt(32)
		x1.AddArg2(x, v2)
		y0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		y0.AddArg(y)
		y1 := b.NewValue0(v.Pos, ssaop.OpWasmI64ShrS, typ.Int64)
		y1.AddArg2(y, v2)
		x0y0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Mul, typ.Int64)
		x0y0.AddArg2(x0, y0)
		tt := b.NewValue0(v.Pos, ssaop.OpWasmI64Add, typ.Int64)
		v7 := b.NewValue0(v.Pos, ssaop.OpWasmI64Mul, typ.Int64)
		v7.AddArg2(x1, y0)
		v8 := b.NewValue0(v.Pos, ssaop.OpWasmI64ShrU, typ.Int64)
		v8.AddArg2(x0y0, v2)
		tt.AddArg2(v7, v8)
		w1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Add, typ.Int64)
		v10 := b.NewValue0(v.Pos, ssaop.OpWasmI64Mul, typ.Int64)
		v10.AddArg2(x0, y1)
		v11 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v11.AddArg(tt)
		w1.AddArg2(v10, v11)
		w2 := b.NewValue0(v.Pos, ssaop.OpWasmI64ShrS, typ.Int64)
		w2.AddArg2(tt, v2)
		v13 := b.NewValue0(v.Pos, ssaop.OpWasmI64Add, typ.Int64)
		v14 := b.NewValue0(v.Pos, ssaop.OpWasmI64Add, typ.Int64)
		v15 := b.NewValue0(v.Pos, ssaop.OpWasmI64Mul, typ.Int64)
		v15.AddArg2(x1, y1)
		v14.AddArg2(v15, w2)
		v16 := b.NewValue0(v.Pos, ssaop.OpWasmI64ShrS, typ.Int64)
		v16.AddArg2(w1, v2)
		v13.AddArg2(v14, v16)
		v.AddArgs(x0, x1, y0, y1, x0y0, tt, w1, w2, v13)
		return true
	}
}
func rewriteValueWasm_OpHmul64u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul64u <t> x y)
	// result: (Last <t> x0: (ZeroExt32to64 x) x1: (I64ShrU x (I64Const [32])) y0: (ZeroExt32to64 y) y1: (I64ShrU y (I64Const [32])) w0: (I64Mul x0 y0) tt: (I64Add (I64Mul x1 y0) (I64ShrU w0 (I64Const [32]))) w1: (I64Add (I64Mul x0 y1) (ZeroExt32to64 tt)) w2: (I64ShrU tt (I64Const [32])) hi: (I64Add (I64Add (I64Mul x1 y1) w2) (I64ShrU w1 (I64Const [32]))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLast)
		v.Type = t
		x0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		x0.AddArg(x)
		x1 := b.NewValue0(v.Pos, ssaop.OpWasmI64ShrU, typ.Int64)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v2.AuxInt = Int64ToAuxInt(32)
		x1.AddArg2(x, v2)
		y0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		y0.AddArg(y)
		y1 := b.NewValue0(v.Pos, ssaop.OpWasmI64ShrU, typ.Int64)
		y1.AddArg2(y, v2)
		w0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Mul, typ.Int64)
		w0.AddArg2(x0, y0)
		tt := b.NewValue0(v.Pos, ssaop.OpWasmI64Add, typ.Int64)
		v7 := b.NewValue0(v.Pos, ssaop.OpWasmI64Mul, typ.Int64)
		v7.AddArg2(x1, y0)
		v8 := b.NewValue0(v.Pos, ssaop.OpWasmI64ShrU, typ.Int64)
		v8.AddArg2(w0, v2)
		tt.AddArg2(v7, v8)
		w1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Add, typ.Int64)
		v10 := b.NewValue0(v.Pos, ssaop.OpWasmI64Mul, typ.Int64)
		v10.AddArg2(x0, y1)
		v11 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v11.AddArg(tt)
		w1.AddArg2(v10, v11)
		w2 := b.NewValue0(v.Pos, ssaop.OpWasmI64ShrU, typ.Int64)
		w2.AddArg2(tt, v2)
		hi := b.NewValue0(v.Pos, ssaop.OpWasmI64Add, typ.Int64)
		v14 := b.NewValue0(v.Pos, ssaop.OpWasmI64Add, typ.Int64)
		v15 := b.NewValue0(v.Pos, ssaop.OpWasmI64Mul, typ.Int64)
		v15.AddArg2(x1, y1)
		v14.AddArg2(v15, w2)
		v16 := b.NewValue0(v.Pos, ssaop.OpWasmI64ShrU, typ.Int64)
		v16.AddArg2(w1, v2)
		hi.AddArg2(v14, v16)
		v.AddArgs(x0, x1, y0, y1, w0, tt, w1, w2, hi)
		return true
	}
}
func rewriteValueWasm_OpIsNonNil(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsNonNil p)
	// result: (I64Eqz (I64Eqz p))
	for {
		p := v_0
		v.Reset(ssaop.OpWasmI64Eqz)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Eqz, typ.Bool)
		v0.AddArg(p)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpLast(v *ssa.Value) bool {
	// match: (Last ___)
	// result: v.Args[len(v.Args)-1]
	for {
		v.CopyOf(v.Args[len(v.Args)-1])
		return true
	}
}
func rewriteValueWasm_OpLeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (I64LeS (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64LeS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpLeq16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// result: (I64LeU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64LeU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpLeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32 x y)
	// result: (I64LeS (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64LeS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpLeq32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32U x y)
	// result: (I64LeU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64LeU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpLeq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (I64LeS (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64LeS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpLeq8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// result: (I64LeU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64LeU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpLess16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (I64LtS (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64LtS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpLess16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// result: (I64LtU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64LtU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpLess32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32 x y)
	// result: (I64LtS (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64LtS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpLess32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32U x y)
	// result: (I64LtU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64LtU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpLess8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (I64LtS (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64LtS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpLess8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// result: (I64LtU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64LtU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpLoad(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Load <t> ptr mem)
	// cond: Is32BitFloat(t)
	// result: (F32Load ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(Is32BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpWasmF32Load)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: Is64BitFloat(t)
	// result: (F64Load ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(Is64BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpWasmF64Load)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 16
	// result: (V128Load ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 16) {
			break
		}
		v.Reset(ssaop.OpWasmV128Load)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 8
	// result: (I64Load ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 8) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 4 && !t.IsSigned()
	// result: (I64Load32U ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 4 && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load32U)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 4 && t.IsSigned()
	// result: (I64Load32S ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 4 && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load32S)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 2 && !t.IsSigned()
	// result: (I64Load16U ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 2 && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load16U)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 2 && t.IsSigned()
	// result: (I64Load16S ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 2 && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load16S)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 1 && !t.IsSigned()
	// result: (I64Load8U ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 1 && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load8U)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 1 && t.IsSigned()
	// result: (I64Load8S ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 1 && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load8S)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpLocalAddr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LocalAddr <t> {sym} base mem)
	// cond: t.Elem().HasPointers()
	// result: (LoweredAddr {sym} (SPanchored base mem))
	for {
		t := v.Type
		sym := AuxToSym(v.Aux)
		base := v_0
		mem := v_1
		if !(t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpWasmLoweredAddr)
		v.Aux = SymToAux(sym)
		v0 := b.NewValue0(v.Pos, ssaop.OpSPanchored, typ.Uintptr)
		v0.AddArg2(base, mem)
		v.AddArg(v0)
		return true
	}
	// match: (LocalAddr <t> {sym} base _)
	// cond: !t.Elem().HasPointers()
	// result: (LoweredAddr {sym} base)
	for {
		t := v.Type
		sym := AuxToSym(v.Aux)
		base := v_0
		if !(!t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpWasmLoweredAddr)
		v.Aux = SymToAux(sym)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValueWasm_OpLsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt16to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpLsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt32to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpLsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt8to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpLsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt16to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpLsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt32to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpLsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt8to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpLsh64x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt16to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpLsh64x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt32to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpLsh64x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x64 x y)
	// cond: ShiftIsBounded(v)
	// result: (I64Shl x y)
	for {
		x := v_0
		y := v_1
		if !(ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpWasmI64Shl)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x64 x (I64Const [c]))
	// cond: uint64(c) < 64
	// result: (I64Shl x (I64Const [c]))
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64Shl)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Lsh64x64 x (I64Const [c]))
	// cond: uint64(c) >= 64
	// result: (I64Const [0])
	for {
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (Lsh64x64 x y)
	// result: (Select (I64Shl x y) (I64Const [0]) (I64LtU y (I64Const [64])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelect)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Shl, typ.Int64)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v1.AuxInt = Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(64)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpLsh64x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt8to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpLsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt16to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpLsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt32to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpLsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 [c] x y)
	// result: (Lsh64x64 [c] x (ZeroExt8to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpMod16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 [false] x y)
	// result: (I64RemS (SignExt16to64 x) (SignExt16to64 y))
	for {
		if AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64RemS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueWasm_OpMod16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (I64RemU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64RemU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpMod32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32 [false] x y)
	// result: (I64RemS (SignExt32to64 x) (SignExt32to64 y))
	for {
		if AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64RemS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueWasm_OpMod32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32u x y)
	// result: (I64RemU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64RemU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpMod64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod64 [false] x y)
	// result: (I64RemS x y)
	for {
		if AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64RemS)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueWasm_OpMod8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (I64RemS (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64RemS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpMod8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (I64RemU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64RemU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpMove(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Move [0] _ _ mem)
	// result: mem
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.CopyOf(mem)
		return true
	}
	// match: (Move [1] dst src mem)
	// result: (I64Store8 dst (I64Load8U src mem) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpWasmI64Store8)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load8U, typ.UInt8)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (I64Store16 dst (I64Load16U src mem) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpWasmI64Store16)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load16U, typ.UInt16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (I64Store32 dst (I64Load32U src mem) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpWasmI64Store32)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load32U, typ.UInt32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [8] dst src mem)
	// result: (I64Store dst (I64Load src mem) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpWasmI64Store)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load, typ.UInt64)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [16] dst src mem)
	// result: (I64Store [8] dst (I64Load [8] src mem) (I64Store dst (I64Load src mem) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 16 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpWasmI64Store)
		v.AuxInt = Int64ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [3] dst src mem)
	// result: (I64Store8 [2] dst (I64Load8U [2] src mem) (I64Store16 dst (I64Load16U src mem) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpWasmI64Store8)
		v.AuxInt = Int64ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load8U, typ.UInt8)
		v0.AuxInt = Int64ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store16, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load16U, typ.UInt16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [5] dst src mem)
	// result: (I64Store8 [4] dst (I64Load8U [4] src mem) (I64Store32 dst (I64Load32U src mem) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 5 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpWasmI64Store8)
		v.AuxInt = Int64ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load8U, typ.UInt8)
		v0.AuxInt = Int64ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store32, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load32U, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [6] dst src mem)
	// result: (I64Store16 [4] dst (I64Load16U [4] src mem) (I64Store32 dst (I64Load32U src mem) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpWasmI64Store16)
		v.AuxInt = Int64ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load16U, typ.UInt16)
		v0.AuxInt = Int64ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store32, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load32U, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [7] dst src mem)
	// result: (I64Store32 [3] dst (I64Load32U [3] src mem) (I64Store32 dst (I64Load32U src mem) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 7 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpWasmI64Store32)
		v.AuxInt = Int64ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load32U, typ.UInt32)
		v0.AuxInt = Int64ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store32, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load32U, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 8 && s < 16
	// result: (I64Store [s-8] dst (I64Load [s-8] src mem) (I64Store dst (I64Load src mem) mem))
	for {
		s := AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 8 && s < 16) {
			break
		}
		v.Reset(ssaop.OpWasmI64Store)
		v.AuxInt = Int64ToAuxInt(s - 8)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(s - 8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Load, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: LogLargeCopyValue(v, s)
	// result: (LoweredMove [s] dst src mem)
	for {
		s := AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.OpWasmLoweredMove)
		v.AuxInt = Int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpNeg16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg16 x)
	// result: (I64Sub (I64Const [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Sub)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueWasm_OpNeg32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg32 x)
	// result: (I64Sub (I64Const [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Sub)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueWasm_OpNeg64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg64 x)
	// result: (I64Sub (I64Const [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Sub)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueWasm_OpNeg8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg8 x)
	// result: (I64Sub (I64Const [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Sub)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueWasm_OpNeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// result: (I64Ne (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64Ne)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpNeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq32 x y)
	// result: (I64Ne (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64Ne)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpNeq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// result: (I64Ne (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64Ne)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpPopCount16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount16 x)
	// result: (I64Popcnt (ZeroExt16to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Popcnt)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpPopCount32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount32 x)
	// result: (I64Popcnt (ZeroExt32to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Popcnt)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpPopCount8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount8 x)
	// result: (I64Popcnt (ZeroExt8to64 x))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Popcnt)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueWasm_OpRotateAllLeftVarInt16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllLeftVarInt16x8 x y)
	// result: (V128Or (I16x8Shl x y) (I16x8ShrU x (I64Sub (I64Const [16]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8ShrU, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(16)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllLeftVarInt32x4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllLeftVarInt32x4 x y)
	// result: (V128Or (I32x4Shl x y) (I32x4ShrU x (I64Sub (I64Const [32]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4ShrU, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(32)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllLeftVarInt64x2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllLeftVarInt64x2 x y)
	// result: (V128Or (I64x2Shl x y) (I64x2ShrU x (I64Sub (I64Const [64]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2ShrU, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(64)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllLeftVarInt8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllLeftVarInt8x16 x y)
	// result: (V128Or (I8x16Shl x y) (I8x16ShrU x (I64Sub (I64Const [8]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16ShrU, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(8)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllLeftVarUint16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllLeftVarUint16x8 x y)
	// result: (V128Or (I16x8Shl x y) (I16x8ShrU x (I64Sub (I64Const [16]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8ShrU, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(16)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllLeftVarUint32x4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllLeftVarUint32x4 x y)
	// result: (V128Or (I32x4Shl x y) (I32x4ShrU x (I64Sub (I64Const [32]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4ShrU, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(32)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllLeftVarUint64x2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllLeftVarUint64x2 x y)
	// result: (V128Or (I64x2Shl x y) (I64x2ShrU x (I64Sub (I64Const [64]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2ShrU, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(64)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllLeftVarUint8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllLeftVarUint8x16 x y)
	// result: (V128Or (I8x16Shl x y) (I8x16ShrU x (I64Sub (I64Const [8]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16ShrU, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(8)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllRightVarInt16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllRightVarInt16x8 x y)
	// result: (V128Or (I16x8ShrU x y) (I16x8Shl x (I64Sub (I64Const [16]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8ShrU, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8Shl, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(16)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllRightVarInt32x4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllRightVarInt32x4 x y)
	// result: (V128Or (I32x4ShrU x y) (I32x4Shl x (I64Sub (I64Const [32]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4ShrU, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4Shl, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(32)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllRightVarInt64x2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllRightVarInt64x2 x y)
	// result: (V128Or (I64x2ShrU x y) (I64x2Shl x (I64Sub (I64Const [64]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2ShrU, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2Shl, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(64)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllRightVarInt8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllRightVarInt8x16 x y)
	// result: (V128Or (I8x16ShrU x y) (I8x16Shl x (I64Sub (I64Const [8]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16ShrU, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16Shl, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(8)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllRightVarUint16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllRightVarUint16x8 x y)
	// result: (V128Or (I16x8ShrU x y) (I16x8Shl x (I64Sub (I64Const [16]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8ShrU, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8Shl, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(16)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllRightVarUint32x4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllRightVarUint32x4 x y)
	// result: (V128Or (I32x4ShrU x y) (I32x4Shl x (I64Sub (I64Const [32]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4ShrU, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4Shl, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(32)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllRightVarUint64x2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllRightVarUint64x2 x y)
	// result: (V128Or (I64x2ShrU x y) (I64x2Shl x (I64Sub (I64Const [64]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2ShrU, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2Shl, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(64)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateAllRightVarUint8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateAllRightVarUint8x16 x y)
	// result: (V128Or (I8x16ShrU x y) (I8x16Shl x (I64Sub (I64Const [8]) y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmV128Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16ShrU, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16Shl, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(8)
		v2.AddArg2(v3, y)
		v1.AddArg2(x, v2)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRotateLeft16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (I64Const [c]))
	// result: (Or16 (Lsh16x64 <t> x (I64Const [c&15])) (Rsh16Ux64 <t> x (I64Const [-c&15])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpOr16)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh16x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v1.AuxInt = Int64ToAuxInt(c & 15)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh16Ux64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(-c & 15)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueWasm_OpRotateLeft8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (I64Const [c]))
	// result: (Or8 (Lsh8x64 <t> x (I64Const [c&7])) (Rsh8Ux64 <t> x (I64Const [-c&7])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpOr8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh8x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v1.AuxInt = Int64ToAuxInt(c & 7)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh8Ux64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(-c & 7)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueWasm_OpRsh16Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh16Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt16to64 x) (ZeroExt32to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh16Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt16to64 x) y)
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValueWasm_OpRsh16Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt16to64 x) (ZeroExt8to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 [c] x y)
	// result: (Rsh64x64 [c] (SignExt16to64 x) (ZeroExt16to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 [c] x y)
	// result: (Rsh64x64 [c] (SignExt16to64 x) (ZeroExt32to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 [c] x y)
	// result: (Rsh64x64 [c] (SignExt16to64 x) y)
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValueWasm_OpRsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 [c] x y)
	// result: (Rsh64x64 [c] (SignExt16to64 x) (ZeroExt8to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh32Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt32to64 x) (ZeroExt16to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh32Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh32Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt32to64 x) y)
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValueWasm_OpRsh32Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt32to64 x) (ZeroExt8to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 [c] x y)
	// result: (Rsh64x64 [c] (SignExt32to64 x) (ZeroExt16to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 [c] x y)
	// result: (Rsh64x64 [c] (SignExt32to64 x) (ZeroExt32to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x64 [c] x y)
	// result: (Rsh64x64 [c] (SignExt32to64 x) y)
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValueWasm_OpRsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 [c] x y)
	// result: (Rsh64x64 [c] (SignExt32to64 x) (ZeroExt8to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh64Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 [c] x y)
	// result: (Rsh64Ux64 [c] x (ZeroExt16to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpRsh64Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 [c] x y)
	// result: (Rsh64Ux64 [c] x (ZeroExt32to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpRsh64Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux64 x y)
	// cond: ShiftIsBounded(v)
	// result: (I64ShrU x y)
	for {
		x := v_0
		y := v_1
		if !(ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpWasmI64ShrU)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux64 x (I64Const [c]))
	// cond: uint64(c) < 64
	// result: (I64ShrU x (I64Const [c]))
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64ShrU)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64Ux64 x (I64Const [c]))
	// cond: uint64(c) >= 64
	// result: (I64Const [0])
	for {
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (Rsh64Ux64 x y)
	// result: (Select (I64ShrU x y) (I64Const [0]) (I64LtU y (I64Const [64])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelect)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64ShrU, typ.Int64)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v1.AuxInt = Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(64)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpRsh64Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 [c] x y)
	// result: (Rsh64Ux64 [c] x (ZeroExt8to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpRsh64x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 [c] x y)
	// result: (Rsh64x64 [c] x (ZeroExt16to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpRsh64x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x32 [c] x y)
	// result: (Rsh64x64 [c] x (ZeroExt32to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpRsh64x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x64 x y)
	// cond: ShiftIsBounded(v)
	// result: (I64ShrS x y)
	for {
		x := v_0
		y := v_1
		if !(ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpWasmI64ShrS)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x64 x (I64Const [c]))
	// cond: uint64(c) < 64
	// result: (I64ShrS x (I64Const [c]))
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64ShrS)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x64 x (I64Const [c]))
	// cond: uint64(c) >= 64
	// result: (I64ShrS x (I64Const [63]))
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64ShrS)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(63)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Rsh64x64 x y)
	// result: (I64ShrS x (Select <typ.Int64> y (I64Const [63]) (I64LtU y (I64Const [64]))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmI64ShrS)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmSelect, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v1.AuxInt = Int64ToAuxInt(63)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(64)
		v2.AddArg2(y, v3)
		v0.AddArg3(y, v1, v2)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpRsh64x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 [c] x y)
	// result: (Rsh64x64 [c] x (ZeroExt8to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpRsh8Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt8to64 x) (ZeroExt16to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh8Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt8to64 x) (ZeroExt32to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh8Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt8to64 x) y)
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValueWasm_OpRsh8Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 [c] x y)
	// result: (Rsh64Ux64 [c] (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 [c] x y)
	// result: (Rsh64x64 [c] (SignExt8to64 x) (ZeroExt16to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 [c] x y)
	// result: (Rsh64x64 [c] (SignExt8to64 x) (ZeroExt32to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpRsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 [c] x y)
	// result: (Rsh64x64 [c] (SignExt8to64 x) y)
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValueWasm_OpRsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 [c] x y)
	// result: (Rsh64x64 [c] (SignExt8to64 x) (ZeroExt8to64 y))
	for {
		c := AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.AuxInt = ssa.BoolToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueWasm_OpSelect0(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Select0 <t> (Mul64uhilo x y))
	// result: (Hmul64u <t> x y)
	for {
		t := v.Type
		if v_0.Op != ssaop.OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpHmul64u)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueWasm_OpSelect1(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Select1 <t> (Mul64uhilo x y))
	// result: (I64Mul x y)
	for {
		if v_0.Op != ssaop.OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpWasmI64Mul)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueWasm_OpShiftAllLeftInt16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftInt16x8 x d:(Const64 [c]))
	// cond: uint64(c) < 16
	// result: (I16x8Shl x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpWasmI16x8Shl)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllLeftInt16x8 x d:(I64Const [c]))
	// cond: uint64(c) < 16
	// result: (I16x8Shl x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpWasmI16x8Shl)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllLeftInt16x8 x y)
	// result: (SelectV (I16x8Shl x y) (V128Xor x x) (I64LtU y (I64Const [16])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmV128Xor, typ.Vec128)
		v1.AddArg2(x, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(16)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpShiftAllLeftInt32x4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftInt32x4 x d:(Const64 [c]))
	// cond: uint64(c) < 32
	// result: (I32x4Shl x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpWasmI32x4Shl)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllLeftInt32x4 x d:(I64Const [c]))
	// cond: uint64(c) < 32
	// result: (I32x4Shl x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpWasmI32x4Shl)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllLeftInt32x4 x y)
	// result: (SelectV (I32x4Shl x y) (V128Xor x x) (I64LtU y (I64Const [32])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmV128Xor, typ.Vec128)
		v1.AddArg2(x, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(32)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpShiftAllLeftInt64x2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftInt64x2 x d:(Const64 [c]))
	// cond: uint64(c) < 64
	// result: (I64x2Shl x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64x2Shl)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllLeftInt64x2 x d:(I64Const [c]))
	// cond: uint64(c) < 64
	// result: (I64x2Shl x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64x2Shl)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllLeftInt64x2 x y)
	// result: (SelectV (I64x2Shl x y) (V128Xor x x) (I64LtU y (I64Const [64])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmV128Xor, typ.Vec128)
		v1.AddArg2(x, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(64)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpShiftAllLeftInt8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftInt8x16 x d:(Const64 [c]))
	// cond: uint64(c) < 8
	// result: (I8x16Shl x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpWasmI8x16Shl)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllLeftInt8x16 x d:(I64Const [c]))
	// cond: uint64(c) < 8
	// result: (I8x16Shl x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpWasmI8x16Shl)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllLeftInt8x16 x y)
	// result: (SelectV (I8x16Shl x y) (V128Xor x x) (I64LtU y (I64Const [8])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmV128Xor, typ.Vec128)
		v1.AddArg2(x, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(8)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpShiftAllLeftUint16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftUint16x8 x d:(Const64 [c]))
	// cond: uint64(c) < 16
	// result: (I16x8Shl x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpWasmI16x8Shl)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllLeftUint16x8 x d:(I64Const [c]))
	// cond: uint64(c) < 16
	// result: (I16x8Shl x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpWasmI16x8Shl)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllLeftUint16x8 x y)
	// result: (SelectV (I16x8Shl x y) (V128Xor x x) (I64LtU y (I64Const [16])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmV128Xor, typ.Vec128)
		v1.AddArg2(x, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(16)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpShiftAllLeftUint32x4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftUint32x4 x d:(Const64 [c]))
	// cond: uint64(c) < 32
	// result: (I32x4Shl x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpWasmI32x4Shl)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllLeftUint32x4 x d:(I64Const [c]))
	// cond: uint64(c) < 32
	// result: (I32x4Shl x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpWasmI32x4Shl)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllLeftUint32x4 x y)
	// result: (SelectV (I32x4Shl x y) (V128Xor x x) (I64LtU y (I64Const [32])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmV128Xor, typ.Vec128)
		v1.AddArg2(x, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(32)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpShiftAllLeftUint64x2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftUint64x2 x d:(Const64 [c]))
	// cond: uint64(c) < 64
	// result: (I64x2Shl x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64x2Shl)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllLeftUint64x2 x d:(I64Const [c]))
	// cond: uint64(c) < 64
	// result: (I64x2Shl x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64x2Shl)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllLeftUint64x2 x y)
	// result: (SelectV (I64x2Shl x y) (V128Xor x x) (I64LtU y (I64Const [64])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmV128Xor, typ.Vec128)
		v1.AddArg2(x, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(64)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpShiftAllLeftUint8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftUint8x16 x d:(Const64 [c]))
	// cond: uint64(c) < 8
	// result: (I8x16Shl x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpWasmI8x16Shl)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllLeftUint8x16 x d:(I64Const [c]))
	// cond: uint64(c) < 8
	// result: (I8x16Shl x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpWasmI8x16Shl)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllLeftUint8x16 x y)
	// result: (SelectV (I8x16Shl x y) (V128Xor x x) (I64LtU y (I64Const [8])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16Shl, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmV128Xor, typ.Vec128)
		v1.AddArg2(x, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(8)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpShiftAllRightInt16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightInt16x8 x d:(Const64 [c]))
	// cond: uint64(c) < 16
	// result: (I16x8ShrS x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpWasmI16x8ShrS)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllRightInt16x8 x d:(I64Const [c]))
	// cond: uint64(c) < 16
	// result: (I16x8ShrS x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpWasmI16x8ShrS)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllRightInt16x8 x y)
	// result: (SelectV (I16x8ShrS x y) (I16x8ShrS x (I64Const [15])) (I64LtU y (I64Const [16])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8ShrS, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8ShrS, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v2.AuxInt = Int64ToAuxInt(15)
		v1.AddArg2(x, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v4.AuxInt = Int64ToAuxInt(16)
		v3.AddArg2(y, v4)
		v.AddArg3(v0, v1, v3)
		return true
	}
}
func rewriteValueWasm_OpShiftAllRightInt32x4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightInt32x4 x d:(Const64 [c]))
	// cond: uint64(c) < 32
	// result: (I32x4ShrS x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpWasmI32x4ShrS)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllRightInt32x4 x d:(I64Const [c]))
	// cond: uint64(c) < 32
	// result: (I32x4ShrS x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpWasmI32x4ShrS)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllRightInt32x4 x y)
	// result: (SelectV (I32x4ShrS x y) (I32x4ShrS x (I64Const [31])) (I64LtU y (I64Const [32])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4ShrS, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4ShrS, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v2.AuxInt = Int64ToAuxInt(31)
		v1.AddArg2(x, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v4.AuxInt = Int64ToAuxInt(32)
		v3.AddArg2(y, v4)
		v.AddArg3(v0, v1, v3)
		return true
	}
}
func rewriteValueWasm_OpShiftAllRightInt64x2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightInt64x2 x d:(Const64 [c]))
	// cond: uint64(c) < 64
	// result: (I64x2ShrS x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64x2ShrS)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllRightInt64x2 x d:(I64Const [c]))
	// cond: uint64(c) < 64
	// result: (I64x2ShrS x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64x2ShrS)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllRightInt64x2 x y)
	// result: (SelectV (I64x2ShrS x y) (I64x2ShrS x (I64Const [63])) (I64LtU y (I64Const [64])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2ShrS, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2ShrS, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v2.AuxInt = Int64ToAuxInt(63)
		v1.AddArg2(x, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v4.AuxInt = Int64ToAuxInt(64)
		v3.AddArg2(y, v4)
		v.AddArg3(v0, v1, v3)
		return true
	}
}
func rewriteValueWasm_OpShiftAllRightInt8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightInt8x16 x d:(Const64 [c]))
	// cond: uint64(c) < 8
	// result: (I8x16ShrS x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpWasmI8x16ShrS)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllRightInt8x16 x d:(I64Const [c]))
	// cond: uint64(c) < 8
	// result: (I8x16ShrS x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpWasmI8x16ShrS)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllRightInt8x16 x y)
	// result: (SelectV (I8x16ShrS x y) (I8x16ShrS x (I64Const [7])) (I64LtU y (I64Const [8])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16ShrS, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16ShrS, typ.Vec128)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v2.AuxInt = Int64ToAuxInt(7)
		v1.AddArg2(x, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v4.AuxInt = Int64ToAuxInt(8)
		v3.AddArg2(y, v4)
		v.AddArg3(v0, v1, v3)
		return true
	}
}
func rewriteValueWasm_OpShiftAllRightUint16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightUint16x8 x d:(Const64 [c]))
	// cond: uint64(c) < 16
	// result: (I16x8ShrU x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpWasmI16x8ShrU)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllRightUint16x8 x d:(I64Const [c]))
	// cond: uint64(c) < 16
	// result: (I16x8ShrU x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpWasmI16x8ShrU)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllRightUint16x8 x y)
	// result: (SelectV (I16x8ShrU x y) (V128Xor x x) (I64LtU y (I64Const [16])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI16x8ShrU, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmV128Xor, typ.Vec128)
		v1.AddArg2(x, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(16)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpShiftAllRightUint32x4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightUint32x4 x d:(Const64 [c]))
	// cond: uint64(c) < 32
	// result: (I32x4ShrU x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpWasmI32x4ShrU)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllRightUint32x4 x d:(I64Const [c]))
	// cond: uint64(c) < 32
	// result: (I32x4ShrU x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpWasmI32x4ShrU)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllRightUint32x4 x y)
	// result: (SelectV (I32x4ShrU x y) (V128Xor x x) (I64LtU y (I64Const [32])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI32x4ShrU, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmV128Xor, typ.Vec128)
		v1.AddArg2(x, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(32)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpShiftAllRightUint64x2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightUint64x2 x d:(Const64 [c]))
	// cond: uint64(c) < 64
	// result: (I64x2ShrU x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64x2ShrU)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllRightUint64x2 x d:(I64Const [c]))
	// cond: uint64(c) < 64
	// result: (I64x2ShrU x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpWasmI64x2ShrU)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllRightUint64x2 x y)
	// result: (SelectV (I64x2ShrU x y) (V128Xor x x) (I64LtU y (I64Const [64])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64x2ShrU, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmV128Xor, typ.Vec128)
		v1.AddArg2(x, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(64)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpShiftAllRightUint8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightUint8x16 x d:(Const64 [c]))
	// cond: uint64(c) < 8
	// result: (I8x16ShrU x (I64Const [c]))
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpWasmI8x16ShrU)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (ShiftAllRightUint8x16 x d:(I64Const [c]))
	// cond: uint64(c) < 8
	// result: (I8x16ShrU x d)
	for {
		x := v_0
		d := v_1
		if d.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(d.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpWasmI8x16ShrU)
		v.AddArg2(x, d)
		return true
	}
	// match: (ShiftAllRightUint8x16 x y)
	// result: (SelectV (I8x16ShrU x y) (V128Xor x x) (I64LtU y (I64Const [8])))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpWasmSelectV)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI8x16ShrU, typ.Vec128)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmV128Xor, typ.Vec128)
		v1.AddArg2(x, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64LtU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v3.AuxInt = Int64ToAuxInt(8)
		v2.AddArg2(y, v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueWasm_OpSignExt16to32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SignExt16to32 x:(I64Load16S _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64Load16S {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (SignExt16to32 x)
	// result: (I64Extend16S x)
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Extend16S)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpSignExt16to64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SignExt16to64 x:(I64Load16S _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64Load16S {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (SignExt16to64 x)
	// result: (I64Extend16S x)
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Extend16S)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpSignExt32to64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SignExt32to64 x:(I64Load32S _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64Load32S {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (SignExt32to64 x)
	// result: (I64Extend32S x)
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Extend32S)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpSignExt8to16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SignExt8to16 x:(I64Load8S _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64Load8S {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (SignExt8to16 x)
	// result: (I64Extend8S x)
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Extend8S)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpSignExt8to32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SignExt8to32 x:(I64Load8S _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64Load8S {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (SignExt8to32 x)
	// result: (I64Extend8S x)
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Extend8S)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpSignExt8to64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SignExt8to64 x:(I64Load8S _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64Load8S {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (SignExt8to64 x)
	// result: (I64Extend8S x)
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64Extend8S)
		v.AddArg(x)
		return true
	}
}
func rewriteValueWasm_OpSlicemask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Slicemask x)
	// result: (I64ShrS (I64Sub (I64Const [0]) x) (I64Const [63]))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64ShrS)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Sub, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v1.AuxInt = Int64ToAuxInt(0)
		v0.AddArg2(v1, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v2.AuxInt = Int64ToAuxInt(63)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueWasm_OpStore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Store {t} ptr val mem)
	// cond: Is64BitFloat(t)
	// result: (F64Store ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(Is64BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpWasmF64Store)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: Is32BitFloat(t)
	// result: (F32Store ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(Is32BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpWasmF32Store)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 16
	// result: (V128Store ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 16) {
			break
		}
		v.Reset(ssaop.OpWasmV128Store)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8
	// result: (I64Store ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8) {
			break
		}
		v.Reset(ssaop.OpWasmI64Store)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4
	// result: (I64Store32 ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4) {
			break
		}
		v.Reset(ssaop.OpWasmI64Store32)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 2
	// result: (I64Store16 ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 2) {
			break
		}
		v.Reset(ssaop.OpWasmI64Store16)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 1
	// result: (I64Store8 ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 1) {
			break
		}
		v.Reset(ssaop.OpWasmI64Store8)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmF32DemoteF64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (F32DemoteF64 (F64Sqrt (F64PromoteF32 x)))
	// result: (F32Sqrt x)
	for {
		if v_0.Op != ssaop.OpWasmF64Sqrt {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpWasmF64PromoteF32 {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpWasmF32Sqrt)
		v.AddArg(x)
		return true
	}
	// match: (F32DemoteF64 (F64Trunc (F64PromoteF32 x)))
	// result: (F32Trunc x)
	for {
		if v_0.Op != ssaop.OpWasmF64Trunc {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpWasmF64PromoteF32 {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpWasmF32Trunc)
		v.AddArg(x)
		return true
	}
	// match: (F32DemoteF64 (F64Ceil (F64PromoteF32 x)))
	// result: (F32Ceil x)
	for {
		if v_0.Op != ssaop.OpWasmF64Ceil {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpWasmF64PromoteF32 {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpWasmF32Ceil)
		v.AddArg(x)
		return true
	}
	// match: (F32DemoteF64 (F64Floor (F64PromoteF32 x)))
	// result: (F32Floor x)
	for {
		if v_0.Op != ssaop.OpWasmF64Floor {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpWasmF64PromoteF32 {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpWasmF32Floor)
		v.AddArg(x)
		return true
	}
	// match: (F32DemoteF64 (F64Nearest (F64PromoteF32 x)))
	// result: (F32Nearest x)
	for {
		if v_0.Op != ssaop.OpWasmF64Nearest {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpWasmF64PromoteF32 {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpWasmF32Nearest)
		v.AddArg(x)
		return true
	}
	// match: (F32DemoteF64 (F64Abs (F64PromoteF32 x)))
	// result: (F32Abs x)
	for {
		if v_0.Op != ssaop.OpWasmF64Abs {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpWasmF64PromoteF32 {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpWasmF32Abs)
		v.AddArg(x)
		return true
	}
	// match: (F32DemoteF64 (F64Copysign (F64PromoteF32 x) (F64PromoteF32 y)))
	// result: (F32Copysign x y)
	for {
		if v_0.Op != ssaop.OpWasmF64Copysign {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpWasmF64PromoteF32 {
			break
		}
		x := v_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpWasmF64PromoteF32 {
			break
		}
		y := v_0_1.Args[0]
		v.Reset(ssaop.OpWasmF32Copysign)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmF64Add(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (F64Add (F64Const [x]) (F64Const [y]))
	// result: (F64Const [x + y])
	for {
		if v_0.Op != ssaop.OpWasmF64Const {
			break
		}
		x := AuxIntToFloat64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmF64Const {
			break
		}
		y := AuxIntToFloat64(v_1.AuxInt)
		v.Reset(ssaop.OpWasmF64Const)
		v.AuxInt = Float64ToAuxInt(x + y)
		return true
	}
	// match: (F64Add (F64Const [x]) y)
	// cond: y.Op != ssaop.OpWasmF64Const
	// result: (F64Add y (F64Const [x]))
	for {
		if v_0.Op != ssaop.OpWasmF64Const {
			break
		}
		x := AuxIntToFloat64(v_0.AuxInt)
		y := v_1
		if !(y.Op != ssaop.OpWasmF64Const) {
			break
		}
		v.Reset(ssaop.OpWasmF64Add)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmF64Const, typ.Float64)
		v0.AuxInt = Float64ToAuxInt(x)
		v.AddArg2(y, v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmF64Mul(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (F64Mul (F64Const [x]) (F64Const [y]))
	// cond: !math.IsNaN(x * y)
	// result: (F64Const [x * y])
	for {
		if v_0.Op != ssaop.OpWasmF64Const {
			break
		}
		x := AuxIntToFloat64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmF64Const {
			break
		}
		y := AuxIntToFloat64(v_1.AuxInt)
		if !(!math.IsNaN(x * y)) {
			break
		}
		v.Reset(ssaop.OpWasmF64Const)
		v.AuxInt = Float64ToAuxInt(x * y)
		return true
	}
	// match: (F64Mul (F64Const [x]) y)
	// cond: y.Op != ssaop.OpWasmF64Const
	// result: (F64Mul y (F64Const [x]))
	for {
		if v_0.Op != ssaop.OpWasmF64Const {
			break
		}
		x := AuxIntToFloat64(v_0.AuxInt)
		y := v_1
		if !(y.Op != ssaop.OpWasmF64Const) {
			break
		}
		v.Reset(ssaop.OpWasmF64Mul)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmF64Const, typ.Float64)
		v0.AuxInt = Float64ToAuxInt(x)
		v.AddArg2(y, v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Add(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64Add (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x + y])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(x + y)
		return true
	}
	// match: (I64Add (I64Const [x]) y)
	// cond: y.Op != ssaop.OpWasmI64Const
	// result: (I64Add y (I64Const [x]))
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(y.Op != ssaop.OpWasmI64Const) {
			break
		}
		v.Reset(ssaop.OpWasmI64Add)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(x)
		v.AddArg2(y, v0)
		return true
	}
	// match: (I64Add x (I64Const <t> [y]))
	// cond: !t.IsPtr()
	// result: (I64AddConst [y] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		t := v_1.Type
		y := AuxIntToInt64(v_1.AuxInt)
		if !(!t.IsPtr()) {
			break
		}
		v.Reset(ssaop.OpWasmI64AddConst)
		v.AuxInt = Int64ToAuxInt(y)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64AddConst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (I64AddConst [0] x)
	// result: x
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (I64AddConst [off] (LoweredAddr {sym} [off2] base))
	// cond: IsU32Bit(off+int64(off2))
	// result: (LoweredAddr {sym} [int32(off)+off2] base)
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmLoweredAddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym := AuxToSym(v_0.Aux)
		base := v_0.Args[0]
		if !(IsU32Bit(off + int64(off2))) {
			break
		}
		v.Reset(ssaop.OpWasmLoweredAddr)
		v.AuxInt = Int32ToAuxInt(int32(off) + off2)
		v.Aux = SymToAux(sym)
		v.AddArg(base)
		return true
	}
	// match: (I64AddConst [off] x:(SP))
	// cond: IsU32Bit(off)
	// result: (LoweredAddr [int32(off)] x)
	for {
		off := AuxIntToInt64(v.AuxInt)
		x := v_0
		if x.Op != ssaop.OpSP || !(IsU32Bit(off)) {
			break
		}
		v.Reset(ssaop.OpWasmLoweredAddr)
		v.AuxInt = Int32ToAuxInt(int32(off))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64And(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64And (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x & y])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(x & y)
		return true
	}
	// match: (I64And x (I64Const [-1]))
	// result: x
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (I64And x (I64Const [0]))
	// result: (I64Const [0])
	for {
		if v_1.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (I64And (I64And x (I64Const [c1])) (I64Const [c2]))
	// result: (I64And x (I64Const [c1 & c2]))
	for {
		if v_0.Op != ssaop.OpWasmI64And {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c1 := AuxIntToInt64(v_0_1.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c2 := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpWasmI64And)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(c1 & c2)
		v.AddArg2(x, v0)
		return true
	}
	// match: (I64And (I64Const [x]) y)
	// cond: y.Op != ssaop.OpWasmI64Const
	// result: (I64And y (I64Const [x]))
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(y.Op != ssaop.OpWasmI64Const) {
			break
		}
		v.Reset(ssaop.OpWasmI64And)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(x)
		v.AddArg2(y, v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Eq(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64Eq (I64Const [x]) (I64Const [y]))
	// cond: x == y
	// result: (I64Const [1])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		if !(x == y) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	// match: (I64Eq (I64Const [x]) (I64Const [y]))
	// cond: x != y
	// result: (I64Const [0])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		if !(x != y) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (I64Eq (I64Const [x]) y)
	// cond: y.Op != ssaop.OpWasmI64Const
	// result: (I64Eq y (I64Const [x]))
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(y.Op != ssaop.OpWasmI64Const) {
			break
		}
		v.Reset(ssaop.OpWasmI64Eq)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(x)
		v.AddArg2(y, v0)
		return true
	}
	// match: (I64Eq x (I64Const [0]))
	// result: (I64Eqz x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpWasmI64Eqz)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Eqz(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (I64Eqz (I64Eqz (I64Eqz x)))
	// result: (I64Eqz x)
	for {
		if v_0.Op != ssaop.OpWasmI64Eqz {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpWasmI64Eqz {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpWasmI64Eqz)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Extend16S(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (I64Extend16S (I64Extend16S x))
	// result: (I64Extend16S x)
	for {
		if v_0.Op != ssaop.OpWasmI64Extend16S {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpWasmI64Extend16S)
		v.AddArg(x)
		return true
	}
	// match: (I64Extend16S (I64Extend8S x))
	// result: (I64Extend8S x)
	for {
		if v_0.Op != ssaop.OpWasmI64Extend8S {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpWasmI64Extend8S)
		v.AddArg(x)
		return true
	}
	// match: (I64Extend16S x:(I64And _ (I64Const [c])))
	// cond: c >= 0 && int64(int16(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64And {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(x_1.AuxInt)
		if !(c >= 0 && int64(int16(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Extend32S(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (I64Extend32S (I64Extend32S x))
	// result: (I64Extend32S x)
	for {
		if v_0.Op != ssaop.OpWasmI64Extend32S {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpWasmI64Extend32S)
		v.AddArg(x)
		return true
	}
	// match: (I64Extend32S (I64Extend16S x))
	// result: (I64Extend16S x)
	for {
		if v_0.Op != ssaop.OpWasmI64Extend16S {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpWasmI64Extend16S)
		v.AddArg(x)
		return true
	}
	// match: (I64Extend32S (I64Extend8S x))
	// result: (I64Extend8S x)
	for {
		if v_0.Op != ssaop.OpWasmI64Extend8S {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpWasmI64Extend8S)
		v.AddArg(x)
		return true
	}
	// match: (I64Extend32S x:(I64And _ (I64Const [c])))
	// cond: c >= 0 && int64(int32(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64And {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(x_1.AuxInt)
		if !(c >= 0 && int64(int32(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Extend8S(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (I64Extend8S (I64Extend8S x))
	// result: (I64Extend8S x)
	for {
		if v_0.Op != ssaop.OpWasmI64Extend8S {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpWasmI64Extend8S)
		v.AddArg(x)
		return true
	}
	// match: (I64Extend8S x:(I64And _ (I64Const [c])))
	// cond: c >= 0 && int64(int8(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64And {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != ssaop.OpWasmI64Const {
			break
		}
		c := AuxIntToInt64(x_1.AuxInt)
		if !(c >= 0 && int64(int8(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64LeU(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64LeU x (I64Const [0]))
	// result: (I64Eqz x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpWasmI64Eqz)
		v.AddArg(x)
		return true
	}
	// match: (I64LeU (I64Const [1]) x)
	// result: (I64Eqz (I64Eqz x))
	for {
		if v_0.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpWasmI64Eqz)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Eqz, typ.Bool)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (I64Load [off] (I64AddConst [off2] ptr) mem)
	// cond: IsU32Bit(off+off2)
	// result: (I64Load [off+off2] ptr mem)
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmI64AddConst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(IsU32Bit(off + off2)) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load)
		v.AuxInt = Int64ToAuxInt(off + off2)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (I64Load [off] (LoweredAddr {sym} [off2] (SB)) _)
	// cond: SymIsRO(sym) && IsU32Bit(off+int64(off2))
	// result: (I64Const [int64(Read64(sym, off+int64(off2), config.Ctxt.Arch.ByteOrder))])
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmLoweredAddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym := AuxToSym(v_0.Aux)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpSB || !(SymIsRO(sym) && IsU32Bit(off+int64(off2))) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(int64(Read64(sym, off+int64(off2), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load16S(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (I64Load16S [off] (I64AddConst [off2] ptr) mem)
	// cond: IsU32Bit(off+off2)
	// result: (I64Load16S [off+off2] ptr mem)
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmI64AddConst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(IsU32Bit(off + off2)) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load16S)
		v.AuxInt = Int64ToAuxInt(off + off2)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (I64Load16S [off] (LoweredAddr {sym} [off2] (SB)) _)
	// cond: SymIsRO(sym) && IsU32Bit(off+int64(off2))
	// result: (I64Const [int64(int16(Read16(sym, off+int64(off2), config.Ctxt.Arch.ByteOrder)))])
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmLoweredAddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym := AuxToSym(v_0.Aux)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpSB || !(SymIsRO(sym) && IsU32Bit(off+int64(off2))) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(int64(int16(Read16(sym, off+int64(off2), config.Ctxt.Arch.ByteOrder))))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (I64Load16U [off] (I64AddConst [off2] ptr) mem)
	// cond: IsU32Bit(off+off2)
	// result: (I64Load16U [off+off2] ptr mem)
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmI64AddConst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(IsU32Bit(off + off2)) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load16U)
		v.AuxInt = Int64ToAuxInt(off + off2)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (I64Load16U [off] (LoweredAddr {sym} [off2] (SB)) _)
	// cond: SymIsRO(sym) && IsU32Bit(off+int64(off2))
	// result: (I64Const [int64(Read16(sym, off+int64(off2), config.Ctxt.Arch.ByteOrder))])
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmLoweredAddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym := AuxToSym(v_0.Aux)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpSB || !(SymIsRO(sym) && IsU32Bit(off+int64(off2))) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(int64(Read16(sym, off+int64(off2), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load32S(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (I64Load32S [off] (I64AddConst [off2] ptr) mem)
	// cond: IsU32Bit(off+off2)
	// result: (I64Load32S [off+off2] ptr mem)
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmI64AddConst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(IsU32Bit(off + off2)) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load32S)
		v.AuxInt = Int64ToAuxInt(off + off2)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (I64Load32S [off] (LoweredAddr {sym} [off2] (SB)) _)
	// cond: SymIsRO(sym) && IsU32Bit(off+int64(off2))
	// result: (I64Const [int64(int32(Read32(sym, off+int64(off2), config.Ctxt.Arch.ByteOrder)))])
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmLoweredAddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym := AuxToSym(v_0.Aux)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpSB || !(SymIsRO(sym) && IsU32Bit(off+int64(off2))) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(int64(int32(Read32(sym, off+int64(off2), config.Ctxt.Arch.ByteOrder))))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (I64Load32U [off] (I64AddConst [off2] ptr) mem)
	// cond: IsU32Bit(off+off2)
	// result: (I64Load32U [off+off2] ptr mem)
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmI64AddConst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(IsU32Bit(off + off2)) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load32U)
		v.AuxInt = Int64ToAuxInt(off + off2)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (I64Load32U [off] (LoweredAddr {sym} [off2] (SB)) _)
	// cond: SymIsRO(sym) && IsU32Bit(off+int64(off2))
	// result: (I64Const [int64(Read32(sym, off+int64(off2), config.Ctxt.Arch.ByteOrder))])
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmLoweredAddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym := AuxToSym(v_0.Aux)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpSB || !(SymIsRO(sym) && IsU32Bit(off+int64(off2))) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(int64(Read32(sym, off+int64(off2), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load8S(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Load8S [off] (I64AddConst [off2] ptr) mem)
	// cond: IsU32Bit(off+off2)
	// result: (I64Load8S [off+off2] ptr mem)
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmI64AddConst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(IsU32Bit(off + off2)) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load8S)
		v.AuxInt = Int64ToAuxInt(off + off2)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (I64Load8S [off] (LoweredAddr {sym} [off2] (SB)) _)
	// cond: SymIsRO(sym) && IsU32Bit(off+int64(off2))
	// result: (I64Const [int64(int8(Read8(sym, off+int64(off2))))])
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmLoweredAddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym := AuxToSym(v_0.Aux)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpSB || !(SymIsRO(sym) && IsU32Bit(off+int64(off2))) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(int64(int8(Read8(sym, off+int64(off2)))))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Load8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Load8U [off] (I64AddConst [off2] ptr) mem)
	// cond: IsU32Bit(off+off2)
	// result: (I64Load8U [off+off2] ptr mem)
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmI64AddConst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(IsU32Bit(off + off2)) {
			break
		}
		v.Reset(ssaop.OpWasmI64Load8U)
		v.AuxInt = Int64ToAuxInt(off + off2)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (I64Load8U [off] (LoweredAddr {sym} [off2] (SB)) _)
	// cond: SymIsRO(sym) && IsU32Bit(off+int64(off2))
	// result: (I64Const [int64(Read8(sym, off+int64(off2)))])
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmLoweredAddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym := AuxToSym(v_0.Aux)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpSB || !(SymIsRO(sym) && IsU32Bit(off+int64(off2))) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(int64(Read8(sym, off+int64(off2))))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64LtU(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64LtU (I64Const [0]) x)
	// result: (I64Eqz (I64Eqz x))
	for {
		if v_0.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpWasmI64Eqz)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Eqz, typ.Bool)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (I64LtU x (I64Const [1]))
	// result: (I64Eqz x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.Reset(ssaop.OpWasmI64Eqz)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Mul(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64Mul (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x * y])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(x * y)
		return true
	}
	// match: (I64Mul x (I64Const [0]))
	// result: (I64Const [0])
	for {
		if v_1.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (I64Mul x (I64Const [1]))
	// result: x
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (I64Mul (I64Const [x]) y)
	// cond: y.Op != ssaop.OpWasmI64Const
	// result: (I64Mul y (I64Const [x]))
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(y.Op != ssaop.OpWasmI64Const) {
			break
		}
		v.Reset(ssaop.OpWasmI64Mul)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(x)
		v.AddArg2(y, v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Ne(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64Ne (I64Const [x]) (I64Const [y]))
	// cond: x == y
	// result: (I64Const [0])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		if !(x == y) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (I64Ne (I64Const [x]) (I64Const [y]))
	// cond: x != y
	// result: (I64Const [1])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		if !(x != y) {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	// match: (I64Ne (I64Const [x]) y)
	// cond: y.Op != ssaop.OpWasmI64Const
	// result: (I64Ne y (I64Const [x]))
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(y.Op != ssaop.OpWasmI64Const) {
			break
		}
		v.Reset(ssaop.OpWasmI64Ne)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(x)
		v.AddArg2(y, v0)
		return true
	}
	// match: (I64Ne x (I64Const [0]))
	// result: (I64Eqz (I64Eqz x))
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpWasmI64Eqz)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Eqz, typ.Bool)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Or(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64Or (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x | y])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(x | y)
		return true
	}
	// match: (I64Or x (I64Const [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (I64Or x (I64Const [-1]))
	// result: (I64Const [-1])
	for {
		if v_1.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(-1)
		return true
	}
	// match: (I64Or (I64Const [x]) y)
	// cond: y.Op != ssaop.OpWasmI64Const
	// result: (I64Or y (I64Const [x]))
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(y.Op != ssaop.OpWasmI64Const) {
			break
		}
		v.Reset(ssaop.OpWasmI64Or)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(x)
		v.AddArg2(y, v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Shl(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Shl (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x << uint64(y)])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(x << uint64(y))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64ShrS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64ShrS (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x >> uint64(y)])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(x >> uint64(y))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64ShrU(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64ShrU (I64Const [x]) (I64Const [y]))
	// result: (I64Const [int64(uint64(x) >> uint64(y))])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(int64(uint64(x) >> uint64(y)))
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Store(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Store [off] (I64AddConst [off2] ptr) val mem)
	// cond: IsU32Bit(off+off2)
	// result: (I64Store [off+off2] ptr val mem)
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmI64AddConst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(IsU32Bit(off + off2)) {
			break
		}
		v.Reset(ssaop.OpWasmI64Store)
		v.AuxInt = Int64ToAuxInt(off + off2)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Store16(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Store16 [off] (I64AddConst [off2] ptr) val mem)
	// cond: IsU32Bit(off+off2)
	// result: (I64Store16 [off+off2] ptr val mem)
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmI64AddConst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(IsU32Bit(off + off2)) {
			break
		}
		v.Reset(ssaop.OpWasmI64Store16)
		v.AuxInt = Int64ToAuxInt(off + off2)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Store32(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Store32 [off] (I64AddConst [off2] ptr) val mem)
	// cond: IsU32Bit(off+off2)
	// result: (I64Store32 [off+off2] ptr val mem)
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmI64AddConst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(IsU32Bit(off + off2)) {
			break
		}
		v.Reset(ssaop.OpWasmI64Store32)
		v.AuxInt = Int64ToAuxInt(off + off2)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Store8(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Store8 [off] (I64AddConst [off2] ptr) val mem)
	// cond: IsU32Bit(off+off2)
	// result: (I64Store8 [off+off2] ptr val mem)
	for {
		off := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpWasmI64AddConst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(IsU32Bit(off + off2)) {
			break
		}
		v.Reset(ssaop.OpWasmI64Store8)
		v.AuxInt = Int64ToAuxInt(off + off2)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Sub(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (I64Sub (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x - y])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(x - y)
		return true
	}
	return false
}
func rewriteValueWasm_OpWasmI64Xor(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (I64Xor (I64Const [x]) (I64Const [y]))
	// result: (I64Const [x ^ y])
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpWasmI64Const {
			break
		}
		y := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpWasmI64Const)
		v.AuxInt = Int64ToAuxInt(x ^ y)
		return true
	}
	// match: (I64Xor x (I64Const [0]))
	// result: x
	for {
		x := v_0
		if v_1.Op != ssaop.OpWasmI64Const || AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (I64Xor (I64Const [x]) y)
	// cond: y.Op != ssaop.OpWasmI64Const
	// result: (I64Xor y (I64Const [x]))
	for {
		if v_0.Op != ssaop.OpWasmI64Const {
			break
		}
		x := AuxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(y.Op != ssaop.OpWasmI64Const) {
			break
		}
		v.Reset(ssaop.OpWasmI64Xor)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(x)
		v.AddArg2(y, v0)
		return true
	}
	return false
}
func rewriteValueWasm_OpZero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Zero [0] _ mem)
	// result: mem
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		mem := v_1
		v.CopyOf(mem)
		return true
	}
	// match: (Zero [1] destptr mem)
	// result: (I64Store8 destptr (I64Const [0]) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpWasmI64Store8)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg3(destptr, v0, mem)
		return true
	}
	// match: (Zero [2] destptr mem)
	// result: (I64Store16 destptr (I64Const [0]) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpWasmI64Store16)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg3(destptr, v0, mem)
		return true
	}
	// match: (Zero [4] destptr mem)
	// result: (I64Store32 destptr (I64Const [0]) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpWasmI64Store32)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg3(destptr, v0, mem)
		return true
	}
	// match: (Zero [8] destptr mem)
	// result: (I64Store destptr (I64Const [0]) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpWasmI64Store)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg3(destptr, v0, mem)
		return true
	}
	// match: (Zero [3] destptr mem)
	// result: (I64Store8 [2] destptr (I64Const [0]) (I64Store16 destptr (I64Const [0]) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpWasmI64Store8)
		v.AuxInt = Int64ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store16, types.TypeMem)
		v1.AddArg3(destptr, v0, mem)
		v.AddArg3(destptr, v0, v1)
		return true
	}
	// match: (Zero [5] destptr mem)
	// result: (I64Store8 [4] destptr (I64Const [0]) (I64Store32 destptr (I64Const [0]) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 5 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpWasmI64Store8)
		v.AuxInt = Int64ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store32, types.TypeMem)
		v1.AddArg3(destptr, v0, mem)
		v.AddArg3(destptr, v0, v1)
		return true
	}
	// match: (Zero [6] destptr mem)
	// result: (I64Store16 [4] destptr (I64Const [0]) (I64Store32 destptr (I64Const [0]) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpWasmI64Store16)
		v.AuxInt = Int64ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store32, types.TypeMem)
		v1.AddArg3(destptr, v0, mem)
		v.AddArg3(destptr, v0, v1)
		return true
	}
	// match: (Zero [7] destptr mem)
	// result: (I64Store32 [3] destptr (I64Const [0]) (I64Store32 destptr (I64Const [0]) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 7 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpWasmI64Store32)
		v.AuxInt = Int64ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store32, types.TypeMem)
		v1.AddArg3(destptr, v0, mem)
		v.AddArg3(destptr, v0, v1)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s%8 != 0 && s > 8 && s < 32
	// result: (Zero [s-s%8] (OffPtr <destptr.Type> destptr [s%8]) (I64Store destptr (I64Const [0]) mem))
	for {
		s := AuxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		if !(s%8 != 0 && s > 8 && s < 32) {
			break
		}
		v.Reset(ssaop.OpZero)
		v.AuxInt = Int64ToAuxInt(s - s%8)
		v0 := b.NewValue0(v.Pos, ssaop.OpOffPtr, destptr.Type)
		v0.AuxInt = Int64ToAuxInt(s % 8)
		v0.AddArg(destptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v2.AuxInt = Int64ToAuxInt(0)
		v1.AddArg3(destptr, v2, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Zero [16] destptr mem)
	// result: (I64Store [8] destptr (I64Const [0]) (I64Store destptr (I64Const [0]) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 16 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpWasmI64Store)
		v.AuxInt = Int64ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store, types.TypeMem)
		v1.AddArg3(destptr, v0, mem)
		v.AddArg3(destptr, v0, v1)
		return true
	}
	// match: (Zero [24] destptr mem)
	// result: (I64Store [16] destptr (I64Const [0]) (I64Store [8] destptr (I64Const [0]) (I64Store destptr (I64Const [0]) mem)))
	for {
		if AuxIntToInt64(v.AuxInt) != 24 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpWasmI64Store)
		v.AuxInt = Int64ToAuxInt(16)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store, types.TypeMem)
		v1.AuxInt = Int64ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store, types.TypeMem)
		v2.AddArg3(destptr, v0, mem)
		v1.AddArg3(destptr, v0, v2)
		v.AddArg3(destptr, v0, v1)
		return true
	}
	// match: (Zero [32] destptr mem)
	// result: (I64Store [24] destptr (I64Const [0]) (I64Store [16] destptr (I64Const [0]) (I64Store [8] destptr (I64Const [0]) (I64Store destptr (I64Const [0]) mem))))
	for {
		if AuxIntToInt64(v.AuxInt) != 32 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpWasmI64Store)
		v.AuxInt = Int64ToAuxInt(24)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store, types.TypeMem)
		v1.AuxInt = Int64ToAuxInt(16)
		v2 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store, types.TypeMem)
		v2.AuxInt = Int64ToAuxInt(8)
		v3 := b.NewValue0(v.Pos, ssaop.OpWasmI64Store, types.TypeMem)
		v3.AddArg3(destptr, v0, mem)
		v2.AddArg3(destptr, v0, v3)
		v1.AddArg3(destptr, v0, v2)
		v.AddArg3(destptr, v0, v1)
		return true
	}
	// match: (Zero [s] destptr mem)
	// result: (LoweredZero [s] destptr mem)
	for {
		s := AuxIntToInt64(v.AuxInt)
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpWasmLoweredZero)
		v.AuxInt = Int64ToAuxInt(s)
		v.AddArg2(destptr, mem)
		return true
	}
}
func rewriteValueWasm_OpZeroExt16to32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ZeroExt16to32 x:(I64Load16U _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64Load16U {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ZeroExt16to32 x)
	// result: (I64And x (I64Const [0xffff]))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64And)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0xffff)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpZeroExt16to64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ZeroExt16to64 x:(I64Load16U _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64Load16U {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ZeroExt16to64 x)
	// result: (I64And x (I64Const [0xffff]))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64And)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0xffff)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpZeroExt32to64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ZeroExt32to64 x:(I64Load32U _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64Load32U {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ZeroExt32to64 x)
	// result: (I64And x (I64Const [0xffffffff]))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64And)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0xffffffff)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpZeroExt8to16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ZeroExt8to16 x:(I64Load8U _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64Load8U {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ZeroExt8to16 x)
	// result: (I64And x (I64Const [0xff]))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64And)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0xff)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpZeroExt8to32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ZeroExt8to32 x:(I64Load8U _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64Load8U {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ZeroExt8to32 x)
	// result: (I64And x (I64Const [0xff]))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64And)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0xff)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueWasm_OpZeroExt8to64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ZeroExt8to64 x:(I64Load8U _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpWasmI64Load8U {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ZeroExt8to64 x)
	// result: (I64And x (I64Const [0xff]))
	for {
		x := v_0
		v.Reset(ssaop.OpWasmI64And)
		v0 := b.NewValue0(v.Pos, ssaop.OpWasmI64Const, typ.Int64)
		v0.AuxInt = Int64ToAuxInt(0xff)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteBlockWasm(b *ssa.Block) bool {
	return false
}
