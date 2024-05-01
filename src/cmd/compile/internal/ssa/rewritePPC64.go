// Code generated from _gen/PPC64.rules using 'go generate'; DO NOT EDIT.

package ssa

import "internal/buildcfg"
import "math"
import "cmd/compile/internal/types"

func rewriteValuePPC64(v *Value) bool {
	switch v.Op {
	case OpAbs:
		v.Op = OpPPC64FABS
		return true
	case OpAdd16:
		v.Op = OpPPC64ADD
		return true
	case OpAdd32:
		v.Op = OpPPC64ADD
		return true
	case OpAdd32F:
		v.Op = OpPPC64FADDS
		return true
	case OpAdd64:
		v.Op = OpPPC64ADD
		return true
	case OpAdd64F:
		v.Op = OpPPC64FADD
		return true
	case OpAdd8:
		v.Op = OpPPC64ADD
		return true
	case OpAddPtr:
		v.Op = OpPPC64ADD
		return true
	case OpAddr:
		return rewriteValuePPC64_OpAddr(v)
	case OpAnd16:
		v.Op = OpPPC64AND
		return true
	case OpAnd32:
		v.Op = OpPPC64AND
		return true
	case OpAnd64:
		v.Op = OpPPC64AND
		return true
	case OpAnd8:
		v.Op = OpPPC64AND
		return true
	case OpAndB:
		v.Op = OpPPC64AND
		return true
	case OpAtomicAdd32:
		v.Op = OpPPC64LoweredAtomicAdd32
		return true
	case OpAtomicAdd64:
		v.Op = OpPPC64LoweredAtomicAdd64
		return true
	case OpAtomicAnd32:
		v.Op = OpPPC64LoweredAtomicAnd32
		return true
	case OpAtomicAnd8:
		v.Op = OpPPC64LoweredAtomicAnd8
		return true
	case OpAtomicCompareAndSwap32:
		return rewriteValuePPC64_OpAtomicCompareAndSwap32(v)
	case OpAtomicCompareAndSwap64:
		return rewriteValuePPC64_OpAtomicCompareAndSwap64(v)
	case OpAtomicCompareAndSwapRel32:
		return rewriteValuePPC64_OpAtomicCompareAndSwapRel32(v)
	case OpAtomicExchange32:
		v.Op = OpPPC64LoweredAtomicExchange32
		return true
	case OpAtomicExchange64:
		v.Op = OpPPC64LoweredAtomicExchange64
		return true
	case OpAtomicLoad32:
		return rewriteValuePPC64_OpAtomicLoad32(v)
	case OpAtomicLoad64:
		return rewriteValuePPC64_OpAtomicLoad64(v)
	case OpAtomicLoad8:
		return rewriteValuePPC64_OpAtomicLoad8(v)
	case OpAtomicLoadAcq32:
		return rewriteValuePPC64_OpAtomicLoadAcq32(v)
	case OpAtomicLoadAcq64:
		return rewriteValuePPC64_OpAtomicLoadAcq64(v)
	case OpAtomicLoadPtr:
		return rewriteValuePPC64_OpAtomicLoadPtr(v)
	case OpAtomicOr32:
		v.Op = OpPPC64LoweredAtomicOr32
		return true
	case OpAtomicOr8:
		v.Op = OpPPC64LoweredAtomicOr8
		return true
	case OpAtomicStore32:
		return rewriteValuePPC64_OpAtomicStore32(v)
	case OpAtomicStore64:
		return rewriteValuePPC64_OpAtomicStore64(v)
	case OpAtomicStore8:
		return rewriteValuePPC64_OpAtomicStore8(v)
	case OpAtomicStoreRel32:
		return rewriteValuePPC64_OpAtomicStoreRel32(v)
	case OpAtomicStoreRel64:
		return rewriteValuePPC64_OpAtomicStoreRel64(v)
	case OpAvg64u:
		return rewriteValuePPC64_OpAvg64u(v)
	case OpBitLen32:
		return rewriteValuePPC64_OpBitLen32(v)
	case OpBitLen64:
		return rewriteValuePPC64_OpBitLen64(v)
	case OpBswap16:
		return rewriteValuePPC64_OpBswap16(v)
	case OpBswap32:
		return rewriteValuePPC64_OpBswap32(v)
	case OpBswap64:
		return rewriteValuePPC64_OpBswap64(v)
	case OpCeil:
		v.Op = OpPPC64FCEIL
		return true
	case OpClosureCall:
		v.Op = OpPPC64CALLclosure
		return true
	case OpCom16:
		return rewriteValuePPC64_OpCom16(v)
	case OpCom32:
		return rewriteValuePPC64_OpCom32(v)
	case OpCom64:
		return rewriteValuePPC64_OpCom64(v)
	case OpCom8:
		return rewriteValuePPC64_OpCom8(v)
	case OpCondSelect:
		return rewriteValuePPC64_OpCondSelect(v)
	case OpConst16:
		return rewriteValuePPC64_OpConst16(v)
	case OpConst32:
		return rewriteValuePPC64_OpConst32(v)
	case OpConst32F:
		v.Op = OpPPC64FMOVSconst
		return true
	case OpConst64:
		return rewriteValuePPC64_OpConst64(v)
	case OpConst64F:
		v.Op = OpPPC64FMOVDconst
		return true
	case OpConst8:
		return rewriteValuePPC64_OpConst8(v)
	case OpConstBool:
		return rewriteValuePPC64_OpConstBool(v)
	case OpConstNil:
		return rewriteValuePPC64_OpConstNil(v)
	case OpCopysign:
		return rewriteValuePPC64_OpCopysign(v)
	case OpCtz16:
		return rewriteValuePPC64_OpCtz16(v)
	case OpCtz32:
		return rewriteValuePPC64_OpCtz32(v)
	case OpCtz32NonZero:
		v.Op = OpCtz32
		return true
	case OpCtz64:
		return rewriteValuePPC64_OpCtz64(v)
	case OpCtz64NonZero:
		v.Op = OpCtz64
		return true
	case OpCtz8:
		return rewriteValuePPC64_OpCtz8(v)
	case OpCvt32Fto32:
		return rewriteValuePPC64_OpCvt32Fto32(v)
	case OpCvt32Fto64:
		return rewriteValuePPC64_OpCvt32Fto64(v)
	case OpCvt32Fto64F:
		v.Op = OpCopy
		return true
	case OpCvt32to32F:
		return rewriteValuePPC64_OpCvt32to32F(v)
	case OpCvt32to64F:
		return rewriteValuePPC64_OpCvt32to64F(v)
	case OpCvt64Fto32:
		return rewriteValuePPC64_OpCvt64Fto32(v)
	case OpCvt64Fto32F:
		v.Op = OpPPC64FRSP
		return true
	case OpCvt64Fto64:
		return rewriteValuePPC64_OpCvt64Fto64(v)
	case OpCvt64to32F:
		return rewriteValuePPC64_OpCvt64to32F(v)
	case OpCvt64to64F:
		return rewriteValuePPC64_OpCvt64to64F(v)
	case OpCvtBoolToUint8:
		v.Op = OpCopy
		return true
	case OpDiv16:
		return rewriteValuePPC64_OpDiv16(v)
	case OpDiv16u:
		return rewriteValuePPC64_OpDiv16u(v)
	case OpDiv32:
		return rewriteValuePPC64_OpDiv32(v)
	case OpDiv32F:
		v.Op = OpPPC64FDIVS
		return true
	case OpDiv32u:
		v.Op = OpPPC64DIVWU
		return true
	case OpDiv64:
		return rewriteValuePPC64_OpDiv64(v)
	case OpDiv64F:
		v.Op = OpPPC64FDIV
		return true
	case OpDiv64u:
		v.Op = OpPPC64DIVDU
		return true
	case OpDiv8:
		return rewriteValuePPC64_OpDiv8(v)
	case OpDiv8u:
		return rewriteValuePPC64_OpDiv8u(v)
	case OpEq16:
		return rewriteValuePPC64_OpEq16(v)
	case OpEq32:
		return rewriteValuePPC64_OpEq32(v)
	case OpEq32F:
		return rewriteValuePPC64_OpEq32F(v)
	case OpEq64:
		return rewriteValuePPC64_OpEq64(v)
	case OpEq64F:
		return rewriteValuePPC64_OpEq64F(v)
	case OpEq8:
		return rewriteValuePPC64_OpEq8(v)
	case OpEqB:
		return rewriteValuePPC64_OpEqB(v)
	case OpEqPtr:
		return rewriteValuePPC64_OpEqPtr(v)
	case OpFMA:
		v.Op = OpPPC64FMADD
		return true
	case OpFloor:
		v.Op = OpPPC64FFLOOR
		return true
	case OpGetCallerPC:
		v.Op = OpPPC64LoweredGetCallerPC
		return true
	case OpGetCallerSP:
		v.Op = OpPPC64LoweredGetCallerSP
		return true
	case OpGetClosurePtr:
		v.Op = OpPPC64LoweredGetClosurePtr
		return true
	case OpHmul32:
		v.Op = OpPPC64MULHW
		return true
	case OpHmul32u:
		v.Op = OpPPC64MULHWU
		return true
	case OpHmul64:
		v.Op = OpPPC64MULHD
		return true
	case OpHmul64u:
		v.Op = OpPPC64MULHDU
		return true
	case OpInterCall:
		v.Op = OpPPC64CALLinter
		return true
	case OpIsInBounds:
		return rewriteValuePPC64_OpIsInBounds(v)
	case OpIsNonNil:
		return rewriteValuePPC64_OpIsNonNil(v)
	case OpIsSliceInBounds:
		return rewriteValuePPC64_OpIsSliceInBounds(v)
	case OpLeq16:
		return rewriteValuePPC64_OpLeq16(v)
	case OpLeq16U:
		return rewriteValuePPC64_OpLeq16U(v)
	case OpLeq32:
		return rewriteValuePPC64_OpLeq32(v)
	case OpLeq32F:
		return rewriteValuePPC64_OpLeq32F(v)
	case OpLeq32U:
		return rewriteValuePPC64_OpLeq32U(v)
	case OpLeq64:
		return rewriteValuePPC64_OpLeq64(v)
	case OpLeq64F:
		return rewriteValuePPC64_OpLeq64F(v)
	case OpLeq64U:
		return rewriteValuePPC64_OpLeq64U(v)
	case OpLeq8:
		return rewriteValuePPC64_OpLeq8(v)
	case OpLeq8U:
		return rewriteValuePPC64_OpLeq8U(v)
	case OpLess16:
		return rewriteValuePPC64_OpLess16(v)
	case OpLess16U:
		return rewriteValuePPC64_OpLess16U(v)
	case OpLess32:
		return rewriteValuePPC64_OpLess32(v)
	case OpLess32F:
		return rewriteValuePPC64_OpLess32F(v)
	case OpLess32U:
		return rewriteValuePPC64_OpLess32U(v)
	case OpLess64:
		return rewriteValuePPC64_OpLess64(v)
	case OpLess64F:
		return rewriteValuePPC64_OpLess64F(v)
	case OpLess64U:
		return rewriteValuePPC64_OpLess64U(v)
	case OpLess8:
		return rewriteValuePPC64_OpLess8(v)
	case OpLess8U:
		return rewriteValuePPC64_OpLess8U(v)
	case OpLoad:
		return rewriteValuePPC64_OpLoad(v)
	case OpLocalAddr:
		return rewriteValuePPC64_OpLocalAddr(v)
	case OpLsh16x16:
		return rewriteValuePPC64_OpLsh16x16(v)
	case OpLsh16x32:
		return rewriteValuePPC64_OpLsh16x32(v)
	case OpLsh16x64:
		return rewriteValuePPC64_OpLsh16x64(v)
	case OpLsh16x8:
		return rewriteValuePPC64_OpLsh16x8(v)
	case OpLsh32x16:
		return rewriteValuePPC64_OpLsh32x16(v)
	case OpLsh32x32:
		return rewriteValuePPC64_OpLsh32x32(v)
	case OpLsh32x64:
		return rewriteValuePPC64_OpLsh32x64(v)
	case OpLsh32x8:
		return rewriteValuePPC64_OpLsh32x8(v)
	case OpLsh64x16:
		return rewriteValuePPC64_OpLsh64x16(v)
	case OpLsh64x32:
		return rewriteValuePPC64_OpLsh64x32(v)
	case OpLsh64x64:
		return rewriteValuePPC64_OpLsh64x64(v)
	case OpLsh64x8:
		return rewriteValuePPC64_OpLsh64x8(v)
	case OpLsh8x16:
		return rewriteValuePPC64_OpLsh8x16(v)
	case OpLsh8x32:
		return rewriteValuePPC64_OpLsh8x32(v)
	case OpLsh8x64:
		return rewriteValuePPC64_OpLsh8x64(v)
	case OpLsh8x8:
		return rewriteValuePPC64_OpLsh8x8(v)
	case OpMax32F:
		return rewriteValuePPC64_OpMax32F(v)
	case OpMax64F:
		return rewriteValuePPC64_OpMax64F(v)
	case OpMin32F:
		return rewriteValuePPC64_OpMin32F(v)
	case OpMin64F:
		return rewriteValuePPC64_OpMin64F(v)
	case OpMod16:
		return rewriteValuePPC64_OpMod16(v)
	case OpMod16u:
		return rewriteValuePPC64_OpMod16u(v)
	case OpMod32:
		return rewriteValuePPC64_OpMod32(v)
	case OpMod32u:
		return rewriteValuePPC64_OpMod32u(v)
	case OpMod64:
		return rewriteValuePPC64_OpMod64(v)
	case OpMod64u:
		return rewriteValuePPC64_OpMod64u(v)
	case OpMod8:
		return rewriteValuePPC64_OpMod8(v)
	case OpMod8u:
		return rewriteValuePPC64_OpMod8u(v)
	case OpMove:
		return rewriteValuePPC64_OpMove(v)
	case OpMul16:
		v.Op = OpPPC64MULLW
		return true
	case OpMul32:
		v.Op = OpPPC64MULLW
		return true
	case OpMul32F:
		v.Op = OpPPC64FMULS
		return true
	case OpMul64:
		v.Op = OpPPC64MULLD
		return true
	case OpMul64F:
		v.Op = OpPPC64FMUL
		return true
	case OpMul8:
		v.Op = OpPPC64MULLW
		return true
	case OpNeg16:
		v.Op = OpPPC64NEG
		return true
	case OpNeg32:
		v.Op = OpPPC64NEG
		return true
	case OpNeg32F:
		v.Op = OpPPC64FNEG
		return true
	case OpNeg64:
		v.Op = OpPPC64NEG
		return true
	case OpNeg64F:
		v.Op = OpPPC64FNEG
		return true
	case OpNeg8:
		v.Op = OpPPC64NEG
		return true
	case OpNeq16:
		return rewriteValuePPC64_OpNeq16(v)
	case OpNeq32:
		return rewriteValuePPC64_OpNeq32(v)
	case OpNeq32F:
		return rewriteValuePPC64_OpNeq32F(v)
	case OpNeq64:
		return rewriteValuePPC64_OpNeq64(v)
	case OpNeq64F:
		return rewriteValuePPC64_OpNeq64F(v)
	case OpNeq8:
		return rewriteValuePPC64_OpNeq8(v)
	case OpNeqB:
		v.Op = OpPPC64XOR
		return true
	case OpNeqPtr:
		return rewriteValuePPC64_OpNeqPtr(v)
	case OpNilCheck:
		v.Op = OpPPC64LoweredNilCheck
		return true
	case OpNot:
		return rewriteValuePPC64_OpNot(v)
	case OpOffPtr:
		return rewriteValuePPC64_OpOffPtr(v)
	case OpOr16:
		v.Op = OpPPC64OR
		return true
	case OpOr32:
		v.Op = OpPPC64OR
		return true
	case OpOr64:
		v.Op = OpPPC64OR
		return true
	case OpOr8:
		v.Op = OpPPC64OR
		return true
	case OpOrB:
		v.Op = OpPPC64OR
		return true
	case OpPPC64ADD:
		return rewriteValuePPC64_OpPPC64ADD(v)
	case OpPPC64ADDE:
		return rewriteValuePPC64_OpPPC64ADDE(v)
	case OpPPC64ADDconst:
		return rewriteValuePPC64_OpPPC64ADDconst(v)
	case OpPPC64AND:
		return rewriteValuePPC64_OpPPC64AND(v)
	case OpPPC64ANDCCconst:
		return rewriteValuePPC64_OpPPC64ANDCCconst(v)
	case OpPPC64ANDN:
		return rewriteValuePPC64_OpPPC64ANDN(v)
	case OpPPC64BRD:
		return rewriteValuePPC64_OpPPC64BRD(v)
	case OpPPC64BRH:
		return rewriteValuePPC64_OpPPC64BRH(v)
	case OpPPC64BRW:
		return rewriteValuePPC64_OpPPC64BRW(v)
	case OpPPC64CLRLSLDI:
		return rewriteValuePPC64_OpPPC64CLRLSLDI(v)
	case OpPPC64CMP:
		return rewriteValuePPC64_OpPPC64CMP(v)
	case OpPPC64CMPU:
		return rewriteValuePPC64_OpPPC64CMPU(v)
	case OpPPC64CMPUconst:
		return rewriteValuePPC64_OpPPC64CMPUconst(v)
	case OpPPC64CMPW:
		return rewriteValuePPC64_OpPPC64CMPW(v)
	case OpPPC64CMPWU:
		return rewriteValuePPC64_OpPPC64CMPWU(v)
	case OpPPC64CMPWUconst:
		return rewriteValuePPC64_OpPPC64CMPWUconst(v)
	case OpPPC64CMPWconst:
		return rewriteValuePPC64_OpPPC64CMPWconst(v)
	case OpPPC64CMPconst:
		return rewriteValuePPC64_OpPPC64CMPconst(v)
	case OpPPC64Equal:
		return rewriteValuePPC64_OpPPC64Equal(v)
	case OpPPC64FABS:
		return rewriteValuePPC64_OpPPC64FABS(v)
	case OpPPC64FADD:
		return rewriteValuePPC64_OpPPC64FADD(v)
	case OpPPC64FADDS:
		return rewriteValuePPC64_OpPPC64FADDS(v)
	case OpPPC64FCEIL:
		return rewriteValuePPC64_OpPPC64FCEIL(v)
	case OpPPC64FFLOOR:
		return rewriteValuePPC64_OpPPC64FFLOOR(v)
	case OpPPC64FGreaterEqual:
		return rewriteValuePPC64_OpPPC64FGreaterEqual(v)
	case OpPPC64FGreaterThan:
		return rewriteValuePPC64_OpPPC64FGreaterThan(v)
	case OpPPC64FLessEqual:
		return rewriteValuePPC64_OpPPC64FLessEqual(v)
	case OpPPC64FLessThan:
		return rewriteValuePPC64_OpPPC64FLessThan(v)
	case OpPPC64FMOVDload:
		return rewriteValuePPC64_OpPPC64FMOVDload(v)
	case OpPPC64FMOVDstore:
		return rewriteValuePPC64_OpPPC64FMOVDstore(v)
	case OpPPC64FMOVSload:
		return rewriteValuePPC64_OpPPC64FMOVSload(v)
	case OpPPC64FMOVSstore:
		return rewriteValuePPC64_OpPPC64FMOVSstore(v)
	case OpPPC64FNEG:
		return rewriteValuePPC64_OpPPC64FNEG(v)
	case OpPPC64FSQRT:
		return rewriteValuePPC64_OpPPC64FSQRT(v)
	case OpPPC64FSUB:
		return rewriteValuePPC64_OpPPC64FSUB(v)
	case OpPPC64FSUBS:
		return rewriteValuePPC64_OpPPC64FSUBS(v)
	case OpPPC64FTRUNC:
		return rewriteValuePPC64_OpPPC64FTRUNC(v)
	case OpPPC64GreaterEqual:
		return rewriteValuePPC64_OpPPC64GreaterEqual(v)
	case OpPPC64GreaterThan:
		return rewriteValuePPC64_OpPPC64GreaterThan(v)
	case OpPPC64ISEL:
		return rewriteValuePPC64_OpPPC64ISEL(v)
	case OpPPC64LessEqual:
		return rewriteValuePPC64_OpPPC64LessEqual(v)
	case OpPPC64LessThan:
		return rewriteValuePPC64_OpPPC64LessThan(v)
	case OpPPC64MFVSRD:
		return rewriteValuePPC64_OpPPC64MFVSRD(v)
	case OpPPC64MOVBZload:
		return rewriteValuePPC64_OpPPC64MOVBZload(v)
	case OpPPC64MOVBZloadidx:
		return rewriteValuePPC64_OpPPC64MOVBZloadidx(v)
	case OpPPC64MOVBZreg:
		return rewriteValuePPC64_OpPPC64MOVBZreg(v)
	case OpPPC64MOVBreg:
		return rewriteValuePPC64_OpPPC64MOVBreg(v)
	case OpPPC64MOVBstore:
		return rewriteValuePPC64_OpPPC64MOVBstore(v)
	case OpPPC64MOVBstoreidx:
		return rewriteValuePPC64_OpPPC64MOVBstoreidx(v)
	case OpPPC64MOVBstorezero:
		return rewriteValuePPC64_OpPPC64MOVBstorezero(v)
	case OpPPC64MOVDaddr:
		return rewriteValuePPC64_OpPPC64MOVDaddr(v)
	case OpPPC64MOVDload:
		return rewriteValuePPC64_OpPPC64MOVDload(v)
	case OpPPC64MOVDloadidx:
		return rewriteValuePPC64_OpPPC64MOVDloadidx(v)
	case OpPPC64MOVDstore:
		return rewriteValuePPC64_OpPPC64MOVDstore(v)
	case OpPPC64MOVDstoreidx:
		return rewriteValuePPC64_OpPPC64MOVDstoreidx(v)
	case OpPPC64MOVDstorezero:
		return rewriteValuePPC64_OpPPC64MOVDstorezero(v)
	case OpPPC64MOVHBRstore:
		return rewriteValuePPC64_OpPPC64MOVHBRstore(v)
	case OpPPC64MOVHZload:
		return rewriteValuePPC64_OpPPC64MOVHZload(v)
	case OpPPC64MOVHZloadidx:
		return rewriteValuePPC64_OpPPC64MOVHZloadidx(v)
	case OpPPC64MOVHZreg:
		return rewriteValuePPC64_OpPPC64MOVHZreg(v)
	case OpPPC64MOVHload:
		return rewriteValuePPC64_OpPPC64MOVHload(v)
	case OpPPC64MOVHloadidx:
		return rewriteValuePPC64_OpPPC64MOVHloadidx(v)
	case OpPPC64MOVHreg:
		return rewriteValuePPC64_OpPPC64MOVHreg(v)
	case OpPPC64MOVHstore:
		return rewriteValuePPC64_OpPPC64MOVHstore(v)
	case OpPPC64MOVHstoreidx:
		return rewriteValuePPC64_OpPPC64MOVHstoreidx(v)
	case OpPPC64MOVHstorezero:
		return rewriteValuePPC64_OpPPC64MOVHstorezero(v)
	case OpPPC64MOVWBRstore:
		return rewriteValuePPC64_OpPPC64MOVWBRstore(v)
	case OpPPC64MOVWZload:
		return rewriteValuePPC64_OpPPC64MOVWZload(v)
	case OpPPC64MOVWZloadidx:
		return rewriteValuePPC64_OpPPC64MOVWZloadidx(v)
	case OpPPC64MOVWZreg:
		return rewriteValuePPC64_OpPPC64MOVWZreg(v)
	case OpPPC64MOVWload:
		return rewriteValuePPC64_OpPPC64MOVWload(v)
	case OpPPC64MOVWloadidx:
		return rewriteValuePPC64_OpPPC64MOVWloadidx(v)
	case OpPPC64MOVWreg:
		return rewriteValuePPC64_OpPPC64MOVWreg(v)
	case OpPPC64MOVWstore:
		return rewriteValuePPC64_OpPPC64MOVWstore(v)
	case OpPPC64MOVWstoreidx:
		return rewriteValuePPC64_OpPPC64MOVWstoreidx(v)
	case OpPPC64MOVWstorezero:
		return rewriteValuePPC64_OpPPC64MOVWstorezero(v)
	case OpPPC64MTVSRD:
		return rewriteValuePPC64_OpPPC64MTVSRD(v)
	case OpPPC64MULLD:
		return rewriteValuePPC64_OpPPC64MULLD(v)
	case OpPPC64MULLW:
		return rewriteValuePPC64_OpPPC64MULLW(v)
	case OpPPC64NEG:
		return rewriteValuePPC64_OpPPC64NEG(v)
	case OpPPC64NOR:
		return rewriteValuePPC64_OpPPC64NOR(v)
	case OpPPC64NotEqual:
		return rewriteValuePPC64_OpPPC64NotEqual(v)
	case OpPPC64OR:
		return rewriteValuePPC64_OpPPC64OR(v)
	case OpPPC64ORN:
		return rewriteValuePPC64_OpPPC64ORN(v)
	case OpPPC64ORconst:
		return rewriteValuePPC64_OpPPC64ORconst(v)
	case OpPPC64ROTL:
		return rewriteValuePPC64_OpPPC64ROTL(v)
	case OpPPC64ROTLW:
		return rewriteValuePPC64_OpPPC64ROTLW(v)
	case OpPPC64ROTLWconst:
		return rewriteValuePPC64_OpPPC64ROTLWconst(v)
	case OpPPC64SETBC:
		return rewriteValuePPC64_OpPPC64SETBC(v)
	case OpPPC64SETBCR:
		return rewriteValuePPC64_OpPPC64SETBCR(v)
	case OpPPC64SLD:
		return rewriteValuePPC64_OpPPC64SLD(v)
	case OpPPC64SLDconst:
		return rewriteValuePPC64_OpPPC64SLDconst(v)
	case OpPPC64SLW:
		return rewriteValuePPC64_OpPPC64SLW(v)
	case OpPPC64SLWconst:
		return rewriteValuePPC64_OpPPC64SLWconst(v)
	case OpPPC64SRAD:
		return rewriteValuePPC64_OpPPC64SRAD(v)
	case OpPPC64SRAW:
		return rewriteValuePPC64_OpPPC64SRAW(v)
	case OpPPC64SRD:
		return rewriteValuePPC64_OpPPC64SRD(v)
	case OpPPC64SRW:
		return rewriteValuePPC64_OpPPC64SRW(v)
	case OpPPC64SRWconst:
		return rewriteValuePPC64_OpPPC64SRWconst(v)
	case OpPPC64SUB:
		return rewriteValuePPC64_OpPPC64SUB(v)
	case OpPPC64SUBE:
		return rewriteValuePPC64_OpPPC64SUBE(v)
	case OpPPC64SUBFCconst:
		return rewriteValuePPC64_OpPPC64SUBFCconst(v)
	case OpPPC64XOR:
		return rewriteValuePPC64_OpPPC64XOR(v)
	case OpPPC64XORconst:
		return rewriteValuePPC64_OpPPC64XORconst(v)
	case OpPanicBounds:
		return rewriteValuePPC64_OpPanicBounds(v)
	case OpPopCount16:
		return rewriteValuePPC64_OpPopCount16(v)
	case OpPopCount32:
		return rewriteValuePPC64_OpPopCount32(v)
	case OpPopCount64:
		v.Op = OpPPC64POPCNTD
		return true
	case OpPopCount8:
		return rewriteValuePPC64_OpPopCount8(v)
	case OpPrefetchCache:
		return rewriteValuePPC64_OpPrefetchCache(v)
	case OpPrefetchCacheStreamed:
		return rewriteValuePPC64_OpPrefetchCacheStreamed(v)
	case OpPubBarrier:
		v.Op = OpPPC64LoweredPubBarrier
		return true
	case OpRotateLeft16:
		return rewriteValuePPC64_OpRotateLeft16(v)
	case OpRotateLeft32:
		v.Op = OpPPC64ROTLW
		return true
	case OpRotateLeft64:
		v.Op = OpPPC64ROTL
		return true
	case OpRotateLeft8:
		return rewriteValuePPC64_OpRotateLeft8(v)
	case OpRound:
		v.Op = OpPPC64FROUND
		return true
	case OpRound32F:
		v.Op = OpPPC64LoweredRound32F
		return true
	case OpRound64F:
		v.Op = OpPPC64LoweredRound64F
		return true
	case OpRsh16Ux16:
		return rewriteValuePPC64_OpRsh16Ux16(v)
	case OpRsh16Ux32:
		return rewriteValuePPC64_OpRsh16Ux32(v)
	case OpRsh16Ux64:
		return rewriteValuePPC64_OpRsh16Ux64(v)
	case OpRsh16Ux8:
		return rewriteValuePPC64_OpRsh16Ux8(v)
	case OpRsh16x16:
		return rewriteValuePPC64_OpRsh16x16(v)
	case OpRsh16x32:
		return rewriteValuePPC64_OpRsh16x32(v)
	case OpRsh16x64:
		return rewriteValuePPC64_OpRsh16x64(v)
	case OpRsh16x8:
		return rewriteValuePPC64_OpRsh16x8(v)
	case OpRsh32Ux16:
		return rewriteValuePPC64_OpRsh32Ux16(v)
	case OpRsh32Ux32:
		return rewriteValuePPC64_OpRsh32Ux32(v)
	case OpRsh32Ux64:
		return rewriteValuePPC64_OpRsh32Ux64(v)
	case OpRsh32Ux8:
		return rewriteValuePPC64_OpRsh32Ux8(v)
	case OpRsh32x16:
		return rewriteValuePPC64_OpRsh32x16(v)
	case OpRsh32x32:
		return rewriteValuePPC64_OpRsh32x32(v)
	case OpRsh32x64:
		return rewriteValuePPC64_OpRsh32x64(v)
	case OpRsh32x8:
		return rewriteValuePPC64_OpRsh32x8(v)
	case OpRsh64Ux16:
		return rewriteValuePPC64_OpRsh64Ux16(v)
	case OpRsh64Ux32:
		return rewriteValuePPC64_OpRsh64Ux32(v)
	case OpRsh64Ux64:
		return rewriteValuePPC64_OpRsh64Ux64(v)
	case OpRsh64Ux8:
		return rewriteValuePPC64_OpRsh64Ux8(v)
	case OpRsh64x16:
		return rewriteValuePPC64_OpRsh64x16(v)
	case OpRsh64x32:
		return rewriteValuePPC64_OpRsh64x32(v)
	case OpRsh64x64:
		return rewriteValuePPC64_OpRsh64x64(v)
	case OpRsh64x8:
		return rewriteValuePPC64_OpRsh64x8(v)
	case OpRsh8Ux16:
		return rewriteValuePPC64_OpRsh8Ux16(v)
	case OpRsh8Ux32:
		return rewriteValuePPC64_OpRsh8Ux32(v)
	case OpRsh8Ux64:
		return rewriteValuePPC64_OpRsh8Ux64(v)
	case OpRsh8Ux8:
		return rewriteValuePPC64_OpRsh8Ux8(v)
	case OpRsh8x16:
		return rewriteValuePPC64_OpRsh8x16(v)
	case OpRsh8x32:
		return rewriteValuePPC64_OpRsh8x32(v)
	case OpRsh8x64:
		return rewriteValuePPC64_OpRsh8x64(v)
	case OpRsh8x8:
		return rewriteValuePPC64_OpRsh8x8(v)
	case OpSelect0:
		return rewriteValuePPC64_OpSelect0(v)
	case OpSelect1:
		return rewriteValuePPC64_OpSelect1(v)
	case OpSelectN:
		return rewriteValuePPC64_OpSelectN(v)
	case OpSignExt16to32:
		v.Op = OpPPC64MOVHreg
		return true
	case OpSignExt16to64:
		v.Op = OpPPC64MOVHreg
		return true
	case OpSignExt32to64:
		v.Op = OpPPC64MOVWreg
		return true
	case OpSignExt8to16:
		v.Op = OpPPC64MOVBreg
		return true
	case OpSignExt8to32:
		v.Op = OpPPC64MOVBreg
		return true
	case OpSignExt8to64:
		v.Op = OpPPC64MOVBreg
		return true
	case OpSlicemask:
		return rewriteValuePPC64_OpSlicemask(v)
	case OpSqrt:
		v.Op = OpPPC64FSQRT
		return true
	case OpSqrt32:
		v.Op = OpPPC64FSQRTS
		return true
	case OpStaticCall:
		v.Op = OpPPC64CALLstatic
		return true
	case OpStore:
		return rewriteValuePPC64_OpStore(v)
	case OpSub16:
		v.Op = OpPPC64SUB
		return true
	case OpSub32:
		v.Op = OpPPC64SUB
		return true
	case OpSub32F:
		v.Op = OpPPC64FSUBS
		return true
	case OpSub64:
		v.Op = OpPPC64SUB
		return true
	case OpSub64F:
		v.Op = OpPPC64FSUB
		return true
	case OpSub8:
		v.Op = OpPPC64SUB
		return true
	case OpSubPtr:
		v.Op = OpPPC64SUB
		return true
	case OpTailCall:
		v.Op = OpPPC64CALLtail
		return true
	case OpTrunc:
		v.Op = OpPPC64FTRUNC
		return true
	case OpTrunc16to8:
		return rewriteValuePPC64_OpTrunc16to8(v)
	case OpTrunc32to16:
		return rewriteValuePPC64_OpTrunc32to16(v)
	case OpTrunc32to8:
		return rewriteValuePPC64_OpTrunc32to8(v)
	case OpTrunc64to16:
		return rewriteValuePPC64_OpTrunc64to16(v)
	case OpTrunc64to32:
		return rewriteValuePPC64_OpTrunc64to32(v)
	case OpTrunc64to8:
		return rewriteValuePPC64_OpTrunc64to8(v)
	case OpWB:
		v.Op = OpPPC64LoweredWB
		return true
	case OpXor16:
		v.Op = OpPPC64XOR
		return true
	case OpXor32:
		v.Op = OpPPC64XOR
		return true
	case OpXor64:
		v.Op = OpPPC64XOR
		return true
	case OpXor8:
		v.Op = OpPPC64XOR
		return true
	case OpZero:
		return rewriteValuePPC64_OpZero(v)
	case OpZeroExt16to32:
		v.Op = OpPPC64MOVHZreg
		return true
	case OpZeroExt16to64:
		v.Op = OpPPC64MOVHZreg
		return true
	case OpZeroExt32to64:
		v.Op = OpPPC64MOVWZreg
		return true
	case OpZeroExt8to16:
		v.Op = OpPPC64MOVBZreg
		return true
	case OpZeroExt8to32:
		v.Op = OpPPC64MOVBZreg
		return true
	case OpZeroExt8to64:
		v.Op = OpPPC64MOVBZreg
		return true
	}
	return false
}
func rewriteValuePPC64_OpAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (MOVDaddr {sym} [0] base)
	for {
		sym := auxToSym(v.Aux)
		base := v_0
		v.reset(OpPPC64MOVDaddr)
		v.AuxInt = int32ToAuxInt(0)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValuePPC64_OpAtomicCompareAndSwap32(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicCompareAndSwap32 ptr old new_ mem)
	// result: (LoweredAtomicCas32 [1] ptr old new_ mem)
	for {
		ptr := v_0
		old := v_1
		new_ := v_2
		mem := v_3
		v.reset(OpPPC64LoweredAtomicCas32)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg4(ptr, old, new_, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicCompareAndSwap64(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicCompareAndSwap64 ptr old new_ mem)
	// result: (LoweredAtomicCas64 [1] ptr old new_ mem)
	for {
		ptr := v_0
		old := v_1
		new_ := v_2
		mem := v_3
		v.reset(OpPPC64LoweredAtomicCas64)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg4(ptr, old, new_, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicCompareAndSwapRel32(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicCompareAndSwapRel32 ptr old new_ mem)
	// result: (LoweredAtomicCas32 [0] ptr old new_ mem)
	for {
		ptr := v_0
		old := v_1
		new_ := v_2
		mem := v_3
		v.reset(OpPPC64LoweredAtomicCas32)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg4(ptr, old, new_, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicLoad32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoad32 ptr mem)
	// result: (LoweredAtomicLoad32 [1] ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpPPC64LoweredAtomicLoad32)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicLoad64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoad64 ptr mem)
	// result: (LoweredAtomicLoad64 [1] ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpPPC64LoweredAtomicLoad64)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicLoad8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoad8 ptr mem)
	// result: (LoweredAtomicLoad8 [1] ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpPPC64LoweredAtomicLoad8)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicLoadAcq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoadAcq32 ptr mem)
	// result: (LoweredAtomicLoad32 [0] ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpPPC64LoweredAtomicLoad32)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicLoadAcq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoadAcq64 ptr mem)
	// result: (LoweredAtomicLoad64 [0] ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpPPC64LoweredAtomicLoad64)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicLoadPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoadPtr ptr mem)
	// result: (LoweredAtomicLoadPtr [1] ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpPPC64LoweredAtomicLoadPtr)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicStore32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicStore32 ptr val mem)
	// result: (LoweredAtomicStore32 [1] ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpPPC64LoweredAtomicStore32)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicStore64(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicStore64 ptr val mem)
	// result: (LoweredAtomicStore64 [1] ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpPPC64LoweredAtomicStore64)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicStore8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicStore8 ptr val mem)
	// result: (LoweredAtomicStore8 [1] ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpPPC64LoweredAtomicStore8)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicStoreRel32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicStoreRel32 ptr val mem)
	// result: (LoweredAtomicStore32 [0] ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpPPC64LoweredAtomicStore32)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicStoreRel64(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicStoreRel64 ptr val mem)
	// result: (LoweredAtomicStore64 [0] ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpPPC64LoweredAtomicStore64)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValuePPC64_OpAvg64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Avg64u <t> x y)
	// result: (ADD (SRDconst <t> (SUB <t> x y) [1]) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ADD)
		v0 := b.NewValue0(v.Pos, OpPPC64SRDconst, t)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpPPC64SUB, t)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValuePPC64_OpBitLen32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen32 x)
	// result: (SUBFCconst [32] (CNTLZW <typ.Int> x))
	for {
		x := v_0
		v.reset(OpPPC64SUBFCconst)
		v.AuxInt = int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, OpPPC64CNTLZW, typ.Int)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpBitLen64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen64 x)
	// result: (SUBFCconst [64] (CNTLZD <typ.Int> x))
	for {
		x := v_0
		v.reset(OpPPC64SUBFCconst)
		v.AuxInt = int64ToAuxInt(64)
		v0 := b.NewValue0(v.Pos, OpPPC64CNTLZD, typ.Int)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpBswap16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Bswap16 x)
	// cond: buildcfg.GOPPC64>=10
	// result: (BRH x)
	for {
		x := v_0
		if !(buildcfg.GOPPC64 >= 10) {
			break
		}
		v.reset(OpPPC64BRH)
		v.AddArg(x)
		return true
	}
	// match: (Bswap16 x:(MOVHZload [off] {sym} ptr mem))
	// result: @x.Block (MOVHBRload (MOVDaddr <ptr.Type> [off] {sym} ptr) mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVHZload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpPPC64MOVHBRload, typ.UInt16)
		v.copyOf(v0)
		v1 := b.NewValue0(x.Pos, OpPPC64MOVDaddr, ptr.Type)
		v1.AuxInt = int32ToAuxInt(off)
		v1.Aux = symToAux(sym)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (Bswap16 x:(MOVHZloadidx ptr idx mem))
	// result: @x.Block (MOVHBRloadidx ptr idx mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVHZloadidx {
			break
		}
		mem := x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHBRloadidx, typ.Int16)
		v.copyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpBswap32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Bswap32 x)
	// cond: buildcfg.GOPPC64>=10
	// result: (BRW x)
	for {
		x := v_0
		if !(buildcfg.GOPPC64 >= 10) {
			break
		}
		v.reset(OpPPC64BRW)
		v.AddArg(x)
		return true
	}
	// match: (Bswap32 x:(MOVWZload [off] {sym} ptr mem))
	// result: @x.Block (MOVWBRload (MOVDaddr <ptr.Type> [off] {sym} ptr) mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVWZload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpPPC64MOVWBRload, typ.UInt32)
		v.copyOf(v0)
		v1 := b.NewValue0(x.Pos, OpPPC64MOVDaddr, ptr.Type)
		v1.AuxInt = int32ToAuxInt(off)
		v1.Aux = symToAux(sym)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (Bswap32 x:(MOVWZloadidx ptr idx mem))
	// result: @x.Block (MOVWBRloadidx ptr idx mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVWZloadidx {
			break
		}
		mem := x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpPPC64MOVWBRloadidx, typ.Int32)
		v.copyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpBswap64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Bswap64 x)
	// cond: buildcfg.GOPPC64>=10
	// result: (BRD x)
	for {
		x := v_0
		if !(buildcfg.GOPPC64 >= 10) {
			break
		}
		v.reset(OpPPC64BRD)
		v.AddArg(x)
		return true
	}
	// match: (Bswap64 x:(MOVDload [off] {sym} ptr mem))
	// result: @x.Block (MOVDBRload (MOVDaddr <ptr.Type> [off] {sym} ptr) mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVDload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpPPC64MOVDBRload, typ.UInt64)
		v.copyOf(v0)
		v1 := b.NewValue0(x.Pos, OpPPC64MOVDaddr, ptr.Type)
		v1.AuxInt = int32ToAuxInt(off)
		v1.Aux = symToAux(sym)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (Bswap64 x:(MOVDloadidx ptr idx mem))
	// result: @x.Block (MOVDBRloadidx ptr idx mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVDloadidx {
			break
		}
		mem := x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDBRloadidx, typ.Int64)
		v.copyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpCom16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Com16 x)
	// result: (NOR x x)
	for {
		x := v_0
		v.reset(OpPPC64NOR)
		v.AddArg2(x, x)
		return true
	}
}
func rewriteValuePPC64_OpCom32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Com32 x)
	// result: (NOR x x)
	for {
		x := v_0
		v.reset(OpPPC64NOR)
		v.AddArg2(x, x)
		return true
	}
}
func rewriteValuePPC64_OpCom64(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Com64 x)
	// result: (NOR x x)
	for {
		x := v_0
		v.reset(OpPPC64NOR)
		v.AddArg2(x, x)
		return true
	}
}
func rewriteValuePPC64_OpCom8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Com8 x)
	// result: (NOR x x)
	for {
		x := v_0
		v.reset(OpPPC64NOR)
		v.AddArg2(x, x)
		return true
	}
}
func rewriteValuePPC64_OpCondSelect(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (CondSelect x y (SETBC [a] cmp))
	// result: (ISEL [a] x y cmp)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64SETBC {
			break
		}
		a := auxIntToInt32(v_2.AuxInt)
		cmp := v_2.Args[0]
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(a)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (CondSelect x y (SETBCR [a] cmp))
	// result: (ISEL [a+4] x y cmp)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64SETBCR {
			break
		}
		a := auxIntToInt32(v_2.AuxInt)
		cmp := v_2.Args[0]
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(a + 4)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (CondSelect x y bool)
	// cond: flagArg(bool) == nil
	// result: (ISEL [6] x y (Select1 <types.TypeFlags> (ANDCCconst [1] bool)))
	for {
		x := v_0
		y := v_1
		bool := v_2
		if !(flagArg(bool) == nil) {
			break
		}
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AuxInt = int64ToAuxInt(1)
		v1.AddArg(bool)
		v0.AddArg(v1)
		v.AddArg3(x, y, v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpConst16(v *Value) bool {
	// match: (Const16 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := auxIntToInt16(v.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValuePPC64_OpConst32(v *Value) bool {
	// match: (Const32 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := auxIntToInt32(v.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValuePPC64_OpConst64(v *Value) bool {
	// match: (Const64 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := auxIntToInt64(v.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValuePPC64_OpConst8(v *Value) bool {
	// match: (Const8 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := auxIntToInt8(v.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValuePPC64_OpConstBool(v *Value) bool {
	// match: (ConstBool [t])
	// result: (MOVDconst [b2i(t)])
	for {
		t := auxIntToBool(v.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(t))
		return true
	}
}
func rewriteValuePPC64_OpConstNil(v *Value) bool {
	// match: (ConstNil)
	// result: (MOVDconst [0])
	for {
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
}
func rewriteValuePPC64_OpCopysign(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Copysign x y)
	// result: (FCPSGN y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64FCPSGN)
		v.AddArg2(y, x)
		return true
	}
}
func rewriteValuePPC64_OpCtz16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 x)
	// result: (POPCNTW (MOVHZreg (ANDN <typ.Int16> (ADDconst <typ.Int16> [-1] x) x)))
	for {
		x := v_0
		v.reset(OpPPC64POPCNTW)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDN, typ.Int16)
		v2 := b.NewValue0(v.Pos, OpPPC64ADDconst, typ.Int16)
		v2.AuxInt = int64ToAuxInt(-1)
		v2.AddArg(x)
		v1.AddArg2(v2, x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCtz32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz32 x)
	// cond: buildcfg.GOPPC64<=8
	// result: (POPCNTW (MOVWZreg (ANDN <typ.Int> (ADDconst <typ.Int> [-1] x) x)))
	for {
		x := v_0
		if !(buildcfg.GOPPC64 <= 8) {
			break
		}
		v.reset(OpPPC64POPCNTW)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVWZreg, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDN, typ.Int)
		v2 := b.NewValue0(v.Pos, OpPPC64ADDconst, typ.Int)
		v2.AuxInt = int64ToAuxInt(-1)
		v2.AddArg(x)
		v1.AddArg2(v2, x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Ctz32 x)
	// result: (CNTTZW (MOVWZreg x))
	for {
		x := v_0
		v.reset(OpPPC64CNTTZW)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVWZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCtz64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz64 x)
	// cond: buildcfg.GOPPC64<=8
	// result: (POPCNTD (ANDN <typ.Int64> (ADDconst <typ.Int64> [-1] x) x))
	for {
		x := v_0
		if !(buildcfg.GOPPC64 <= 8) {
			break
		}
		v.reset(OpPPC64POPCNTD)
		v0 := b.NewValue0(v.Pos, OpPPC64ANDN, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpPPC64ADDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(-1)
		v1.AddArg(x)
		v0.AddArg2(v1, x)
		v.AddArg(v0)
		return true
	}
	// match: (Ctz64 x)
	// result: (CNTTZD x)
	for {
		x := v_0
		v.reset(OpPPC64CNTTZD)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpCtz8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 x)
	// result: (POPCNTB (MOVBZreg (ANDN <typ.UInt8> (ADDconst <typ.UInt8> [-1] x) x)))
	for {
		x := v_0
		v.reset(OpPPC64POPCNTB)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDN, typ.UInt8)
		v2 := b.NewValue0(v.Pos, OpPPC64ADDconst, typ.UInt8)
		v2.AuxInt = int64ToAuxInt(-1)
		v2.AddArg(x)
		v1.AddArg2(v2, x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt32Fto32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32Fto32 x)
	// result: (MFVSRD (FCTIWZ x))
	for {
		x := v_0
		v.reset(OpPPC64MFVSRD)
		v0 := b.NewValue0(v.Pos, OpPPC64FCTIWZ, typ.Float64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt32Fto64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32Fto64 x)
	// result: (MFVSRD (FCTIDZ x))
	for {
		x := v_0
		v.reset(OpPPC64MFVSRD)
		v0 := b.NewValue0(v.Pos, OpPPC64FCTIDZ, typ.Float64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt32to32F(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32to32F x)
	// result: (FCFIDS (MTVSRD (SignExt32to64 x)))
	for {
		x := v_0
		v.reset(OpPPC64FCFIDS)
		v0 := b.NewValue0(v.Pos, OpPPC64MTVSRD, typ.Float64)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt32to64F(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32to64F x)
	// result: (FCFID (MTVSRD (SignExt32to64 x)))
	for {
		x := v_0
		v.reset(OpPPC64FCFID)
		v0 := b.NewValue0(v.Pos, OpPPC64MTVSRD, typ.Float64)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt64Fto32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt64Fto32 x)
	// result: (MFVSRD (FCTIWZ x))
	for {
		x := v_0
		v.reset(OpPPC64MFVSRD)
		v0 := b.NewValue0(v.Pos, OpPPC64FCTIWZ, typ.Float64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt64Fto64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt64Fto64 x)
	// result: (MFVSRD (FCTIDZ x))
	for {
		x := v_0
		v.reset(OpPPC64MFVSRD)
		v0 := b.NewValue0(v.Pos, OpPPC64FCTIDZ, typ.Float64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt64to32F(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt64to32F x)
	// result: (FCFIDS (MTVSRD x))
	for {
		x := v_0
		v.reset(OpPPC64FCFIDS)
		v0 := b.NewValue0(v.Pos, OpPPC64MTVSRD, typ.Float64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt64to64F(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt64to64F x)
	// result: (FCFID (MTVSRD x))
	for {
		x := v_0
		v.reset(OpPPC64FCFID)
		v0 := b.NewValue0(v.Pos, OpPPC64MTVSRD, typ.Float64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpDiv16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 [false] x y)
	// result: (DIVW (SignExt16to32 x) (SignExt16to32 y))
	for {
		if auxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.reset(OpPPC64DIVW)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValuePPC64_OpDiv16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (DIVWU (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64DIVWU)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpDiv32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div32 [false] x y)
	// result: (DIVW x y)
	for {
		if auxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.reset(OpPPC64DIVW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpDiv64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64 [false] x y)
	// result: (DIVD x y)
	for {
		if auxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.reset(OpPPC64DIVD)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpDiv8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (DIVW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64DIVW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpDiv8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (DIVWU (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64DIVWU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpEq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// cond: x.Type.IsSigned() && y.Type.IsSigned()
	// result: (Equal (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			y := v_1
			if !(x.Type.IsSigned() && y.Type.IsSigned()) {
				continue
			}
			v.reset(OpPPC64Equal)
			v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
			v2.AddArg(y)
			v0.AddArg2(v1, v2)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Eq16 x y)
	// result: (Equal (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32 x y)
	// result: (Equal (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32F x y)
	// result: (Equal (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64 x y)
	// result: (Equal (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64F x y)
	// result: (Equal (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// cond: x.Type.IsSigned() && y.Type.IsSigned()
	// result: (Equal (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			y := v_1
			if !(x.Type.IsSigned() && y.Type.IsSigned()) {
				continue
			}
			v.reset(OpPPC64Equal)
			v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
			v2.AddArg(y)
			v0.AddArg2(v1, v2)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Eq8 x y)
	// result: (Equal (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// result: (Select0 <typ.Int> (ANDCCconst [1] (EQV x y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpSelect0)
		v.Type = typ.Int
		v0 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpPPC64EQV, typ.Int64)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (EqPtr x y)
	// result: (Equal (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpIsInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsInBounds idx len)
	// result: (LessThan (CMPU idx len))
	for {
		idx := v_0
		len := v_1
		v.reset(OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPU, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpIsNonNil(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsNonNil ptr)
	// result: (NotEqual (CMPconst [0] ptr))
	for {
		ptr := v_0
		v.reset(OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(0)
		v0.AddArg(ptr)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpIsSliceInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsSliceInBounds idx len)
	// result: (LessEqual (CMPU idx len))
	for {
		idx := v_0
		len := v_1
		v.reset(OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPU, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (LessEqual (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// result: (LessEqual (CMPWU (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPWU, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32 x y)
	// result: (LessEqual (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32F x y)
	// result: (FLessEqual (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64FLessEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32U x y)
	// result: (LessEqual (CMPWU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPWU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64 x y)
	// result: (LessEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64F x y)
	// result: (FLessEqual (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64FLessEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64U x y)
	// result: (LessEqual (CMPU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (LessEqual (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// result: (LessEqual (CMPWU (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPWU, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (LessThan (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// result: (LessThan (CMPWU (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPWU, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32 x y)
	// result: (LessThan (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32F x y)
	// result: (FLessThan (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64FLessThan)
		v0 := b.NewValue0(v.Pos, OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32U x y)
	// result: (LessThan (CMPWU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPWU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64 x y)
	// result: (LessThan (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64F x y)
	// result: (FLessThan (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64FLessThan)
		v0 := b.NewValue0(v.Pos, OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64U x y)
	// result: (LessThan (CMPU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (LessThan (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// result: (LessThan (CMPWU (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPWU, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLoad(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Load <t> ptr mem)
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (MOVDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpPPC64MOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is32BitInt(t) && t.IsSigned()
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitInt(t) && t.IsSigned()) {
			break
		}
		v.reset(OpPPC64MOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is32BitInt(t) && !t.IsSigned()
	// result: (MOVWZload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitInt(t) && !t.IsSigned()) {
			break
		}
		v.reset(OpPPC64MOVWZload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is16BitInt(t) && t.IsSigned()
	// result: (MOVHload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is16BitInt(t) && t.IsSigned()) {
			break
		}
		v.reset(OpPPC64MOVHload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is16BitInt(t) && !t.IsSigned()
	// result: (MOVHZload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is16BitInt(t) && !t.IsSigned()) {
			break
		}
		v.reset(OpPPC64MOVHZload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsBoolean()
	// result: (MOVBZload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.IsBoolean()) {
			break
		}
		v.reset(OpPPC64MOVBZload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is8BitInt(t) && t.IsSigned()
	// result: (MOVBreg (MOVBZload ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is8BitInt(t) && t.IsSigned()) {
			break
		}
		v.reset(OpPPC64MOVBreg)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZload, typ.UInt8)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is8BitInt(t) && !t.IsSigned()
	// result: (MOVBZload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is8BitInt(t) && !t.IsSigned()) {
			break
		}
		v.reset(OpPPC64MOVBZload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is32BitFloat(t)
	// result: (FMOVSload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitFloat(t)) {
			break
		}
		v.reset(OpPPC64FMOVSload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is64BitFloat(t)
	// result: (FMOVDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is64BitFloat(t)) {
			break
		}
		v.reset(OpPPC64FMOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpLocalAddr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LocalAddr <t> {sym} base mem)
	// cond: t.Elem().HasPointers()
	// result: (MOVDaddr {sym} (SPanchored base mem))
	for {
		t := v.Type
		sym := auxToSym(v.Aux)
		base := v_0
		mem := v_1
		if !(t.Elem().HasPointers()) {
			break
		}
		v.reset(OpPPC64MOVDaddr)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpSPanchored, typ.Uintptr)
		v0.AddArg2(base, mem)
		v.AddArg(v0)
		return true
	}
	// match: (LocalAddr <t> {sym} base _)
	// cond: !t.Elem().HasPointers()
	// result: (MOVDaddr {sym} base)
	for {
		t := v.Type
		sym := auxToSym(v.Aux)
		base := v_0
		if !(!t.Elem().HasPointers()) {
			break
		}
		v.reset(OpPPC64MOVDaddr)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValuePPC64_OpLsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x16 <t> x y)
	// result: (ISEL [2] (SLD <t> (MOVHZreg x) y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0xFFF0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v4.AuxInt = int64ToAuxInt(0xFFF0)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x32 <t> x y)
	// result: (ISEL [0] (SLD <t> (MOVHZreg x) y) (MOVDconst [0]) (CMPWUconst y [16]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(16)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x64 x (MOVDconst [c]))
	// cond: uint64(c) < 16
	// result: (SLWconst x [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.reset(OpPPC64SLWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Lsh16x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x64 <t> x y)
	// result: (ISEL [0] (SLD <t> (MOVHZreg x) y) (MOVDconst [0]) (CMPUconst y [16]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(16)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x8 <t> x y)
	// result: (ISEL [2] (SLD <t> (MOVHZreg x) y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0x00F0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v4.AuxInt = int64ToAuxInt(0x00F0)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SLW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x16 <t> x y)
	// result: (ISEL [2] (SLW <t> x y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0xFFE0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SLW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(0xFFE0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SLW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x32 <t> x y)
	// result: (ISEL [0] (SLW <t> x y) (MOVDconst [0]) (CMPWUconst y [32]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SLW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x64 x (MOVDconst [c]))
	// cond: uint64(c) < 32
	// result: (SLWconst x [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.reset(OpPPC64SLWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Lsh32x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SLW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x64 <t> x y)
	// result: (ISEL [0] (SLW <t> x y) (MOVDconst [0]) (CMPUconst y [32]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SLW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SLW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x8 <t> x y)
	// result: (ISEL [2] (SLW <t> x y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0x00E0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SLW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(0x00E0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x16 <t> x y)
	// result: (ISEL [2] (SLD <t> x y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0xFFC0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SLD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(0xFFC0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x32 <t> x y)
	// result: (ISEL [0] (SLD <t> x y) (MOVDconst [0]) (CMPWUconst y [64]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SLD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x64 x (MOVDconst [c]))
	// cond: uint64(c) < 64
	// result: (SLDconst x [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.reset(OpPPC64SLDconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Lsh64x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x64 <t> x y)
	// result: (ISEL [0] (SLD <t> x y) (MOVDconst [0]) (CMPUconst y [64]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SLD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x8 <t> x y)
	// result: (ISEL [2] (SLD <t> x y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0x00C0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SLD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(0x00C0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x16 <t> x y)
	// result: (ISEL [2] (SLD <t> (MOVBZreg x) y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0xFFF8] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v4.AuxInt = int64ToAuxInt(0xFFF8)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x32 <t> x y)
	// result: (ISEL [0] (SLD <t> (MOVBZreg x) y) (MOVDconst [0]) (CMPWUconst y [8]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(8)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x64 x (MOVDconst [c]))
	// cond: uint64(c) < 8
	// result: (SLWconst x [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.reset(OpPPC64SLWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Lsh8x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x64 <t> x y)
	// result: (ISEL [0] (SLD <t> (MOVBZreg x) y) (MOVDconst [0]) (CMPUconst y [8]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(8)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x8 <t> x y)
	// result: (ISEL [2] (SLD <t> (MOVBZreg x) y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0x00F8] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v4.AuxInt = int64ToAuxInt(0x00F8)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpMax32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Max32F x y)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (XSMAXJDP x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64XSMAXJDP)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMax64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Max64F x y)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (XSMAXJDP x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64XSMAXJDP)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMin32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Min32F x y)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (XSMINJDP x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64XSMINJDP)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMin64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Min64F x y)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (XSMINJDP x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64XSMINJDP)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMod16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// result: (Mod32 (SignExt16to32 x) (SignExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpMod32)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpMod16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (Mod32u (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpMod32u)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpMod32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32 x y)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (MODSW x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64MODSW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Mod32 x y)
	// cond: buildcfg.GOPPC64 <= 8
	// result: (SUB x (MULLW y (DIVW x y)))
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOPPC64 <= 8) {
			break
		}
		v.reset(OpPPC64SUB)
		v0 := b.NewValue0(v.Pos, OpPPC64MULLW, typ.Int32)
		v1 := b.NewValue0(v.Pos, OpPPC64DIVW, typ.Int32)
		v1.AddArg2(x, y)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMod32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32u x y)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (MODUW x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64MODUW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Mod32u x y)
	// cond: buildcfg.GOPPC64 <= 8
	// result: (SUB x (MULLW y (DIVWU x y)))
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOPPC64 <= 8) {
			break
		}
		v.reset(OpPPC64SUB)
		v0 := b.NewValue0(v.Pos, OpPPC64MULLW, typ.Int32)
		v1 := b.NewValue0(v.Pos, OpPPC64DIVWU, typ.Int32)
		v1.AddArg2(x, y)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMod64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod64 x y)
	// cond: buildcfg.GOPPC64 >=9
	// result: (MODSD x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64MODSD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Mod64 x y)
	// cond: buildcfg.GOPPC64 <=8
	// result: (SUB x (MULLD y (DIVD x y)))
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOPPC64 <= 8) {
			break
		}
		v.reset(OpPPC64SUB)
		v0 := b.NewValue0(v.Pos, OpPPC64MULLD, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpPPC64DIVD, typ.Int64)
		v1.AddArg2(x, y)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMod64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod64u x y)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (MODUD x y)
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64MODUD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Mod64u x y)
	// cond: buildcfg.GOPPC64 <= 8
	// result: (SUB x (MULLD y (DIVDU x y)))
	for {
		x := v_0
		y := v_1
		if !(buildcfg.GOPPC64 <= 8) {
			break
		}
		v.reset(OpPPC64SUB)
		v0 := b.NewValue0(v.Pos, OpPPC64MULLD, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpPPC64DIVDU, typ.Int64)
		v1.AddArg2(x, y)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMod8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (Mod32 (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpMod32)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpMod8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (Mod32u (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpMod32u)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpMove(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Move [0] _ _ mem)
	// result: mem
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.copyOf(mem)
		return true
	}
	// match: (Move [1] dst src mem)
	// result: (MOVBstore dst (MOVBZload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpPPC64MOVBstore)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZload, typ.UInt8)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (MOVHstore dst (MOVHZload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpPPC64MOVHstore)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHZload, typ.UInt16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (MOVWstore dst (MOVWZload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpPPC64MOVWstore)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVWZload, typ.UInt32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [8] {t} dst src mem)
	// result: (MOVDstore dst (MOVDload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpPPC64MOVDstore)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDload, typ.Int64)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [3] dst src mem)
	// result: (MOVBstore [2] dst (MOVBZload [2] src mem) (MOVHstore dst (MOVHload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVHload, typ.Int16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [5] dst src mem)
	// result: (MOVBstore [4] dst (MOVBZload [4] src mem) (MOVWstore dst (MOVWZload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 5 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVWZload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [6] dst src mem)
	// result: (MOVHstore [4] dst (MOVHZload [4] src mem) (MOVWstore dst (MOVWZload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 6 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpPPC64MOVHstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHZload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVWZload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [7] dst src mem)
	// result: (MOVBstore [6] dst (MOVBZload [6] src mem) (MOVHstore [4] dst (MOVHZload [4] src mem) (MOVWstore dst (MOVWZload src mem) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 7 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(6)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVHZload, typ.UInt16)
		v2.AuxInt = int32ToAuxInt(4)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, OpPPC64MOVWstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, OpPPC64MOVWZload, typ.UInt32)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 8 && buildcfg.GOPPC64 <= 8 && logLargeCopy(v, s)
	// result: (LoweredMove [s] dst src mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 8 && buildcfg.GOPPC64 <= 8 && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpPPC64LoweredMove)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 8 && s <= 64 && buildcfg.GOPPC64 >= 9
	// result: (LoweredQuadMoveShort [s] dst src mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 8 && s <= 64 && buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64LoweredQuadMoveShort)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 8 && buildcfg.GOPPC64 >= 9 && logLargeCopy(v, s)
	// result: (LoweredQuadMove [s] dst src mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 8 && buildcfg.GOPPC64 >= 9 && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpPPC64LoweredQuadMove)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpNeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// cond: x.Type.IsSigned() && y.Type.IsSigned()
	// result: (NotEqual (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			y := v_1
			if !(x.Type.IsSigned() && y.Type.IsSigned()) {
				continue
			}
			v.reset(OpPPC64NotEqual)
			v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
			v2.AddArg(y)
			v0.AddArg2(v1, v2)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Neq16 x y)
	// result: (NotEqual (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32 x y)
	// result: (NotEqual (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32F x y)
	// result: (NotEqual (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64 x y)
	// result: (NotEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64F x y)
	// result: (NotEqual (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// cond: x.Type.IsSigned() && y.Type.IsSigned()
	// result: (NotEqual (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			y := v_1
			if !(x.Type.IsSigned() && y.Type.IsSigned()) {
				continue
			}
			v.reset(OpPPC64NotEqual)
			v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
			v2.AddArg(y)
			v0.AddArg2(v1, v2)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (Neq8 x y)
	// result: (NotEqual (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNeqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (NeqPtr x y)
	// result: (NotEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNot(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Not x)
	// result: (XORconst [1] x)
	for {
		x := v_0
		v.reset(OpPPC64XORconst)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpOffPtr(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OffPtr [off] ptr)
	// result: (ADD (MOVDconst <typ.Int64> [off]) ptr)
	for {
		off := auxIntToInt64(v.AuxInt)
		ptr := v_0
		v.reset(OpPPC64ADD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = int64ToAuxInt(off)
		v.AddArg2(v0, ptr)
		return true
	}
}
func rewriteValuePPC64_OpPPC64ADD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADD l:(MULLD x y) z)
	// cond: buildcfg.GOPPC64 >= 9 && l.Uses == 1 && clobber(l)
	// result: (MADDLD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			l := v_0
			if l.Op != OpPPC64MULLD {
				continue
			}
			y := l.Args[1]
			x := l.Args[0]
			z := v_1
			if !(buildcfg.GOPPC64 >= 9 && l.Uses == 1 && clobber(l)) {
				continue
			}
			v.reset(OpPPC64MADDLD)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (ADD x (MOVDconst <t> [c]))
	// cond: is32Bit(c) && !t.IsPtr()
	// result: (ADDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpPPC64MOVDconst {
				continue
			}
			t := v_1.Type
			c := auxIntToInt64(v_1.AuxInt)
			if !(is32Bit(c) && !t.IsPtr()) {
				continue
			}
			v.reset(OpPPC64ADDconst)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64ADDE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDE x y (Select1 <typ.UInt64> (ADDCconst (MOVDconst [0]) [-1])))
	// result: (ADDC x y)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpSelect1 || v_2.Type != typ.UInt64 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpPPC64ADDCconst || auxIntToInt64(v_2_0.AuxInt) != -1 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpPPC64MOVDconst || auxIntToInt64(v_2_0_0.AuxInt) != 0 {
			break
		}
		v.reset(OpPPC64ADDC)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDE (MOVDconst [0]) y c)
	// result: (ADDZE y c)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst || auxIntToInt64(v_0.AuxInt) != 0 {
				continue
			}
			y := v_1
			c := v_2
			v.reset(OpPPC64ADDZE)
			v.AddArg2(y, c)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64ADDconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ADDconst [c] (ADDconst [d] x))
	// cond: is32Bit(c+d)
	// result: (ADDconst [c+d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpPPC64ADDconst)
		v.AuxInt = int64ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [0] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (ADDconst [c] (MOVDaddr [d] {sym} x))
	// cond: is32Bit(c+int64(d))
	// result: (MOVDaddr [int32(c+int64(d))] {sym} x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64MOVDaddr {
			break
		}
		d := auxIntToInt32(v_0.AuxInt)
		sym := auxToSym(v_0.Aux)
		x := v_0.Args[0]
		if !(is32Bit(c + int64(d))) {
			break
		}
		v.reset(OpPPC64MOVDaddr)
		v.AuxInt = int32ToAuxInt(int32(c + int64(d)))
		v.Aux = symToAux(sym)
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] x:(SP))
	// cond: is32Bit(c)
	// result: (MOVDaddr [int32(c)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		x := v_0
		if x.Op != OpSP || !(is32Bit(c)) {
			break
		}
		v.reset(OpPPC64MOVDaddr)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (SUBFCconst [d] x))
	// cond: is32Bit(c+d)
	// result: (SUBFCconst [c+d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64SUBFCconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpPPC64SUBFCconst)
		v.AuxInt = int64ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64AND(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AND (MOVDconst [m]) (ROTLWconst [r] x))
	// cond: isPPC64WordRotateMask(m)
	// result: (RLWINM [encodePPC64RotateMask(r,m,32)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst {
				continue
			}
			m := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpPPC64ROTLWconst {
				continue
			}
			r := auxIntToInt64(v_1.AuxInt)
			x := v_1.Args[0]
			if !(isPPC64WordRotateMask(m)) {
				continue
			}
			v.reset(OpPPC64RLWINM)
			v.AuxInt = int64ToAuxInt(encodePPC64RotateMask(r, m, 32))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [m]) (ROTLW x r))
	// cond: isPPC64WordRotateMask(m)
	// result: (RLWNM [encodePPC64RotateMask(0,m,32)] x r)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst {
				continue
			}
			m := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpPPC64ROTLW {
				continue
			}
			r := v_1.Args[1]
			x := v_1.Args[0]
			if !(isPPC64WordRotateMask(m)) {
				continue
			}
			v.reset(OpPPC64RLWNM)
			v.AuxInt = int64ToAuxInt(encodePPC64RotateMask(0, m, 32))
			v.AddArg2(x, r)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [m]) (SRWconst x [s]))
	// cond: mergePPC64RShiftMask(m,s,32) == 0
	// result: (MOVDconst [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst {
				continue
			}
			m := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpPPC64SRWconst {
				continue
			}
			s := auxIntToInt64(v_1.AuxInt)
			if !(mergePPC64RShiftMask(m, s, 32) == 0) {
				continue
			}
			v.reset(OpPPC64MOVDconst)
			v.AuxInt = int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [m]) (SRWconst x [s]))
	// cond: mergePPC64AndSrwi(m,s) != 0
	// result: (RLWINM [mergePPC64AndSrwi(m,s)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst {
				continue
			}
			m := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpPPC64SRWconst {
				continue
			}
			s := auxIntToInt64(v_1.AuxInt)
			x := v_1.Args[0]
			if !(mergePPC64AndSrwi(m, s) != 0) {
				continue
			}
			v.reset(OpPPC64RLWINM)
			v.AuxInt = int64ToAuxInt(mergePPC64AndSrwi(m, s))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND x (NOR y y))
	// result: (ANDN x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpPPC64NOR {
				continue
			}
			y := v_1.Args[1]
			if y != v_1.Args[0] {
				continue
			}
			v.reset(OpPPC64ANDN)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [c&d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpPPC64MOVDconst {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpPPC64MOVDconst)
			v.AuxInt = int64ToAuxInt(c & d)
			return true
		}
		break
	}
	// match: (AND x (MOVDconst [-1]))
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpPPC64MOVDconst || auxIntToInt64(v_1.AuxInt) != -1 {
				continue
			}
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (AND x (MOVDconst [c]))
	// cond: isU16Bit(c)
	// result: (Select0 (ANDCCconst [c] x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isU16Bit(c)) {
				continue
			}
			v.reset(OpSelect0)
			v0 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [c]) y:(MOVWZreg _))
	// cond: c&0xFFFFFFFF == 0xFFFFFFFF
	// result: y
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			y := v_1
			if y.Op != OpPPC64MOVWZreg || !(c&0xFFFFFFFF == 0xFFFFFFFF) {
				continue
			}
			v.copyOf(y)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [0xFFFFFFFF]) y:(MOVWreg x))
	// result: (MOVWZreg x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst || auxIntToInt64(v_0.AuxInt) != 0xFFFFFFFF {
				continue
			}
			y := v_1
			if y.Op != OpPPC64MOVWreg {
				continue
			}
			x := y.Args[0]
			v.reset(OpPPC64MOVWZreg)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [c]) x:(MOVBZload _ _))
	// result: (Select0 (ANDCCconst [c&0xFF] x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			x := v_1
			if x.Op != OpPPC64MOVBZload {
				continue
			}
			v.reset(OpSelect0)
			v0 := b.NewValue0(x.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
			v0.AuxInt = int64ToAuxInt(c & 0xFF)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64ANDCCconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ANDCCconst [c] (Select0 (ANDCCconst [d] x)))
	// result: (ANDCCconst [c&d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpSelect0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64ANDCCconst {
			break
		}
		d := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		v.reset(OpPPC64ANDCCconst)
		v.AuxInt = int64ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64ANDN(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ANDN (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [c&^d])
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(c &^ d)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64BRD(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BRD x:(MOVDload [off] {sym} ptr mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVDBRload (MOVDaddr <ptr.Type> [off] {sym} ptr) mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVDload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpPPC64MOVDBRload, typ.UInt64)
		v.copyOf(v0)
		v1 := b.NewValue0(x.Pos, OpPPC64MOVDaddr, ptr.Type)
		v1.AuxInt = int32ToAuxInt(off)
		v1.Aux = symToAux(sym)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (BRD x:(MOVDloadidx ptr idx mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVDBRloadidx ptr idx mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVDloadidx {
			break
		}
		mem := x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDBRloadidx, typ.Int64)
		v.copyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64BRH(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BRH x:(MOVHZload [off] {sym} ptr mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVHBRload (MOVDaddr <ptr.Type> [off] {sym} ptr) mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVHZload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpPPC64MOVHBRload, typ.UInt16)
		v.copyOf(v0)
		v1 := b.NewValue0(x.Pos, OpPPC64MOVDaddr, ptr.Type)
		v1.AuxInt = int32ToAuxInt(off)
		v1.Aux = symToAux(sym)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (BRH x:(MOVHZloadidx ptr idx mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVHBRloadidx ptr idx mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVHZloadidx {
			break
		}
		mem := x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHBRloadidx, typ.Int16)
		v.copyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64BRW(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BRW x:(MOVWZload [off] {sym} ptr mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVWBRload (MOVDaddr <ptr.Type> [off] {sym} ptr) mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVWZload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpPPC64MOVWBRload, typ.UInt32)
		v.copyOf(v0)
		v1 := b.NewValue0(x.Pos, OpPPC64MOVDaddr, ptr.Type)
		v1.AuxInt = int32ToAuxInt(off)
		v1.Aux = symToAux(sym)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (BRW x:(MOVWZloadidx ptr idx mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVWBRloadidx ptr idx mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVWZloadidx {
			break
		}
		mem := x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpPPC64MOVWBRloadidx, typ.Int32)
		v.copyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CLRLSLDI(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CLRLSLDI [c] (SRWconst [s] x))
	// cond: mergePPC64ClrlsldiSrw(int64(c),s) != 0
	// result: (RLWINM [mergePPC64ClrlsldiSrw(int64(c),s)] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		s := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(mergePPC64ClrlsldiSrw(int64(c), s) != 0) {
			break
		}
		v.reset(OpPPC64RLWINM)
		v.AuxInt = int64ToAuxInt(mergePPC64ClrlsldiSrw(int64(c), s))
		v.AddArg(x)
		return true
	}
	// match: (CLRLSLDI [c] i:(RLWINM [s] x))
	// cond: mergePPC64ClrlsldiRlwinm(c,s) != 0
	// result: (RLWINM [mergePPC64ClrlsldiRlwinm(c,s)] x)
	for {
		c := auxIntToInt32(v.AuxInt)
		i := v_0
		if i.Op != OpPPC64RLWINM {
			break
		}
		s := auxIntToInt64(i.AuxInt)
		x := i.Args[0]
		if !(mergePPC64ClrlsldiRlwinm(c, s) != 0) {
			break
		}
		v.reset(OpPPC64RLWINM)
		v.AuxInt = int64ToAuxInt(mergePPC64ClrlsldiRlwinm(c, s))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMP(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMP x (MOVDconst [c]))
	// cond: is16Bit(c)
	// result: (CMPconst x [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(is16Bit(c)) {
			break
		}
		v.reset(OpPPC64CMPconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMP (MOVDconst [c]) y)
	// cond: is16Bit(c)
	// result: (InvertFlags (CMPconst y [c]))
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(is16Bit(c)) {
			break
		}
		v.reset(OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x y)
	// cond: canonLessThan(x,y)
	// result: (InvertFlags (CMP y x))
	for {
		x := v_0
		y := v_1
		if !(canonLessThan(x, y)) {
			break
		}
		v.reset(OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPU(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPU x (MOVDconst [c]))
	// cond: isU16Bit(c)
	// result: (CMPUconst x [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isU16Bit(c)) {
			break
		}
		v.reset(OpPPC64CMPUconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMPU (MOVDconst [c]) y)
	// cond: isU16Bit(c)
	// result: (InvertFlags (CMPUconst y [c]))
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(isU16Bit(c)) {
			break
		}
		v.reset(OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMPU x y)
	// cond: canonLessThan(x,y)
	// result: (InvertFlags (CMPU y x))
	for {
		x := v_0
		y := v_1
		if !(canonLessThan(x, y)) {
			break
		}
		v.reset(OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPU, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPUconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CMPUconst [d] (Select0 (ANDCCconst z [c])))
	// cond: uint64(d) > uint64(c)
	// result: (FlagLT)
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpSelect0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(v_0_0.AuxInt)
		if !(uint64(d) > uint64(c)) {
			break
		}
		v.reset(OpPPC64FlagLT)
		return true
	}
	// match: (CMPUconst (MOVDconst [x]) [y])
	// cond: x==y
	// result: (FlagEQ)
	for {
		y := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(x == y) {
			break
		}
		v.reset(OpPPC64FlagEQ)
		return true
	}
	// match: (CMPUconst (MOVDconst [x]) [y])
	// cond: uint64(x)<uint64(y)
	// result: (FlagLT)
	for {
		y := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(uint64(x) < uint64(y)) {
			break
		}
		v.reset(OpPPC64FlagLT)
		return true
	}
	// match: (CMPUconst (MOVDconst [x]) [y])
	// cond: uint64(x)>uint64(y)
	// result: (FlagGT)
	for {
		y := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(uint64(x) > uint64(y)) {
			break
		}
		v.reset(OpPPC64FlagGT)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPW x (MOVWreg y))
	// result: (CMPW x y)
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVWreg {
			break
		}
		y := v_1.Args[0]
		v.reset(OpPPC64CMPW)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPW (MOVWreg x) y)
	// result: (CMPW x y)
	for {
		if v_0.Op != OpPPC64MOVWreg {
			break
		}
		x := v_0.Args[0]
		y := v_1
		v.reset(OpPPC64CMPW)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPW x (MOVDconst [c]))
	// cond: is16Bit(c)
	// result: (CMPWconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(is16Bit(c)) {
			break
		}
		v.reset(OpPPC64CMPWconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPW (MOVDconst [c]) y)
	// cond: is16Bit(c)
	// result: (InvertFlags (CMPWconst y [int32(c)]))
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(is16Bit(c)) {
			break
		}
		v.reset(OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMPW x y)
	// cond: canonLessThan(x,y)
	// result: (InvertFlags (CMPW y x))
	for {
		x := v_0
		y := v_1
		if !(canonLessThan(x, y)) {
			break
		}
		v.reset(OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPW, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPWU(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPWU x (MOVWZreg y))
	// result: (CMPWU x y)
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVWZreg {
			break
		}
		y := v_1.Args[0]
		v.reset(OpPPC64CMPWU)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPWU (MOVWZreg x) y)
	// result: (CMPWU x y)
	for {
		if v_0.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_0.Args[0]
		y := v_1
		v.reset(OpPPC64CMPWU)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPWU x (MOVDconst [c]))
	// cond: isU16Bit(c)
	// result: (CMPWUconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isU16Bit(c)) {
			break
		}
		v.reset(OpPPC64CMPWUconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPWU (MOVDconst [c]) y)
	// cond: isU16Bit(c)
	// result: (InvertFlags (CMPWUconst y [int32(c)]))
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(isU16Bit(c)) {
			break
		}
		v.reset(OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMPWU x y)
	// cond: canonLessThan(x,y)
	// result: (InvertFlags (CMPWU y x))
	for {
		x := v_0
		y := v_1
		if !(canonLessThan(x, y)) {
			break
		}
		v.reset(OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpPPC64CMPWU, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPWUconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CMPWUconst [d] (Select0 (ANDCCconst z [c])))
	// cond: uint64(d) > uint64(c)
	// result: (FlagLT)
	for {
		d := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpSelect0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(v_0_0.AuxInt)
		if !(uint64(d) > uint64(c)) {
			break
		}
		v.reset(OpPPC64FlagLT)
		return true
	}
	// match: (CMPWUconst (MOVDconst [x]) [y])
	// cond: int32(x)==int32(y)
	// result: (FlagEQ)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(int32(x) == int32(y)) {
			break
		}
		v.reset(OpPPC64FlagEQ)
		return true
	}
	// match: (CMPWUconst (MOVDconst [x]) [y])
	// cond: uint32(x)<uint32(y)
	// result: (FlagLT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(uint32(x) < uint32(y)) {
			break
		}
		v.reset(OpPPC64FlagLT)
		return true
	}
	// match: (CMPWUconst (MOVDconst [x]) [y])
	// cond: uint32(x)>uint32(y)
	// result: (FlagGT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(uint32(x) > uint32(y)) {
			break
		}
		v.reset(OpPPC64FlagGT)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CMPWconst (MOVDconst [x]) [y])
	// cond: int32(x)==int32(y)
	// result: (FlagEQ)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(int32(x) == int32(y)) {
			break
		}
		v.reset(OpPPC64FlagEQ)
		return true
	}
	// match: (CMPWconst (MOVDconst [x]) [y])
	// cond: int32(x)<int32(y)
	// result: (FlagLT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(int32(x) < int32(y)) {
			break
		}
		v.reset(OpPPC64FlagLT)
		return true
	}
	// match: (CMPWconst (MOVDconst [x]) [y])
	// cond: int32(x)>int32(y)
	// result: (FlagGT)
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(int32(x) > int32(y)) {
			break
		}
		v.reset(OpPPC64FlagGT)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: x==y
	// result: (FlagEQ)
	for {
		y := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(x == y) {
			break
		}
		v.reset(OpPPC64FlagEQ)
		return true
	}
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: x<y
	// result: (FlagLT)
	for {
		y := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(x < y) {
			break
		}
		v.reset(OpPPC64FlagLT)
		return true
	}
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: x>y
	// result: (FlagGT)
	for {
		y := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		if !(x > y) {
			break
		}
		v.reset(OpPPC64FlagGT)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64Equal(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Equal (FlagEQ))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != OpPPC64FlagEQ {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (Equal (FlagLT))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpPPC64FlagLT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Equal (FlagGT))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpPPC64FlagGT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Equal (InvertFlags x))
	// result: (Equal x)
	for {
		if v_0.Op != OpPPC64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpPPC64Equal)
		v.AddArg(x)
		return true
	}
	// match: (Equal cmp)
	// result: (SETBC [2] cmp)
	for {
		cmp := v_0
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(2)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64FABS(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FABS (FMOVDconst [x]))
	// result: (FMOVDconst [math.Abs(x)])
	for {
		if v_0.Op != OpPPC64FMOVDconst {
			break
		}
		x := auxIntToFloat64(v_0.AuxInt)
		v.reset(OpPPC64FMOVDconst)
		v.AuxInt = float64ToAuxInt(math.Abs(x))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FADD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FADD (FMUL x y) z)
	// cond: x.Block.Func.useFMA(v)
	// result: (FMADD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64FMUL {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				z := v_1
				if !(x.Block.Func.useFMA(v)) {
					continue
				}
				v.reset(OpPPC64FMADD)
				v.AddArg3(x, y, z)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64FADDS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FADDS (FMULS x y) z)
	// cond: x.Block.Func.useFMA(v)
	// result: (FMADDS x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64FMULS {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				z := v_1
				if !(x.Block.Func.useFMA(v)) {
					continue
				}
				v.reset(OpPPC64FMADDS)
				v.AddArg3(x, y, z)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64FCEIL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FCEIL (FMOVDconst [x]))
	// result: (FMOVDconst [math.Ceil(x)])
	for {
		if v_0.Op != OpPPC64FMOVDconst {
			break
		}
		x := auxIntToFloat64(v_0.AuxInt)
		v.reset(OpPPC64FMOVDconst)
		v.AuxInt = float64ToAuxInt(math.Ceil(x))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FFLOOR(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FFLOOR (FMOVDconst [x]))
	// result: (FMOVDconst [math.Floor(x)])
	for {
		if v_0.Op != OpPPC64FMOVDconst {
			break
		}
		x := auxIntToFloat64(v_0.AuxInt)
		v.reset(OpPPC64FMOVDconst)
		v.AuxInt = float64ToAuxInt(math.Floor(x))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FGreaterEqual(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (FGreaterEqual cmp)
	// result: (OR (SETBC [2] cmp) (SETBC [1] cmp))
	for {
		cmp := v_0
		v.reset(OpPPC64OR)
		v0 := b.NewValue0(v.Pos, OpPPC64SETBC, typ.Int32)
		v0.AuxInt = int32ToAuxInt(2)
		v0.AddArg(cmp)
		v1 := b.NewValue0(v.Pos, OpPPC64SETBC, typ.Int32)
		v1.AuxInt = int32ToAuxInt(1)
		v1.AddArg(cmp)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpPPC64FGreaterThan(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FGreaterThan cmp)
	// result: (SETBC [1] cmp)
	for {
		cmp := v_0
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(1)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64FLessEqual(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (FLessEqual cmp)
	// result: (OR (SETBC [2] cmp) (SETBC [0] cmp))
	for {
		cmp := v_0
		v.reset(OpPPC64OR)
		v0 := b.NewValue0(v.Pos, OpPPC64SETBC, typ.Int32)
		v0.AuxInt = int32ToAuxInt(2)
		v0.AddArg(cmp)
		v1 := b.NewValue0(v.Pos, OpPPC64SETBC, typ.Int32)
		v1.AuxInt = int32ToAuxInt(0)
		v1.AddArg(cmp)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpPPC64FLessThan(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FLessThan cmp)
	// result: (SETBC [0] cmp)
	for {
		cmp := v_0
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(0)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64FMOVDload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVDload [off] {sym} ptr (MOVDstore [off] {sym} ptr x _))
	// result: (MTVSRD x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVDstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpPPC64MTVSRD)
		v.AddArg(x)
		return true
	}
	// match: (FMOVDload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (FMOVDload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64FMOVDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVDload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (FMOVDload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64FMOVDload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FMOVDstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVDstore [off] {sym} ptr (MTVSRD x) mem)
	// result: (MOVDstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MTVSRD {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVDstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (FMOVDstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (FMOVDstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64FMOVDstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVDstore [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (FMOVDstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64FMOVDstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FMOVSload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVSload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (FMOVSload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64FMOVSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVSload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (FMOVSload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64FMOVSload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FMOVSstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVSstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (FMOVSstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64FMOVSstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVSstore [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (FMOVSstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64FMOVSstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FNEG(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FNEG (FABS x))
	// result: (FNABS x)
	for {
		if v_0.Op != OpPPC64FABS {
			break
		}
		x := v_0.Args[0]
		v.reset(OpPPC64FNABS)
		v.AddArg(x)
		return true
	}
	// match: (FNEG (FNABS x))
	// result: (FABS x)
	for {
		if v_0.Op != OpPPC64FNABS {
			break
		}
		x := v_0.Args[0]
		v.reset(OpPPC64FABS)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FSQRT(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FSQRT (FMOVDconst [x]))
	// cond: x >= 0
	// result: (FMOVDconst [math.Sqrt(x)])
	for {
		if v_0.Op != OpPPC64FMOVDconst {
			break
		}
		x := auxIntToFloat64(v_0.AuxInt)
		if !(x >= 0) {
			break
		}
		v.reset(OpPPC64FMOVDconst)
		v.AuxInt = float64ToAuxInt(math.Sqrt(x))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FSUB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FSUB (FMUL x y) z)
	// cond: x.Block.Func.useFMA(v)
	// result: (FMSUB x y z)
	for {
		if v_0.Op != OpPPC64FMUL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			z := v_1
			if !(x.Block.Func.useFMA(v)) {
				continue
			}
			v.reset(OpPPC64FMSUB)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64FSUBS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FSUBS (FMULS x y) z)
	// cond: x.Block.Func.useFMA(v)
	// result: (FMSUBS x y z)
	for {
		if v_0.Op != OpPPC64FMULS {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			z := v_1
			if !(x.Block.Func.useFMA(v)) {
				continue
			}
			v.reset(OpPPC64FMSUBS)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64FTRUNC(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FTRUNC (FMOVDconst [x]))
	// result: (FMOVDconst [math.Trunc(x)])
	for {
		if v_0.Op != OpPPC64FMOVDconst {
			break
		}
		x := auxIntToFloat64(v_0.AuxInt)
		v.reset(OpPPC64FMOVDconst)
		v.AuxInt = float64ToAuxInt(math.Trunc(x))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64GreaterEqual(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterEqual (FlagEQ))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != OpPPC64FlagEQ {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (GreaterEqual (FlagLT))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpPPC64FlagLT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (GreaterEqual (FlagGT))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != OpPPC64FlagGT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (GreaterEqual (InvertFlags x))
	// result: (LessEqual x)
	for {
		if v_0.Op != OpPPC64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpPPC64LessEqual)
		v.AddArg(x)
		return true
	}
	// match: (GreaterEqual cmp)
	// result: (SETBCR [0] cmp)
	for {
		cmp := v_0
		v.reset(OpPPC64SETBCR)
		v.AuxInt = int32ToAuxInt(0)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64GreaterThan(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterThan (FlagEQ))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpPPC64FlagEQ {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (GreaterThan (FlagLT))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpPPC64FlagLT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (GreaterThan (FlagGT))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != OpPPC64FlagGT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (GreaterThan (InvertFlags x))
	// result: (LessThan x)
	for {
		if v_0.Op != OpPPC64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpPPC64LessThan)
		v.AddArg(x)
		return true
	}
	// match: (GreaterThan cmp)
	// result: (SETBC [1] cmp)
	for {
		cmp := v_0
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(1)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64ISEL(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ISEL [6] x y (Select1 (ANDCCconst [1] (SETBC [c] cmp))))
	// result: (ISEL [c] x y cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != OpSelect1 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_2_0.AuxInt) != 1 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpPPC64SETBC {
			break
		}
		c := auxIntToInt32(v_2_0_0.AuxInt)
		cmp := v_2_0_0.Args[0]
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (ISEL [6] x y (CMPconst [0] (SETBC [c] cmp)))
	// result: (ISEL [c] x y cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64CMPconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpPPC64SETBC {
			break
		}
		c := auxIntToInt32(v_2_0.AuxInt)
		cmp := v_2_0.Args[0]
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (ISEL [6] x y (CMPWconst [0] (SETBC [c] cmp)))
	// result: (ISEL [c] x y cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64CMPWconst || auxIntToInt32(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpPPC64SETBC {
			break
		}
		c := auxIntToInt32(v_2_0.AuxInt)
		cmp := v_2_0.Args[0]
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(c)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (ISEL [6] x y (CMPconst [0] (SETBCR [c] cmp)))
	// result: (ISEL [c+4] x y cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64CMPconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpPPC64SETBCR {
			break
		}
		c := auxIntToInt32(v_2_0.AuxInt)
		cmp := v_2_0.Args[0]
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(c + 4)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (ISEL [6] x y (CMPWconst [0] (SETBCR [c] cmp)))
	// result: (ISEL [c+4] x y cmp)
	for {
		if auxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64CMPWconst || auxIntToInt32(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpPPC64SETBCR {
			break
		}
		c := auxIntToInt32(v_2_0.AuxInt)
		cmp := v_2_0.Args[0]
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(c + 4)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (ISEL [2] x _ (FlagEQ))
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		x := v_0
		if v_2.Op != OpPPC64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ISEL [2] _ y (FlagLT))
	// result: y
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		y := v_1
		if v_2.Op != OpPPC64FlagLT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (ISEL [2] _ y (FlagGT))
	// result: y
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		y := v_1
		if v_2.Op != OpPPC64FlagGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (ISEL [6] _ y (FlagEQ))
	// result: y
	for {
		if auxIntToInt32(v.AuxInt) != 6 {
			break
		}
		y := v_1
		if v_2.Op != OpPPC64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (ISEL [6] x _ (FlagLT))
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		if v_2.Op != OpPPC64FlagLT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ISEL [6] x _ (FlagGT))
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		if v_2.Op != OpPPC64FlagGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ISEL [0] _ y (FlagEQ))
	// result: y
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		y := v_1
		if v_2.Op != OpPPC64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (ISEL [0] _ y (FlagGT))
	// result: y
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		y := v_1
		if v_2.Op != OpPPC64FlagGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (ISEL [0] x _ (FlagLT))
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		if v_2.Op != OpPPC64FlagLT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ISEL [5] _ x (FlagEQ))
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 5 {
			break
		}
		x := v_1
		if v_2.Op != OpPPC64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ISEL [5] _ x (FlagLT))
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 5 {
			break
		}
		x := v_1
		if v_2.Op != OpPPC64FlagLT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ISEL [5] y _ (FlagGT))
	// result: y
	for {
		if auxIntToInt32(v.AuxInt) != 5 {
			break
		}
		y := v_0
		if v_2.Op != OpPPC64FlagGT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (ISEL [1] _ y (FlagEQ))
	// result: y
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		y := v_1
		if v_2.Op != OpPPC64FlagEQ {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (ISEL [1] _ y (FlagLT))
	// result: y
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		y := v_1
		if v_2.Op != OpPPC64FlagLT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (ISEL [1] x _ (FlagGT))
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		x := v_0
		if v_2.Op != OpPPC64FlagGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ISEL [4] x _ (FlagEQ))
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 4 {
			break
		}
		x := v_0
		if v_2.Op != OpPPC64FlagEQ {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ISEL [4] x _ (FlagGT))
	// result: x
	for {
		if auxIntToInt32(v.AuxInt) != 4 {
			break
		}
		x := v_0
		if v_2.Op != OpPPC64FlagGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ISEL [4] _ y (FlagLT))
	// result: y
	for {
		if auxIntToInt32(v.AuxInt) != 4 {
			break
		}
		y := v_1
		if v_2.Op != OpPPC64FlagLT {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (ISEL [2] x y (CMPconst [0] (Select0 (ANDCCconst [n] z))))
	// result: (ISEL [2] x y (Select1 <types.TypeFlags> (ANDCCconst [n] z )))
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64CMPconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpSelect0 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpPPC64ANDCCconst {
			break
		}
		n := auxIntToInt64(v_2_0_0.AuxInt)
		z := v_2_0_0.Args[0]
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AuxInt = int64ToAuxInt(n)
		v1.AddArg(z)
		v0.AddArg(v1)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (ISEL [2] x y (CMPWconst [0] (Select0 (ANDCCconst [n] z))))
	// result: (ISEL [2] x y (Select1 <types.TypeFlags> (ANDCCconst [n] z )))
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64CMPWconst || auxIntToInt32(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpSelect0 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpPPC64ANDCCconst {
			break
		}
		n := auxIntToInt64(v_2_0_0.AuxInt)
		z := v_2_0_0.Args[0]
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AuxInt = int64ToAuxInt(n)
		v1.AddArg(z)
		v0.AddArg(v1)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (ISEL [6] x y (CMPconst [0] (Select0 (ANDCCconst [n] z))))
	// result: (ISEL [6] x y (Select1 <types.TypeFlags> (ANDCCconst [n] z )))
	for {
		if auxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64CMPconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpSelect0 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpPPC64ANDCCconst {
			break
		}
		n := auxIntToInt64(v_2_0_0.AuxInt)
		z := v_2_0_0.Args[0]
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AuxInt = int64ToAuxInt(n)
		v1.AddArg(z)
		v0.AddArg(v1)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (ISEL [6] x y (CMPWconst [0] (Select0 (ANDCCconst [n] z))))
	// result: (ISEL [6] x y (Select1 <types.TypeFlags> (ANDCCconst [n] z )))
	for {
		if auxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64CMPWconst || auxIntToInt32(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpSelect0 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpPPC64ANDCCconst {
			break
		}
		n := auxIntToInt64(v_2_0_0.AuxInt)
		z := v_2_0_0.Args[0]
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AuxInt = int64ToAuxInt(n)
		v1.AddArg(z)
		v0.AddArg(v1)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (ISEL [n] x y (InvertFlags bool))
	// cond: n%4 == 0
	// result: (ISEL [n+1] x y bool)
	for {
		n := auxIntToInt32(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64InvertFlags {
			break
		}
		bool := v_2.Args[0]
		if !(n%4 == 0) {
			break
		}
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(n + 1)
		v.AddArg3(x, y, bool)
		return true
	}
	// match: (ISEL [n] x y (InvertFlags bool))
	// cond: n%4 == 1
	// result: (ISEL [n-1] x y bool)
	for {
		n := auxIntToInt32(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64InvertFlags {
			break
		}
		bool := v_2.Args[0]
		if !(n%4 == 1) {
			break
		}
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(n - 1)
		v.AddArg3(x, y, bool)
		return true
	}
	// match: (ISEL [n] x y (InvertFlags bool))
	// cond: n%4 == 2
	// result: (ISEL [n] x y bool)
	for {
		n := auxIntToInt32(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != OpPPC64InvertFlags {
			break
		}
		bool := v_2.Args[0]
		if !(n%4 == 2) {
			break
		}
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(n)
		v.AddArg3(x, y, bool)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64LessEqual(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LessEqual (FlagEQ))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != OpPPC64FlagEQ {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (LessEqual (FlagLT))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != OpPPC64FlagLT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (LessEqual (FlagGT))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpPPC64FlagGT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (LessEqual (InvertFlags x))
	// result: (GreaterEqual x)
	for {
		if v_0.Op != OpPPC64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpPPC64GreaterEqual)
		v.AddArg(x)
		return true
	}
	// match: (LessEqual cmp)
	// result: (SETBCR [1] cmp)
	for {
		cmp := v_0
		v.reset(OpPPC64SETBCR)
		v.AuxInt = int32ToAuxInt(1)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64LessThan(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LessThan (FlagEQ))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpPPC64FlagEQ {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (LessThan (FlagLT))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != OpPPC64FlagLT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (LessThan (FlagGT))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpPPC64FlagGT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (LessThan (InvertFlags x))
	// result: (GreaterThan x)
	for {
		if v_0.Op != OpPPC64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpPPC64GreaterThan)
		v.AddArg(x)
		return true
	}
	// match: (LessThan cmp)
	// result: (SETBC [0] cmp)
	for {
		cmp := v_0
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(0)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64MFVSRD(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MFVSRD (FMOVDconst [c]))
	// result: (MOVDconst [int64(math.Float64bits(c))])
	for {
		if v_0.Op != OpPPC64FMOVDconst {
			break
		}
		c := auxIntToFloat64(v_0.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(math.Float64bits(c)))
		return true
	}
	// match: (MFVSRD x:(FMOVDload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVDload [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpPPC64FMOVDload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpPPC64MOVDload, typ.Int64)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBZload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBZload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVBZload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVBZload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBZload [off1] {sym} (ADDconst [off2] x) mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (MOVBZload [off1+int32(off2)] {sym} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64MOVBZload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVBZload [0] {sym} p:(ADD ptr idx) mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVBZloadidx ptr idx mem)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		mem := v_1
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVBZloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBZloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBZloadidx ptr (MOVDconst [c]) mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVBZload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVBZload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBZloadidx (MOVDconst [c]) ptr mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVBZload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVBZload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBZreg(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVBZreg y:(Select0 (ANDCCconst [c] _)))
	// cond: uint64(c) <= 0xFF
	// result: y
	for {
		y := v_0
		if y.Op != OpSelect0 {
			break
		}
		y_0 := y.Args[0]
		if y_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(y_0.AuxInt)
		if !(uint64(c) <= 0xFF) {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVBZreg (SRWconst [c] (MOVBZreg x)))
	// result: (SRWconst [c] (MOVBZreg x))
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVBZreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVBZreg (SRWconst [c] x))
	// cond: sizeof(x.Type) == 8
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sizeof(x.Type) == 8) {
			break
		}
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBZreg (SRDconst [c] x))
	// cond: c>=56
	// result: (SRDconst [c] x)
	for {
		if v_0.Op != OpPPC64SRDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 56) {
			break
		}
		v.reset(OpPPC64SRDconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBZreg (SRWconst [c] x))
	// cond: c>=24
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 24) {
			break
		}
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBZreg y:(MOVBZreg _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVBZreg {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVBZreg (MOVBreg x))
	// result: (MOVBZreg x)
	for {
		if v_0.Op != OpPPC64MOVBreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpPPC64MOVBZreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBZreg (OR <t> x (MOVWZreg y)))
	// result: (MOVBZreg (OR <t> x y))
	for {
		if v_0.Op != OpPPC64OR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64OR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (XOR <t> x (MOVWZreg y)))
	// result: (MOVBZreg (XOR <t> x y))
	for {
		if v_0.Op != OpPPC64XOR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64XOR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (AND <t> x (MOVWZreg y)))
	// result: (MOVBZreg (AND <t> x y))
	for {
		if v_0.Op != OpPPC64AND {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64AND, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (OR <t> x (MOVHZreg y)))
	// result: (MOVBZreg (OR <t> x y))
	for {
		if v_0.Op != OpPPC64OR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVHZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64OR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (XOR <t> x (MOVHZreg y)))
	// result: (MOVBZreg (XOR <t> x y))
	for {
		if v_0.Op != OpPPC64XOR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVHZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64XOR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (AND <t> x (MOVHZreg y)))
	// result: (MOVBZreg (AND <t> x y))
	for {
		if v_0.Op != OpPPC64AND {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVHZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64AND, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (OR <t> x (MOVBZreg y)))
	// result: (MOVBZreg (OR <t> x y))
	for {
		if v_0.Op != OpPPC64OR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVBZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64OR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (XOR <t> x (MOVBZreg y)))
	// result: (MOVBZreg (XOR <t> x y))
	for {
		if v_0.Op != OpPPC64XOR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVBZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64XOR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (AND <t> x (MOVBZreg y)))
	// result: (MOVBZreg (AND <t> x y))
	for {
		if v_0.Op != OpPPC64AND {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVBZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64AND, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg z:(Select0 (ANDCCconst [c] (MOVBZload ptr x))))
	// result: z
	for {
		z := v_0
		if z.Op != OpSelect0 {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != OpPPC64ANDCCconst {
			break
		}
		z_0_0 := z_0.Args[0]
		if z_0_0.Op != OpPPC64MOVBZload {
			break
		}
		v.copyOf(z)
		return true
	}
	// match: (MOVBZreg z:(AND y (MOVBZload ptr x)))
	// result: z
	for {
		z := v_0
		if z.Op != OpPPC64AND {
			break
		}
		_ = z.Args[1]
		z_0 := z.Args[0]
		z_1 := z.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
			if z_1.Op != OpPPC64MOVBZload {
				continue
			}
			v.copyOf(z)
			return true
		}
		break
	}
	// match: (MOVBZreg x:(MOVBZload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVBZload {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBZreg x:(MOVBZloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVBZloadidx {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBZreg x:(Select0 (LoweredAtomicLoad8 _ _)))
	// result: x
	for {
		x := v_0
		if x.Op != OpSelect0 {
			break
		}
		x_0 := x.Args[0]
		if x_0.Op != OpPPC64LoweredAtomicLoad8 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBZreg x:(Arg <t>))
	// cond: is8BitInt(t) && !t.IsSigned()
	// result: x
	for {
		x := v_0
		if x.Op != OpArg {
			break
		}
		t := x.Type
		if !(is8BitInt(t) && !t.IsSigned()) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBZreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint8(c))])
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint8(c)))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBreg(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVBreg y:(Select0 (ANDCCconst [c] _)))
	// cond: uint64(c) <= 0x7F
	// result: y
	for {
		y := v_0
		if y.Op != OpSelect0 {
			break
		}
		y_0 := y.Args[0]
		if y_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(y_0.AuxInt)
		if !(uint64(c) <= 0x7F) {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVBreg (SRAWconst [c] (MOVBreg x)))
	// result: (SRAWconst [c] (MOVBreg x))
	for {
		if v_0.Op != OpPPC64SRAWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVBreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVBreg (SRAWconst [c] x))
	// cond: sizeof(x.Type) == 8
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRAWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sizeof(x.Type) == 8) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SRDconst [c] x))
	// cond: c>56
	// result: (SRDconst [c] x)
	for {
		if v_0.Op != OpPPC64SRDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > 56) {
			break
		}
		v.reset(OpPPC64SRDconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SRDconst [c] x))
	// cond: c==56
	// result: (SRADconst [c] x)
	for {
		if v_0.Op != OpPPC64SRDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c == 56) {
			break
		}
		v.reset(OpPPC64SRADconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SRADconst [c] x))
	// cond: c>=56
	// result: (SRADconst [c] x)
	for {
		if v_0.Op != OpPPC64SRADconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 56) {
			break
		}
		v.reset(OpPPC64SRADconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SRWconst [c] x))
	// cond: c>24
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > 24) {
			break
		}
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SRWconst [c] x))
	// cond: c==24
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c == 24) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SRAWconst [c] x))
	// cond: c>=24
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRAWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 24) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg y:(MOVBreg _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVBreg {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVBreg (MOVBZreg x))
	// result: (MOVBreg x)
	for {
		if v_0.Op != OpPPC64MOVBZreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpPPC64MOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(Arg <t>))
	// cond: is8BitInt(t) && t.IsSigned()
	// result: x
	for {
		x := v_0
		if x.Op != OpArg {
			break
		}
		t := x.Type
		if !(is8BitInt(t) && t.IsSigned()) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int8(c))])
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int8(c)))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVBstore [off1] {sym} (ADDconst [off2] x) val mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (MOVBstore [off1+int32(off2)] {sym} x val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(x, val, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVBstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVBstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpPPC64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstore [0] {sym} p:(ADD ptr idx) val mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVBstoreidx ptr idx val mem)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVBreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBZreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVBZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHZreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVHZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWZreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (SRWconst (MOVHreg x) [c]) mem)
	// cond: c <= 8
	// result: (MOVBstore [off] {sym} ptr (SRWconst <typ.UInt32> x [c]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpPPC64MOVHreg {
			break
		}
		x := v_1_0.Args[0]
		mem := v_2
		if !(c <= 8) {
			break
		}
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (SRWconst (MOVHZreg x) [c]) mem)
	// cond: c <= 8
	// result: (MOVBstore [off] {sym} ptr (SRWconst <typ.UInt32> x [c]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpPPC64MOVHZreg {
			break
		}
		x := v_1_0.Args[0]
		mem := v_2
		if !(c <= 8) {
			break
		}
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (SRWconst (MOVWreg x) [c]) mem)
	// cond: c <= 24
	// result: (MOVBstore [off] {sym} ptr (SRWconst <typ.UInt32> x [c]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpPPC64MOVWreg {
			break
		}
		x := v_1_0.Args[0]
		mem := v_2
		if !(c <= 24) {
			break
		}
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (SRWconst (MOVWZreg x) [c]) mem)
	// cond: c <= 24
	// result: (MOVBstore [off] {sym} ptr (SRWconst <typ.UInt32> x [c]) mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_1_0.Args[0]
		mem := v_2
		if !(c <= 24) {
			break
		}
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVBstoreidx ptr (MOVDconst [c]) val mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVBstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstoreidx (MOVDconst [c]) ptr val mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVBstore [int32(c)] ptr val mem)
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		val := v_2
		mem := v_3
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVBstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVBreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64MOVBreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVBZreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64MOVBZreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVHreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64MOVHreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVHZreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64MOVHZreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVWreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVWZreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (SRWconst (MOVHreg x) [c]) mem)
	// cond: c <= 8
	// result: (MOVBstoreidx ptr idx (SRWconst <typ.UInt32> x [c]) mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpPPC64MOVHreg {
			break
		}
		x := v_2_0.Args[0]
		mem := v_3
		if !(c <= 8) {
			break
		}
		v.reset(OpPPC64MOVBstoreidx)
		v0 := b.NewValue0(v.Pos, OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg4(ptr, idx, v0, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (SRWconst (MOVHZreg x) [c]) mem)
	// cond: c <= 8
	// result: (MOVBstoreidx ptr idx (SRWconst <typ.UInt32> x [c]) mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpPPC64MOVHZreg {
			break
		}
		x := v_2_0.Args[0]
		mem := v_3
		if !(c <= 8) {
			break
		}
		v.reset(OpPPC64MOVBstoreidx)
		v0 := b.NewValue0(v.Pos, OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg4(ptr, idx, v0, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (SRWconst (MOVWreg x) [c]) mem)
	// cond: c <= 24
	// result: (MOVBstoreidx ptr idx (SRWconst <typ.UInt32> x [c]) mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpPPC64MOVWreg {
			break
		}
		x := v_2_0.Args[0]
		mem := v_3
		if !(c <= 24) {
			break
		}
		v.reset(OpPPC64MOVBstoreidx)
		v0 := b.NewValue0(v.Pos, OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg4(ptr, idx, v0, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (SRWconst (MOVWZreg x) [c]) mem)
	// cond: c <= 24
	// result: (MOVBstoreidx ptr idx (SRWconst <typ.UInt32> x [c]) mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_2_0.Args[0]
		mem := v_3
		if !(c <= 24) {
			break
		}
		v.reset(OpPPC64MOVBstoreidx)
		v0 := b.NewValue0(v.Pos, OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg4(ptr, idx, v0, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstorezero [off1] {sym} (ADDconst [off2] x) mem)
	// cond: ((supportsPPC64PCRel() && is32Bit(int64(off1)+off2)) || (is16Bit(int64(off1)+off2)))
	// result: (MOVBstorezero [off1+int32(off2)] {sym} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !((supportsPPC64PCRel() && is32Bit(int64(off1)+off2)) || (is16Bit(int64(off1) + off2))) {
			break
		}
		v.reset(OpPPC64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVBstorezero [off1] {sym1} p:(MOVDaddr [off2] {sym2} x) mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (x.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVBstorezero [off1+off2] {mergeSym(sym1,sym2)} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		x := p.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (x.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVDaddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDaddr {sym} [n] p:(ADD x y))
	// cond: sym == nil && n == 0
	// result: p
	for {
		n := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64ADD {
			break
		}
		if !(sym == nil && n == 0) {
			break
		}
		v.copyOf(p)
		return true
	}
	// match: (MOVDaddr {sym} [n] ptr)
	// cond: sym == nil && n == 0 && (ptr.Op == OpArgIntReg || ptr.Op == OpPhi)
	// result: ptr
	for {
		n := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if !(sym == nil && n == 0 && (ptr.Op == OpArgIntReg || ptr.Op == OpPhi)) {
			break
		}
		v.copyOf(ptr)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVDload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDload [off] {sym} ptr (FMOVDstore [off] {sym} ptr x _))
	// result: (MFVSRD x)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64FMOVDstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpPPC64MFVSRD)
		v.AddArg(x)
		return true
	}
	// match: (MOVDload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVDload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off1] {sym} (ADDconst [off2] x) mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (MOVDload [off1+int32(off2)] {sym} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64MOVDload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVDload [0] {sym} p:(ADD ptr idx) mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVDloadidx ptr idx mem)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		mem := v_1
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVDloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVDloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDloadidx ptr (MOVDconst [c]) mem)
	// cond: ((is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVDload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !((is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVDload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDloadidx (MOVDconst [c]) ptr mem)
	// cond: ((is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVDload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !((is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVDload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVDstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVDstore [off] {sym} ptr (MFVSRD x) mem)
	// result: (FMOVDstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MFVSRD {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64FMOVDstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (ADDconst [off2] x) val mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (MOVDstore [off1+int32(off2)] {sym} x val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64MOVDstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(x, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVDstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVDstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVDstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpPPC64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDstore [0] {sym} p:(ADD ptr idx) val mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVDstoreidx ptr idx val mem)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVDstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} ptr r:(BRD val) mem)
	// cond: r.Uses == 1
	// result: (MOVDBRstore (MOVDaddr <ptr.Type> [off] {sym} ptr) val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		r := v_1
		if r.Op != OpPPC64BRD {
			break
		}
		val := r.Args[0]
		mem := v_2
		if !(r.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVDBRstore)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDaddr, ptr.Type)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg(ptr)
		v.AddArg3(v0, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} ptr (Bswap64 val) mem)
	// result: (MOVDBRstore (MOVDaddr <ptr.Type> [off] {sym} ptr) val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpBswap64 {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVDBRstore)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDaddr, ptr.Type)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg(ptr)
		v.AddArg3(v0, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVDstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstoreidx ptr (MOVDconst [c]) val mem)
	// cond: ((is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVDstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !((is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVDstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstoreidx (MOVDconst [c]) ptr val mem)
	// cond: ((is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVDstore [int32(c)] ptr val mem)
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		val := v_2
		mem := v_3
		if !((is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVDstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstoreidx ptr idx r:(BRD val) mem)
	// cond: r.Uses == 1
	// result: (MOVDBRstoreidx ptr idx val mem)
	for {
		ptr := v_0
		idx := v_1
		r := v_2
		if r.Op != OpPPC64BRD {
			break
		}
		val := r.Args[0]
		mem := v_3
		if !(r.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVDBRstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVDstoreidx ptr idx (Bswap64 val) mem)
	// result: (MOVDBRstoreidx ptr idx val mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpBswap64 {
			break
		}
		val := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVDBRstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVDstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstorezero [off1] {sym} (ADDconst [off2] x) mem)
	// cond: ((supportsPPC64PCRel() && is32Bit(int64(off1)+off2)) || (is16Bit(int64(off1)+off2)))
	// result: (MOVDstorezero [off1+int32(off2)] {sym} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !((supportsPPC64PCRel() && is32Bit(int64(off1)+off2)) || (is16Bit(int64(off1) + off2))) {
			break
		}
		v.reset(OpPPC64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVDstorezero [off1] {sym1} p:(MOVDaddr [off2] {sym2} x) mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (x.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVDstorezero [off1+off2] {mergeSym(sym1,sym2)} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		x := p.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (x.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHBRstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHBRstore ptr (MOVHreg x) mem)
	// result: (MOVHBRstore ptr x mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVHBRstore)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHBRstore ptr (MOVHZreg x) mem)
	// result: (MOVHBRstore ptr x mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVHZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVHBRstore)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHBRstore ptr (MOVWreg x) mem)
	// result: (MOVHBRstore ptr x mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVHBRstore)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHBRstore ptr (MOVWZreg x) mem)
	// result: (MOVHBRstore ptr x mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVHBRstore)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHZload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHZload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVHZload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVHZload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHZload [off1] {sym} (ADDconst [off2] x) mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (MOVHZload [off1+int32(off2)] {sym} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64MOVHZload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVHZload [0] {sym} p:(ADD ptr idx) mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVHZloadidx ptr idx mem)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		mem := v_1
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVHZloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHZloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHZloadidx ptr (MOVDconst [c]) mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVHZload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVHZload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHZloadidx (MOVDconst [c]) ptr mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVHZload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVHZload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHZreg(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVHZreg y:(Select0 (ANDCCconst [c] _)))
	// cond: uint64(c) <= 0xFFFF
	// result: y
	for {
		y := v_0
		if y.Op != OpSelect0 {
			break
		}
		y_0 := y.Args[0]
		if y_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(y_0.AuxInt)
		if !(uint64(c) <= 0xFFFF) {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVHZreg (SRWconst [c] (MOVBZreg x)))
	// result: (SRWconst [c] (MOVBZreg x))
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVBZreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVHZreg (SRWconst [c] (MOVHZreg x)))
	// result: (SRWconst [c] (MOVHZreg x))
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVHZreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVHZreg (SRWconst [c] x))
	// cond: sizeof(x.Type) <= 16
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sizeof(x.Type) <= 16) {
			break
		}
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHZreg (SRDconst [c] x))
	// cond: c>=48
	// result: (SRDconst [c] x)
	for {
		if v_0.Op != OpPPC64SRDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 48) {
			break
		}
		v.reset(OpPPC64SRDconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHZreg (SRWconst [c] x))
	// cond: c>=16
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 16) {
			break
		}
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHZreg y:(MOVHZreg _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVHZreg {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVHZreg y:(MOVBZreg _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVBZreg {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVHZreg y:(MOVHBRload _ _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVHBRload {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVHZreg y:(MOVHreg x))
	// result: (MOVHZreg x)
	for {
		y := v_0
		if y.Op != OpPPC64MOVHreg {
			break
		}
		x := y.Args[0]
		v.reset(OpPPC64MOVHZreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHZreg (OR <t> x (MOVWZreg y)))
	// result: (MOVHZreg (OR <t> x y))
	for {
		if v_0.Op != OpPPC64OR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVHZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64OR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVHZreg (XOR <t> x (MOVWZreg y)))
	// result: (MOVHZreg (XOR <t> x y))
	for {
		if v_0.Op != OpPPC64XOR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVHZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64XOR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVHZreg (AND <t> x (MOVWZreg y)))
	// result: (MOVHZreg (AND <t> x y))
	for {
		if v_0.Op != OpPPC64AND {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVHZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64AND, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVHZreg (OR <t> x (MOVHZreg y)))
	// result: (MOVHZreg (OR <t> x y))
	for {
		if v_0.Op != OpPPC64OR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVHZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVHZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64OR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVHZreg (XOR <t> x (MOVHZreg y)))
	// result: (MOVHZreg (XOR <t> x y))
	for {
		if v_0.Op != OpPPC64XOR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVHZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVHZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64XOR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVHZreg (AND <t> x (MOVHZreg y)))
	// result: (MOVHZreg (AND <t> x y))
	for {
		if v_0.Op != OpPPC64AND {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVHZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVHZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64AND, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVHZreg z:(Select0 (ANDCCconst [c] (MOVBZload ptr x))))
	// result: z
	for {
		z := v_0
		if z.Op != OpSelect0 {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != OpPPC64ANDCCconst {
			break
		}
		z_0_0 := z_0.Args[0]
		if z_0_0.Op != OpPPC64MOVBZload {
			break
		}
		v.copyOf(z)
		return true
	}
	// match: (MOVHZreg z:(AND y (MOVHZload ptr x)))
	// result: z
	for {
		z := v_0
		if z.Op != OpPPC64AND {
			break
		}
		_ = z.Args[1]
		z_0 := z.Args[0]
		z_1 := z.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
			if z_1.Op != OpPPC64MOVHZload {
				continue
			}
			v.copyOf(z)
			return true
		}
		break
	}
	// match: (MOVHZreg z:(Select0 (ANDCCconst [c] (MOVHZload ptr x))))
	// result: z
	for {
		z := v_0
		if z.Op != OpSelect0 {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != OpPPC64ANDCCconst {
			break
		}
		z_0_0 := z_0.Args[0]
		if z_0_0.Op != OpPPC64MOVHZload {
			break
		}
		v.copyOf(z)
		return true
	}
	// match: (MOVHZreg x:(MOVBZload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVBZload {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVHZreg x:(MOVBZloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVBZloadidx {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVHZreg x:(MOVHZload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVHZload {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVHZreg x:(MOVHZloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVHZloadidx {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVHZreg x:(Arg <t>))
	// cond: (is8BitInt(t) || is16BitInt(t)) && !t.IsSigned()
	// result: x
	for {
		x := v_0
		if x.Op != OpArg {
			break
		}
		t := x.Type
		if !((is8BitInt(t) || is16BitInt(t)) && !t.IsSigned()) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVHZreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint16(c))])
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint16(c)))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVHload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVHload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off1] {sym} (ADDconst [off2] x) mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (MOVHload [off1+int32(off2)] {sym} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64MOVHload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVHload [0] {sym} p:(ADD ptr idx) mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVHloadidx ptr idx mem)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		mem := v_1
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVHloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHloadidx ptr (MOVDconst [c]) mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVHload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVHload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHloadidx (MOVDconst [c]) ptr mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVHload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVHload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHreg(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVHreg y:(Select0 (ANDCCconst [c] _)))
	// cond: uint64(c) <= 0x7FFF
	// result: y
	for {
		y := v_0
		if y.Op != OpSelect0 {
			break
		}
		y_0 := y.Args[0]
		if y_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(y_0.AuxInt)
		if !(uint64(c) <= 0x7FFF) {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVHreg (SRAWconst [c] (MOVBreg x)))
	// result: (SRAWconst [c] (MOVBreg x))
	for {
		if v_0.Op != OpPPC64SRAWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVBreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVHreg (SRAWconst [c] (MOVHreg x)))
	// result: (SRAWconst [c] (MOVHreg x))
	for {
		if v_0.Op != OpPPC64SRAWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVHreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVHreg (SRAWconst [c] x))
	// cond: sizeof(x.Type) <= 16
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRAWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sizeof(x.Type) <= 16) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SRDconst [c] x))
	// cond: c>48
	// result: (SRDconst [c] x)
	for {
		if v_0.Op != OpPPC64SRDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > 48) {
			break
		}
		v.reset(OpPPC64SRDconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SRDconst [c] x))
	// cond: c==48
	// result: (SRADconst [c] x)
	for {
		if v_0.Op != OpPPC64SRDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c == 48) {
			break
		}
		v.reset(OpPPC64SRADconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SRADconst [c] x))
	// cond: c>=48
	// result: (SRADconst [c] x)
	for {
		if v_0.Op != OpPPC64SRADconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 48) {
			break
		}
		v.reset(OpPPC64SRADconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SRWconst [c] x))
	// cond: c>16
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > 16) {
			break
		}
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SRAWconst [c] x))
	// cond: c>=16
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRAWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 16) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SRWconst [c] x))
	// cond: c==16
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c == 16) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg y:(MOVHreg _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVHreg {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVHreg y:(MOVBreg _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVBreg {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVHreg y:(MOVHZreg x))
	// result: (MOVHreg x)
	for {
		y := v_0
		if y.Op != OpPPC64MOVHZreg {
			break
		}
		x := y.Args[0]
		v.reset(OpPPC64MOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVHload {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVHreg x:(MOVHloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVHloadidx {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVHreg x:(Arg <t>))
	// cond: (is8BitInt(t) || is16BitInt(t)) && t.IsSigned()
	// result: x
	for {
		x := v_0
		if x.Op != OpArg {
			break
		}
		t := x.Type
		if !((is8BitInt(t) || is16BitInt(t)) && t.IsSigned()) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVHreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int16(c))])
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int16(c)))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVHstore [off1] {sym} (ADDconst [off2] x) val mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (MOVHstore [off1+int32(off2)] {sym} x val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64MOVHstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(x, val, mem)
		return true
	}
	// match: (MOVHstore [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVHstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVHstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVHstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpPPC64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstore [0] {sym} p:(ADD ptr idx) val mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVHstoreidx ptr idx val mem)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVHstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVHstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHZreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVHZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVHstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVHstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWZreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVHstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr r:(BRH val) mem)
	// cond: r.Uses == 1
	// result: (MOVHBRstore (MOVDaddr <ptr.Type> [off] {sym} ptr) val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		r := v_1
		if r.Op != OpPPC64BRH {
			break
		}
		val := r.Args[0]
		mem := v_2
		if !(r.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVHBRstore)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDaddr, ptr.Type)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg(ptr)
		v.AddArg3(v0, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (Bswap16 val) mem)
	// result: (MOVHBRstore (MOVDaddr <ptr.Type> [off] {sym} ptr) val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpBswap16 {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVHBRstore)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDaddr, ptr.Type)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg(ptr)
		v.AddArg3(v0, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstoreidx ptr (MOVDconst [c]) val mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVHstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVHstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstoreidx (MOVDconst [c]) ptr val mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVHstore [int32(c)] ptr val mem)
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		val := v_2
		mem := v_3
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVHstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVHreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64MOVHreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVHZreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64MOVHZreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVWreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVWZreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx r:(BRH val) mem)
	// cond: r.Uses == 1
	// result: (MOVHBRstoreidx ptr idx val mem)
	for {
		ptr := v_0
		idx := v_1
		r := v_2
		if r.Op != OpPPC64BRH {
			break
		}
		val := r.Args[0]
		mem := v_3
		if !(r.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVHBRstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (Bswap16 val) mem)
	// result: (MOVHBRstoreidx ptr idx val mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpBswap16 {
			break
		}
		val := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVHBRstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstorezero [off1] {sym} (ADDconst [off2] x) mem)
	// cond: ((supportsPPC64PCRel() && is32Bit(int64(off1)+off2)) || (is16Bit(int64(off1)+off2)))
	// result: (MOVHstorezero [off1+int32(off2)] {sym} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !((supportsPPC64PCRel() && is32Bit(int64(off1)+off2)) || (is16Bit(int64(off1) + off2))) {
			break
		}
		v.reset(OpPPC64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVHstorezero [off1] {sym1} p:(MOVDaddr [off2] {sym2} x) mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (x.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVHstorezero [off1+off2] {mergeSym(sym1,sym2)} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		x := p.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (x.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWBRstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWBRstore ptr (MOVWreg x) mem)
	// result: (MOVWBRstore ptr x mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVWBRstore)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWBRstore ptr (MOVWZreg x) mem)
	// result: (MOVWBRstore ptr x mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVWBRstore)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWZload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWZload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVWZload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVWZload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWZload [off1] {sym} (ADDconst [off2] x) mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (MOVWZload [off1+int32(off2)] {sym} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64MOVWZload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVWZload [0] {sym} p:(ADD ptr idx) mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVWZloadidx ptr idx mem)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		mem := v_1
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVWZloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWZloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWZloadidx ptr (MOVDconst [c]) mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVWZload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVWZload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWZloadidx (MOVDconst [c]) ptr mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVWZload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVWZload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWZreg(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVWZreg y:(Select0 (ANDCCconst [c] _)))
	// cond: uint64(c) <= 0xFFFFFFFF
	// result: y
	for {
		y := v_0
		if y.Op != OpSelect0 {
			break
		}
		y_0 := y.Args[0]
		if y_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(y_0.AuxInt)
		if !(uint64(c) <= 0xFFFFFFFF) {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVWZreg y:(AND (MOVDconst [c]) _))
	// cond: uint64(c) <= 0xFFFFFFFF
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64AND {
			break
		}
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			if y_0.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(y_0.AuxInt)
			if !(uint64(c) <= 0xFFFFFFFF) {
				continue
			}
			v.copyOf(y)
			return true
		}
		break
	}
	// match: (MOVWZreg (SRWconst [c] (MOVBZreg x)))
	// result: (SRWconst [c] (MOVBZreg x))
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVBZreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWZreg (SRWconst [c] (MOVHZreg x)))
	// result: (SRWconst [c] (MOVHZreg x))
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVHZreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWZreg (SRWconst [c] (MOVWZreg x)))
	// result: (SRWconst [c] (MOVWZreg x))
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVWZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWZreg (SRWconst [c] x))
	// cond: sizeof(x.Type) <= 32
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sizeof(x.Type) <= 32) {
			break
		}
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWZreg (SRDconst [c] x))
	// cond: c>=32
	// result: (SRDconst [c] x)
	for {
		if v_0.Op != OpPPC64SRDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 32) {
			break
		}
		v.reset(OpPPC64SRDconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWZreg y:(MOVWZreg _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVWZreg {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVWZreg y:(MOVHZreg _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVHZreg {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVWZreg y:(MOVBZreg _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVBZreg {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVWZreg y:(MOVHBRload _ _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVHBRload {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVWZreg y:(MOVWBRload _ _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVWBRload {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVWZreg y:(MOVWreg x))
	// result: (MOVWZreg x)
	for {
		y := v_0
		if y.Op != OpPPC64MOVWreg {
			break
		}
		x := y.Args[0]
		v.reset(OpPPC64MOVWZreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWZreg (OR <t> x (MOVWZreg y)))
	// result: (MOVWZreg (OR <t> x y))
	for {
		if v_0.Op != OpPPC64OR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVWZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64OR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVWZreg (XOR <t> x (MOVWZreg y)))
	// result: (MOVWZreg (XOR <t> x y))
	for {
		if v_0.Op != OpPPC64XOR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVWZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64XOR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVWZreg (AND <t> x (MOVWZreg y)))
	// result: (MOVWZreg (AND <t> x y))
	for {
		if v_0.Op != OpPPC64AND {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.reset(OpPPC64MOVWZreg)
			v0 := b.NewValue0(v.Pos, OpPPC64AND, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVWZreg z:(Select0 (ANDCCconst [c] (MOVBZload ptr x))))
	// result: z
	for {
		z := v_0
		if z.Op != OpSelect0 {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != OpPPC64ANDCCconst {
			break
		}
		z_0_0 := z_0.Args[0]
		if z_0_0.Op != OpPPC64MOVBZload {
			break
		}
		v.copyOf(z)
		return true
	}
	// match: (MOVWZreg z:(AND y (MOVWZload ptr x)))
	// result: z
	for {
		z := v_0
		if z.Op != OpPPC64AND {
			break
		}
		_ = z.Args[1]
		z_0 := z.Args[0]
		z_1 := z.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
			if z_1.Op != OpPPC64MOVWZload {
				continue
			}
			v.copyOf(z)
			return true
		}
		break
	}
	// match: (MOVWZreg z:(Select0 (ANDCCconst [c] (MOVHZload ptr x))))
	// result: z
	for {
		z := v_0
		if z.Op != OpSelect0 {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != OpPPC64ANDCCconst {
			break
		}
		z_0_0 := z_0.Args[0]
		if z_0_0.Op != OpPPC64MOVHZload {
			break
		}
		v.copyOf(z)
		return true
	}
	// match: (MOVWZreg z:(Select0 (ANDCCconst [c] (MOVWZload ptr x))))
	// result: z
	for {
		z := v_0
		if z.Op != OpSelect0 {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != OpPPC64ANDCCconst {
			break
		}
		z_0_0 := z_0.Args[0]
		if z_0_0.Op != OpPPC64MOVWZload {
			break
		}
		v.copyOf(z)
		return true
	}
	// match: (MOVWZreg x:(MOVBZload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVBZload {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWZreg x:(MOVBZloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVBZloadidx {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWZreg x:(MOVHZload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVHZload {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWZreg x:(MOVHZloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVHZloadidx {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWZreg x:(MOVWZload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVWZload {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWZreg x:(MOVWZloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVWZloadidx {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWZreg x:(Select0 (LoweredAtomicLoad32 _ _)))
	// result: x
	for {
		x := v_0
		if x.Op != OpSelect0 {
			break
		}
		x_0 := x.Args[0]
		if x_0.Op != OpPPC64LoweredAtomicLoad32 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWZreg x:(Arg <t>))
	// cond: (is8BitInt(t) || is16BitInt(t) || is32BitInt(t)) && !t.IsSigned()
	// result: x
	for {
		x := v_0
		if x.Op != OpArg {
			break
		}
		t := x.Type
		if !((is8BitInt(t) || is16BitInt(t) || is32BitInt(t)) && !t.IsSigned()) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWZreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint32(c))])
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint32(c)))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVWload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVWload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off1] {sym} (ADDconst [off2] x) mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (MOVWload [off1+int32(off2)] {sym} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64MOVWload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVWload [0] {sym} p:(ADD ptr idx) mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVWloadidx ptr idx mem)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		mem := v_1
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVWloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWloadidx ptr (MOVDconst [c]) mem)
	// cond: ((is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVWload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !((is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVWload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWloadidx (MOVDconst [c]) ptr mem)
	// cond: ((is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVWload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !((is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVWload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWreg(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVWreg y:(Select0 (ANDCCconst [c] _)))
	// cond: uint64(c) <= 0xFFFF
	// result: y
	for {
		y := v_0
		if y.Op != OpSelect0 {
			break
		}
		y_0 := y.Args[0]
		if y_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(y_0.AuxInt)
		if !(uint64(c) <= 0xFFFF) {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVWreg y:(AND (MOVDconst [c]) _))
	// cond: uint64(c) <= 0x7FFFFFFF
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64AND {
			break
		}
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			if y_0.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(y_0.AuxInt)
			if !(uint64(c) <= 0x7FFFFFFF) {
				continue
			}
			v.copyOf(y)
			return true
		}
		break
	}
	// match: (MOVWreg (SRAWconst [c] (MOVBreg x)))
	// result: (SRAWconst [c] (MOVBreg x))
	for {
		if v_0.Op != OpPPC64SRAWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVBreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWreg (SRAWconst [c] (MOVHreg x)))
	// result: (SRAWconst [c] (MOVHreg x))
	for {
		if v_0.Op != OpPPC64SRAWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVHreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWreg (SRAWconst [c] (MOVWreg x)))
	// result: (SRAWconst [c] (MOVWreg x))
	for {
		if v_0.Op != OpPPC64SRAWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVWreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVWreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWreg (SRAWconst [c] x))
	// cond: sizeof(x.Type) <= 32
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != OpPPC64SRAWconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sizeof(x.Type) <= 32) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (SRDconst [c] x))
	// cond: c>32
	// result: (SRDconst [c] x)
	for {
		if v_0.Op != OpPPC64SRDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > 32) {
			break
		}
		v.reset(OpPPC64SRDconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (SRADconst [c] x))
	// cond: c>=32
	// result: (SRADconst [c] x)
	for {
		if v_0.Op != OpPPC64SRADconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 32) {
			break
		}
		v.reset(OpPPC64SRADconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (SRDconst [c] x))
	// cond: c==32
	// result: (SRADconst [c] x)
	for {
		if v_0.Op != OpPPC64SRDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c == 32) {
			break
		}
		v.reset(OpPPC64SRADconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg y:(MOVWreg _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVWreg {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVWreg y:(MOVHreg _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVHreg {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVWreg y:(MOVBreg _))
	// result: y
	for {
		y := v_0
		if y.Op != OpPPC64MOVBreg {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (MOVWreg y:(MOVWZreg x))
	// result: (MOVWreg x)
	for {
		y := v_0
		if y.Op != OpPPC64MOVWZreg {
			break
		}
		x := y.Args[0]
		v.reset(OpPPC64MOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVHload {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWreg x:(MOVHloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVHloadidx {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWreg x:(MOVWload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVWload {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWreg x:(MOVWloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpPPC64MOVWloadidx {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWreg x:(Arg <t>))
	// cond: (is8BitInt(t) || is16BitInt(t) || is32BitInt(t)) && t.IsSigned()
	// result: x
	for {
		x := v_0
		if x.Op != OpArg {
			break
		}
		t := x.Type
		if !((is8BitInt(t) || is16BitInt(t) || is32BitInt(t)) && t.IsSigned()) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int32(c))])
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int32(c)))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVWstore [off1] {sym} (ADDconst [off2] x) val mem)
	// cond: (is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2)))
	// result: (MOVWstore [off1+int32(off2)] {sym} x val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is16Bit(int64(off1)+off2) || (supportsPPC64PCRel() && is32Bit(int64(off1)+off2))) {
			break
		}
		v.reset(OpPPC64MOVWstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(x, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVWstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (ptr.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVWstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVWstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpPPC64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstore [0] {sym} p:(ADD ptr idx) val mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVWstoreidx ptr idx val mem)
	for {
		if auxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVWstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWZreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr r:(BRW val) mem)
	// cond: r.Uses == 1
	// result: (MOVWBRstore (MOVDaddr <ptr.Type> [off] {sym} ptr) val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		r := v_1
		if r.Op != OpPPC64BRW {
			break
		}
		val := r.Args[0]
		mem := v_2
		if !(r.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVWBRstore)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDaddr, ptr.Type)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg(ptr)
		v.AddArg3(v0, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (Bswap32 val) mem)
	// result: (MOVWBRstore (MOVDaddr <ptr.Type> [off] {sym} ptr) val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpBswap32 {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpPPC64MOVWBRstore)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDaddr, ptr.Type)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg(ptr)
		v.AddArg3(v0, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreidx ptr (MOVDconst [c]) val mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVWstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVWstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstoreidx (MOVDconst [c]) ptr val mem)
	// cond: (is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c)))
	// result: (MOVWstore [int32(c)] ptr val mem)
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		val := v_2
		mem := v_3
		if !(is16Bit(c) || (buildcfg.GOPPC64 >= 10 && is32Bit(c))) {
			break
		}
		v.reset(OpPPC64MOVWstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVWreg x) mem)
	// result: (MOVWstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVWstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVWZreg x) mem)
	// result: (MOVWstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVWstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx r:(BRW val) mem)
	// cond: r.Uses == 1
	// result: (MOVWBRstoreidx ptr idx val mem)
	for {
		ptr := v_0
		idx := v_1
		r := v_2
		if r.Op != OpPPC64BRW {
			break
		}
		val := r.Args[0]
		mem := v_3
		if !(r.Uses == 1) {
			break
		}
		v.reset(OpPPC64MOVWBRstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (Bswap32 val) mem)
	// result: (MOVWBRstoreidx ptr idx val mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpBswap32 {
			break
		}
		val := v_2.Args[0]
		mem := v_3
		v.reset(OpPPC64MOVWBRstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstorezero [off1] {sym} (ADDconst [off2] x) mem)
	// cond: ((supportsPPC64PCRel() && is32Bit(int64(off1)+off2)) || (is16Bit(int64(off1)+off2)))
	// result: (MOVWstorezero [off1+int32(off2)] {sym} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !((supportsPPC64PCRel() && is32Bit(int64(off1)+off2)) || (is16Bit(int64(off1) + off2))) {
			break
		}
		v.reset(OpPPC64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVWstorezero [off1] {sym1} p:(MOVDaddr [off2] {sym2} x) mem)
	// cond: canMergeSym(sym1,sym2) && ((is16Bit(int64(off1+off2)) && (x.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))
	// result: (MOVWstorezero [off1+off2] {mergeSym(sym1,sym2)} x mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		p := v_0
		if p.Op != OpPPC64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(p.AuxInt)
		sym2 := auxToSym(p.Aux)
		x := p.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && ((is16Bit(int64(off1+off2)) && (x.Op != OpSB || p.Uses == 1)) || (supportsPPC64PCRel() && is32Bit(int64(off1+off2))))) {
			break
		}
		v.reset(OpPPC64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(x, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MTVSRD(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MTVSRD (MOVDconst [c]))
	// cond: !math.IsNaN(math.Float64frombits(uint64(c)))
	// result: (FMOVDconst [math.Float64frombits(uint64(c))])
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if !(!math.IsNaN(math.Float64frombits(uint64(c)))) {
			break
		}
		v.reset(OpPPC64FMOVDconst)
		v.AuxInt = float64ToAuxInt(math.Float64frombits(uint64(c)))
		return true
	}
	// match: (MTVSRD x:(MOVDload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (FMOVDload [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != OpPPC64MOVDload {
			break
		}
		off := auxIntToInt32(x.AuxInt)
		sym := auxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, OpPPC64FMOVDload, typ.Float64)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MULLD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULLD x (MOVDconst [c]))
	// cond: is16Bit(c)
	// result: (MULLDconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(is16Bit(c)) {
				continue
			}
			v.reset(OpPPC64MULLDconst)
			v.AuxInt = int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64MULLW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULLW x (MOVDconst [c]))
	// cond: is16Bit(c)
	// result: (MULLWconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(is16Bit(c)) {
				continue
			}
			v.reset(OpPPC64MULLWconst)
			v.AuxInt = int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64NEG(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NEG (ADDconst [c] x))
	// cond: is32Bit(-c)
	// result: (SUBFCconst [-c] x)
	for {
		if v_0.Op != OpPPC64ADDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(-c)) {
			break
		}
		v.reset(OpPPC64SUBFCconst)
		v.AuxInt = int64ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	// match: (NEG (SUBFCconst [c] x))
	// cond: is32Bit(-c)
	// result: (ADDconst [-c] x)
	for {
		if v_0.Op != OpPPC64SUBFCconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(-c)) {
			break
		}
		v.reset(OpPPC64ADDconst)
		v.AuxInt = int64ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	// match: (NEG (SUB x y))
	// result: (SUB y x)
	for {
		if v_0.Op != OpPPC64SUB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpPPC64SUB)
		v.AddArg2(y, x)
		return true
	}
	// match: (NEG (NEG x))
	// result: x
	for {
		if v_0.Op != OpPPC64NEG {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64NOR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NOR (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [^(c|d)])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpPPC64MOVDconst {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpPPC64MOVDconst)
			v.AuxInt = int64ToAuxInt(^(c | d))
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64NotEqual(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NotEqual (FlagEQ))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpPPC64FlagEQ {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (NotEqual (FlagLT))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != OpPPC64FlagLT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (NotEqual (FlagGT))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != OpPPC64FlagGT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (NotEqual (InvertFlags x))
	// result: (NotEqual x)
	for {
		if v_0.Op != OpPPC64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpPPC64NotEqual)
		v.AddArg(x)
		return true
	}
	// match: (NotEqual cmp)
	// result: (SETBCR [2] cmp)
	for {
		cmp := v_0
		v.reset(OpPPC64SETBCR)
		v.AuxInt = int32ToAuxInt(2)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64OR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (OR x (NOR y y))
	// result: (ORN x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpPPC64NOR {
				continue
			}
			y := v_1.Args[1]
			if y != v_1.Args[0] {
				continue
			}
			v.reset(OpPPC64ORN)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [c|d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpPPC64MOVDconst {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpPPC64MOVDconst)
			v.AuxInt = int64ToAuxInt(c | d)
			return true
		}
		break
	}
	// match: (OR x (MOVDconst [c]))
	// cond: isU32Bit(c)
	// result: (ORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isU32Bit(c)) {
				continue
			}
			v.reset(OpPPC64ORconst)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64ORN(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORN x (MOVDconst [-1]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst || auxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (ORN (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [c|^d])
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(c | ^d)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64ORconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ORconst [c] (ORconst [d] x))
	// result: (ORconst [c|d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64ORconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpPPC64ORconst)
		v.AuxInt = int64ToAuxInt(c | d)
		v.AddArg(x)
		return true
	}
	// match: (ORconst [-1] _)
	// result: (MOVDconst [-1])
	for {
		if auxIntToInt64(v.AuxInt) != -1 {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	// match: (ORconst [0] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64ROTL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROTL x (MOVDconst [c]))
	// result: (ROTLconst x [c&63])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpPPC64ROTLconst)
		v.AuxInt = int64ToAuxInt(c & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64ROTLW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROTLW x (MOVDconst [c]))
	// result: (ROTLWconst x [c&31])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpPPC64ROTLWconst)
		v.AuxInt = int64ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64ROTLWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ROTLWconst [r] (AND (MOVDconst [m]) x))
	// cond: isPPC64WordRotateMask(m)
	// result: (RLWINM [encodePPC64RotateMask(r,rotateLeft32(m,r),32)] x)
	for {
		r := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpPPC64MOVDconst {
				continue
			}
			m := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_1
			if !(isPPC64WordRotateMask(m)) {
				continue
			}
			v.reset(OpPPC64RLWINM)
			v.AuxInt = int64ToAuxInt(encodePPC64RotateMask(r, rotateLeft32(m, r), 32))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ROTLWconst [r] (Select0 (ANDCCconst [m] x)))
	// cond: isPPC64WordRotateMask(m)
	// result: (RLWINM [encodePPC64RotateMask(r,rotateLeft32(m,r),32)] x)
	for {
		r := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpSelect0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64ANDCCconst {
			break
		}
		m := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if !(isPPC64WordRotateMask(m)) {
			break
		}
		v.reset(OpPPC64RLWINM)
		v.AuxInt = int64ToAuxInt(encodePPC64RotateMask(r, rotateLeft32(m, r), 32))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SETBC(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETBC [0] (FlagLT))
	// result: (MOVDconst [1])
	for {
		if auxIntToInt32(v.AuxInt) != 0 || v_0.Op != OpPPC64FlagLT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SETBC [0] (FlagGT))
	// result: (MOVDconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 0 || v_0.Op != OpPPC64FlagGT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SETBC [0] (FlagEQ))
	// result: (MOVDconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 0 || v_0.Op != OpPPC64FlagEQ {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SETBC [1] (FlagGT))
	// result: (MOVDconst [1])
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpPPC64FlagGT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SETBC [1] (FlagLT))
	// result: (MOVDconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpPPC64FlagLT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SETBC [1] (FlagEQ))
	// result: (MOVDconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpPPC64FlagEQ {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SETBC [2] (FlagEQ))
	// result: (MOVDconst [1])
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64FlagEQ {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SETBC [2] (FlagLT))
	// result: (MOVDconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64FlagLT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SETBC [2] (FlagGT))
	// result: (MOVDconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64FlagGT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SETBC [0] (InvertFlags bool))
	// result: (SETBC [1] bool)
	for {
		if auxIntToInt32(v.AuxInt) != 0 || v_0.Op != OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(1)
		v.AddArg(bool)
		return true
	}
	// match: (SETBC [1] (InvertFlags bool))
	// result: (SETBC [0] bool)
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(0)
		v.AddArg(bool)
		return true
	}
	// match: (SETBC [2] (InvertFlags bool))
	// result: (SETBC [2] bool)
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(2)
		v.AddArg(bool)
		return true
	}
	// match: (SETBC [n] (InvertFlags bool))
	// result: (SETBCR [n] bool)
	for {
		n := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.reset(OpPPC64SETBCR)
		v.AuxInt = int32ToAuxInt(n)
		v.AddArg(bool)
		return true
	}
	// match: (SETBC [2] (CMPconst [0] (Select0 (ANDCCconst [1] z))))
	// result: (XORconst [1] (Select0 <typ.UInt64> (ANDCCconst [1] z )))
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSelect0 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
			break
		}
		z := v_0_0_0.Args[0]
		v.reset(OpPPC64XORconst)
		v.AuxInt = int64ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AuxInt = int64ToAuxInt(1)
		v1.AddArg(z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SETBC [2] (CMPWconst [0] (Select0 (ANDCCconst [1] z))))
	// result: (XORconst [1] (Select0 <typ.UInt64> (ANDCCconst [1] z )))
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSelect0 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
			break
		}
		z := v_0_0_0.Args[0]
		v.reset(OpPPC64XORconst)
		v.AuxInt = int64ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AuxInt = int64ToAuxInt(1)
		v1.AddArg(z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SETBC [2] (CMPWconst [0] (Select0 (ANDCCconst [n] z))))
	// result: (SETBC [2] (Select1 <types.TypeFlags> (ANDCCconst [n] z )))
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSelect0 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpPPC64ANDCCconst {
			break
		}
		n := auxIntToInt64(v_0_0_0.AuxInt)
		z := v_0_0_0.Args[0]
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AuxInt = int64ToAuxInt(n)
		v1.AddArg(z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SETBC [2] (CMPconst [0] a:(AND y z)))
	// cond: a.Uses == 1
	// result: (SETBC [2] (Select1 <types.TypeFlags> (ANDCC y z )))
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		a := v_0.Args[0]
		if a.Op != OpPPC64AND {
			break
		}
		z := a.Args[1]
		y := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
		v1.AddArg2(y, z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SETBC [2] (CMPconst [0] o:(OR y z)))
	// cond: o.Uses == 1
	// result: (SETBC [2] (Select1 <types.TypeFlags> (ORCC y z )))
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		o := v_0.Args[0]
		if o.Op != OpPPC64OR {
			break
		}
		z := o.Args[1]
		y := o.Args[0]
		if !(o.Uses == 1) {
			break
		}
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AddArg2(y, z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SETBC [2] (CMPconst [0] a:(XOR y z)))
	// cond: a.Uses == 1
	// result: (SETBC [2] (Select1 <types.TypeFlags> (XORCC y z )))
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		a := v_0.Args[0]
		if a.Op != OpPPC64XOR {
			break
		}
		z := a.Args[1]
		y := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AddArg2(y, z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SETBCR(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETBCR [0] (FlagLT))
	// result: (MOVDconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 0 || v_0.Op != OpPPC64FlagLT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SETBCR [0] (FlagGT))
	// result: (MOVDconst [1])
	for {
		if auxIntToInt32(v.AuxInt) != 0 || v_0.Op != OpPPC64FlagGT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SETBCR [0] (FlagEQ))
	// result: (MOVDconst [1])
	for {
		if auxIntToInt32(v.AuxInt) != 0 || v_0.Op != OpPPC64FlagEQ {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SETBCR [1] (FlagGT))
	// result: (MOVDconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpPPC64FlagGT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SETBCR [1] (FlagLT))
	// result: (MOVDconst [1])
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpPPC64FlagLT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SETBCR [1] (FlagEQ))
	// result: (MOVDconst [1])
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpPPC64FlagEQ {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SETBCR [2] (FlagEQ))
	// result: (MOVDconst [0])
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64FlagEQ {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SETBCR [2] (FlagLT))
	// result: (MOVDconst [1])
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64FlagLT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SETBCR [2] (FlagGT))
	// result: (MOVDconst [1])
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64FlagGT {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SETBCR [0] (InvertFlags bool))
	// result: (SETBCR [1] bool)
	for {
		if auxIntToInt32(v.AuxInt) != 0 || v_0.Op != OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.reset(OpPPC64SETBCR)
		v.AuxInt = int32ToAuxInt(1)
		v.AddArg(bool)
		return true
	}
	// match: (SETBCR [1] (InvertFlags bool))
	// result: (SETBCR [0] bool)
	for {
		if auxIntToInt32(v.AuxInt) != 1 || v_0.Op != OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.reset(OpPPC64SETBCR)
		v.AuxInt = int32ToAuxInt(0)
		v.AddArg(bool)
		return true
	}
	// match: (SETBCR [2] (InvertFlags bool))
	// result: (SETBCR [2] bool)
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.reset(OpPPC64SETBCR)
		v.AuxInt = int32ToAuxInt(2)
		v.AddArg(bool)
		return true
	}
	// match: (SETBCR [n] (InvertFlags bool))
	// result: (SETBC [n] bool)
	for {
		n := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(n)
		v.AddArg(bool)
		return true
	}
	// match: (SETBCR [2] (CMPconst [0] (Select0 (ANDCCconst [1] z))))
	// result: (Select0 <typ.UInt64> (ANDCCconst [1] z ))
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSelect0 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
			break
		}
		z := v_0_0_0.Args[0]
		v.reset(OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v0.AuxInt = int64ToAuxInt(1)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (SETBCR [2] (CMPWconst [0] (Select0 (ANDCCconst [1] z))))
	// result: (Select0 <typ.UInt64> (ANDCCconst [1] z ))
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSelect0 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
			break
		}
		z := v_0_0_0.Args[0]
		v.reset(OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v0.AuxInt = int64ToAuxInt(1)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (SETBCR [2] (CMPWconst [0] (Select0 (ANDCCconst [n] z))))
	// result: (SETBCR [2] (Select1 <types.TypeFlags> (ANDCCconst [n] z )))
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSelect0 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpPPC64ANDCCconst {
			break
		}
		n := auxIntToInt64(v_0_0_0.AuxInt)
		z := v_0_0_0.Args[0]
		v.reset(OpPPC64SETBCR)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AuxInt = int64ToAuxInt(n)
		v1.AddArg(z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SETBCR [2] (CMPconst [0] a:(AND y z)))
	// cond: a.Uses == 1
	// result: (SETBCR [2] (Select1 <types.TypeFlags> (ANDCC y z )))
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		a := v_0.Args[0]
		if a.Op != OpPPC64AND {
			break
		}
		z := a.Args[1]
		y := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpPPC64SETBCR)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
		v1.AddArg2(y, z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SETBCR [2] (CMPconst [0] o:(OR y z)))
	// cond: o.Uses == 1
	// result: (SETBCR [2] (Select1 <types.TypeFlags> (ORCC y z )))
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		o := v_0.Args[0]
		if o.Op != OpPPC64OR {
			break
		}
		z := o.Args[1]
		y := o.Args[0]
		if !(o.Uses == 1) {
			break
		}
		v.reset(OpPPC64SETBCR)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AddArg2(y, z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SETBCR [2] (CMPconst [0] a:(XOR y z)))
	// cond: a.Uses == 1
	// result: (SETBCR [2] (Select1 <types.TypeFlags> (XORCC y z )))
	for {
		if auxIntToInt32(v.AuxInt) != 2 || v_0.Op != OpPPC64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		a := v_0.Args[0]
		if a.Op != OpPPC64XOR {
			break
		}
		z := a.Args[1]
		y := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.reset(OpPPC64SETBCR)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AddArg2(y, z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SLD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLD x (MOVDconst [c]))
	// result: (SLDconst [c&63 | (c>>6&1*63)] x)
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpPPC64SLDconst)
		v.AuxInt = int64ToAuxInt(c&63 | (c >> 6 & 1 * 63))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SLDconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SLDconst [l] (SRWconst [r] x))
	// cond: mergePPC64SldiSrw(l,r) != 0
	// result: (RLWINM [mergePPC64SldiSrw(l,r)] x)
	for {
		l := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64SRWconst {
			break
		}
		r := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(mergePPC64SldiSrw(l, r) != 0) {
			break
		}
		v.reset(OpPPC64RLWINM)
		v.AuxInt = int64ToAuxInt(mergePPC64SldiSrw(l, r))
		v.AddArg(x)
		return true
	}
	// match: (SLDconst [c] z:(MOVBZreg x))
	// cond: c < 8 && z.Uses == 1
	// result: (CLRLSLDI [newPPC64ShiftAuxInt(c,56,63,64)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != OpPPC64MOVBZreg {
			break
		}
		x := z.Args[0]
		if !(c < 8 && z.Uses == 1) {
			break
		}
		v.reset(OpPPC64CLRLSLDI)
		v.AuxInt = int32ToAuxInt(newPPC64ShiftAuxInt(c, 56, 63, 64))
		v.AddArg(x)
		return true
	}
	// match: (SLDconst [c] z:(MOVHZreg x))
	// cond: c < 16 && z.Uses == 1
	// result: (CLRLSLDI [newPPC64ShiftAuxInt(c,48,63,64)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != OpPPC64MOVHZreg {
			break
		}
		x := z.Args[0]
		if !(c < 16 && z.Uses == 1) {
			break
		}
		v.reset(OpPPC64CLRLSLDI)
		v.AuxInt = int32ToAuxInt(newPPC64ShiftAuxInt(c, 48, 63, 64))
		v.AddArg(x)
		return true
	}
	// match: (SLDconst [c] z:(MOVWZreg x))
	// cond: c < 32 && z.Uses == 1
	// result: (CLRLSLDI [newPPC64ShiftAuxInt(c,32,63,64)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != OpPPC64MOVWZreg {
			break
		}
		x := z.Args[0]
		if !(c < 32 && z.Uses == 1) {
			break
		}
		v.reset(OpPPC64CLRLSLDI)
		v.AuxInt = int32ToAuxInt(newPPC64ShiftAuxInt(c, 32, 63, 64))
		v.AddArg(x)
		return true
	}
	// match: (SLDconst [c] z:(Select0 (ANDCCconst [d] x)))
	// cond: z.Uses == 1 && isPPC64ValidShiftMask(d) && c <= (64-getPPC64ShiftMaskLength(d))
	// result: (CLRLSLDI [newPPC64ShiftAuxInt(c,64-getPPC64ShiftMaskLength(d),63,64)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != OpSelect0 {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != OpPPC64ANDCCconst {
			break
		}
		d := auxIntToInt64(z_0.AuxInt)
		x := z_0.Args[0]
		if !(z.Uses == 1 && isPPC64ValidShiftMask(d) && c <= (64-getPPC64ShiftMaskLength(d))) {
			break
		}
		v.reset(OpPPC64CLRLSLDI)
		v.AuxInt = int32ToAuxInt(newPPC64ShiftAuxInt(c, 64-getPPC64ShiftMaskLength(d), 63, 64))
		v.AddArg(x)
		return true
	}
	// match: (SLDconst [c] z:(AND (MOVDconst [d]) x))
	// cond: z.Uses == 1 && isPPC64ValidShiftMask(d) && c<=(64-getPPC64ShiftMaskLength(d))
	// result: (CLRLSLDI [newPPC64ShiftAuxInt(c,64-getPPC64ShiftMaskLength(d),63,64)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != OpPPC64AND {
			break
		}
		_ = z.Args[1]
		z_0 := z.Args[0]
		z_1 := z.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
			if z_0.Op != OpPPC64MOVDconst {
				continue
			}
			d := auxIntToInt64(z_0.AuxInt)
			x := z_1
			if !(z.Uses == 1 && isPPC64ValidShiftMask(d) && c <= (64-getPPC64ShiftMaskLength(d))) {
				continue
			}
			v.reset(OpPPC64CLRLSLDI)
			v.AuxInt = int32ToAuxInt(newPPC64ShiftAuxInt(c, 64-getPPC64ShiftMaskLength(d), 63, 64))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (SLDconst [c] z:(MOVWreg x))
	// cond: c < 32 && buildcfg.GOPPC64 >= 9
	// result: (EXTSWSLconst [c] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != OpPPC64MOVWreg {
			break
		}
		x := z.Args[0]
		if !(c < 32 && buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64EXTSWSLconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SLW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLW x (MOVDconst [c]))
	// result: (SLWconst [c&31 | (c>>5&1*31)] x)
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpPPC64SLWconst)
		v.AuxInt = int64ToAuxInt(c&31 | (c >> 5 & 1 * 31))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SLWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SLWconst [c] z:(MOVBZreg x))
	// cond: z.Uses == 1 && c < 8
	// result: (CLRLSLWI [newPPC64ShiftAuxInt(c,24,31,32)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != OpPPC64MOVBZreg {
			break
		}
		x := z.Args[0]
		if !(z.Uses == 1 && c < 8) {
			break
		}
		v.reset(OpPPC64CLRLSLWI)
		v.AuxInt = int32ToAuxInt(newPPC64ShiftAuxInt(c, 24, 31, 32))
		v.AddArg(x)
		return true
	}
	// match: (SLWconst [c] z:(MOVHZreg x))
	// cond: z.Uses == 1 && c < 16
	// result: (CLRLSLWI [newPPC64ShiftAuxInt(c,16,31,32)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != OpPPC64MOVHZreg {
			break
		}
		x := z.Args[0]
		if !(z.Uses == 1 && c < 16) {
			break
		}
		v.reset(OpPPC64CLRLSLWI)
		v.AuxInt = int32ToAuxInt(newPPC64ShiftAuxInt(c, 16, 31, 32))
		v.AddArg(x)
		return true
	}
	// match: (SLWconst [c] z:(Select0 (ANDCCconst [d] x)))
	// cond: z.Uses == 1 && isPPC64ValidShiftMask(d) && c<=(32-getPPC64ShiftMaskLength(d))
	// result: (CLRLSLWI [newPPC64ShiftAuxInt(c,32-getPPC64ShiftMaskLength(d),31,32)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != OpSelect0 {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != OpPPC64ANDCCconst {
			break
		}
		d := auxIntToInt64(z_0.AuxInt)
		x := z_0.Args[0]
		if !(z.Uses == 1 && isPPC64ValidShiftMask(d) && c <= (32-getPPC64ShiftMaskLength(d))) {
			break
		}
		v.reset(OpPPC64CLRLSLWI)
		v.AuxInt = int32ToAuxInt(newPPC64ShiftAuxInt(c, 32-getPPC64ShiftMaskLength(d), 31, 32))
		v.AddArg(x)
		return true
	}
	// match: (SLWconst [c] z:(AND (MOVDconst [d]) x))
	// cond: z.Uses == 1 && isPPC64ValidShiftMask(d) && c<=(32-getPPC64ShiftMaskLength(d))
	// result: (CLRLSLWI [newPPC64ShiftAuxInt(c,32-getPPC64ShiftMaskLength(d),31,32)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != OpPPC64AND {
			break
		}
		_ = z.Args[1]
		z_0 := z.Args[0]
		z_1 := z.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
			if z_0.Op != OpPPC64MOVDconst {
				continue
			}
			d := auxIntToInt64(z_0.AuxInt)
			x := z_1
			if !(z.Uses == 1 && isPPC64ValidShiftMask(d) && c <= (32-getPPC64ShiftMaskLength(d))) {
				continue
			}
			v.reset(OpPPC64CLRLSLWI)
			v.AuxInt = int32ToAuxInt(newPPC64ShiftAuxInt(c, 32-getPPC64ShiftMaskLength(d), 31, 32))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (SLWconst [c] z:(MOVWreg x))
	// cond: c < 32 && buildcfg.GOPPC64 >= 9
	// result: (EXTSWSLconst [c] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != OpPPC64MOVWreg {
			break
		}
		x := z.Args[0]
		if !(c < 32 && buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64EXTSWSLconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SRAD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRAD x (MOVDconst [c]))
	// result: (SRADconst [c&63 | (c>>6&1*63)] x)
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpPPC64SRADconst)
		v.AuxInt = int64ToAuxInt(c&63 | (c >> 6 & 1 * 63))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SRAW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRAW x (MOVDconst [c]))
	// result: (SRAWconst [c&31 | (c>>5&1*31)] x)
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c&31 | (c >> 5 & 1 * 31))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SRD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRD x (MOVDconst [c]))
	// result: (SRDconst [c&63 | (c>>6&1*63)] x)
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpPPC64SRDconst)
		v.AuxInt = int64ToAuxInt(c&63 | (c >> 6 & 1 * 63))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SRW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRW x (MOVDconst [c]))
	// result: (SRWconst [c&31 | (c>>5&1*31)] x)
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c&31 | (c >> 5 & 1 * 31))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SRWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SRWconst (Select0 (ANDCCconst [m] x)) [s])
	// cond: mergePPC64RShiftMask(m>>uint(s),s,32) == 0
	// result: (MOVDconst [0])
	for {
		s := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpSelect0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64ANDCCconst {
			break
		}
		m := auxIntToInt64(v_0_0.AuxInt)
		if !(mergePPC64RShiftMask(m>>uint(s), s, 32) == 0) {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SRWconst (Select0 (ANDCCconst [m] x)) [s])
	// cond: mergePPC64AndSrwi(m>>uint(s),s) != 0
	// result: (RLWINM [mergePPC64AndSrwi(m>>uint(s),s)] x)
	for {
		s := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpSelect0 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64ANDCCconst {
			break
		}
		m := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if !(mergePPC64AndSrwi(m>>uint(s), s) != 0) {
			break
		}
		v.reset(OpPPC64RLWINM)
		v.AuxInt = int64ToAuxInt(mergePPC64AndSrwi(m>>uint(s), s))
		v.AddArg(x)
		return true
	}
	// match: (SRWconst (AND (MOVDconst [m]) x) [s])
	// cond: mergePPC64RShiftMask(m>>uint(s),s,32) == 0
	// result: (MOVDconst [0])
	for {
		s := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpPPC64MOVDconst {
				continue
			}
			m := auxIntToInt64(v_0_0.AuxInt)
			if !(mergePPC64RShiftMask(m>>uint(s), s, 32) == 0) {
				continue
			}
			v.reset(OpPPC64MOVDconst)
			v.AuxInt = int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (SRWconst (AND (MOVDconst [m]) x) [s])
	// cond: mergePPC64AndSrwi(m>>uint(s),s) != 0
	// result: (RLWINM [mergePPC64AndSrwi(m>>uint(s),s)] x)
	for {
		s := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != OpPPC64MOVDconst {
				continue
			}
			m := auxIntToInt64(v_0_0.AuxInt)
			x := v_0_1
			if !(mergePPC64AndSrwi(m>>uint(s), s) != 0) {
				continue
			}
			v.reset(OpPPC64RLWINM)
			v.AuxInt = int64ToAuxInt(mergePPC64AndSrwi(m>>uint(s), s))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64SUB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUB x (MOVDconst [c]))
	// cond: is32Bit(-c)
	// result: (ADDconst [-c] x)
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(is32Bit(-c)) {
			break
		}
		v.reset(OpPPC64ADDconst)
		v.AuxInt = int64ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	// match: (SUB (MOVDconst [c]) x)
	// cond: is32Bit(c)
	// result: (SUBFCconst [c] x)
	for {
		if v_0.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpPPC64SUBFCconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SUBE(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SUBE x y (Select1 <typ.UInt64> (SUBCconst (MOVDconst [0]) [0])))
	// result: (SUBC x y)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpSelect1 || v_2.Type != typ.UInt64 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpPPC64SUBCconst || auxIntToInt64(v_2_0.AuxInt) != 0 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpPPC64MOVDconst || auxIntToInt64(v_2_0_0.AuxInt) != 0 {
			break
		}
		v.reset(OpPPC64SUBC)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SUBFCconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SUBFCconst [c] (NEG x))
	// result: (ADDconst [c] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64NEG {
			break
		}
		x := v_0.Args[0]
		v.reset(OpPPC64ADDconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUBFCconst [c] (SUBFCconst [d] x))
	// cond: is32Bit(c-d)
	// result: (ADDconst [c-d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64SUBFCconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(c - d)) {
			break
		}
		v.reset(OpPPC64ADDconst)
		v.AuxInt = int64ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	// match: (SUBFCconst [0] x)
	// result: (NEG x)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.reset(OpPPC64NEG)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64XOR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XOR (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [c^d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpPPC64MOVDconst {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpPPC64MOVDconst)
			v.AuxInt = int64ToAuxInt(c ^ d)
			return true
		}
		break
	}
	// match: (XOR x (MOVDconst [c]))
	// cond: isU32Bit(c)
	// result: (XORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpPPC64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isU32Bit(c)) {
				continue
			}
			v.reset(OpPPC64XORconst)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64XORconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (XORconst [c] (XORconst [d] x))
	// result: (XORconst [c^d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpPPC64XORconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpPPC64XORconst)
		v.AuxInt = int64ToAuxInt(c ^ d)
		v.AddArg(x)
		return true
	}
	// match: (XORconst [0] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (XORconst [1] (SETBCR [n] cmp))
	// result: (SETBC [n] cmp)
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpPPC64SETBCR {
			break
		}
		n := auxIntToInt32(v_0.AuxInt)
		cmp := v_0.Args[0]
		v.reset(OpPPC64SETBC)
		v.AuxInt = int32ToAuxInt(n)
		v.AddArg(cmp)
		return true
	}
	// match: (XORconst [1] (SETBC [n] cmp))
	// result: (SETBCR [n] cmp)
	for {
		if auxIntToInt64(v.AuxInt) != 1 || v_0.Op != OpPPC64SETBC {
			break
		}
		n := auxIntToInt32(v_0.AuxInt)
		cmp := v_0.Args[0]
		v.reset(OpPPC64SETBCR)
		v.AuxInt = int32ToAuxInt(n)
		v.AddArg(cmp)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPanicBounds(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PanicBounds [kind] x y mem)
	// cond: boundsABI(kind) == 0
	// result: (LoweredPanicBoundsA [kind] x y mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		x := v_0
		y := v_1
		mem := v_2
		if !(boundsABI(kind) == 0) {
			break
		}
		v.reset(OpPPC64LoweredPanicBoundsA)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg3(x, y, mem)
		return true
	}
	// match: (PanicBounds [kind] x y mem)
	// cond: boundsABI(kind) == 1
	// result: (LoweredPanicBoundsB [kind] x y mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		x := v_0
		y := v_1
		mem := v_2
		if !(boundsABI(kind) == 1) {
			break
		}
		v.reset(OpPPC64LoweredPanicBoundsB)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg3(x, y, mem)
		return true
	}
	// match: (PanicBounds [kind] x y mem)
	// cond: boundsABI(kind) == 2
	// result: (LoweredPanicBoundsC [kind] x y mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		x := v_0
		y := v_1
		mem := v_2
		if !(boundsABI(kind) == 2) {
			break
		}
		v.reset(OpPPC64LoweredPanicBoundsC)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg3(x, y, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPopCount16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount16 x)
	// result: (POPCNTW (MOVHZreg x))
	for {
		x := v_0
		v.reset(OpPPC64POPCNTW)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpPopCount32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount32 x)
	// result: (POPCNTW (MOVWZreg x))
	for {
		x := v_0
		v.reset(OpPPC64POPCNTW)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVWZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpPopCount8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount8 x)
	// result: (POPCNTB (MOVBZreg x))
	for {
		x := v_0
		v.reset(OpPPC64POPCNTB)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpPrefetchCache(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PrefetchCache ptr mem)
	// result: (DCBT ptr mem [0])
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpPPC64DCBT)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpPrefetchCacheStreamed(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PrefetchCacheStreamed ptr mem)
	// result: (DCBT ptr mem [16])
	for {
		ptr := v_0
		mem := v_1
		v.reset(OpPPC64DCBT)
		v.AuxInt = int64ToAuxInt(16)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpRotateLeft16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVDconst [c]))
	// result: (Or16 (Lsh16x64 <t> x (MOVDconst [c&15])) (Rsh16Ux64 <t> x (MOVDconst [-c&15])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpLsh16x64, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(c & 15)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpRsh16Ux64, t)
		v3 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v3.AuxInt = int64ToAuxInt(-c & 15)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValuePPC64_OpRotateLeft8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVDconst [c]))
	// result: (Or8 (Lsh8x64 <t> x (MOVDconst [c&7])) (Rsh8Ux64 <t> x (MOVDconst [-c&7])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpLsh8x64, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(c & 7)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpRsh8Ux64, t)
		v3 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v3.AuxInt = int64ToAuxInt(-c & 7)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValuePPC64_OpRsh16Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRD (MOVHZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux16 <t> x y)
	// result: (ISEL [2] (SRD <t> (MOVHZreg x) y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0xFFF0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v4.AuxInt = int64ToAuxInt(0xFFF0)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRD (MOVHZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux32 <t> x y)
	// result: (ISEL [0] (SRD <t> (MOVHZreg x) y) (MOVDconst [0]) (CMPWUconst y [16]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(16)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 x (MOVDconst [c]))
	// cond: uint64(c) < 16
	// result: (SRWconst (ZeroExt16to32 x) [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRD (MOVHZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux64 <t> x y)
	// result: (ISEL [0] (SRD <t> (MOVHZreg x) y) (MOVDconst [0]) (CMPUconst y [16]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(16)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRD (MOVHZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux8 <t> x y)
	// result: (ISEL [2] (SRD <t> (MOVHZreg x) y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0x00F0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v4.AuxInt = int64ToAuxInt(0x00F0)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAD (MOVHreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x16 <t> x y)
	// result: (ISEL [2] (SRAD <t> (MOVHreg x) y) (SRADconst <t> (MOVHreg x) [15]) (Select1 <types.TypeFlags> (ANDCCconst [0xFFF0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64SRADconst, t)
		v2.AuxInt = int64ToAuxInt(15)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v4.AuxInt = int64ToAuxInt(0xFFF0)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAD (MOVHreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x32 <t> x y)
	// result: (ISEL [0] (SRAD <t> (MOVHreg x) y) (SRADconst <t> (MOVHreg x) [15]) (CMPWUconst y [16]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64SRADconst, t)
		v2.AuxInt = int64ToAuxInt(15)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(16)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 x (MOVDconst [c]))
	// cond: uint64(c) >= 16
	// result: (SRAWconst (SignExt16to32 x) [63])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x64 x (MOVDconst [c]))
	// cond: uint64(c) < 16
	// result: (SRAWconst (SignExt16to32 x) [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAD (MOVHreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x64 <t> x y)
	// result: (ISEL [0] (SRAD <t> (MOVHreg x) y) (SRADconst <t> (MOVHreg x) [15]) (CMPUconst y [16]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64SRADconst, t)
		v2.AuxInt = int64ToAuxInt(15)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(16)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAD (MOVHreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x8 <t> x y)
	// result: (ISEL [2] (SRAD <t> (MOVHreg x) y) (SRADconst <t> (MOVHreg x) [15]) (Select1 <types.TypeFlags> (ANDCCconst [0x00F0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVHreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64SRADconst, t)
		v2.AuxInt = int64ToAuxInt(15)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v4.AuxInt = int64ToAuxInt(0x00F0)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh32Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32Ux16 <t> x y)
	// result: (ISEL [2] (SRW <t> x y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0xFFE0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(0xFFE0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32Ux32 <t> x y)
	// result: (ISEL [0] (SRW <t> x y) (MOVDconst [0]) (CMPWUconst y [32]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 x (MOVDconst [c]))
	// cond: uint64(c) < 32
	// result: (SRWconst x [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Rsh32Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32Ux64 <t> x y)
	// result: (ISEL [0] (SRW <t> x y) (MOVDconst [0]) (CMPUconst y [32]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32Ux8 <t> x y)
	// result: (ISEL [2] (SRW <t> x y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0x00E0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(0x00E0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32x16 <t> x y)
	// result: (ISEL [2] (SRAW <t> x y) (SRAWconst <t> x [31]) (Select1 <types.TypeFlags> (ANDCCconst [0xFFE0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64SRAWconst, t)
		v1.AuxInt = int64ToAuxInt(31)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(0xFFE0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32x32 <t> x y)
	// result: (ISEL [0] (SRAW <t> x y) (SRAWconst <t> x [31]) (CMPWUconst y [32]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64SRAWconst, t)
		v1.AuxInt = int64ToAuxInt(31)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x64 x (MOVDconst [c]))
	// cond: uint64(c) >= 32
	// result: (SRAWconst x [63])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(63)
		v.AddArg(x)
		return true
	}
	// match: (Rsh32x64 x (MOVDconst [c]))
	// cond: uint64(c) < 32
	// result: (SRAWconst x [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Rsh32x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32x64 <t> x y)
	// result: (ISEL [0] (SRAW <t> x y) (SRAWconst <t> x [31]) (CMPUconst y [32]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64SRAWconst, t)
		v1.AuxInt = int64ToAuxInt(31)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAW x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32x8 <t> x y)
	// result: (ISEL [2] (SRAW <t> x y) (SRAWconst <t> x [31]) (Select1 <types.TypeFlags> (ANDCCconst [0x00E0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64SRAWconst, t)
		v1.AuxInt = int64ToAuxInt(31)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(0x00E0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux16 <t> x y)
	// result: (ISEL [2] (SRD <t> x y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0xFFC0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(0xFFC0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux32 <t> x y)
	// result: (ISEL [0] (SRD <t> x y) (MOVDconst [0]) (CMPWUconst y [64]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux64 x (MOVDconst [c]))
	// cond: uint64(c) < 64
	// result: (SRDconst x [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.reset(OpPPC64SRDconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Rsh64Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux64 <t> x y)
	// result: (ISEL [0] (SRD <t> x y) (MOVDconst [0]) (CMPUconst y [64]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux8 <t> x y)
	// result: (ISEL [2] (SRD <t> x y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0x00C0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(0x00C0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x16 <t> x y)
	// result: (ISEL [2] (SRAD <t> x y) (SRADconst <t> x [63]) (Select1 <types.TypeFlags> (ANDCCconst [0xFFC0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64SRADconst, t)
		v1.AuxInt = int64ToAuxInt(63)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(0xFFC0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x32 <t> x y)
	// result: (ISEL [0] (SRAD <t> x y) (SRADconst <t> x [63]) (CMPWUconst y [64]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64SRADconst, t)
		v1.AuxInt = int64ToAuxInt(63)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v2.AuxInt = int32ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x64 x (MOVDconst [c]))
	// cond: uint64(c) >= 64
	// result: (SRADconst x [63])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpPPC64SRADconst)
		v.AuxInt = int64ToAuxInt(63)
		v.AddArg(x)
		return true
	}
	// match: (Rsh64x64 x (MOVDconst [c]))
	// cond: uint64(c) < 64
	// result: (SRADconst x [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.reset(OpPPC64SRADconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Rsh64x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x64 <t> x y)
	// result: (ISEL [0] (SRAD <t> x y) (SRADconst <t> x [63]) (CMPUconst y [64]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64SRADconst, t)
		v1.AuxInt = int64ToAuxInt(63)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAD x y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x8 <t> x y)
	// result: (ISEL [2] (SRAD <t> x y) (SRADconst <t> x [63]) (Select1 <types.TypeFlags> (ANDCCconst [0x00C0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpPPC64SRADconst, t)
		v1.AuxInt = int64ToAuxInt(63)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(0x00C0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh8Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRD (MOVBZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux16 <t> x y)
	// result: (ISEL [2] (SRD <t> (MOVBZreg x) y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0xFFF8] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v4.AuxInt = int64ToAuxInt(0xFFF8)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRD (MOVBZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux32 <t> x y)
	// result: (ISEL [0] (SRD <t> (MOVBZreg x) y) (MOVDconst [0]) (CMPWUconst y [8]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(8)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 x (MOVDconst [c]))
	// cond: uint64(c) < 8
	// result: (SRWconst (ZeroExt8to32 x) [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.reset(OpPPC64SRWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8Ux64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRD (MOVBZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux64 <t> x y)
	// result: (ISEL [0] (SRD <t> (MOVBZreg x) y) (MOVDconst [0]) (CMPUconst y [8]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(8)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRD (MOVBZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux8 <t> x y)
	// result: (ISEL [2] (SRD <t> (MOVBZreg x) y) (MOVDconst [0]) (Select1 <types.TypeFlags> (ANDCCconst [0x00F8] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v4.AuxInt = int64ToAuxInt(0x00F8)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAD (MOVBreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x16 <t> x y)
	// result: (ISEL [2] (SRAD <t> (MOVBreg x) y) (SRADconst <t> (MOVBreg x) [7]) (Select1 <types.TypeFlags> (ANDCCconst [0xFFF8] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVBreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64SRADconst, t)
		v2.AuxInt = int64ToAuxInt(7)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v4.AuxInt = int64ToAuxInt(0xFFF8)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAD (MOVBreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x32 <t> x y)
	// result: (ISEL [0] (SRAD <t> (MOVBreg x) y) (SRADconst <t> (MOVBreg x) [7]) (CMPWUconst y [8]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVBreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64SRADconst, t)
		v2.AuxInt = int64ToAuxInt(7)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpPPC64CMPWUconst, types.TypeFlags)
		v3.AuxInt = int32ToAuxInt(8)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 x (MOVDconst [c]))
	// cond: uint64(c) >= 8
	// result: (SRAWconst (SignExt8to32 x) [63])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8x64 x (MOVDconst [c]))
	// cond: uint64(c) < 8
	// result: (SRAWconst (SignExt8to32 x) [c])
	for {
		x := v_0
		if v_1.Op != OpPPC64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.reset(OpPPC64SRAWconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8x64 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAD (MOVBreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x64 <t> x y)
	// result: (ISEL [0] (SRAD <t> (MOVBreg x) y) (SRADconst <t> (MOVBreg x) [7]) (CMPUconst y [8]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVBreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64SRADconst, t)
		v2.AuxInt = int64ToAuxInt(7)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpPPC64CMPUconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(8)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 x y)
	// cond: shiftIsBounded(v)
	// result: (SRAD (MOVBreg x) y)
	for {
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVBreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x8 <t> x y)
	// result: (ISEL [2] (SRAD <t> (MOVBreg x) y) (SRADconst <t> (MOVBreg x) [7]) (Select1 <types.TypeFlags> (ANDCCconst [0x00F8] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpPPC64ISEL)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVBreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpPPC64SRADconst, t)
		v2.AuxInt = int64ToAuxInt(7)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v4.AuxInt = int64ToAuxInt(0x00F8)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpSelect0(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Mul64uhilo x y))
	// result: (MULHDU x y)
	for {
		if v_0.Op != OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpPPC64MULHDU)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select0 (Add64carry x y c))
	// result: (Select0 <typ.UInt64> (ADDE x y (Select1 <typ.UInt64> (ADDCconst c [-1]))))
	for {
		if v_0.Op != OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpPPC64ADDE, types.NewTuple(typ.UInt64, typ.UInt64))
		v1 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpPPC64ADDCconst, types.NewTuple(typ.UInt64, typ.UInt64))
		v2.AuxInt = int64ToAuxInt(-1)
		v2.AddArg(c)
		v1.AddArg(v2)
		v0.AddArg3(x, y, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 (Sub64borrow x y c))
	// result: (Select0 <typ.UInt64> (SUBE x y (Select1 <typ.UInt64> (SUBCconst c [0]))))
	for {
		if v_0.Op != OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpPPC64SUBE, types.NewTuple(typ.UInt64, typ.UInt64))
		v1 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpPPC64SUBCconst, types.NewTuple(typ.UInt64, typ.UInt64))
		v2.AuxInt = int64ToAuxInt(0)
		v2.AddArg(c)
		v1.AddArg(v2)
		v0.AddArg3(x, y, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 (ANDCCconst [m] (ROTLWconst [r] x)))
	// cond: isPPC64WordRotateMask(m)
	// result: (RLWINM [encodePPC64RotateMask(r,m,32)] x)
	for {
		if v_0.Op != OpPPC64ANDCCconst {
			break
		}
		m := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64ROTLWconst {
			break
		}
		r := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if !(isPPC64WordRotateMask(m)) {
			break
		}
		v.reset(OpPPC64RLWINM)
		v.AuxInt = int64ToAuxInt(encodePPC64RotateMask(r, m, 32))
		v.AddArg(x)
		return true
	}
	// match: (Select0 (ANDCCconst [m] (ROTLW x r)))
	// cond: isPPC64WordRotateMask(m)
	// result: (RLWNM [encodePPC64RotateMask(0,m,32)] x r)
	for {
		if v_0.Op != OpPPC64ANDCCconst {
			break
		}
		m := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64ROTLW {
			break
		}
		r := v_0_0.Args[1]
		x := v_0_0.Args[0]
		if !(isPPC64WordRotateMask(m)) {
			break
		}
		v.reset(OpPPC64RLWNM)
		v.AuxInt = int64ToAuxInt(encodePPC64RotateMask(0, m, 32))
		v.AddArg2(x, r)
		return true
	}
	// match: (Select0 (ANDCCconst [m] (SRWconst x [s])))
	// cond: mergePPC64RShiftMask(m,s,32) == 0
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpPPC64ANDCCconst {
			break
		}
		m := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64SRWconst {
			break
		}
		s := auxIntToInt64(v_0_0.AuxInt)
		if !(mergePPC64RShiftMask(m, s, 32) == 0) {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Select0 (ANDCCconst [m] (SRWconst x [s])))
	// cond: mergePPC64AndSrwi(m,s) != 0
	// result: (RLWINM [mergePPC64AndSrwi(m,s)] x)
	for {
		if v_0.Op != OpPPC64ANDCCconst {
			break
		}
		m := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64SRWconst {
			break
		}
		s := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if !(mergePPC64AndSrwi(m, s) != 0) {
			break
		}
		v.reset(OpPPC64RLWINM)
		v.AuxInt = int64ToAuxInt(mergePPC64AndSrwi(m, s))
		v.AddArg(x)
		return true
	}
	// match: (Select0 (ANDCCconst [-1] x))
	// result: x
	for {
		if v_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0.AuxInt) != -1 {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
		return true
	}
	// match: (Select0 (ANDCCconst [0] _))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpPPC64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (Select0 (ANDCCconst [c] y:(MOVBZreg _)))
	// cond: c&0xFF == 0xFF
	// result: y
	for {
		if v_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		y := v_0.Args[0]
		if y.Op != OpPPC64MOVBZreg || !(c&0xFF == 0xFF) {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (Select0 (ANDCCconst [0xFF] (MOVBreg x)))
	// result: (MOVBZreg x)
	for {
		if v_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0.AuxInt) != 0xFF {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVBreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64MOVBZreg)
		v.AddArg(x)
		return true
	}
	// match: (Select0 (ANDCCconst [c] y:(MOVHZreg _)))
	// cond: c&0xFFFF == 0xFFFF
	// result: y
	for {
		if v_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		y := v_0.Args[0]
		if y.Op != OpPPC64MOVHZreg || !(c&0xFFFF == 0xFFFF) {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (Select0 (ANDCCconst [0xFFFF] (MOVHreg x)))
	// result: (MOVHZreg x)
	for {
		if v_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0.AuxInt) != 0xFFFF {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVHreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpPPC64MOVHZreg)
		v.AddArg(x)
		return true
	}
	// match: (Select0 (ANDCCconst [c] (MOVBZreg x)))
	// result: (Select0 (ANDCCconst [c&0xFF] x))
	for {
		if v_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVBZreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v0.AuxInt = int64ToAuxInt(c & 0xFF)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 (ANDCCconst [c] (MOVHZreg x)))
	// result: (Select0 (ANDCCconst [c&0xFFFF] x))
	for {
		if v_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVHZreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v0.AuxInt = int64ToAuxInt(c & 0xFFFF)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 (ANDCCconst [c] (MOVWZreg x)))
	// result: (Select0 (ANDCCconst [c&0xFFFFFFFF] x))
	for {
		if v_0.Op != OpPPC64ANDCCconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpPPC64MOVWZreg {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
		v0.AuxInt = int64ToAuxInt(c & 0xFFFFFFFF)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 (ANDCCconst [1] z:(SRADconst [63] x)))
	// cond: z.Uses == 1
	// result: (SRDconst [63] x)
	for {
		if v_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpPPC64SRADconst || auxIntToInt64(z.AuxInt) != 63 {
			break
		}
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpPPC64SRDconst)
		v.AuxInt = int64ToAuxInt(63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpSelect1(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Mul64uhilo x y))
	// result: (MULLD x y)
	for {
		if v_0.Op != OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpPPC64MULLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select1 (Add64carry x y c))
	// result: (ADDZEzero (Select1 <typ.UInt64> (ADDE x y (Select1 <typ.UInt64> (ADDCconst c [-1])))))
	for {
		if v_0.Op != OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpPPC64ADDZEzero)
		v0 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpPPC64ADDE, types.NewTuple(typ.UInt64, typ.UInt64))
		v2 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v3 := b.NewValue0(v.Pos, OpPPC64ADDCconst, types.NewTuple(typ.UInt64, typ.UInt64))
		v3.AuxInt = int64ToAuxInt(-1)
		v3.AddArg(c)
		v2.AddArg(v3)
		v1.AddArg3(x, y, v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select1 (ADDCconst n:(ADDZEzero x) [-1]))
	// cond: n.Uses <= 2
	// result: x
	for {
		if v_0.Op != OpPPC64ADDCconst || auxIntToInt64(v_0.AuxInt) != -1 {
			break
		}
		n := v_0.Args[0]
		if n.Op != OpPPC64ADDZEzero {
			break
		}
		x := n.Args[0]
		if !(n.Uses <= 2) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Select1 (Sub64borrow x y c))
	// result: (NEG (SUBZEzero (Select1 <typ.UInt64> (SUBE x y (Select1 <typ.UInt64> (SUBCconst c [0]))))))
	for {
		if v_0.Op != OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpPPC64NEG)
		v0 := b.NewValue0(v.Pos, OpPPC64SUBZEzero, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpPPC64SUBE, types.NewTuple(typ.UInt64, typ.UInt64))
		v3 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v4 := b.NewValue0(v.Pos, OpPPC64SUBCconst, types.NewTuple(typ.UInt64, typ.UInt64))
		v4.AuxInt = int64ToAuxInt(0)
		v4.AddArg(c)
		v3.AddArg(v4)
		v2.AddArg3(x, y, v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select1 (SUBCconst n:(NEG (SUBZEzero x)) [0]))
	// cond: n.Uses <= 2
	// result: x
	for {
		if v_0.Op != OpPPC64SUBCconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		n := v_0.Args[0]
		if n.Op != OpPPC64NEG {
			break
		}
		n_0 := n.Args[0]
		if n_0.Op != OpPPC64SUBZEzero {
			break
		}
		x := n_0.Args[0]
		if !(n.Uses <= 2) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (Select1 (ANDCCconst [0] _))
	// result: (FlagEQ)
	for {
		if v_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpPPC64FlagEQ)
		return true
	}
	return false
}
func rewriteValuePPC64_OpSelectN(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SelectN [0] call:(CALLstatic {sym} s1:(MOVDstore _ (MOVDconst [sz]) s2:(MOVDstore _ src s3:(MOVDstore {t} _ dst mem)))))
	// cond: sz >= 0 && isSameCall(sym, "runtime.memmove") && s1.Uses == 1 && s2.Uses == 1 && s3.Uses == 1 && isInlinableMemmove(dst, src, sz, config) && clobber(s1, s2, s3, call)
	// result: (Move [sz] dst src mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpPPC64CALLstatic || len(call.Args) != 1 {
			break
		}
		sym := auxToCall(call.Aux)
		s1 := call.Args[0]
		if s1.Op != OpPPC64MOVDstore {
			break
		}
		_ = s1.Args[2]
		s1_1 := s1.Args[1]
		if s1_1.Op != OpPPC64MOVDconst {
			break
		}
		sz := auxIntToInt64(s1_1.AuxInt)
		s2 := s1.Args[2]
		if s2.Op != OpPPC64MOVDstore {
			break
		}
		_ = s2.Args[2]
		src := s2.Args[1]
		s3 := s2.Args[2]
		if s3.Op != OpPPC64MOVDstore {
			break
		}
		mem := s3.Args[2]
		dst := s3.Args[1]
		if !(sz >= 0 && isSameCall(sym, "runtime.memmove") && s1.Uses == 1 && s2.Uses == 1 && s3.Uses == 1 && isInlinableMemmove(dst, src, sz, config) && clobber(s1, s2, s3, call)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(sz)
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (SelectN [0] call:(CALLstatic {sym} dst src (MOVDconst [sz]) mem))
	// cond: sz >= 0 && isSameCall(sym, "runtime.memmove") && call.Uses == 1 && isInlinableMemmove(dst, src, sz, config) && clobber(call)
	// result: (Move [sz] dst src mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpPPC64CALLstatic || len(call.Args) != 4 {
			break
		}
		sym := auxToCall(call.Aux)
		mem := call.Args[3]
		dst := call.Args[0]
		src := call.Args[1]
		call_2 := call.Args[2]
		if call_2.Op != OpPPC64MOVDconst {
			break
		}
		sz := auxIntToInt64(call_2.AuxInt)
		if !(sz >= 0 && isSameCall(sym, "runtime.memmove") && call.Uses == 1 && isInlinableMemmove(dst, src, sz, config) && clobber(call)) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(sz)
		v.AddArg3(dst, src, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpSlicemask(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SRADconst (NEG <t> x) [63])
	for {
		t := v.Type
		x := v_0
		v.reset(OpPPC64SRADconst)
		v.AuxInt = int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, OpPPC64NEG, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpStore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && t.IsFloat()
	// result: (FMOVDstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && t.IsFloat()) {
			break
		}
		v.reset(OpPPC64FMOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && t.IsFloat()
	// result: (FMOVSstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && t.IsFloat()) {
			break
		}
		v.reset(OpPPC64FMOVSstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && !t.IsFloat()
	// result: (MOVDstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && !t.IsFloat()) {
			break
		}
		v.reset(OpPPC64MOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && !t.IsFloat()
	// result: (MOVWstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && !t.IsFloat()) {
			break
		}
		v.reset(OpPPC64MOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 2
	// result: (MOVHstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 2) {
			break
		}
		v.reset(OpPPC64MOVHstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 1
	// result: (MOVBstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 1) {
			break
		}
		v.reset(OpPPC64MOVBstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpTrunc16to8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc16to8 <t> x)
	// cond: t.IsSigned()
	// result: (MOVBreg x)
	for {
		t := v.Type
		x := v_0
		if !(t.IsSigned()) {
			break
		}
		v.reset(OpPPC64MOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (Trunc16to8 x)
	// result: (MOVBZreg x)
	for {
		x := v_0
		v.reset(OpPPC64MOVBZreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpTrunc32to16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc32to16 <t> x)
	// cond: t.IsSigned()
	// result: (MOVHreg x)
	for {
		t := v.Type
		x := v_0
		if !(t.IsSigned()) {
			break
		}
		v.reset(OpPPC64MOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to16 x)
	// result: (MOVHZreg x)
	for {
		x := v_0
		v.reset(OpPPC64MOVHZreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpTrunc32to8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc32to8 <t> x)
	// cond: t.IsSigned()
	// result: (MOVBreg x)
	for {
		t := v.Type
		x := v_0
		if !(t.IsSigned()) {
			break
		}
		v.reset(OpPPC64MOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to8 x)
	// result: (MOVBZreg x)
	for {
		x := v_0
		v.reset(OpPPC64MOVBZreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpTrunc64to16(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc64to16 <t> x)
	// cond: t.IsSigned()
	// result: (MOVHreg x)
	for {
		t := v.Type
		x := v_0
		if !(t.IsSigned()) {
			break
		}
		v.reset(OpPPC64MOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to16 x)
	// result: (MOVHZreg x)
	for {
		x := v_0
		v.reset(OpPPC64MOVHZreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpTrunc64to32(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc64to32 <t> x)
	// cond: t.IsSigned()
	// result: (MOVWreg x)
	for {
		t := v.Type
		x := v_0
		if !(t.IsSigned()) {
			break
		}
		v.reset(OpPPC64MOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 x)
	// result: (MOVWZreg x)
	for {
		x := v_0
		v.reset(OpPPC64MOVWZreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpTrunc64to8(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Trunc64to8 <t> x)
	// cond: t.IsSigned()
	// result: (MOVBreg x)
	for {
		t := v.Type
		x := v_0
		if !(t.IsSigned()) {
			break
		}
		v.reset(OpPPC64MOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to8 x)
	// result: (MOVBZreg x)
	for {
		x := v_0
		v.reset(OpPPC64MOVBZreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpZero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Zero [0] _ mem)
	// result: mem
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		mem := v_1
		v.copyOf(mem)
		return true
	}
	// match: (Zero [1] destptr mem)
	// result: (MOVBstorezero destptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 1 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpPPC64MOVBstorezero)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [2] destptr mem)
	// result: (MOVHstorezero destptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 2 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpPPC64MOVHstorezero)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [3] destptr mem)
	// result: (MOVBstorezero [2] destptr (MOVHstorezero destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 3 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpPPC64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHstorezero, types.TypeMem)
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [4] destptr mem)
	// result: (MOVWstorezero destptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpPPC64MOVWstorezero)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [5] destptr mem)
	// result: (MOVBstorezero [4] destptr (MOVWstorezero destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 5 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpPPC64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVWstorezero, types.TypeMem)
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [6] destptr mem)
	// result: (MOVHstorezero [4] destptr (MOVWstorezero destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 6 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpPPC64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVWstorezero, types.TypeMem)
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [7] destptr mem)
	// result: (MOVBstorezero [6] destptr (MOVHstorezero [4] destptr (MOVWstorezero destptr mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 7 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpPPC64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVHstorezero, types.TypeMem)
		v0.AuxInt = int32ToAuxInt(4)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVWstorezero, types.TypeMem)
		v1.AddArg2(destptr, mem)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [8] {t} destptr mem)
	// result: (MOVDstorezero destptr mem)
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpPPC64MOVDstorezero)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [12] {t} destptr mem)
	// result: (MOVWstorezero [8] destptr (MOVDstorezero [0] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 12 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpPPC64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDstorezero, types.TypeMem)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [16] {t} destptr mem)
	// result: (MOVDstorezero [8] destptr (MOVDstorezero [0] destptr mem))
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpPPC64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDstorezero, types.TypeMem)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [24] {t} destptr mem)
	// result: (MOVDstorezero [16] destptr (MOVDstorezero [8] destptr (MOVDstorezero [0] destptr mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 24 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpPPC64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(16)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDstorezero, types.TypeMem)
		v0.AuxInt = int32ToAuxInt(8)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDstorezero, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(0)
		v1.AddArg2(destptr, mem)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [32] {t} destptr mem)
	// result: (MOVDstorezero [24] destptr (MOVDstorezero [16] destptr (MOVDstorezero [8] destptr (MOVDstorezero [0] destptr mem))))
	for {
		if auxIntToInt64(v.AuxInt) != 32 {
			break
		}
		destptr := v_0
		mem := v_1
		v.reset(OpPPC64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(24)
		v0 := b.NewValue0(v.Pos, OpPPC64MOVDstorezero, types.TypeMem)
		v0.AuxInt = int32ToAuxInt(16)
		v1 := b.NewValue0(v.Pos, OpPPC64MOVDstorezero, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, OpPPC64MOVDstorezero, types.TypeMem)
		v2.AuxInt = int32ToAuxInt(0)
		v2.AddArg2(destptr, mem)
		v1.AddArg2(destptr, v2)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: buildcfg.GOPPC64 <= 8 && s < 64
	// result: (LoweredZeroShort [s] ptr mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(buildcfg.GOPPC64 <= 8 && s < 64) {
			break
		}
		v.reset(OpPPC64LoweredZeroShort)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: buildcfg.GOPPC64 <= 8
	// result: (LoweredZero [s] ptr mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(buildcfg.GOPPC64 <= 8) {
			break
		}
		v.reset(OpPPC64LoweredZero)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s < 128 && buildcfg.GOPPC64 >= 9
	// result: (LoweredQuadZeroShort [s] ptr mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s < 128 && buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64LoweredQuadZeroShort)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (LoweredQuadZero [s] ptr mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.reset(OpPPC64LoweredQuadZero)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteBlockPPC64(b *Block) bool {
	typ := &b.Func.Config.Types
	switch b.Kind {
	case BlockPPC64EQ:
		// match: (EQ (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpPPC64FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (EQ (FlagLT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpPPC64FlagLT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpPPC64FlagGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (InvertFlags cmp) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == OpPPC64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockPPC64EQ, cmp)
			return true
		}
		// match: (EQ (CMPconst [0] (Select0 z:(ANDCCconst [c] x))) yes no)
		// result: (EQ (Select1 <types.TypeFlags> z) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			z := v_0_0.Args[0]
			if z.Op != OpPPC64ANDCCconst {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(z)
			b.resetWithControl(BlockPPC64EQ, v0)
			return true
		}
		// match: (EQ (CMPWconst [0] (Select0 z:(ANDCCconst [c] x))) yes no)
		// result: (EQ (Select1 <types.TypeFlags> z) yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			z := v_0_0.Args[0]
			if z.Op != OpPPC64ANDCCconst {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(z)
			b.resetWithControl(BlockPPC64EQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (Select1 <types.TypeFlags> (ANDCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64AND {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPconst [0] z:(OR x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (Select1 <types.TypeFlags> (ORCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64OR {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPconst [0] z:(XOR x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (Select1 <types.TypeFlags> (XORCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64XOR {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64EQ, v0)
				return true
			}
			break
		}
	case BlockPPC64GE:
		// match: (GE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpPPC64FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (GE (FlagLT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpPPC64FlagLT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GE (FlagGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpPPC64FlagGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (GE (InvertFlags cmp) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == OpPPC64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockPPC64LE, cmp)
			return true
		}
		// match: (GE (CMPconst [0] (Select0 z:(ANDCCconst [c] x))) yes no)
		// result: (GE (Select1 <types.TypeFlags> z) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			z := v_0_0.Args[0]
			if z.Op != OpPPC64ANDCCconst {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(z)
			b.resetWithControl(BlockPPC64GE, v0)
			return true
		}
		// match: (GE (CMPWconst [0] (Select0 z:(ANDCCconst [c] x))) yes no)
		// result: (GE (Select1 <types.TypeFlags> z) yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			z := v_0_0.Args[0]
			if z.Op != OpPPC64ANDCCconst {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(z)
			b.resetWithControl(BlockPPC64GE, v0)
			return true
		}
		// match: (GE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (Select1 <types.TypeFlags> (ANDCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64AND {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64GE, v0)
				return true
			}
			break
		}
		// match: (GE (CMPconst [0] z:(OR x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (Select1 <types.TypeFlags> (ORCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64OR {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64GE, v0)
				return true
			}
			break
		}
		// match: (GE (CMPconst [0] z:(XOR x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (Select1 <types.TypeFlags> (XORCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64XOR {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64GE, v0)
				return true
			}
			break
		}
	case BlockPPC64GT:
		// match: (GT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpPPC64FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagLT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpPPC64FlagLT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpPPC64FlagGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (GT (InvertFlags cmp) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == OpPPC64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockPPC64LT, cmp)
			return true
		}
		// match: (GT (CMPconst [0] (Select0 z:(ANDCCconst [c] x))) yes no)
		// result: (GT (Select1 <types.TypeFlags> z) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			z := v_0_0.Args[0]
			if z.Op != OpPPC64ANDCCconst {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(z)
			b.resetWithControl(BlockPPC64GT, v0)
			return true
		}
		// match: (GT (CMPWconst [0] (Select0 z:(ANDCCconst [c] x))) yes no)
		// result: (GT (Select1 <types.TypeFlags> z) yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			z := v_0_0.Args[0]
			if z.Op != OpPPC64ANDCCconst {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(z)
			b.resetWithControl(BlockPPC64GT, v0)
			return true
		}
		// match: (GT (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (Select1 <types.TypeFlags> (ANDCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64AND {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64GT, v0)
				return true
			}
			break
		}
		// match: (GT (CMPconst [0] z:(OR x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (Select1 <types.TypeFlags> (ORCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64OR {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64GT, v0)
				return true
			}
			break
		}
		// match: (GT (CMPconst [0] z:(XOR x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (Select1 <types.TypeFlags> (XORCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64XOR {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64GT, v0)
				return true
			}
			break
		}
	case BlockIf:
		// match: (If (Equal cc) yes no)
		// result: (EQ cc yes no)
		for b.Controls[0].Op == OpPPC64Equal {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockPPC64EQ, cc)
			return true
		}
		// match: (If (NotEqual cc) yes no)
		// result: (NE cc yes no)
		for b.Controls[0].Op == OpPPC64NotEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockPPC64NE, cc)
			return true
		}
		// match: (If (LessThan cc) yes no)
		// result: (LT cc yes no)
		for b.Controls[0].Op == OpPPC64LessThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockPPC64LT, cc)
			return true
		}
		// match: (If (LessEqual cc) yes no)
		// result: (LE cc yes no)
		for b.Controls[0].Op == OpPPC64LessEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockPPC64LE, cc)
			return true
		}
		// match: (If (GreaterThan cc) yes no)
		// result: (GT cc yes no)
		for b.Controls[0].Op == OpPPC64GreaterThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockPPC64GT, cc)
			return true
		}
		// match: (If (GreaterEqual cc) yes no)
		// result: (GE cc yes no)
		for b.Controls[0].Op == OpPPC64GreaterEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockPPC64GE, cc)
			return true
		}
		// match: (If (FLessThan cc) yes no)
		// result: (FLT cc yes no)
		for b.Controls[0].Op == OpPPC64FLessThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockPPC64FLT, cc)
			return true
		}
		// match: (If (FLessEqual cc) yes no)
		// result: (FLE cc yes no)
		for b.Controls[0].Op == OpPPC64FLessEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockPPC64FLE, cc)
			return true
		}
		// match: (If (FGreaterThan cc) yes no)
		// result: (FGT cc yes no)
		for b.Controls[0].Op == OpPPC64FGreaterThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockPPC64FGT, cc)
			return true
		}
		// match: (If (FGreaterEqual cc) yes no)
		// result: (FGE cc yes no)
		for b.Controls[0].Op == OpPPC64FGreaterEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockPPC64FGE, cc)
			return true
		}
		// match: (If cond yes no)
		// result: (NE (CMPWconst [0] (Select0 <typ.UInt32> (ANDCCconst [1] cond))) yes no)
		for {
			cond := b.Controls[0]
			v0 := b.NewValue0(cond.Pos, OpPPC64CMPWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(0)
			v1 := b.NewValue0(cond.Pos, OpSelect0, typ.UInt32)
			v2 := b.NewValue0(cond.Pos, OpPPC64ANDCCconst, types.NewTuple(typ.Int, types.TypeFlags))
			v2.AuxInt = int64ToAuxInt(1)
			v2.AddArg(cond)
			v1.AddArg(v2)
			v0.AddArg(v1)
			b.resetWithControl(BlockPPC64NE, v0)
			return true
		}
	case BlockPPC64LE:
		// match: (LE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpPPC64FlagEQ {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LE (FlagLT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpPPC64FlagLT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LE (FlagGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpPPC64FlagGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LE (InvertFlags cmp) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == OpPPC64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockPPC64GE, cmp)
			return true
		}
		// match: (LE (CMPconst [0] (Select0 z:(ANDCCconst [c] x))) yes no)
		// result: (LE (Select1 <types.TypeFlags> z) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			z := v_0_0.Args[0]
			if z.Op != OpPPC64ANDCCconst {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(z)
			b.resetWithControl(BlockPPC64LE, v0)
			return true
		}
		// match: (LE (CMPWconst [0] (Select0 z:(ANDCCconst [c] x))) yes no)
		// result: (LE (Select1 <types.TypeFlags> z) yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			z := v_0_0.Args[0]
			if z.Op != OpPPC64ANDCCconst {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(z)
			b.resetWithControl(BlockPPC64LE, v0)
			return true
		}
		// match: (LE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (Select1 <types.TypeFlags> (ANDCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64AND {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64LE, v0)
				return true
			}
			break
		}
		// match: (LE (CMPconst [0] z:(OR x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (Select1 <types.TypeFlags> (ORCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64OR {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64LE, v0)
				return true
			}
			break
		}
		// match: (LE (CMPconst [0] z:(XOR x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (Select1 <types.TypeFlags> (XORCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64XOR {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64LE, v0)
				return true
			}
			break
		}
	case BlockPPC64LT:
		// match: (LT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpPPC64FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LT (FlagLT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpPPC64FlagLT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (LT (FlagGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpPPC64FlagGT {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LT (InvertFlags cmp) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == OpPPC64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockPPC64GT, cmp)
			return true
		}
		// match: (LT (CMPconst [0] (Select0 z:(ANDCCconst [c] x))) yes no)
		// result: (LT (Select1 <types.TypeFlags> z) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			z := v_0_0.Args[0]
			if z.Op != OpPPC64ANDCCconst {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(z)
			b.resetWithControl(BlockPPC64LT, v0)
			return true
		}
		// match: (LT (CMPWconst [0] (Select0 z:(ANDCCconst [c] x))) yes no)
		// result: (LT (Select1 <types.TypeFlags> z) yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			z := v_0_0.Args[0]
			if z.Op != OpPPC64ANDCCconst {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(z)
			b.resetWithControl(BlockPPC64LT, v0)
			return true
		}
		// match: (LT (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (Select1 <types.TypeFlags> (ANDCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64AND {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64LT, v0)
				return true
			}
			break
		}
		// match: (LT (CMPconst [0] z:(OR x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (Select1 <types.TypeFlags> (ORCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64OR {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64LT, v0)
				return true
			}
			break
		}
		// match: (LT (CMPconst [0] z:(XOR x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (Select1 <types.TypeFlags> (XORCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64XOR {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64LT, v0)
				return true
			}
			break
		}
	case BlockPPC64NE:
		// match: (NE (CMPWconst [0] (Select0 (ANDCCconst [1] (Equal cc)))) yes no)
		// result: (EQ cc yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0_0 := v_0_0_0.Args[0]
			if v_0_0_0_0.Op != OpPPC64Equal {
				break
			}
			cc := v_0_0_0_0.Args[0]
			b.resetWithControl(BlockPPC64EQ, cc)
			return true
		}
		// match: (NE (CMPWconst [0] (Select0 (ANDCCconst [1] (NotEqual cc)))) yes no)
		// result: (NE cc yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0_0 := v_0_0_0.Args[0]
			if v_0_0_0_0.Op != OpPPC64NotEqual {
				break
			}
			cc := v_0_0_0_0.Args[0]
			b.resetWithControl(BlockPPC64NE, cc)
			return true
		}
		// match: (NE (CMPWconst [0] (Select0 (ANDCCconst [1] (LessThan cc)))) yes no)
		// result: (LT cc yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0_0 := v_0_0_0.Args[0]
			if v_0_0_0_0.Op != OpPPC64LessThan {
				break
			}
			cc := v_0_0_0_0.Args[0]
			b.resetWithControl(BlockPPC64LT, cc)
			return true
		}
		// match: (NE (CMPWconst [0] (Select0 (ANDCCconst [1] (LessEqual cc)))) yes no)
		// result: (LE cc yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0_0 := v_0_0_0.Args[0]
			if v_0_0_0_0.Op != OpPPC64LessEqual {
				break
			}
			cc := v_0_0_0_0.Args[0]
			b.resetWithControl(BlockPPC64LE, cc)
			return true
		}
		// match: (NE (CMPWconst [0] (Select0 (ANDCCconst [1] (GreaterThan cc)))) yes no)
		// result: (GT cc yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0_0 := v_0_0_0.Args[0]
			if v_0_0_0_0.Op != OpPPC64GreaterThan {
				break
			}
			cc := v_0_0_0_0.Args[0]
			b.resetWithControl(BlockPPC64GT, cc)
			return true
		}
		// match: (NE (CMPWconst [0] (Select0 (ANDCCconst [1] (GreaterEqual cc)))) yes no)
		// result: (GE cc yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0_0 := v_0_0_0.Args[0]
			if v_0_0_0_0.Op != OpPPC64GreaterEqual {
				break
			}
			cc := v_0_0_0_0.Args[0]
			b.resetWithControl(BlockPPC64GE, cc)
			return true
		}
		// match: (NE (CMPWconst [0] (Select0 (ANDCCconst [1] (FLessThan cc)))) yes no)
		// result: (FLT cc yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0_0 := v_0_0_0.Args[0]
			if v_0_0_0_0.Op != OpPPC64FLessThan {
				break
			}
			cc := v_0_0_0_0.Args[0]
			b.resetWithControl(BlockPPC64FLT, cc)
			return true
		}
		// match: (NE (CMPWconst [0] (Select0 (ANDCCconst [1] (FLessEqual cc)))) yes no)
		// result: (FLE cc yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0_0 := v_0_0_0.Args[0]
			if v_0_0_0_0.Op != OpPPC64FLessEqual {
				break
			}
			cc := v_0_0_0_0.Args[0]
			b.resetWithControl(BlockPPC64FLE, cc)
			return true
		}
		// match: (NE (CMPWconst [0] (Select0 (ANDCCconst [1] (FGreaterThan cc)))) yes no)
		// result: (FGT cc yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0_0 := v_0_0_0.Args[0]
			if v_0_0_0_0.Op != OpPPC64FGreaterThan {
				break
			}
			cc := v_0_0_0_0.Args[0]
			b.resetWithControl(BlockPPC64FGT, cc)
			return true
		}
		// match: (NE (CMPWconst [0] (Select0 (ANDCCconst [1] (FGreaterEqual cc)))) yes no)
		// result: (FGE cc yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != OpPPC64ANDCCconst || auxIntToInt64(v_0_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0_0 := v_0_0_0.Args[0]
			if v_0_0_0_0.Op != OpPPC64FGreaterEqual {
				break
			}
			cc := v_0_0_0_0.Args[0]
			b.resetWithControl(BlockPPC64FGE, cc)
			return true
		}
		// match: (NE (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpPPC64FlagEQ {
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (NE (FlagLT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpPPC64FlagLT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (NE (FlagGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpPPC64FlagGT {
			b.Reset(BlockFirst)
			return true
		}
		// match: (NE (InvertFlags cmp) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == OpPPC64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockPPC64NE, cmp)
			return true
		}
		// match: (NE (CMPconst [0] (Select0 z:(ANDCCconst [c] x))) yes no)
		// result: (NE (Select1 <types.TypeFlags> z) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			z := v_0_0.Args[0]
			if z.Op != OpPPC64ANDCCconst {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(z)
			b.resetWithControl(BlockPPC64NE, v0)
			return true
		}
		// match: (NE (CMPWconst [0] (Select0 z:(ANDCCconst [c] x))) yes no)
		// result: (NE (Select1 <types.TypeFlags> z) yes no)
		for b.Controls[0].Op == OpPPC64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpSelect0 {
				break
			}
			z := v_0_0.Args[0]
			if z.Op != OpPPC64ANDCCconst {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
			v0.AddArg(z)
			b.resetWithControl(BlockPPC64NE, v0)
			return true
		}
		// match: (NE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (Select1 <types.TypeFlags> (ANDCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64AND {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64NE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPconst [0] z:(OR x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (Select1 <types.TypeFlags> (ORCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64OR {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64NE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPconst [0] z:(XOR x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (Select1 <types.TypeFlags> (XORCC x y)) yes no)
		for b.Controls[0].Op == OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpPPC64XOR {
				break
			}
			_ = z.Args[1]
			z_0 := z.Args[0]
			z_1 := z.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
				x := z_0
				y := z_1
				if !(z.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.resetWithControl(BlockPPC64NE, v0)
				return true
			}
			break
		}
	}
	return false
}
