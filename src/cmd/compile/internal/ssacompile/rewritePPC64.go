// Code generated from _gen/PPC64.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "internal/buildcfg"
import "math"
import "cmd/compile/internal/types"
import "cmd/compile/internal/ssa/block"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func rewriteValuePPC64(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpAbs:
		v.Op = ssaop.OpPPC64FABS
		return true
	case ssaop.OpAdd16:
		v.Op = ssaop.OpPPC64ADD
		return true
	case ssaop.OpAdd32:
		v.Op = ssaop.OpPPC64ADD
		return true
	case ssaop.OpAdd32F:
		v.Op = ssaop.OpPPC64FADDS
		return true
	case ssaop.OpAdd64:
		v.Op = ssaop.OpPPC64ADD
		return true
	case ssaop.OpAdd64F:
		v.Op = ssaop.OpPPC64FADD
		return true
	case ssaop.OpAdd8:
		v.Op = ssaop.OpPPC64ADD
		return true
	case ssaop.OpAddPtr:
		v.Op = ssaop.OpPPC64ADD
		return true
	case ssaop.OpAddr:
		return rewriteValuePPC64_OpAddr(v)
	case ssaop.OpAnd16:
		v.Op = ssaop.OpPPC64AND
		return true
	case ssaop.OpAnd32:
		v.Op = ssaop.OpPPC64AND
		return true
	case ssaop.OpAnd64:
		v.Op = ssaop.OpPPC64AND
		return true
	case ssaop.OpAnd8:
		v.Op = ssaop.OpPPC64AND
		return true
	case ssaop.OpAndB:
		v.Op = ssaop.OpPPC64AND
		return true
	case ssaop.OpAtomicAdd32:
		v.Op = ssaop.OpPPC64LoweredAtomicAdd32
		return true
	case ssaop.OpAtomicAdd64:
		v.Op = ssaop.OpPPC64LoweredAtomicAdd64
		return true
	case ssaop.OpAtomicAnd32:
		v.Op = ssaop.OpPPC64LoweredAtomicAnd32
		return true
	case ssaop.OpAtomicAnd8:
		v.Op = ssaop.OpPPC64LoweredAtomicAnd8
		return true
	case ssaop.OpAtomicCompareAndSwap32:
		return rewriteValuePPC64_OpAtomicCompareAndSwap32(v)
	case ssaop.OpAtomicCompareAndSwap64:
		return rewriteValuePPC64_OpAtomicCompareAndSwap64(v)
	case ssaop.OpAtomicCompareAndSwapRel32:
		return rewriteValuePPC64_OpAtomicCompareAndSwapRel32(v)
	case ssaop.OpAtomicExchange32:
		v.Op = ssaop.OpPPC64LoweredAtomicExchange32
		return true
	case ssaop.OpAtomicExchange64:
		v.Op = ssaop.OpPPC64LoweredAtomicExchange64
		return true
	case ssaop.OpAtomicExchange8:
		v.Op = ssaop.OpPPC64LoweredAtomicExchange8
		return true
	case ssaop.OpAtomicLoad32:
		return rewriteValuePPC64_OpAtomicLoad32(v)
	case ssaop.OpAtomicLoad64:
		return rewriteValuePPC64_OpAtomicLoad64(v)
	case ssaop.OpAtomicLoad8:
		return rewriteValuePPC64_OpAtomicLoad8(v)
	case ssaop.OpAtomicLoadAcq32:
		return rewriteValuePPC64_OpAtomicLoadAcq32(v)
	case ssaop.OpAtomicLoadAcq64:
		return rewriteValuePPC64_OpAtomicLoadAcq64(v)
	case ssaop.OpAtomicLoadPtr:
		return rewriteValuePPC64_OpAtomicLoadPtr(v)
	case ssaop.OpAtomicOr32:
		v.Op = ssaop.OpPPC64LoweredAtomicOr32
		return true
	case ssaop.OpAtomicOr8:
		v.Op = ssaop.OpPPC64LoweredAtomicOr8
		return true
	case ssaop.OpAtomicStore32:
		return rewriteValuePPC64_OpAtomicStore32(v)
	case ssaop.OpAtomicStore64:
		return rewriteValuePPC64_OpAtomicStore64(v)
	case ssaop.OpAtomicStore8:
		return rewriteValuePPC64_OpAtomicStore8(v)
	case ssaop.OpAtomicStoreRel32:
		return rewriteValuePPC64_OpAtomicStoreRel32(v)
	case ssaop.OpAtomicStoreRel64:
		return rewriteValuePPC64_OpAtomicStoreRel64(v)
	case ssaop.OpAvg64u:
		return rewriteValuePPC64_OpAvg64u(v)
	case ssaop.OpBitLen16:
		return rewriteValuePPC64_OpBitLen16(v)
	case ssaop.OpBitLen32:
		return rewriteValuePPC64_OpBitLen32(v)
	case ssaop.OpBitLen64:
		return rewriteValuePPC64_OpBitLen64(v)
	case ssaop.OpBitLen8:
		return rewriteValuePPC64_OpBitLen8(v)
	case ssaop.OpBswap16:
		return rewriteValuePPC64_OpBswap16(v)
	case ssaop.OpBswap32:
		return rewriteValuePPC64_OpBswap32(v)
	case ssaop.OpBswap64:
		return rewriteValuePPC64_OpBswap64(v)
	case ssaop.OpCeil:
		v.Op = ssaop.OpPPC64FCEIL
		return true
	case ssaop.OpClosureCall:
		v.Op = ssaop.OpPPC64CALLclosure
		return true
	case ssaop.OpCom16:
		return rewriteValuePPC64_OpCom16(v)
	case ssaop.OpCom32:
		return rewriteValuePPC64_OpCom32(v)
	case ssaop.OpCom64:
		return rewriteValuePPC64_OpCom64(v)
	case ssaop.OpCom8:
		return rewriteValuePPC64_OpCom8(v)
	case ssaop.OpCondSelect:
		return rewriteValuePPC64_OpCondSelect(v)
	case ssaop.OpConst16:
		return rewriteValuePPC64_OpConst16(v)
	case ssaop.OpConst32:
		return rewriteValuePPC64_OpConst32(v)
	case ssaop.OpConst32F:
		v.Op = ssaop.OpPPC64FMOVSconst
		return true
	case ssaop.OpConst64:
		return rewriteValuePPC64_OpConst64(v)
	case ssaop.OpConst64F:
		v.Op = ssaop.OpPPC64FMOVDconst
		return true
	case ssaop.OpConst8:
		return rewriteValuePPC64_OpConst8(v)
	case ssaop.OpConstBool:
		return rewriteValuePPC64_OpConstBool(v)
	case ssaop.OpConstNil:
		return rewriteValuePPC64_OpConstNil(v)
	case ssaop.OpCopysign:
		return rewriteValuePPC64_OpCopysign(v)
	case ssaop.OpCtz16:
		return rewriteValuePPC64_OpCtz16(v)
	case ssaop.OpCtz16NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCtz32:
		return rewriteValuePPC64_OpCtz32(v)
	case ssaop.OpCtz32NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCtz64:
		return rewriteValuePPC64_OpCtz64(v)
	case ssaop.OpCtz64NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCtz8:
		return rewriteValuePPC64_OpCtz8(v)
	case ssaop.OpCtz8NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCvt32Fto32:
		return rewriteValuePPC64_OpCvt32Fto32(v)
	case ssaop.OpCvt32Fto64:
		return rewriteValuePPC64_OpCvt32Fto64(v)
	case ssaop.OpCvt32Fto64F:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpCvt32to32F:
		return rewriteValuePPC64_OpCvt32to32F(v)
	case ssaop.OpCvt32to64F:
		return rewriteValuePPC64_OpCvt32to64F(v)
	case ssaop.OpCvt64Fto32:
		return rewriteValuePPC64_OpCvt64Fto32(v)
	case ssaop.OpCvt64Fto32F:
		v.Op = ssaop.OpPPC64FRSP
		return true
	case ssaop.OpCvt64Fto64:
		return rewriteValuePPC64_OpCvt64Fto64(v)
	case ssaop.OpCvt64to32F:
		return rewriteValuePPC64_OpCvt64to32F(v)
	case ssaop.OpCvt64to64F:
		return rewriteValuePPC64_OpCvt64to64F(v)
	case ssaop.OpCvtBoolToUint8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpDiv16:
		return rewriteValuePPC64_OpDiv16(v)
	case ssaop.OpDiv16u:
		return rewriteValuePPC64_OpDiv16u(v)
	case ssaop.OpDiv32:
		return rewriteValuePPC64_OpDiv32(v)
	case ssaop.OpDiv32F:
		v.Op = ssaop.OpPPC64FDIVS
		return true
	case ssaop.OpDiv32u:
		v.Op = ssaop.OpPPC64DIVWU
		return true
	case ssaop.OpDiv64:
		return rewriteValuePPC64_OpDiv64(v)
	case ssaop.OpDiv64F:
		v.Op = ssaop.OpPPC64FDIV
		return true
	case ssaop.OpDiv64u:
		v.Op = ssaop.OpPPC64DIVDU
		return true
	case ssaop.OpDiv8:
		return rewriteValuePPC64_OpDiv8(v)
	case ssaop.OpDiv8u:
		return rewriteValuePPC64_OpDiv8u(v)
	case ssaop.OpEq16:
		return rewriteValuePPC64_OpEq16(v)
	case ssaop.OpEq32:
		return rewriteValuePPC64_OpEq32(v)
	case ssaop.OpEq32F:
		return rewriteValuePPC64_OpEq32F(v)
	case ssaop.OpEq64:
		return rewriteValuePPC64_OpEq64(v)
	case ssaop.OpEq64F:
		return rewriteValuePPC64_OpEq64F(v)
	case ssaop.OpEq8:
		return rewriteValuePPC64_OpEq8(v)
	case ssaop.OpEqB:
		return rewriteValuePPC64_OpEqB(v)
	case ssaop.OpEqPtr:
		return rewriteValuePPC64_OpEqPtr(v)
	case ssaop.OpFMA:
		v.Op = ssaop.OpPPC64FMADD
		return true
	case ssaop.OpFloor:
		v.Op = ssaop.OpPPC64FFLOOR
		return true
	case ssaop.OpGetCallerPC:
		v.Op = ssaop.OpPPC64LoweredGetCallerPC
		return true
	case ssaop.OpGetCallerSP:
		v.Op = ssaop.OpPPC64LoweredGetCallerSP
		return true
	case ssaop.OpGetClosurePtr:
		v.Op = ssaop.OpPPC64LoweredGetClosurePtr
		return true
	case ssaop.OpHmul32:
		v.Op = ssaop.OpPPC64MULHW
		return true
	case ssaop.OpHmul32u:
		v.Op = ssaop.OpPPC64MULHWU
		return true
	case ssaop.OpHmul64:
		v.Op = ssaop.OpPPC64MULHD
		return true
	case ssaop.OpHmul64u:
		v.Op = ssaop.OpPPC64MULHDU
		return true
	case ssaop.OpInterCall:
		v.Op = ssaop.OpPPC64CALLinter
		return true
	case ssaop.OpIsInBounds:
		return rewriteValuePPC64_OpIsInBounds(v)
	case ssaop.OpIsNonNil:
		return rewriteValuePPC64_OpIsNonNil(v)
	case ssaop.OpIsSliceInBounds:
		return rewriteValuePPC64_OpIsSliceInBounds(v)
	case ssaop.OpLeq16:
		return rewriteValuePPC64_OpLeq16(v)
	case ssaop.OpLeq16U:
		return rewriteValuePPC64_OpLeq16U(v)
	case ssaop.OpLeq32:
		return rewriteValuePPC64_OpLeq32(v)
	case ssaop.OpLeq32F:
		return rewriteValuePPC64_OpLeq32F(v)
	case ssaop.OpLeq32U:
		return rewriteValuePPC64_OpLeq32U(v)
	case ssaop.OpLeq64:
		return rewriteValuePPC64_OpLeq64(v)
	case ssaop.OpLeq64F:
		return rewriteValuePPC64_OpLeq64F(v)
	case ssaop.OpLeq64U:
		return rewriteValuePPC64_OpLeq64U(v)
	case ssaop.OpLeq8:
		return rewriteValuePPC64_OpLeq8(v)
	case ssaop.OpLeq8U:
		return rewriteValuePPC64_OpLeq8U(v)
	case ssaop.OpLess16:
		return rewriteValuePPC64_OpLess16(v)
	case ssaop.OpLess16U:
		return rewriteValuePPC64_OpLess16U(v)
	case ssaop.OpLess32:
		return rewriteValuePPC64_OpLess32(v)
	case ssaop.OpLess32F:
		return rewriteValuePPC64_OpLess32F(v)
	case ssaop.OpLess32U:
		return rewriteValuePPC64_OpLess32U(v)
	case ssaop.OpLess64:
		return rewriteValuePPC64_OpLess64(v)
	case ssaop.OpLess64F:
		return rewriteValuePPC64_OpLess64F(v)
	case ssaop.OpLess64U:
		return rewriteValuePPC64_OpLess64U(v)
	case ssaop.OpLess8:
		return rewriteValuePPC64_OpLess8(v)
	case ssaop.OpLess8U:
		return rewriteValuePPC64_OpLess8U(v)
	case ssaop.OpLoad:
		return rewriteValuePPC64_OpLoad(v)
	case ssaop.OpLocalAddr:
		return rewriteValuePPC64_OpLocalAddr(v)
	case ssaop.OpLsh16x16:
		return rewriteValuePPC64_OpLsh16x16(v)
	case ssaop.OpLsh16x32:
		return rewriteValuePPC64_OpLsh16x32(v)
	case ssaop.OpLsh16x64:
		return rewriteValuePPC64_OpLsh16x64(v)
	case ssaop.OpLsh16x8:
		return rewriteValuePPC64_OpLsh16x8(v)
	case ssaop.OpLsh32x16:
		return rewriteValuePPC64_OpLsh32x16(v)
	case ssaop.OpLsh32x32:
		return rewriteValuePPC64_OpLsh32x32(v)
	case ssaop.OpLsh32x64:
		return rewriteValuePPC64_OpLsh32x64(v)
	case ssaop.OpLsh32x8:
		return rewriteValuePPC64_OpLsh32x8(v)
	case ssaop.OpLsh64x16:
		return rewriteValuePPC64_OpLsh64x16(v)
	case ssaop.OpLsh64x32:
		return rewriteValuePPC64_OpLsh64x32(v)
	case ssaop.OpLsh64x64:
		return rewriteValuePPC64_OpLsh64x64(v)
	case ssaop.OpLsh64x8:
		return rewriteValuePPC64_OpLsh64x8(v)
	case ssaop.OpLsh8x16:
		return rewriteValuePPC64_OpLsh8x16(v)
	case ssaop.OpLsh8x32:
		return rewriteValuePPC64_OpLsh8x32(v)
	case ssaop.OpLsh8x64:
		return rewriteValuePPC64_OpLsh8x64(v)
	case ssaop.OpLsh8x8:
		return rewriteValuePPC64_OpLsh8x8(v)
	case ssaop.OpMax32F:
		return rewriteValuePPC64_OpMax32F(v)
	case ssaop.OpMax64F:
		return rewriteValuePPC64_OpMax64F(v)
	case ssaop.OpMin32F:
		return rewriteValuePPC64_OpMin32F(v)
	case ssaop.OpMin64F:
		return rewriteValuePPC64_OpMin64F(v)
	case ssaop.OpMod16:
		return rewriteValuePPC64_OpMod16(v)
	case ssaop.OpMod16u:
		return rewriteValuePPC64_OpMod16u(v)
	case ssaop.OpMod32:
		return rewriteValuePPC64_OpMod32(v)
	case ssaop.OpMod32u:
		return rewriteValuePPC64_OpMod32u(v)
	case ssaop.OpMod64:
		return rewriteValuePPC64_OpMod64(v)
	case ssaop.OpMod64u:
		return rewriteValuePPC64_OpMod64u(v)
	case ssaop.OpMod8:
		return rewriteValuePPC64_OpMod8(v)
	case ssaop.OpMod8u:
		return rewriteValuePPC64_OpMod8u(v)
	case ssaop.OpMove:
		return rewriteValuePPC64_OpMove(v)
	case ssaop.OpMul16:
		v.Op = ssaop.OpPPC64MULLW
		return true
	case ssaop.OpMul32:
		v.Op = ssaop.OpPPC64MULLW
		return true
	case ssaop.OpMul32F:
		v.Op = ssaop.OpPPC64FMULS
		return true
	case ssaop.OpMul64:
		v.Op = ssaop.OpPPC64MULLD
		return true
	case ssaop.OpMul64F:
		v.Op = ssaop.OpPPC64FMUL
		return true
	case ssaop.OpMul8:
		v.Op = ssaop.OpPPC64MULLW
		return true
	case ssaop.OpNeg16:
		v.Op = ssaop.OpPPC64NEG
		return true
	case ssaop.OpNeg32:
		v.Op = ssaop.OpPPC64NEG
		return true
	case ssaop.OpNeg32F:
		v.Op = ssaop.OpPPC64FNEG
		return true
	case ssaop.OpNeg64:
		v.Op = ssaop.OpPPC64NEG
		return true
	case ssaop.OpNeg64F:
		v.Op = ssaop.OpPPC64FNEG
		return true
	case ssaop.OpNeg8:
		v.Op = ssaop.OpPPC64NEG
		return true
	case ssaop.OpNeq16:
		return rewriteValuePPC64_OpNeq16(v)
	case ssaop.OpNeq32:
		return rewriteValuePPC64_OpNeq32(v)
	case ssaop.OpNeq32F:
		return rewriteValuePPC64_OpNeq32F(v)
	case ssaop.OpNeq64:
		return rewriteValuePPC64_OpNeq64(v)
	case ssaop.OpNeq64F:
		return rewriteValuePPC64_OpNeq64F(v)
	case ssaop.OpNeq8:
		return rewriteValuePPC64_OpNeq8(v)
	case ssaop.OpNeqB:
		v.Op = ssaop.OpPPC64XOR
		return true
	case ssaop.OpNeqPtr:
		return rewriteValuePPC64_OpNeqPtr(v)
	case ssaop.OpNilCheck:
		v.Op = ssaop.OpPPC64LoweredNilCheck
		return true
	case ssaop.OpNot:
		return rewriteValuePPC64_OpNot(v)
	case ssaop.OpOffPtr:
		return rewriteValuePPC64_OpOffPtr(v)
	case ssaop.OpOr16:
		v.Op = ssaop.OpPPC64OR
		return true
	case ssaop.OpOr32:
		v.Op = ssaop.OpPPC64OR
		return true
	case ssaop.OpOr64:
		v.Op = ssaop.OpPPC64OR
		return true
	case ssaop.OpOr8:
		v.Op = ssaop.OpPPC64OR
		return true
	case ssaop.OpOrB:
		v.Op = ssaop.OpPPC64OR
		return true
	case ssaop.OpPPC64ADD:
		return rewriteValuePPC64_OpPPC64ADD(v)
	case ssaop.OpPPC64ADDC:
		return rewriteValuePPC64_OpPPC64ADDC(v)
	case ssaop.OpPPC64ADDE:
		return rewriteValuePPC64_OpPPC64ADDE(v)
	case ssaop.OpPPC64ADDconst:
		return rewriteValuePPC64_OpPPC64ADDconst(v)
	case ssaop.OpPPC64AND:
		return rewriteValuePPC64_OpPPC64AND(v)
	case ssaop.OpPPC64ANDN:
		return rewriteValuePPC64_OpPPC64ANDN(v)
	case ssaop.OpPPC64ANDconst:
		return rewriteValuePPC64_OpPPC64ANDconst(v)
	case ssaop.OpPPC64BRD:
		return rewriteValuePPC64_OpPPC64BRD(v)
	case ssaop.OpPPC64BRH:
		return rewriteValuePPC64_OpPPC64BRH(v)
	case ssaop.OpPPC64BRW:
		return rewriteValuePPC64_OpPPC64BRW(v)
	case ssaop.OpPPC64CLRLSLDI:
		return rewriteValuePPC64_OpPPC64CLRLSLDI(v)
	case ssaop.OpPPC64CMP:
		return rewriteValuePPC64_OpPPC64CMP(v)
	case ssaop.OpPPC64CMPU:
		return rewriteValuePPC64_OpPPC64CMPU(v)
	case ssaop.OpPPC64CMPUconst:
		return rewriteValuePPC64_OpPPC64CMPUconst(v)
	case ssaop.OpPPC64CMPW:
		return rewriteValuePPC64_OpPPC64CMPW(v)
	case ssaop.OpPPC64CMPWU:
		return rewriteValuePPC64_OpPPC64CMPWU(v)
	case ssaop.OpPPC64CMPWUconst:
		return rewriteValuePPC64_OpPPC64CMPWUconst(v)
	case ssaop.OpPPC64CMPWconst:
		return rewriteValuePPC64_OpPPC64CMPWconst(v)
	case ssaop.OpPPC64CMPconst:
		return rewriteValuePPC64_OpPPC64CMPconst(v)
	case ssaop.OpPPC64Equal:
		return rewriteValuePPC64_OpPPC64Equal(v)
	case ssaop.OpPPC64FABS:
		return rewriteValuePPC64_OpPPC64FABS(v)
	case ssaop.OpPPC64FADD:
		return rewriteValuePPC64_OpPPC64FADD(v)
	case ssaop.OpPPC64FADDS:
		return rewriteValuePPC64_OpPPC64FADDS(v)
	case ssaop.OpPPC64FCEIL:
		return rewriteValuePPC64_OpPPC64FCEIL(v)
	case ssaop.OpPPC64FFLOOR:
		return rewriteValuePPC64_OpPPC64FFLOOR(v)
	case ssaop.OpPPC64FGreaterEqual:
		return rewriteValuePPC64_OpPPC64FGreaterEqual(v)
	case ssaop.OpPPC64FGreaterThan:
		return rewriteValuePPC64_OpPPC64FGreaterThan(v)
	case ssaop.OpPPC64FLessEqual:
		return rewriteValuePPC64_OpPPC64FLessEqual(v)
	case ssaop.OpPPC64FLessThan:
		return rewriteValuePPC64_OpPPC64FLessThan(v)
	case ssaop.OpPPC64FMOVDload:
		return rewriteValuePPC64_OpPPC64FMOVDload(v)
	case ssaop.OpPPC64FMOVDstore:
		return rewriteValuePPC64_OpPPC64FMOVDstore(v)
	case ssaop.OpPPC64FMOVSload:
		return rewriteValuePPC64_OpPPC64FMOVSload(v)
	case ssaop.OpPPC64FMOVSstore:
		return rewriteValuePPC64_OpPPC64FMOVSstore(v)
	case ssaop.OpPPC64FNEG:
		return rewriteValuePPC64_OpPPC64FNEG(v)
	case ssaop.OpPPC64FSQRT:
		return rewriteValuePPC64_OpPPC64FSQRT(v)
	case ssaop.OpPPC64FSUB:
		return rewriteValuePPC64_OpPPC64FSUB(v)
	case ssaop.OpPPC64FSUBS:
		return rewriteValuePPC64_OpPPC64FSUBS(v)
	case ssaop.OpPPC64FTRUNC:
		return rewriteValuePPC64_OpPPC64FTRUNC(v)
	case ssaop.OpPPC64GreaterEqual:
		return rewriteValuePPC64_OpPPC64GreaterEqual(v)
	case ssaop.OpPPC64GreaterThan:
		return rewriteValuePPC64_OpPPC64GreaterThan(v)
	case ssaop.OpPPC64ISEL:
		return rewriteValuePPC64_OpPPC64ISEL(v)
	case ssaop.OpPPC64LessEqual:
		return rewriteValuePPC64_OpPPC64LessEqual(v)
	case ssaop.OpPPC64LessThan:
		return rewriteValuePPC64_OpPPC64LessThan(v)
	case ssaop.OpPPC64LoweredPanicBoundsCR:
		return rewriteValuePPC64_OpPPC64LoweredPanicBoundsCR(v)
	case ssaop.OpPPC64LoweredPanicBoundsRC:
		return rewriteValuePPC64_OpPPC64LoweredPanicBoundsRC(v)
	case ssaop.OpPPC64LoweredPanicBoundsRR:
		return rewriteValuePPC64_OpPPC64LoweredPanicBoundsRR(v)
	case ssaop.OpPPC64MFVSRD:
		return rewriteValuePPC64_OpPPC64MFVSRD(v)
	case ssaop.OpPPC64MOVBZload:
		return rewriteValuePPC64_OpPPC64MOVBZload(v)
	case ssaop.OpPPC64MOVBZloadidx:
		return rewriteValuePPC64_OpPPC64MOVBZloadidx(v)
	case ssaop.OpPPC64MOVBZreg:
		return rewriteValuePPC64_OpPPC64MOVBZreg(v)
	case ssaop.OpPPC64MOVBreg:
		return rewriteValuePPC64_OpPPC64MOVBreg(v)
	case ssaop.OpPPC64MOVBstore:
		return rewriteValuePPC64_OpPPC64MOVBstore(v)
	case ssaop.OpPPC64MOVBstoreidx:
		return rewriteValuePPC64_OpPPC64MOVBstoreidx(v)
	case ssaop.OpPPC64MOVBstorezero:
		return rewriteValuePPC64_OpPPC64MOVBstorezero(v)
	case ssaop.OpPPC64MOVDaddr:
		return rewriteValuePPC64_OpPPC64MOVDaddr(v)
	case ssaop.OpPPC64MOVDload:
		return rewriteValuePPC64_OpPPC64MOVDload(v)
	case ssaop.OpPPC64MOVDloadidx:
		return rewriteValuePPC64_OpPPC64MOVDloadidx(v)
	case ssaop.OpPPC64MOVDstore:
		return rewriteValuePPC64_OpPPC64MOVDstore(v)
	case ssaop.OpPPC64MOVDstoreidx:
		return rewriteValuePPC64_OpPPC64MOVDstoreidx(v)
	case ssaop.OpPPC64MOVDstorezero:
		return rewriteValuePPC64_OpPPC64MOVDstorezero(v)
	case ssaop.OpPPC64MOVHBRstore:
		return rewriteValuePPC64_OpPPC64MOVHBRstore(v)
	case ssaop.OpPPC64MOVHZload:
		return rewriteValuePPC64_OpPPC64MOVHZload(v)
	case ssaop.OpPPC64MOVHZloadidx:
		return rewriteValuePPC64_OpPPC64MOVHZloadidx(v)
	case ssaop.OpPPC64MOVHZreg:
		return rewriteValuePPC64_OpPPC64MOVHZreg(v)
	case ssaop.OpPPC64MOVHload:
		return rewriteValuePPC64_OpPPC64MOVHload(v)
	case ssaop.OpPPC64MOVHloadidx:
		return rewriteValuePPC64_OpPPC64MOVHloadidx(v)
	case ssaop.OpPPC64MOVHreg:
		return rewriteValuePPC64_OpPPC64MOVHreg(v)
	case ssaop.OpPPC64MOVHstore:
		return rewriteValuePPC64_OpPPC64MOVHstore(v)
	case ssaop.OpPPC64MOVHstoreidx:
		return rewriteValuePPC64_OpPPC64MOVHstoreidx(v)
	case ssaop.OpPPC64MOVHstorezero:
		return rewriteValuePPC64_OpPPC64MOVHstorezero(v)
	case ssaop.OpPPC64MOVWBRstore:
		return rewriteValuePPC64_OpPPC64MOVWBRstore(v)
	case ssaop.OpPPC64MOVWZload:
		return rewriteValuePPC64_OpPPC64MOVWZload(v)
	case ssaop.OpPPC64MOVWZloadidx:
		return rewriteValuePPC64_OpPPC64MOVWZloadidx(v)
	case ssaop.OpPPC64MOVWZreg:
		return rewriteValuePPC64_OpPPC64MOVWZreg(v)
	case ssaop.OpPPC64MOVWload:
		return rewriteValuePPC64_OpPPC64MOVWload(v)
	case ssaop.OpPPC64MOVWloadidx:
		return rewriteValuePPC64_OpPPC64MOVWloadidx(v)
	case ssaop.OpPPC64MOVWreg:
		return rewriteValuePPC64_OpPPC64MOVWreg(v)
	case ssaop.OpPPC64MOVWstore:
		return rewriteValuePPC64_OpPPC64MOVWstore(v)
	case ssaop.OpPPC64MOVWstoreidx:
		return rewriteValuePPC64_OpPPC64MOVWstoreidx(v)
	case ssaop.OpPPC64MOVWstorezero:
		return rewriteValuePPC64_OpPPC64MOVWstorezero(v)
	case ssaop.OpPPC64MTVSRD:
		return rewriteValuePPC64_OpPPC64MTVSRD(v)
	case ssaop.OpPPC64MULLD:
		return rewriteValuePPC64_OpPPC64MULLD(v)
	case ssaop.OpPPC64MULLW:
		return rewriteValuePPC64_OpPPC64MULLW(v)
	case ssaop.OpPPC64NEG:
		return rewriteValuePPC64_OpPPC64NEG(v)
	case ssaop.OpPPC64NOR:
		return rewriteValuePPC64_OpPPC64NOR(v)
	case ssaop.OpPPC64NotEqual:
		return rewriteValuePPC64_OpPPC64NotEqual(v)
	case ssaop.OpPPC64OR:
		return rewriteValuePPC64_OpPPC64OR(v)
	case ssaop.OpPPC64ORN:
		return rewriteValuePPC64_OpPPC64ORN(v)
	case ssaop.OpPPC64ORconst:
		return rewriteValuePPC64_OpPPC64ORconst(v)
	case ssaop.OpPPC64RLWINM:
		return rewriteValuePPC64_OpPPC64RLWINM(v)
	case ssaop.OpPPC64ROTL:
		return rewriteValuePPC64_OpPPC64ROTL(v)
	case ssaop.OpPPC64ROTLW:
		return rewriteValuePPC64_OpPPC64ROTLW(v)
	case ssaop.OpPPC64ROTLWconst:
		return rewriteValuePPC64_OpPPC64ROTLWconst(v)
	case ssaop.OpPPC64SETBC:
		return rewriteValuePPC64_OpPPC64SETBC(v)
	case ssaop.OpPPC64SETBCR:
		return rewriteValuePPC64_OpPPC64SETBCR(v)
	case ssaop.OpPPC64SLD:
		return rewriteValuePPC64_OpPPC64SLD(v)
	case ssaop.OpPPC64SLDconst:
		return rewriteValuePPC64_OpPPC64SLDconst(v)
	case ssaop.OpPPC64SLW:
		return rewriteValuePPC64_OpPPC64SLW(v)
	case ssaop.OpPPC64SLWconst:
		return rewriteValuePPC64_OpPPC64SLWconst(v)
	case ssaop.OpPPC64SRAD:
		return rewriteValuePPC64_OpPPC64SRAD(v)
	case ssaop.OpPPC64SRAW:
		return rewriteValuePPC64_OpPPC64SRAW(v)
	case ssaop.OpPPC64SRD:
		return rewriteValuePPC64_OpPPC64SRD(v)
	case ssaop.OpPPC64SRW:
		return rewriteValuePPC64_OpPPC64SRW(v)
	case ssaop.OpPPC64SRWconst:
		return rewriteValuePPC64_OpPPC64SRWconst(v)
	case ssaop.OpPPC64SUB:
		return rewriteValuePPC64_OpPPC64SUB(v)
	case ssaop.OpPPC64SUBE:
		return rewriteValuePPC64_OpPPC64SUBE(v)
	case ssaop.OpPPC64SUBFCconst:
		return rewriteValuePPC64_OpPPC64SUBFCconst(v)
	case ssaop.OpPPC64XOR:
		return rewriteValuePPC64_OpPPC64XOR(v)
	case ssaop.OpPPC64XORconst:
		return rewriteValuePPC64_OpPPC64XORconst(v)
	case ssaop.OpPanicBounds:
		v.Op = ssaop.OpPPC64LoweredPanicBoundsRR
		return true
	case ssaop.OpPopCount16:
		return rewriteValuePPC64_OpPopCount16(v)
	case ssaop.OpPopCount32:
		return rewriteValuePPC64_OpPopCount32(v)
	case ssaop.OpPopCount64:
		v.Op = ssaop.OpPPC64POPCNTD
		return true
	case ssaop.OpPopCount8:
		return rewriteValuePPC64_OpPopCount8(v)
	case ssaop.OpPrefetchCache:
		return rewriteValuePPC64_OpPrefetchCache(v)
	case ssaop.OpPrefetchCacheStreamed:
		return rewriteValuePPC64_OpPrefetchCacheStreamed(v)
	case ssaop.OpPubBarrier:
		v.Op = ssaop.OpPPC64LoweredPubBarrier
		return true
	case ssaop.OpRotateLeft16:
		return rewriteValuePPC64_OpRotateLeft16(v)
	case ssaop.OpRotateLeft32:
		v.Op = ssaop.OpPPC64ROTLW
		return true
	case ssaop.OpRotateLeft64:
		v.Op = ssaop.OpPPC64ROTL
		return true
	case ssaop.OpRotateLeft8:
		return rewriteValuePPC64_OpRotateLeft8(v)
	case ssaop.OpRound:
		v.Op = ssaop.OpPPC64FROUND
		return true
	case ssaop.OpRound32F:
		v.Op = ssaop.OpPPC64LoweredRound32F
		return true
	case ssaop.OpRound64F:
		v.Op = ssaop.OpPPC64LoweredRound64F
		return true
	case ssaop.OpRsh16Ux16:
		return rewriteValuePPC64_OpRsh16Ux16(v)
	case ssaop.OpRsh16Ux32:
		return rewriteValuePPC64_OpRsh16Ux32(v)
	case ssaop.OpRsh16Ux64:
		return rewriteValuePPC64_OpRsh16Ux64(v)
	case ssaop.OpRsh16Ux8:
		return rewriteValuePPC64_OpRsh16Ux8(v)
	case ssaop.OpRsh16x16:
		return rewriteValuePPC64_OpRsh16x16(v)
	case ssaop.OpRsh16x32:
		return rewriteValuePPC64_OpRsh16x32(v)
	case ssaop.OpRsh16x64:
		return rewriteValuePPC64_OpRsh16x64(v)
	case ssaop.OpRsh16x8:
		return rewriteValuePPC64_OpRsh16x8(v)
	case ssaop.OpRsh32Ux16:
		return rewriteValuePPC64_OpRsh32Ux16(v)
	case ssaop.OpRsh32Ux32:
		return rewriteValuePPC64_OpRsh32Ux32(v)
	case ssaop.OpRsh32Ux64:
		return rewriteValuePPC64_OpRsh32Ux64(v)
	case ssaop.OpRsh32Ux8:
		return rewriteValuePPC64_OpRsh32Ux8(v)
	case ssaop.OpRsh32x16:
		return rewriteValuePPC64_OpRsh32x16(v)
	case ssaop.OpRsh32x32:
		return rewriteValuePPC64_OpRsh32x32(v)
	case ssaop.OpRsh32x64:
		return rewriteValuePPC64_OpRsh32x64(v)
	case ssaop.OpRsh32x8:
		return rewriteValuePPC64_OpRsh32x8(v)
	case ssaop.OpRsh64Ux16:
		return rewriteValuePPC64_OpRsh64Ux16(v)
	case ssaop.OpRsh64Ux32:
		return rewriteValuePPC64_OpRsh64Ux32(v)
	case ssaop.OpRsh64Ux64:
		return rewriteValuePPC64_OpRsh64Ux64(v)
	case ssaop.OpRsh64Ux8:
		return rewriteValuePPC64_OpRsh64Ux8(v)
	case ssaop.OpRsh64x16:
		return rewriteValuePPC64_OpRsh64x16(v)
	case ssaop.OpRsh64x32:
		return rewriteValuePPC64_OpRsh64x32(v)
	case ssaop.OpRsh64x64:
		return rewriteValuePPC64_OpRsh64x64(v)
	case ssaop.OpRsh64x8:
		return rewriteValuePPC64_OpRsh64x8(v)
	case ssaop.OpRsh8Ux16:
		return rewriteValuePPC64_OpRsh8Ux16(v)
	case ssaop.OpRsh8Ux32:
		return rewriteValuePPC64_OpRsh8Ux32(v)
	case ssaop.OpRsh8Ux64:
		return rewriteValuePPC64_OpRsh8Ux64(v)
	case ssaop.OpRsh8Ux8:
		return rewriteValuePPC64_OpRsh8Ux8(v)
	case ssaop.OpRsh8x16:
		return rewriteValuePPC64_OpRsh8x16(v)
	case ssaop.OpRsh8x32:
		return rewriteValuePPC64_OpRsh8x32(v)
	case ssaop.OpRsh8x64:
		return rewriteValuePPC64_OpRsh8x64(v)
	case ssaop.OpRsh8x8:
		return rewriteValuePPC64_OpRsh8x8(v)
	case ssaop.OpSelect0:
		return rewriteValuePPC64_OpSelect0(v)
	case ssaop.OpSelect1:
		return rewriteValuePPC64_OpSelect1(v)
	case ssaop.OpSelectN:
		return rewriteValuePPC64_OpSelectN(v)
	case ssaop.OpSignExt16to32:
		v.Op = ssaop.OpPPC64MOVHreg
		return true
	case ssaop.OpSignExt16to64:
		v.Op = ssaop.OpPPC64MOVHreg
		return true
	case ssaop.OpSignExt32to64:
		v.Op = ssaop.OpPPC64MOVWreg
		return true
	case ssaop.OpSignExt8to16:
		v.Op = ssaop.OpPPC64MOVBreg
		return true
	case ssaop.OpSignExt8to32:
		v.Op = ssaop.OpPPC64MOVBreg
		return true
	case ssaop.OpSignExt8to64:
		v.Op = ssaop.OpPPC64MOVBreg
		return true
	case ssaop.OpSlicemask:
		return rewriteValuePPC64_OpSlicemask(v)
	case ssaop.OpSqrt:
		v.Op = ssaop.OpPPC64FSQRT
		return true
	case ssaop.OpSqrt32:
		v.Op = ssaop.OpPPC64FSQRTS
		return true
	case ssaop.OpStaticCall:
		v.Op = ssaop.OpPPC64CALLstatic
		return true
	case ssaop.OpStore:
		return rewriteValuePPC64_OpStore(v)
	case ssaop.OpSub16:
		v.Op = ssaop.OpPPC64SUB
		return true
	case ssaop.OpSub32:
		v.Op = ssaop.OpPPC64SUB
		return true
	case ssaop.OpSub32F:
		v.Op = ssaop.OpPPC64FSUBS
		return true
	case ssaop.OpSub64:
		v.Op = ssaop.OpPPC64SUB
		return true
	case ssaop.OpSub64F:
		v.Op = ssaop.OpPPC64FSUB
		return true
	case ssaop.OpSub8:
		v.Op = ssaop.OpPPC64SUB
		return true
	case ssaop.OpSubPtr:
		v.Op = ssaop.OpPPC64SUB
		return true
	case ssaop.OpTailCall:
		v.Op = ssaop.OpPPC64CALLtail
		return true
	case ssaop.OpTailCallInter:
		v.Op = ssaop.OpPPC64CALLtailinter
		return true
	case ssaop.OpTrunc:
		v.Op = ssaop.OpPPC64FTRUNC
		return true
	case ssaop.OpTrunc16to8:
		return rewriteValuePPC64_OpTrunc16to8(v)
	case ssaop.OpTrunc32to16:
		return rewriteValuePPC64_OpTrunc32to16(v)
	case ssaop.OpTrunc32to8:
		return rewriteValuePPC64_OpTrunc32to8(v)
	case ssaop.OpTrunc64to16:
		return rewriteValuePPC64_OpTrunc64to16(v)
	case ssaop.OpTrunc64to32:
		return rewriteValuePPC64_OpTrunc64to32(v)
	case ssaop.OpTrunc64to8:
		return rewriteValuePPC64_OpTrunc64to8(v)
	case ssaop.OpWB:
		v.Op = ssaop.OpPPC64LoweredWB
		return true
	case ssaop.OpXor16:
		v.Op = ssaop.OpPPC64XOR
		return true
	case ssaop.OpXor32:
		v.Op = ssaop.OpPPC64XOR
		return true
	case ssaop.OpXor64:
		v.Op = ssaop.OpPPC64XOR
		return true
	case ssaop.OpXor8:
		v.Op = ssaop.OpPPC64XOR
		return true
	case ssaop.OpZero:
		return rewriteValuePPC64_OpZero(v)
	case ssaop.OpZeroExt16to32:
		v.Op = ssaop.OpPPC64MOVHZreg
		return true
	case ssaop.OpZeroExt16to64:
		v.Op = ssaop.OpPPC64MOVHZreg
		return true
	case ssaop.OpZeroExt32to64:
		v.Op = ssaop.OpPPC64MOVWZreg
		return true
	case ssaop.OpZeroExt8to16:
		v.Op = ssaop.OpPPC64MOVBZreg
		return true
	case ssaop.OpZeroExt8to32:
		v.Op = ssaop.OpPPC64MOVBZreg
		return true
	case ssaop.OpZeroExt8to64:
		v.Op = ssaop.OpPPC64MOVBZreg
		return true
	}
	return false
}
func rewriteValuePPC64_OpAddr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (MOVDaddr {sym} [0] base)
	for {
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		v.Reset(ssaop.OpPPC64MOVDaddr)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValuePPC64_OpAtomicCompareAndSwap32(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64LoweredAtomicCas32)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg4(ptr, old, new_, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicCompareAndSwap64(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64LoweredAtomicCas64)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg4(ptr, old, new_, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicCompareAndSwapRel32(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64LoweredAtomicCas32)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg4(ptr, old, new_, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicLoad32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoad32 ptr mem)
	// result: (LoweredAtomicLoad32 [1] ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64LoweredAtomicLoad32)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicLoad64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoad64 ptr mem)
	// result: (LoweredAtomicLoad64 [1] ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64LoweredAtomicLoad64)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicLoad8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoad8 ptr mem)
	// result: (LoweredAtomicLoad8 [1] ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64LoweredAtomicLoad8)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicLoadAcq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoadAcq32 ptr mem)
	// result: (LoweredAtomicLoad32 [0] ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64LoweredAtomicLoad32)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicLoadAcq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoadAcq64 ptr mem)
	// result: (LoweredAtomicLoad64 [0] ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64LoweredAtomicLoad64)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicLoadPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicLoadPtr ptr mem)
	// result: (LoweredAtomicLoadPtr [1] ptr mem)
	for {
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64LoweredAtomicLoadPtr)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicStore32(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicStore32 ptr val mem)
	// result: (LoweredAtomicStore32 [1] ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64LoweredAtomicStore32)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicStore64(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicStore64 ptr val mem)
	// result: (LoweredAtomicStore64 [1] ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64LoweredAtomicStore64)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicStore8(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicStore8 ptr val mem)
	// result: (LoweredAtomicStore8 [1] ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64LoweredAtomicStore8)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicStoreRel32(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicStoreRel32 ptr val mem)
	// result: (LoweredAtomicStore32 [0] ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64LoweredAtomicStore32)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValuePPC64_OpAtomicStoreRel64(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AtomicStoreRel64 ptr val mem)
	// result: (LoweredAtomicStore64 [0] ptr val mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64LoweredAtomicStore64)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, val, mem)
		return true
	}
}
func rewriteValuePPC64_OpAvg64u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Avg64u <t> x y)
	// result: (ADD (SRDconst <t> (SUB <t> x y) [1]) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRDconst, t)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64SUB, t)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValuePPC64_OpBitLen16(v *ssa.Value) bool {
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
func rewriteValuePPC64_OpBitLen32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen32 x)
	// result: (SUBFCconst [32] (CNTLZW <typ.Int> x))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64SUBFCconst)
		v.AuxInt = ssa.Int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CNTLZW, typ.Int)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpBitLen64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen64 x)
	// result: (SUBFCconst [64] (CNTLZD <typ.Int> x))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64SUBFCconst)
		v.AuxInt = ssa.Int64ToAuxInt(64)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CNTLZD, typ.Int)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpBitLen8(v *ssa.Value) bool {
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
func rewriteValuePPC64_OpBswap16(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64BRH)
		v.AddArg(x)
		return true
	}
	// match: (Bswap16 x:(MOVHZload [off] {sym} ptr mem))
	// result: @x.Block (MOVHBRload (MOVDaddr <ptr.Type> [off] {sym} ptr) mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVHZload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVHBRload, typ.UInt16)
		v.CopyOf(v0)
		v1 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVDaddr, ptr.Type)
		v1.AuxInt = ssa.Int32ToAuxInt(off)
		v1.Aux = ssa.SymToAux(sym)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (Bswap16 x:(MOVHZloadidx ptr idx mem))
	// result: @x.Block (MOVHBRloadidx ptr idx mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVHZloadidx {
			break
		}
		mem := x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHBRloadidx, typ.Int16)
		v.CopyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpBswap32(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64BRW)
		v.AddArg(x)
		return true
	}
	// match: (Bswap32 x:(MOVWZload [off] {sym} ptr mem))
	// result: @x.Block (MOVWBRload (MOVDaddr <ptr.Type> [off] {sym} ptr) mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVWZload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVWBRload, typ.UInt32)
		v.CopyOf(v0)
		v1 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVDaddr, ptr.Type)
		v1.AuxInt = ssa.Int32ToAuxInt(off)
		v1.Aux = ssa.SymToAux(sym)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (Bswap32 x:(MOVWZloadidx ptr idx mem))
	// result: @x.Block (MOVWBRloadidx ptr idx mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVWZloadidx {
			break
		}
		mem := x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWBRloadidx, typ.Int32)
		v.CopyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpBswap64(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64BRD)
		v.AddArg(x)
		return true
	}
	// match: (Bswap64 x:(MOVDload [off] {sym} ptr mem))
	// result: @x.Block (MOVDBRload (MOVDaddr <ptr.Type> [off] {sym} ptr) mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVDload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVDBRload, typ.UInt64)
		v.CopyOf(v0)
		v1 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVDaddr, ptr.Type)
		v1.AuxInt = ssa.Int32ToAuxInt(off)
		v1.Aux = ssa.SymToAux(sym)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (Bswap64 x:(MOVDloadidx ptr idx mem))
	// result: @x.Block (MOVDBRloadidx ptr idx mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVDloadidx {
			break
		}
		mem := x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDBRloadidx, typ.Int64)
		v.CopyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpCom16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Com16 x)
	// result: (NOR x x)
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64NOR)
		v.AddArg2(x, x)
		return true
	}
}
func rewriteValuePPC64_OpCom32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Com32 x)
	// result: (NOR x x)
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64NOR)
		v.AddArg2(x, x)
		return true
	}
}
func rewriteValuePPC64_OpCom64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Com64 x)
	// result: (NOR x x)
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64NOR)
		v.AddArg2(x, x)
		return true
	}
}
func rewriteValuePPC64_OpCom8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Com8 x)
	// result: (NOR x x)
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64NOR)
		v.AddArg2(x, x)
		return true
	}
}
func rewriteValuePPC64_OpCondSelect(v *ssa.Value) bool {
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
		if v_2.Op != ssaop.OpPPC64SETBC {
			break
		}
		a := ssa.AuxIntToInt32(v_2.AuxInt)
		cmp := v_2.Args[0]
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(a)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (CondSelect x y (SETBCR [a] cmp))
	// result: (ISEL [a+4] x y cmp)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpPPC64SETBCR {
			break
		}
		a := ssa.AuxIntToInt32(v_2.AuxInt)
		cmp := v_2.Args[0]
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(a + 4)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (CondSelect x y bool)
	// cond: ssa.FlagArg(bool) == nil
	// result: (ISEL [6] x y (CMPconst [0] (ANDconst [1] bool)))
	for {
		x := v_0
		y := v_1
		bool := v_2
		if !(ssa.FlagArg(bool) == nil) {
			break
		}
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v1.AuxInt = ssa.Int64ToAuxInt(1)
		v1.AddArg(bool)
		v0.AddArg(v1)
		v.AddArg3(x, y, v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpConst16(v *ssa.Value) bool {
	// match: (Const16 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := ssa.AuxIntToInt16(v.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValuePPC64_OpConst32(v *ssa.Value) bool {
	// match: (Const32 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := ssa.AuxIntToInt32(v.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValuePPC64_OpConst64(v *ssa.Value) bool {
	// match: (Const64 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := ssa.AuxIntToInt64(v.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValuePPC64_OpConst8(v *ssa.Value) bool {
	// match: (Const8 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := ssa.AuxIntToInt8(v.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValuePPC64_OpConstBool(v *ssa.Value) bool {
	// match: (ConstBool [t])
	// result: (MOVDconst [ssa.B2i(t)])
	for {
		t := ssa.AuxIntToBool(v.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(t))
		return true
	}
}
func rewriteValuePPC64_OpConstNil(v *ssa.Value) bool {
	// match: (ConstNil)
	// result: (MOVDconst [0])
	for {
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
}
func rewriteValuePPC64_OpCopysign(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Copysign x y)
	// result: (FCPSGN y x)
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64FCPSGN)
		v.AddArg2(y, x)
		return true
	}
}
func rewriteValuePPC64_OpCtz16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 x)
	// cond: buildcfg.GOPPC64 <= 8
	// result: (POPCNTW (MOVHZreg (ANDN <typ.Int16> (ADDconst <typ.Int16> [-1] x) x)))
	for {
		x := v_0
		if !(buildcfg.GOPPC64 <= 8) {
			break
		}
		v.Reset(ssaop.OpPPC64POPCNTW)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDN, typ.Int16)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64ADDconst, typ.Int16)
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v2.AddArg(x)
		v1.AddArg2(v2, x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Ctz16 x)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (CNTTZD (OR <typ.UInt64> x (MOVDconst [1<<16])))
	for {
		x := v_0
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.Reset(ssaop.OpPPC64CNTTZD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64OR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(1 << 16)
		v0.AddArg2(x, v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpCtz32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz32 x)
	// cond: buildcfg.GOPPC64 <= 8
	// result: (POPCNTW (MOVWZreg (ANDN <typ.Int> (ADDconst <typ.Int> [-1] x) x)))
	for {
		x := v_0
		if !(buildcfg.GOPPC64 <= 8) {
			break
		}
		v.Reset(ssaop.OpPPC64POPCNTW)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWZreg, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDN, typ.Int)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64ADDconst, typ.Int)
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v2.AddArg(x)
		v1.AddArg2(v2, x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Ctz32 x)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (CNTTZW (MOVWZreg x))
	for {
		x := v_0
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.Reset(ssaop.OpPPC64CNTTZW)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpCtz64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz64 x)
	// cond: buildcfg.GOPPC64 <= 8
	// result: (POPCNTD (ANDN <typ.Int64> (ADDconst <typ.Int64> [-1] x) x))
	for {
		x := v_0
		if !(buildcfg.GOPPC64 <= 8) {
			break
		}
		v.Reset(ssaop.OpPPC64POPCNTD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDN, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64ADDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(-1)
		v1.AddArg(x)
		v0.AddArg2(v1, x)
		v.AddArg(v0)
		return true
	}
	// match: (Ctz64 x)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (CNTTZD x)
	for {
		x := v_0
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.Reset(ssaop.OpPPC64CNTTZD)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpCtz8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 x)
	// cond: buildcfg.GOPPC64 <= 8
	// result: (POPCNTB (MOVBZreg (ANDN <typ.UInt8> (ADDconst <typ.UInt8> [-1] x) x)))
	for {
		x := v_0
		if !(buildcfg.GOPPC64 <= 8) {
			break
		}
		v.Reset(ssaop.OpPPC64POPCNTB)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDN, typ.UInt8)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64ADDconst, typ.UInt8)
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v2.AddArg(x)
		v1.AddArg2(v2, x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Ctz8 x)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (CNTTZD (OR <typ.UInt64> x (MOVDconst [1<<8])))
	for {
		x := v_0
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.Reset(ssaop.OpPPC64CNTTZD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64OR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(1 << 8)
		v0.AddArg2(x, v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpCvt32Fto32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32Fto32 x)
	// result: (MFVSRD (FCTIWZ x))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64MFVSRD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64FCTIWZ, typ.Float64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt32Fto64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32Fto64 x)
	// result: (MFVSRD (FCTIDZ x))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64MFVSRD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64FCTIDZ, typ.Float64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt32to32F(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32to32F x)
	// result: (FCFIDS (MTVSRD (SignExt32to64 x)))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64FCFIDS)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MTVSRD, typ.Float64)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt32to64F(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt32to64F x)
	// result: (FCFID (MTVSRD (SignExt32to64 x)))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64FCFID)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MTVSRD, typ.Float64)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt64Fto32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt64Fto32 x)
	// result: (MFVSRD (FCTIWZ x))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64MFVSRD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64FCTIWZ, typ.Float64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt64Fto64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt64Fto64 x)
	// result: (MFVSRD (FCTIDZ x))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64MFVSRD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64FCTIDZ, typ.Float64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt64to32F(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt64to32F x)
	// result: (FCFIDS (MTVSRD x))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64FCFIDS)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MTVSRD, typ.Float64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpCvt64to64F(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Cvt64to64F x)
	// result: (FCFID (MTVSRD x))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64FCFID)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MTVSRD, typ.Float64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpDiv16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 [false] x y)
	// result: (DIVW (SignExt16to32 x) (SignExt16to32 y))
	for {
		if ssa.AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64DIVW)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValuePPC64_OpDiv16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (DIVWU (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64DIVWU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpDiv32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div32 [false] x y)
	// result: (DIVW x y)
	for {
		if ssa.AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64DIVW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpDiv64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64 [false] x y)
	// result: (DIVD x y)
	for {
		if ssa.AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64DIVD)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpDiv8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (DIVW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64DIVW)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpDiv8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (DIVWU (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64DIVWU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpEq16(v *ssa.Value) bool {
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
			v.Reset(ssaop.OpPPC64Equal)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
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
		v.Reset(ssaop.OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32 x y)
	// result: (Equal (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32F x y)
	// result: (Equal (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64 x y)
	// result: (Equal (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64F x y)
	// result: (Equal (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEq8(v *ssa.Value) bool {
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
			v.Reset(ssaop.OpPPC64Equal)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
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
		v.Reset(ssaop.OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEqB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// result: (ANDconst [1] (EQV x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64EQV, typ.Int64)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpEqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (EqPtr x y)
	// result: (Equal (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpIsInBounds(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsInBounds idx len)
	// result: (LessThan (CMPU idx len))
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPU, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpIsNonNil(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsNonNil ptr)
	// result: (NotEqual (CMPconst [0] ptr))
	for {
		ptr := v_0
		v.Reset(ssaop.OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v0.AddArg(ptr)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpIsSliceInBounds(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsSliceInBounds idx len)
	// result: (LessEqual (CMPU idx len))
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPU, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (LessEqual (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// result: (LessEqual (CMPWU (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWU, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32 x y)
	// result: (LessEqual (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32F x y)
	// result: (FLessEqual (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64FLessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32U x y)
	// result: (LessEqual (CMPWU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64 x y)
	// result: (LessEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64F x y)
	// result: (FLessEqual (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64FLessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq64U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64U x y)
	// result: (LessEqual (CMPU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (LessEqual (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLeq8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// result: (LessEqual (CMPWU (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWU, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (LessThan (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// result: (LessThan (CMPWU (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWU, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32 x y)
	// result: (LessThan (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32F x y)
	// result: (FLessThan (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64FLessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32U x y)
	// result: (LessThan (CMPWU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64 x y)
	// result: (LessThan (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64F x y)
	// result: (FLessThan (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64FLessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess64U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64U x y)
	// result: (LessThan (CMPU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (LessThan (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLess8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// result: (LessThan (CMPWU (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWU, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpLoad(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is64BitInt(t) ||ssa.IsPtr(t))
	// result: (MOVDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is64BitInt(t) || ssa.IsPtr(t)) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is32BitInt(t) && t.IsSigned()
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is32BitInt(t) && !t.IsSigned()
	// result: (MOVWZload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWZload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is16BitInt(t) && t.IsSigned()
	// result: (MOVHload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is16BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is16BitInt(t) && !t.IsSigned()
	// result: (MOVHZload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is16BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHZload)
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
		v.Reset(ssaop.OpPPC64MOVBZload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is8BitInt(t) && t.IsSigned()
	// result: (MOVBreg (MOVBZload ptr mem))
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is8BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZload, typ.UInt8)
		v0.AddArg2(ptr, mem)
		v.AddArg(v0)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is8BitInt(t) && !t.IsSigned()
	// result: (MOVBZload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is8BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBZload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is32BitFloat(t)
	// result: (FMOVSload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVSload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is64BitFloat(t)
	// result: (FMOVDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is64BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpLocalAddr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LocalAddr <t> {sym} base mem)
	// cond: t.Elem().HasPointers()
	// result: (MOVDaddr {sym} (SPanchored base mem))
	for {
		t := v.Type
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		mem := v_1
		if !(t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDaddr)
		v.Aux = ssa.SymToAux(sym)
		v0 := b.NewValue0(v.Pos, ssaop.OpSPanchored, typ.Uintptr)
		v0.AddArg2(base, mem)
		v.AddArg(v0)
		return true
	}
	// match: (LocalAddr <t> {sym} base _)
	// cond: !t.Elem().HasPointers()
	// result: (MOVDaddr {sym} base)
	for {
		t := v.Type
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		if !(!t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDaddr)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValuePPC64_OpLsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x16 <t> x y)
	// result: (ISEL [2] (SLD <t> (MOVHZreg x) y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0xFFF0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v4.AuxInt = ssa.Int64ToAuxInt(0xFFF0)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x32 <t> x y)
	// result: (ISEL [0] (SLD <t> (MOVHZreg x) y) (MOVDconst [0]) (CMPWUconst y [16]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v3.AuxInt = ssa.Int32ToAuxInt(16)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x64 x (MOVDconst [c]))
	// cond: uint64(c) < 16
	// result: (SLWconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpPPC64SLWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Lsh16x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x64 <t> x y)
	// result: (ISEL [0] (SLD <t> (MOVHZreg x) y) (MOVDconst [0]) (CMPUconst y [16]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(16)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x8 <t> x y)
	// result: (ISEL [2] (SLD <t> (MOVHZreg x) y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0x00F0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v4.AuxInt = ssa.Int64ToAuxInt(0x00F0)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x16 <t> x y)
	// result: (ISEL [2] (SLW <t> x y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0xFFE0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v3.AuxInt = ssa.Int64ToAuxInt(0xFFE0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x32 <t> x y)
	// result: (ISEL [0] (SLW <t> x y) (MOVDconst [0]) (CMPWUconst y [32]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x64 x (MOVDconst [c]))
	// cond: uint64(c) < 32
	// result: (SLWconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpPPC64SLWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Lsh32x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x64 <t> x y)
	// result: (ISEL [0] (SLW <t> x y) (MOVDconst [0]) (CMPUconst y [32]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x8 <t> x y)
	// result: (ISEL [2] (SLW <t> x y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0x00E0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v3.AuxInt = ssa.Int64ToAuxInt(0x00E0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh64x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x16 <t> x y)
	// result: (ISEL [2] (SLD <t> x y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0xFFC0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v3.AuxInt = ssa.Int64ToAuxInt(0xFFC0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh64x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x32 <t> x y)
	// result: (ISEL [0] (SLD <t> x y) (MOVDconst [0]) (CMPWUconst y [64]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh64x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x64 x (MOVDconst [c]))
	// cond: uint64(c) < 64
	// result: (SLDconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpPPC64SLDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Lsh64x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x64 <t> x y)
	// result: (ISEL [0] (SLD <t> x y) (MOVDconst [0]) (CMPUconst y [64]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh64x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x8 <t> x y)
	// result: (ISEL [2] (SLD <t> x y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0x00C0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v3.AuxInt = ssa.Int64ToAuxInt(0x00C0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpLsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x16 <t> x y)
	// result: (ISEL [2] (SLD <t> (MOVBZreg x) y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0xFFF8] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v4.AuxInt = ssa.Int64ToAuxInt(0xFFF8)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x32 <t> x y)
	// result: (ISEL [0] (SLD <t> (MOVBZreg x) y) (MOVDconst [0]) (CMPWUconst y [8]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v3.AuxInt = ssa.Int32ToAuxInt(8)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x64 x (MOVDconst [c]))
	// cond: uint64(c) < 8
	// result: (SLWconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpPPC64SLWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Lsh8x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x64 <t> x y)
	// result: (ISEL [0] (SLD <t> (MOVBZreg x) y) (MOVDconst [0]) (CMPUconst y [8]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(8)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpLsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x8 <t> x y)
	// result: (ISEL [2] (SLD <t> (MOVBZreg x) y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0x00F8] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SLD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v4.AuxInt = ssa.Int64ToAuxInt(0x00F8)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpMax32F(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64XSMAXJDP)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMax64F(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64XSMAXJDP)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMin32F(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64XSMINJDP)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMin64F(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64XSMINJDP)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMod16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// result: (Mod32 (SignExt16to32 x) (SignExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMod32)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpMod16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (Mod32u (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMod32u)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpMod32(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64MODSW)
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
		v.Reset(ssaop.OpPPC64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MULLW, typ.Int32)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64DIVW, typ.Int32)
		v1.AddArg2(x, y)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMod32u(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64MODUW)
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
		v.Reset(ssaop.OpPPC64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MULLW, typ.Int32)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64DIVWU, typ.Int32)
		v1.AddArg2(x, y)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMod64(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64MODSD)
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
		v.Reset(ssaop.OpPPC64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MULLD, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64DIVD, typ.Int64)
		v1.AddArg2(x, y)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMod64u(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64MODUD)
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
		v.Reset(ssaop.OpPPC64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MULLD, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64DIVDU, typ.Int64)
		v1.AddArg2(x, y)
		v0.AddArg2(y, v1)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpMod8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (Mod32 (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMod32)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpMod8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (Mod32u (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMod32u)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpMove(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Move [0] _ _ mem)
	// result: mem
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.CopyOf(mem)
		return true
	}
	// match: (Move [1] dst src mem)
	// result: (MOVBstore dst (MOVBZload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZload, typ.UInt8)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (MOVHstore dst (MOVHZload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZload, typ.UInt16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (MOVWstore dst (MOVWZload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWZload, typ.UInt32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [8] {t} dst src mem)
	// result: (MOVDstore dst (MOVDload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVDstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDload, typ.Int64)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [3] dst src mem)
	// result: (MOVBstore [2] dst (MOVBZload [2] src mem) (MOVHstore dst (MOVHload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHload, typ.Int16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [5] dst src mem)
	// result: (MOVBstore [4] dst (MOVBZload [4] src mem) (MOVWstore dst (MOVWZload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 5 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWZload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [6] dst src mem)
	// result: (MOVHstore [4] dst (MOVHZload [4] src mem) (MOVWstore dst (MOVWZload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZload, typ.UInt16)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWZload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [7] dst src mem)
	// result: (MOVBstore [6] dst (MOVBZload [6] src mem) (MOVHstore [4] dst (MOVHZload [4] src mem) (MOVWstore dst (MOVWZload src mem) mem)))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 7 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(6)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHstore, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZload, typ.UInt16)
		v2.AuxInt = ssa.Int32ToAuxInt(4)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWZload, typ.UInt32)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 8 && buildcfg.GOPPC64 <= 8 && ssa.LogLargeCopyValue(v, s)
	// result: (LoweredMove [s] dst src mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 8 && buildcfg.GOPPC64 <= 8 && ssa.LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.OpPPC64LoweredMove)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 8 && s <= 64 && buildcfg.GOPPC64 >= 9
	// result: (LoweredQuadMoveShort [s] dst src mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 8 && s <= 64 && buildcfg.GOPPC64 >= 9) {
			break
		}
		v.Reset(ssaop.OpPPC64LoweredQuadMoveShort)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 8 && buildcfg.GOPPC64 >= 9 && ssa.LogLargeCopyValue(v, s)
	// result: (LoweredQuadMove [s] dst src mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 8 && buildcfg.GOPPC64 >= 9 && ssa.LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.OpPPC64LoweredQuadMove)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpNeq16(v *ssa.Value) bool {
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
			v.Reset(ssaop.OpPPC64NotEqual)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
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
		v.Reset(ssaop.OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32 x y)
	// result: (NotEqual (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNeq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32F x y)
	// result: (NotEqual (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNeq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64 x y)
	// result: (NotEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNeq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64F x y)
	// result: (NotEqual (FCMPU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64FCMPU, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNeq8(v *ssa.Value) bool {
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
			v.Reset(ssaop.OpPPC64NotEqual)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
			v1.AddArg(x)
			v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
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
		v.Reset(ssaop.OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNeqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (NeqPtr x y)
	// result: (NotEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpNot(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Not x)
	// result: (XORconst [1] x)
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpOffPtr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OffPtr [off] ptr)
	// result: (ADD (MOVDconst <typ.Int64> [off]) ptr)
	for {
		off := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		v.Reset(ssaop.OpPPC64ADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(off)
		v.AddArg2(v0, ptr)
		return true
	}
}
func rewriteValuePPC64_OpPPC64ADD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADD z l:(MULLD x y))
	// cond: buildcfg.GOPPC64 >= 9 && l.Uses == 1 && ssa.Clobber(l)
	// result: (MADDLD x y z )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			z := v_0
			l := v_1
			if l.Op != ssaop.OpPPC64MULLD {
				continue
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(buildcfg.GOPPC64 >= 9 && l.Uses == 1 && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.OpPPC64MADDLD)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (ADD z l:(MULLDconst <mt> [x] y))
	// cond: buildcfg.GOPPC64 >= 9 && l.Uses == 1 && ssa.Clobber(l)
	// result: (MADDLD (MOVDconst <mt> [int64(x)]) y z )
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			z := v_0
			l := v_1
			if l.Op != ssaop.OpPPC64MULLDconst {
				continue
			}
			mt := l.Type
			x := ssa.AuxIntToInt32(l.AuxInt)
			y := l.Args[0]
			if !(buildcfg.GOPPC64 >= 9 && l.Uses == 1 && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.OpPPC64MADDLD)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, mt)
			v0.AuxInt = ssa.Int64ToAuxInt(int64(x))
			v.AddArg3(v0, y, z)
			return true
		}
		break
	}
	// match: (ADD x (MOVDconst <t> [c]))
	// cond: ssa.Is32Bit(c) && !t.IsPtr()
	// result: (ADDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			t := v_1.Type
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.Is32Bit(c) && !t.IsPtr()) {
				continue
			}
			v.Reset(ssaop.OpPPC64ADDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64ADDC(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDC x (MOVDconst [y]))
	// cond: ssa.Is16Bit(y)
	// result: (ADDCconst [y] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			y := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.Is16Bit(y)) {
				continue
			}
			v.Reset(ssaop.OpPPC64ADDCconst)
			v.AuxInt = ssa.Int64ToAuxInt(y)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64ADDE(v *ssa.Value) bool {
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
		if v_2.Op != ssaop.OpSelect1 || v_2.Type != typ.UInt64 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpPPC64ADDCconst || ssa.AuxIntToInt64(v_2_0.AuxInt) != -1 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != ssaop.OpPPC64MOVDconst || ssa.AuxIntToInt64(v_2_0_0.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpPPC64ADDC)
		v.AddArg2(x, y)
		return true
	}
	// match: (ADDE (MOVDconst [0]) y c)
	// result: (ADDZE y c)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				continue
			}
			y := v_1
			c := v_2
			v.Reset(ssaop.OpPPC64ADDZE)
			v.AddArg2(y, c)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64ADDconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDconst <at> [z] l:(MULLD x y))
	// cond: buildcfg.GOPPC64 >= 9 && l.Uses == 1 && ssa.Clobber(l)
	// result: (MADDLD x y (MOVDconst <at> [int64(z)]))
	for {
		at := v.Type
		z := ssa.AuxIntToInt64(v.AuxInt)
		l := v_0
		if l.Op != ssaop.OpPPC64MULLD {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(buildcfg.GOPPC64 >= 9 && l.Uses == 1 && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.OpPPC64MADDLD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, at)
		v0.AuxInt = ssa.Int64ToAuxInt(int64(z))
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (ADDconst <at> [z] l:(MULLDconst <mt> [x] y))
	// cond: buildcfg.GOPPC64 >= 9 && l.Uses == 1 && ssa.Clobber(l)
	// result: (MADDLD (MOVDconst <mt> [int64(x)]) y (MOVDconst <at> [int64(z)]))
	for {
		at := v.Type
		z := ssa.AuxIntToInt64(v.AuxInt)
		l := v_0
		if l.Op != ssaop.OpPPC64MULLDconst {
			break
		}
		mt := l.Type
		x := ssa.AuxIntToInt32(l.AuxInt)
		y := l.Args[0]
		if !(buildcfg.GOPPC64 >= 9 && l.Uses == 1 && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.OpPPC64MADDLD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, mt)
		v0.AuxInt = ssa.Int64ToAuxInt(int64(x))
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, at)
		v1.AuxInt = ssa.Int64ToAuxInt(int64(z))
		v.AddArg3(v0, y, v1)
		return true
	}
	// match: (ADDconst [c] (ADDconst [d] x))
	// cond: ssa.Is32Bit(c+d)
	// result: (ADDconst [c+d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(c + d)) {
			break
		}
		v.Reset(ssaop.OpPPC64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ADDconst [c] (MOVDaddr [d] {sym} x))
	// cond: ssa.Is32Bit(c+int64(d))
	// result: (MOVDaddr [int32(c+int64(d))] {sym} x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		d := ssa.AuxIntToInt32(v_0.AuxInt)
		sym := ssa.AuxToSym(v_0.Aux)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(c + int64(d))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDaddr)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c + int64(d)))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] x:(SP))
	// cond: ssa.Is32Bit(c)
	// result: (MOVDaddr [int32(c)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if x.Op != ssaop.OpSP || !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDaddr)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (SUBFCconst [d] x))
	// cond: ssa.Is32Bit(c+d)
	// result: (SUBFCconst [c+d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64SUBFCconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(c + d)) {
			break
		}
		v.Reset(ssaop.OpPPC64SUBFCconst)
		v.AuxInt = ssa.Int64ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64AND(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AND <t> x (MOVDconst [m]))
	// cond: t.IsUnsigned() && t.Size() == 1 && m != int64(uint8(m))
	// result: (ANDconst [int64(uint8(m))] x)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(t.IsUnsigned() && t.Size() == 1 && m != int64(uint8(m))) {
				continue
			}
			v.Reset(ssaop.OpPPC64ANDconst)
			v.AuxInt = ssa.Int64ToAuxInt(int64(uint8(m)))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND <t> x (MOVDconst [m]))
	// cond: t.IsUnsigned() && t.Size() == 2 && m != int64(uint16(m))
	// result: (ANDconst [int64(uint16(m))] x)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(t.IsUnsigned() && t.Size() == 2 && m != int64(uint16(m))) {
				continue
			}
			v.Reset(ssaop.OpPPC64ANDconst)
			v.AuxInt = ssa.Int64ToAuxInt(int64(uint16(m)))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [m]) (ROTLWconst [r] x))
	// cond: ssa.IsPPC64WordRotateMask(m)
	// result: (RLWINM [ssa.EncodePPC64RotateMask(r,m,32)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpPPC64ROTLWconst {
				continue
			}
			r := ssa.AuxIntToInt64(v_1.AuxInt)
			x := v_1.Args[0]
			if !(ssa.IsPPC64WordRotateMask(m)) {
				continue
			}
			v.Reset(ssaop.OpPPC64RLWINM)
			v.AuxInt = ssa.Int64ToAuxInt(ssa.EncodePPC64RotateMask(r, m, 32))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [m]) (ROTLW x r))
	// cond: ssa.IsPPC64WordRotateMask(m)
	// result: (RLWNM [ssa.EncodePPC64RotateMask(0,m,32)] x r)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpPPC64ROTLW {
				continue
			}
			r := v_1.Args[1]
			x := v_1.Args[0]
			if !(ssa.IsPPC64WordRotateMask(m)) {
				continue
			}
			v.Reset(ssaop.OpPPC64RLWNM)
			v.AuxInt = ssa.Int64ToAuxInt(ssa.EncodePPC64RotateMask(0, m, 32))
			v.AddArg2(x, r)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [m]) (SRWconst x [s]))
	// cond: ssa.MergePPC64RShiftMask(m,s,32) == 0
	// result: (MOVDconst [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpPPC64SRWconst {
				continue
			}
			s := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.MergePPC64RShiftMask(m, s, 32) == 0) {
				continue
			}
			v.Reset(ssaop.OpPPC64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [m]) (SRWconst x [s]))
	// cond: ssa.MergePPC64AndSrwi(m,s) != 0
	// result: (RLWINM [ssa.MergePPC64AndSrwi(m,s)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpPPC64SRWconst {
				continue
			}
			s := ssa.AuxIntToInt64(v_1.AuxInt)
			x := v_1.Args[0]
			if !(ssa.MergePPC64AndSrwi(m, s) != 0) {
				continue
			}
			v.Reset(ssaop.OpPPC64RLWINM)
			v.AuxInt = ssa.Int64ToAuxInt(ssa.MergePPC64AndSrwi(m, s))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [m]) (SRDconst x [s]))
	// cond: mergePPC64AndSrdi(m,s) != 0
	// result: (RLWINM [mergePPC64AndSrdi(m,s)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpPPC64SRDconst {
				continue
			}
			s := ssa.AuxIntToInt64(v_1.AuxInt)
			x := v_1.Args[0]
			if !(mergePPC64AndSrdi(m, s) != 0) {
				continue
			}
			v.Reset(ssaop.OpPPC64RLWINM)
			v.AuxInt = ssa.Int64ToAuxInt(mergePPC64AndSrdi(m, s))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [m]) (SLDconst x [s]))
	// cond: mergePPC64AndSldi(m,s) != 0
	// result: (RLWINM [mergePPC64AndSldi(m,s)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpPPC64SLDconst {
				continue
			}
			s := ssa.AuxIntToInt64(v_1.AuxInt)
			x := v_1.Args[0]
			if !(mergePPC64AndSldi(m, s) != 0) {
				continue
			}
			v.Reset(ssaop.OpPPC64RLWINM)
			v.AuxInt = ssa.Int64ToAuxInt(mergePPC64AndSldi(m, s))
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
			if v_1.Op != ssaop.OpPPC64NOR {
				continue
			}
			y := v_1.Args[1]
			if y != v_1.Args[0] {
				continue
			}
			v.Reset(ssaop.OpPPC64ANDN)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [c&d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			d := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpPPC64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c & d)
			return true
		}
		break
	}
	// match: (AND x (MOVDconst [-1]))
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != -1 {
				continue
			}
			v.CopyOf(x)
			return true
		}
		break
	}
	// match: (AND x (MOVDconst [c]))
	// cond: isU16Bit(c)
	// result: (ANDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(isU16Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpPPC64ANDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [c]) y:(MOVWZreg _))
	// cond: c&0xFFFFFFFF == 0xFFFFFFFF
	// result: y
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			y := v_1
			if y.Op != ssaop.OpPPC64MOVWZreg || !(c&0xFFFFFFFF == 0xFFFFFFFF) {
				continue
			}
			v.CopyOf(y)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [0xFFFFFFFF]) y:(MOVWreg x))
	// result: (MOVWZreg x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0xFFFFFFFF {
				continue
			}
			y := v_1
			if y.Op != ssaop.OpPPC64MOVWreg {
				continue
			}
			x := y.Args[0]
			v.Reset(ssaop.OpPPC64MOVWZreg)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND (MOVDconst [c]) x:(MOVBZload _ _))
	// result: (ANDconst [c&0xFF] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			x := v_1
			if x.Op != ssaop.OpPPC64MOVBZload {
				continue
			}
			v.Reset(ssaop.OpPPC64ANDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c & 0xFF)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64ANDN(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ANDN (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [c&^d])
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c &^ d)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64ANDconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ANDconst [m] (ROTLWconst [r] x))
	// cond: ssa.IsPPC64WordRotateMask(m)
	// result: (RLWINM [ssa.EncodePPC64RotateMask(r,m,32)] x)
	for {
		m := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ROTLWconst {
			break
		}
		r := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.IsPPC64WordRotateMask(m)) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.EncodePPC64RotateMask(r, m, 32))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [m] (ROTLW x r))
	// cond: ssa.IsPPC64WordRotateMask(m)
	// result: (RLWNM [ssa.EncodePPC64RotateMask(0,m,32)] x r)
	for {
		m := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ROTLW {
			break
		}
		r := v_0.Args[1]
		x := v_0.Args[0]
		if !(ssa.IsPPC64WordRotateMask(m)) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWNM)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.EncodePPC64RotateMask(0, m, 32))
		v.AddArg2(x, r)
		return true
	}
	// match: (ANDconst [m] (SRWconst x [s]))
	// cond: ssa.MergePPC64RShiftMask(m,s,32) == 0
	// result: (MOVDconst [0])
	for {
		m := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		s := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(ssa.MergePPC64RShiftMask(m, s, 32) == 0) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (ANDconst [m] (SRWconst x [s]))
	// cond: ssa.MergePPC64AndSrwi(m,s) != 0
	// result: (RLWINM [ssa.MergePPC64AndSrwi(m,s)] x)
	for {
		m := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		s := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.MergePPC64AndSrwi(m, s) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.MergePPC64AndSrwi(m, s))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [m] (SRDconst x [s]))
	// cond: mergePPC64AndSrdi(m,s) != 0
	// result: (RLWINM [mergePPC64AndSrdi(m,s)] x)
	for {
		m := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64SRDconst {
			break
		}
		s := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(mergePPC64AndSrdi(m, s) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(mergePPC64AndSrdi(m, s))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [m] (SLDconst x [s]))
	// cond: mergePPC64AndSldi(m,s) != 0
	// result: (RLWINM [mergePPC64AndSldi(m,s)] x)
	for {
		m := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64SLDconst {
			break
		}
		s := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(mergePPC64AndSldi(m, s) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(mergePPC64AndSldi(m, s))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (ANDconst [d] x))
	// result: (ANDconst [c&d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ANDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [-1] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ANDconst [0] _)
	// result: (MOVDconst [0])
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (ANDconst [c] y:(MOVBZreg _))
	// cond: c&0xFF == 0xFF
	// result: y
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpPPC64MOVBZreg || !(c&0xFF == 0xFF) {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ANDconst [0xFF] (MOVBreg x))
	// result: (MOVBZreg x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0xFF || v_0.Op != ssaop.OpPPC64MOVBreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64MOVBZreg)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] y:(MOVHZreg _))
	// cond: c&0xFFFF == 0xFFFF
	// result: y
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpPPC64MOVHZreg || !(c&0xFFFF == 0xFFFF) {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ANDconst [0xFFFF] (MOVHreg x))
	// result: (MOVHZreg x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0xFFFF || v_0.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64MOVHZreg)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVBZreg x))
	// result: (ANDconst [c&0xFF] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVBZreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 0xFF)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVHZreg x))
	// result: (ANDconst [c&0xFFFF] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 0xFFFF)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVWZreg x))
	// result: (ANDconst [c&0xFFFFFFFF] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 0xFFFFFFFF)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [m] (RLWINM [r] y))
	// cond: mergePPC64AndRlwinm(uint32(m),r) != 0
	// result: (RLWINM [mergePPC64AndRlwinm(uint32(m),r)] y)
	for {
		m := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64RLWINM {
			break
		}
		r := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_0.Args[0]
		if !(mergePPC64AndRlwinm(uint32(m), r) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(mergePPC64AndRlwinm(uint32(m), r))
		v.AddArg(y)
		return true
	}
	// match: (ANDconst [1] z:(SRADconst [63] x))
	// cond: z.Uses == 1
	// result: (SRDconst [63] x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		z := v_0
		if z.Op != ssaop.OpPPC64SRADconst || ssa.AuxIntToInt64(z.AuxInt) != 63 {
			break
		}
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64SRDconst)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64BRD(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BRD x:(MOVDload [off] {sym} ptr mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVDBRload (MOVDaddr <ptr.Type> [off] {sym} ptr) mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVDload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVDBRload, typ.UInt64)
		v.CopyOf(v0)
		v1 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVDaddr, ptr.Type)
		v1.AuxInt = ssa.Int32ToAuxInt(off)
		v1.Aux = ssa.SymToAux(sym)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (BRD x:(MOVDloadidx ptr idx mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVDBRloadidx ptr idx mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVDloadidx {
			break
		}
		mem := x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDBRloadidx, typ.Int64)
		v.CopyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64BRH(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BRH x:(MOVHZload [off] {sym} ptr mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVHBRload (MOVDaddr <ptr.Type> [off] {sym} ptr) mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVHZload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVHBRload, typ.UInt16)
		v.CopyOf(v0)
		v1 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVDaddr, ptr.Type)
		v1.AuxInt = ssa.Int32ToAuxInt(off)
		v1.Aux = ssa.SymToAux(sym)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (BRH x:(MOVHZloadidx ptr idx mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVHBRloadidx ptr idx mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVHZloadidx {
			break
		}
		mem := x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHBRloadidx, typ.Int16)
		v.CopyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64BRW(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BRW x:(MOVWZload [off] {sym} ptr mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVWBRload (MOVDaddr <ptr.Type> [off] {sym} ptr) mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVWZload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVWBRload, typ.UInt32)
		v.CopyOf(v0)
		v1 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVDaddr, ptr.Type)
		v1.AuxInt = ssa.Int32ToAuxInt(off)
		v1.Aux = ssa.SymToAux(sym)
		v1.AddArg(ptr)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (BRW x:(MOVWZloadidx ptr idx mem))
	// cond: x.Uses == 1
	// result: @x.Block (MOVWBRloadidx ptr idx mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVWZloadidx {
			break
		}
		mem := x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		if !(x.Uses == 1) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWBRloadidx, typ.Int32)
		v.CopyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CLRLSLDI(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (CLRLSLDI [c] (SRWconst [s] x))
	// cond: ssa.MergePPC64ClrlsldiSrw(int64(c),s) != 0
	// result: (RLWINM [ssa.MergePPC64ClrlsldiSrw(int64(c),s)] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		s := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.MergePPC64ClrlsldiSrw(int64(c), s) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.MergePPC64ClrlsldiSrw(int64(c), s))
		v.AddArg(x)
		return true
	}
	// match: (CLRLSLDI [c] (SRDconst [s] x))
	// cond: mergePPC64ClrlsldiSrd(int64(c),s) != 0
	// result: (RLWINM [mergePPC64ClrlsldiSrd(int64(c),s)] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64SRDconst {
			break
		}
		s := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(mergePPC64ClrlsldiSrd(int64(c), s) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(mergePPC64ClrlsldiSrd(int64(c), s))
		v.AddArg(x)
		return true
	}
	// match: (CLRLSLDI [c] i:(RLWINM [s] x))
	// cond: ssa.MergePPC64ClrlsldiRlwinm(c,s) != 0
	// result: (RLWINM [ssa.MergePPC64ClrlsldiRlwinm(c,s)] x)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		i := v_0
		if i.Op != ssaop.OpPPC64RLWINM {
			break
		}
		s := ssa.AuxIntToInt64(i.AuxInt)
		x := i.Args[0]
		if !(ssa.MergePPC64ClrlsldiRlwinm(c, s) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.MergePPC64ClrlsldiRlwinm(c, s))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMP(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMP x (MOVDconst [c]))
	// cond: ssa.Is16Bit(c)
	// result: (CMPconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.Is16Bit(c)) {
			break
		}
		v.Reset(ssaop.OpPPC64CMPconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMP (MOVDconst [c]) y)
	// cond: ssa.Is16Bit(c)
	// result: (InvertFlags (CMPconst y [c]))
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(ssa.Is16Bit(c)) {
			break
		}
		v.Reset(ssaop.OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x y)
	// cond: ssa.CanonLessThan(x,y)
	// result: (InvertFlags (CMP y x))
	for {
		x := v_0
		y := v_1
		if !(ssa.CanonLessThan(x, y)) {
			break
		}
		v.Reset(ssaop.OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMP, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPU(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPU x (MOVDconst [c]))
	// cond: isU16Bit(c)
	// result: (CMPUconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(isU16Bit(c)) {
			break
		}
		v.Reset(ssaop.OpPPC64CMPUconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMPU (MOVDconst [c]) y)
	// cond: isU16Bit(c)
	// result: (InvertFlags (CMPUconst y [c]))
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(isU16Bit(c)) {
			break
		}
		v.Reset(ssaop.OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMPU x y)
	// cond: ssa.CanonLessThan(x,y)
	// result: (InvertFlags (CMPU y x))
	for {
		x := v_0
		y := v_1
		if !(ssa.CanonLessThan(x, y)) {
			break
		}
		v.Reset(ssaop.OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPU, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPUconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (CMPUconst [d] (ANDconst z [c]))
	// cond: uint64(d) > uint64(c)
	// result: (FlagLT)
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(uint64(d) > uint64(c)) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagLT)
		return true
	}
	// match: (CMPUconst (MOVDconst [x]) [y])
	// cond: x==y
	// result: (FlagEQ)
	for {
		y := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(x == y) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagEQ)
		return true
	}
	// match: (CMPUconst (MOVDconst [x]) [y])
	// cond: uint64(x)<uint64(y)
	// result: (FlagLT)
	for {
		y := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(uint64(x) < uint64(y)) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagLT)
		return true
	}
	// match: (CMPUconst (MOVDconst [x]) [y])
	// cond: uint64(x)>uint64(y)
	// result: (FlagGT)
	for {
		y := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(uint64(x) > uint64(y)) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagGT)
		return true
	}
	// match: (CMPUconst [0] a:(ANDconst [n] z))
	// result: (CMPconst [0] a)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		a := v_0
		if a.Op != ssaop.OpPPC64ANDconst {
			break
		}
		v.Reset(ssaop.OpPPC64CMPconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPW x (MOVWreg y))
	// result: (CMPW x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpPPC64CMPW)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPW (MOVWreg x) y)
	// result: (CMPW x y)
	for {
		if v_0.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := v_0.Args[0]
		y := v_1
		v.Reset(ssaop.OpPPC64CMPW)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPW x (MOVDconst [c]))
	// cond: ssa.Is16Bit(c)
	// result: (CMPWconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.Is16Bit(c)) {
			break
		}
		v.Reset(ssaop.OpPPC64CMPWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPW (MOVDconst [c]) y)
	// cond: ssa.Is16Bit(c)
	// result: (InvertFlags (CMPWconst y [int32(c)]))
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(ssa.Is16Bit(c)) {
			break
		}
		v.Reset(ssaop.OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMPW x y)
	// cond: ssa.CanonLessThan(x,y)
	// result: (InvertFlags (CMPW y x))
	for {
		x := v_0
		y := v_1
		if !(ssa.CanonLessThan(x, y)) {
			break
		}
		v.Reset(ssaop.OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPW, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPWU(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPWU x (MOVWZreg y))
	// result: (CMPWU x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpPPC64CMPWU)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPWU (MOVWZreg x) y)
	// result: (CMPWU x y)
	for {
		if v_0.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_0.Args[0]
		y := v_1
		v.Reset(ssaop.OpPPC64CMPWU)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMPWU x (MOVDconst [c]))
	// cond: isU16Bit(c)
	// result: (CMPWUconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(isU16Bit(c)) {
			break
		}
		v.Reset(ssaop.OpPPC64CMPWUconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPWU (MOVDconst [c]) y)
	// cond: isU16Bit(c)
	// result: (InvertFlags (CMPWUconst y [int32(c)]))
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_1
		if !(isU16Bit(c)) {
			break
		}
		v.Reset(ssaop.OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMPWU x y)
	// cond: ssa.CanonLessThan(x,y)
	// result: (InvertFlags (CMPWU y x))
	for {
		x := v_0
		y := v_1
		if !(ssa.CanonLessThan(x, y)) {
			break
		}
		v.Reset(ssaop.OpPPC64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWU, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPWUconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (CMPWUconst [d] (ANDconst z [c]))
	// cond: uint64(d) > uint64(c)
	// result: (FlagLT)
	for {
		d := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(uint64(d) > uint64(c)) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagLT)
		return true
	}
	// match: (CMPWUconst (MOVDconst [x]) [y])
	// cond: int32(x)==int32(y)
	// result: (FlagEQ)
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(int32(x) == int32(y)) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagEQ)
		return true
	}
	// match: (CMPWUconst (MOVDconst [x]) [y])
	// cond: uint32(x)<uint32(y)
	// result: (FlagLT)
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(uint32(x) < uint32(y)) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagLT)
		return true
	}
	// match: (CMPWUconst (MOVDconst [x]) [y])
	// cond: uint32(x)>uint32(y)
	// result: (FlagGT)
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(uint32(x) > uint32(y)) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagGT)
		return true
	}
	// match: (CMPWUconst [0] a:(ANDconst [n] z))
	// result: (CMPconst [0] a)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		a := v_0
		if a.Op != ssaop.OpPPC64ANDconst {
			break
		}
		v.Reset(ssaop.OpPPC64CMPconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPWconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (CMPWconst (MOVDconst [x]) [y])
	// cond: int32(x)==int32(y)
	// result: (FlagEQ)
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(int32(x) == int32(y)) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagEQ)
		return true
	}
	// match: (CMPWconst (MOVDconst [x]) [y])
	// cond: int32(x)<int32(y)
	// result: (FlagLT)
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(int32(x) < int32(y)) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagLT)
		return true
	}
	// match: (CMPWconst (MOVDconst [x]) [y])
	// cond: int32(x)>int32(y)
	// result: (FlagGT)
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(int32(x) > int32(y)) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagGT)
		return true
	}
	// match: (CMPWconst [0] a:(ANDconst [n] z))
	// result: (CMPconst [0] a)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		a := v_0
		if a.Op != ssaop.OpPPC64ANDconst {
			break
		}
		v.Reset(ssaop.OpPPC64CMPconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64CMPconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: x==y
	// result: (FlagEQ)
	for {
		y := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(x == y) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagEQ)
		return true
	}
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: x<y
	// result: (FlagLT)
	for {
		y := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(x < y) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagLT)
		return true
	}
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: x>y
	// result: (FlagGT)
	for {
		y := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(x > y) {
			break
		}
		v.Reset(ssaop.OpPPC64FlagGT)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64Equal(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Equal (FlagEQ))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (Equal (FlagLT))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (Equal (FlagGT))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (Equal (InvertFlags x))
	// result: (Equal x)
	for {
		if v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64Equal)
		v.AddArg(x)
		return true
	}
	// match: (Equal cmp)
	// result: (SETBC [2] cmp)
	for {
		cmp := v_0
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64FABS(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (FABS (FMOVDconst [x]))
	// result: (FMOVDconst [math.Abs(x)])
	for {
		if v_0.Op != ssaop.OpPPC64FMOVDconst {
			break
		}
		x := ssa.AuxIntToFloat64(v_0.AuxInt)
		v.Reset(ssaop.OpPPC64FMOVDconst)
		v.AuxInt = ssa.Float64ToAuxInt(math.Abs(x))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FADD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FADD (FMUL x y) z)
	// cond: x.Block.Func.UseFMA(v)
	// result: (FMADD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64FMUL {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				z := v_1
				if !(x.Block.Func.UseFMA(v)) {
					continue
				}
				v.Reset(ssaop.OpPPC64FMADD)
				v.AddArg3(x, y, z)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64FADDS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FADDS (FMULS x y) z)
	// cond: x.Block.Func.UseFMA(v)
	// result: (FMADDS x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64FMULS {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0, v_0_1 = _i1+1, v_0_1, v_0_0 {
				x := v_0_0
				y := v_0_1
				z := v_1
				if !(x.Block.Func.UseFMA(v)) {
					continue
				}
				v.Reset(ssaop.OpPPC64FMADDS)
				v.AddArg3(x, y, z)
				return true
			}
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64FCEIL(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (FCEIL (FMOVDconst [x]))
	// result: (FMOVDconst [math.Ceil(x)])
	for {
		if v_0.Op != ssaop.OpPPC64FMOVDconst {
			break
		}
		x := ssa.AuxIntToFloat64(v_0.AuxInt)
		v.Reset(ssaop.OpPPC64FMOVDconst)
		v.AuxInt = ssa.Float64ToAuxInt(math.Ceil(x))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FFLOOR(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (FFLOOR (FMOVDconst [x]))
	// result: (FMOVDconst [math.Floor(x)])
	for {
		if v_0.Op != ssaop.OpPPC64FMOVDconst {
			break
		}
		x := ssa.AuxIntToFloat64(v_0.AuxInt)
		v.Reset(ssaop.OpPPC64FMOVDconst)
		v.AuxInt = ssa.Float64ToAuxInt(math.Floor(x))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FGreaterEqual(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (FGreaterEqual cmp)
	// result: (OR (SETBC [2] cmp) (SETBC [1] cmp))
	for {
		cmp := v_0
		v.Reset(ssaop.OpPPC64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SETBC, typ.Int32)
		v0.AuxInt = ssa.Int32ToAuxInt(2)
		v0.AddArg(cmp)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64SETBC, typ.Int32)
		v1.AuxInt = ssa.Int32ToAuxInt(1)
		v1.AddArg(cmp)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpPPC64FGreaterThan(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (FGreaterThan cmp)
	// result: (SETBC [1] cmp)
	for {
		cmp := v_0
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64FLessEqual(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (FLessEqual cmp)
	// result: (OR (SETBC [2] cmp) (SETBC [0] cmp))
	for {
		cmp := v_0
		v.Reset(ssaop.OpPPC64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SETBC, typ.Int32)
		v0.AuxInt = ssa.Int32ToAuxInt(2)
		v0.AddArg(cmp)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64SETBC, typ.Int32)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v1.AddArg(cmp)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValuePPC64_OpPPC64FLessThan(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (FLessThan cmp)
	// result: (SETBC [0] cmp)
	for {
		cmp := v_0
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64FMOVDload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVDload [off] {sym} ptr (MOVDstore [off] {sym} ptr x _))
	// result: (MTVSRD x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpPPC64MTVSRD)
		v.AddArg(x)
		return true
	}
	// match: (FMOVDload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (FMOVDload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVDload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (FMOVDload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FMOVDstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVDstore [off] {sym} ptr (MTVSRD x) mem)
	// result: (MOVDstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MTVSRD {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (FMOVDstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (FMOVDstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVDstore [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (FMOVDstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FMOVSload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVSload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (FMOVSload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVSload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (FMOVSload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FMOVSstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVSstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (FMOVSstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVSstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVSstore [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (FMOVSstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVSstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FNEG(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (FNEG (FABS x))
	// result: (FNABS x)
	for {
		if v_0.Op != ssaop.OpPPC64FABS {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64FNABS)
		v.AddArg(x)
		return true
	}
	// match: (FNEG (FNABS x))
	// result: (FABS x)
	for {
		if v_0.Op != ssaop.OpPPC64FNABS {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64FABS)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FSQRT(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (FSQRT (FMOVDconst [x]))
	// cond: x >= 0
	// result: (FMOVDconst [math.Sqrt(x)])
	for {
		if v_0.Op != ssaop.OpPPC64FMOVDconst {
			break
		}
		x := ssa.AuxIntToFloat64(v_0.AuxInt)
		if !(x >= 0) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVDconst)
		v.AuxInt = ssa.Float64ToAuxInt(math.Sqrt(x))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64FSUB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FSUB (FMUL x y) z)
	// cond: x.Block.Func.UseFMA(v)
	// result: (FMSUB x y z)
	for {
		if v_0.Op != ssaop.OpPPC64FMUL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			z := v_1
			if !(x.Block.Func.UseFMA(v)) {
				continue
			}
			v.Reset(ssaop.OpPPC64FMSUB)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64FSUBS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FSUBS (FMULS x y) z)
	// cond: x.Block.Func.UseFMA(v)
	// result: (FMSUBS x y z)
	for {
		if v_0.Op != ssaop.OpPPC64FMULS {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			z := v_1
			if !(x.Block.Func.UseFMA(v)) {
				continue
			}
			v.Reset(ssaop.OpPPC64FMSUBS)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64FTRUNC(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (FTRUNC (FMOVDconst [x]))
	// result: (FMOVDconst [math.Trunc(x)])
	for {
		if v_0.Op != ssaop.OpPPC64FMOVDconst {
			break
		}
		x := ssa.AuxIntToFloat64(v_0.AuxInt)
		v.Reset(ssaop.OpPPC64FMOVDconst)
		v.AuxInt = ssa.Float64ToAuxInt(math.Trunc(x))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64GreaterEqual(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterEqual (FlagEQ))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (GreaterEqual (FlagLT))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (GreaterEqual (FlagGT))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (GreaterEqual (InvertFlags x))
	// result: (LessEqual x)
	for {
		if v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64LessEqual)
		v.AddArg(x)
		return true
	}
	// match: (GreaterEqual cmp)
	// result: (SETBCR [0] cmp)
	for {
		cmp := v_0
		v.Reset(ssaop.OpPPC64SETBCR)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64GreaterThan(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterThan (FlagEQ))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (GreaterThan (FlagLT))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (GreaterThan (FlagGT))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (GreaterThan (InvertFlags x))
	// result: (LessThan x)
	for {
		if v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64LessThan)
		v.AddArg(x)
		return true
	}
	// match: (GreaterThan cmp)
	// result: (SETBC [1] cmp)
	for {
		cmp := v_0
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64ISEL(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ISEL [6] x y (CMPconst [0] (ANDconst [1] (SETBC [c] cmp))))
	// result: (ISEL [c] x y cmp)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpPPC64CMPconst || ssa.AuxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(v_2_0.AuxInt) != 1 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != ssaop.OpPPC64SETBC {
			break
		}
		c := ssa.AuxIntToInt32(v_2_0_0.AuxInt)
		cmp := v_2_0_0.Args[0]
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (ISEL [6] x y (CMPconst [0] (SETBC [c] cmp)))
	// result: (ISEL [c] x y cmp)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpPPC64CMPconst || ssa.AuxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpPPC64SETBC {
			break
		}
		c := ssa.AuxIntToInt32(v_2_0.AuxInt)
		cmp := v_2_0.Args[0]
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (ISEL [6] x y (CMPWconst [0] (SETBC [c] cmp)))
	// result: (ISEL [c] x y cmp)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpPPC64CMPWconst || ssa.AuxIntToInt32(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpPPC64SETBC {
			break
		}
		c := ssa.AuxIntToInt32(v_2_0.AuxInt)
		cmp := v_2_0.Args[0]
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(c)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (ISEL [6] x y (CMPconst [0] (SETBCR [c] cmp)))
	// result: (ISEL [c+4] x y cmp)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpPPC64CMPconst || ssa.AuxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpPPC64SETBCR {
			break
		}
		c := ssa.AuxIntToInt32(v_2_0.AuxInt)
		cmp := v_2_0.Args[0]
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(c + 4)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (ISEL [6] x y (CMPWconst [0] (SETBCR [c] cmp)))
	// result: (ISEL [c+4] x y cmp)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpPPC64CMPWconst || ssa.AuxIntToInt32(v_2.AuxInt) != 0 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpPPC64SETBCR {
			break
		}
		c := ssa.AuxIntToInt32(v_2_0.AuxInt)
		cmp := v_2_0.Args[0]
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(c + 4)
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (ISEL [2] x _ (FlagEQ))
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 {
			break
		}
		x := v_0
		if v_2.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ISEL [2] _ y (FlagLT))
	// result: y
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 {
			break
		}
		y := v_1
		if v_2.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ISEL [2] _ y (FlagGT))
	// result: y
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 {
			break
		}
		y := v_1
		if v_2.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ISEL [6] _ y (FlagEQ))
	// result: y
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 6 {
			break
		}
		y := v_1
		if v_2.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ISEL [6] x _ (FlagLT))
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		if v_2.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ISEL [6] x _ (FlagGT))
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 6 {
			break
		}
		x := v_0
		if v_2.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ISEL [0] _ y (FlagEQ))
	// result: y
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		y := v_1
		if v_2.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ISEL [0] _ y (FlagGT))
	// result: y
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		y := v_1
		if v_2.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ISEL [0] x _ (FlagLT))
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		if v_2.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ISEL [5] _ x (FlagEQ))
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 5 {
			break
		}
		x := v_1
		if v_2.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ISEL [5] _ x (FlagLT))
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 5 {
			break
		}
		x := v_1
		if v_2.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ISEL [5] y _ (FlagGT))
	// result: y
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 5 {
			break
		}
		y := v_0
		if v_2.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ISEL [1] _ y (FlagEQ))
	// result: y
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 1 {
			break
		}
		y := v_1
		if v_2.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ISEL [1] _ y (FlagLT))
	// result: y
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 1 {
			break
		}
		y := v_1
		if v_2.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ISEL [1] x _ (FlagGT))
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 1 {
			break
		}
		x := v_0
		if v_2.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ISEL [4] x _ (FlagEQ))
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 4 {
			break
		}
		x := v_0
		if v_2.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ISEL [4] x _ (FlagGT))
	// result: x
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 4 {
			break
		}
		x := v_0
		if v_2.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ISEL [4] _ y (FlagLT))
	// result: y
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 4 {
			break
		}
		y := v_1
		if v_2.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ISEL [n] x y (InvertFlags bool))
	// cond: n%4 == 0
	// result: (ISEL [n+1] x y bool)
	for {
		n := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		bool := v_2.Args[0]
		if !(n%4 == 0) {
			break
		}
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(n + 1)
		v.AddArg3(x, y, bool)
		return true
	}
	// match: (ISEL [n] x y (InvertFlags bool))
	// cond: n%4 == 1
	// result: (ISEL [n-1] x y bool)
	for {
		n := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		bool := v_2.Args[0]
		if !(n%4 == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(n - 1)
		v.AddArg3(x, y, bool)
		return true
	}
	// match: (ISEL [n] x y (InvertFlags bool))
	// cond: n%4 == 2
	// result: (ISEL [n] x y bool)
	for {
		n := ssa.AuxIntToInt32(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		bool := v_2.Args[0]
		if !(n%4 == 2) {
			break
		}
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(n)
		v.AddArg3(x, y, bool)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64LessEqual(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (LessEqual (FlagEQ))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (LessEqual (FlagLT))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (LessEqual (FlagGT))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (LessEqual (InvertFlags x))
	// result: (GreaterEqual x)
	for {
		if v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64GreaterEqual)
		v.AddArg(x)
		return true
	}
	// match: (LessEqual cmp)
	// result: (SETBCR [1] cmp)
	for {
		cmp := v_0
		v.Reset(ssaop.OpPPC64SETBCR)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64LessThan(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (LessThan (FlagEQ))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (LessThan (FlagLT))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (LessThan (FlagGT))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (LessThan (InvertFlags x))
	// result: (GreaterThan x)
	for {
		if v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64GreaterThan)
		v.AddArg(x)
		return true
	}
	// match: (LessThan cmp)
	// result: (SETBC [0] cmp)
	for {
		cmp := v_0
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64LoweredPanicBoundsCR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsCR [kind] {p} (MOVDconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:p.C, Cy:c}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		p := ssa.AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.OpPPC64LoweredPanicBoundsCC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: p.C, Cy: c})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64LoweredPanicBoundsRC(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRC [kind] {p} (MOVDconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:c, Cy:p.C}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		p := ssa.AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.OpPPC64LoweredPanicBoundsCC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: c, Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64LoweredPanicBoundsRR(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRR [kind] x (MOVDconst [c]) mem)
	// result: (LoweredPanicBoundsRC [kind] x {ssa.PanicBoundsC{C:c}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpPPC64LoweredPanicBoundsRC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: c})
		v.AddArg2(x, mem)
		return true
	}
	// match: (LoweredPanicBoundsRR [kind] (MOVDconst [c]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {ssa.PanicBoundsC{C:c}} y mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_1
		mem := v_2
		v.Reset(ssaop.OpPPC64LoweredPanicBoundsCR)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: c})
		v.AddArg2(y, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MFVSRD(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MFVSRD (FMOVDconst [c]))
	// result: (MOVDconst [int64(math.Float64bits(c))])
	for {
		if v_0.Op != ssaop.OpPPC64FMOVDconst {
			break
		}
		c := ssa.AuxIntToFloat64(v_0.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(math.Float64bits(c)))
		return true
	}
	// match: (MFVSRD x:(FMOVDload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (MOVDload [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64FMOVDload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && ssa.Clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpPPC64MOVDload, typ.Int64)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBZload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBZload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVBZload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBZload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBZload [off1] {sym} (ADDconst [off2] x) mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (MOVBZload [off1+int32(off2)] {sym} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBZload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVBZload [0] {sym} p:(ADD ptr idx) mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVBZloadidx ptr idx mem)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		mem := v_1
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBZloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBZloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBZloadidx ptr (MOVDconst [c]) mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVBZload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBZload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBZloadidx (MOVDconst [c]) ptr mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVBZload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBZload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBZreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVBZreg y:(ANDconst [c] _))
	// cond: uint64(c) <= 0xFF
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(y.AuxInt)
		if !(uint64(c) <= 0xFF) {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVBZreg (SRWconst [c] (MOVBZreg x)))
	// result: (SRWconst [c] (MOVBZreg x))
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpPPC64MOVBZreg {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVBZreg (SRWconst [c] x))
	// cond: x.Type.Size() == 8
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(x.Type.Size() == 8) {
			break
		}
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBZreg (SRDconst [c] x))
	// cond: c>=56
	// result: (SRDconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 56) {
			break
		}
		v.Reset(ssaop.OpPPC64SRDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBZreg (SRWconst [c] x))
	// cond: c>=24
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 24) {
			break
		}
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBZreg y:(MOVBZreg _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVBZreg {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVBZreg (MOVBreg x))
	// result: (MOVBZreg x)
	for {
		if v_0.Op != ssaop.OpPPC64MOVBreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64MOVBZreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBZreg (SRWconst x [s]))
	// cond: ssa.MergePPC64AndSrwi(0xFF,s) != 0
	// result: (RLWINM [ssa.MergePPC64AndSrwi(0xFF,s)] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		s := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.MergePPC64AndSrwi(0xFF, s) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.MergePPC64AndSrwi(0xFF, s))
		v.AddArg(x)
		return true
	}
	// match: (MOVBZreg (RLWINM [r] y))
	// cond: mergePPC64AndRlwinm(0xFF,r) != 0
	// result: (RLWINM [mergePPC64AndRlwinm(0xFF,r)] y)
	for {
		if v_0.Op != ssaop.OpPPC64RLWINM {
			break
		}
		r := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_0.Args[0]
		if !(mergePPC64AndRlwinm(0xFF, r) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(mergePPC64AndRlwinm(0xFF, r))
		v.AddArg(y)
		return true
	}
	// match: (MOVBZreg (OR <t> x (MOVWZreg y)))
	// result: (MOVBZreg (OR <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64OR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64OR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (XOR <t> x (MOVWZreg y)))
	// result: (MOVBZreg (XOR <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64XOR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64XOR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (AND <t> x (MOVWZreg y)))
	// result: (MOVBZreg (AND <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64AND {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64AND, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (OR <t> x (MOVHZreg y)))
	// result: (MOVBZreg (OR <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64OR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVHZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64OR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (XOR <t> x (MOVHZreg y)))
	// result: (MOVBZreg (XOR <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64XOR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVHZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64XOR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (AND <t> x (MOVHZreg y)))
	// result: (MOVBZreg (AND <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64AND {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVHZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64AND, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (OR <t> x (MOVBZreg y)))
	// result: (MOVBZreg (OR <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64OR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVBZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64OR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (XOR <t> x (MOVBZreg y)))
	// result: (MOVBZreg (XOR <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64XOR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVBZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64XOR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg (AND <t> x (MOVBZreg y)))
	// result: (MOVBZreg (AND <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64AND {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVBZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVBZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64AND, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVBZreg z:(ANDconst [c] (MOVBZload ptr x)))
	// result: z
	for {
		z := v_0
		if z.Op != ssaop.OpPPC64ANDconst {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != ssaop.OpPPC64MOVBZload {
			break
		}
		v.CopyOf(z)
		return true
	}
	// match: (MOVBZreg z:(AND y (MOVBZload ptr x)))
	// result: z
	for {
		z := v_0
		if z.Op != ssaop.OpPPC64AND {
			break
		}
		_ = z.Args[1]
		z_0 := z.Args[0]
		z_1 := z.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
			if z_1.Op != ssaop.OpPPC64MOVBZload {
				continue
			}
			v.CopyOf(z)
			return true
		}
		break
	}
	// match: (MOVBZreg x:(MOVBZload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVBZload {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBZreg x:(MOVBZloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVBZloadidx {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBZreg x:(Select0 (LoweredAtomicLoad8 _ _)))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpSelect0 {
			break
		}
		x_0 := x.Args[0]
		if x_0.Op != ssaop.OpPPC64LoweredAtomicLoad8 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBZreg x:(Arg <t>))
	// cond: ssa.Is8BitInt(t) && !t.IsSigned()
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpArg {
			break
		}
		t := x.Type
		if !(ssa.Is8BitInt(t) && !t.IsSigned()) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBZreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint8(c))])
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint8(c)))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVBreg y:(ANDconst [c] _))
	// cond: uint64(c) <= 0x7F
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(y.AuxInt)
		if !(uint64(c) <= 0x7F) {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVBreg (SRAWconst [c] (MOVBreg x)))
	// result: (SRAWconst [c] (MOVBreg x))
	for {
		if v_0.Op != ssaop.OpPPC64SRAWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpPPC64MOVBreg {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVBreg (SRAWconst [c] x))
	// cond: x.Type.Size() == 8
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRAWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(x.Type.Size() == 8) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SRDconst [c] x))
	// cond: c>56
	// result: (SRDconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > 56) {
			break
		}
		v.Reset(ssaop.OpPPC64SRDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SRDconst [c] x))
	// cond: c==56
	// result: (SRADconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c == 56) {
			break
		}
		v.Reset(ssaop.OpPPC64SRADconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SRADconst [c] x))
	// cond: c>=56
	// result: (SRADconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRADconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 56) {
			break
		}
		v.Reset(ssaop.OpPPC64SRADconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SRWconst [c] x))
	// cond: c>24
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > 24) {
			break
		}
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SRWconst [c] x))
	// cond: c==24
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c == 24) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SRAWconst [c] x))
	// cond: c>=24
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRAWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 24) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg y:(MOVBreg _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVBreg {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVBreg (MOVBZreg x))
	// result: (MOVBreg x)
	for {
		if v_0.Op != ssaop.OpPPC64MOVBZreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64MOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(Arg <t>))
	// cond: ssa.Is8BitInt(t) && t.IsSigned()
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpArg {
			break
		}
		t := x.Type
		if !(ssa.Is8BitInt(t) && t.IsSigned()) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int8(c))])
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int8(c)))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVBstore [off1] {sym} (ADDconst [off2] x) val mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (MOVBstore [off1+int32(off2)] {sym} x val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(x, val, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVBstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVBstorezero [off] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVBstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstore [0] {sym} p:(ADD ptr idx) val mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVBstoreidx ptr idx val mem)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVBreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBZreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVBZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHZreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWZreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (SRWconst (MOVHreg x) [c]) mem)
	// cond: c <= 8
	// result: (MOVBstore [off] {sym} ptr (SRWconst <typ.UInt32> x [c]) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		x := v_1_0.Args[0]
		mem := v_2
		if !(c <= 8) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (SRWconst (MOVHZreg x) [c]) mem)
	// cond: c <= 8
	// result: (MOVBstore [off] {sym} ptr (SRWconst <typ.UInt32> x [c]) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := v_1_0.Args[0]
		mem := v_2
		if !(c <= 8) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (SRWconst (MOVWreg x) [c]) mem)
	// cond: c <= 24
	// result: (MOVBstore [off] {sym} ptr (SRWconst <typ.UInt32> x [c]) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := v_1_0.Args[0]
		mem := v_2
		if !(c <= 24) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (SRWconst (MOVWZreg x) [c]) mem)
	// cond: c <= 24
	// result: (MOVBstore [off] {sym} ptr (SRWconst <typ.UInt32> x [c]) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_1_0.Args[0]
		mem := v_2
		if !(c <= 24) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVBstoreidx ptr (MOVDconst [c]) val mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVBstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstoreidx (MOVDconst [c]) ptr val mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVBstore [int32(c)] ptr val mem)
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVBreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64MOVBreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVBZreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64MOVBZreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVHreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVHZreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVWreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVWZreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (SRWconst (MOVHreg x) [c]) mem)
	// cond: c <= 8
	// result: (MOVBstoreidx ptr idx (SRWconst <typ.UInt32> x [c]) mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		x := v_2_0.Args[0]
		mem := v_3
		if !(c <= 8) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstoreidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
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
		if v_2.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := v_2_0.Args[0]
		mem := v_3
		if !(c <= 8) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstoreidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
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
		if v_2.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := v_2_0.Args[0]
		mem := v_3
		if !(c <= 24) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstoreidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
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
		if v_2.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_2_0.Args[0]
		mem := v_3
		if !(c <= 24) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstoreidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRWconst, typ.UInt32)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg4(ptr, idx, v0, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVBstorezero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstorezero [off1] {sym} (ADDconst [off2] x) mem)
	// cond: ((ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)) || (ssa.Is16Bit(int64(off1)+off2)))
	// result: (MOVBstorezero [off1+int32(off2)] {sym} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !((ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)) || (ssa.Is16Bit(int64(off1) + off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVBstorezero [off1] {sym1} p:(MOVDaddr [off2] {sym2} x) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (x.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVBstorezero [off1+off2] {ssa.MergeSym(sym1,sym2)} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		x := p.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (x.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(x, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVDaddr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDaddr {sym} [n] p:(ADD x y))
	// cond: sym == nil && n == 0
	// result: p
	for {
		n := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64ADD {
			break
		}
		if !(sym == nil && n == 0) {
			break
		}
		v.CopyOf(p)
		return true
	}
	// match: (MOVDaddr {sym} [n] ptr)
	// cond: sym == nil && n == 0 && (ptr.Op == ssaop.OpArgIntReg || ptr.Op == ssaop.OpPhi)
	// result: ptr
	for {
		n := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if !(sym == nil && n == 0 && (ptr.Op == ssaop.OpArgIntReg || ptr.Op == ssaop.OpPhi)) {
			break
		}
		v.CopyOf(ptr)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVDload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDload [off] {sym} ptr (FMOVDstore [off] {sym} ptr x _))
	// result: (MFVSRD x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64FMOVDstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpPPC64MFVSRD)
		v.AddArg(x)
		return true
	}
	// match: (MOVDload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVDload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off1] {sym} (ADDconst [off2] x) mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (MOVDload [off1+int32(off2)] {sym} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVDload [0] {sym} p:(ADD ptr idx) mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVDloadidx ptr idx mem)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		mem := v_1
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVDloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDloadidx ptr (MOVDconst [c]) mem)
	// cond: ((ssa.Is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVDload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !((ssa.Is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDloadidx (MOVDconst [c]) ptr mem)
	// cond: ((ssa.Is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVDload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !((ssa.Is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVDstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVDstore [off] {sym} ptr (MFVSRD x) mem)
	// result: (FMOVDstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MFVSRD {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64FMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (ADDconst [off2] x) val mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (MOVDstore [off1+int32(off2)] {sym} x val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(x, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVDstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVDstorezero [off] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVDstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDstore [0] {sym} p:(ADD ptr idx) val mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVDstoreidx ptr idx val mem)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} ptr r:(BRD val) mem)
	// cond: r.Uses == 1
	// result: (MOVDBRstore (MOVDaddr <ptr.Type> [off] {sym} ptr) val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		r := v_1
		if r.Op != ssaop.OpPPC64BRD {
			break
		}
		val := r.Args[0]
		mem := v_2
		if !(r.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDBRstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDaddr, ptr.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg(ptr)
		v.AddArg3(v0, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} ptr (Bswap64 val) mem)
	// result: (MOVDBRstore (MOVDaddr <ptr.Type> [off] {sym} ptr) val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpBswap64 {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVDBRstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDaddr, ptr.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg(ptr)
		v.AddArg3(v0, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVDstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstoreidx ptr (MOVDconst [c]) val mem)
	// cond: ((ssa.Is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVDstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !((ssa.Is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstoreidx (MOVDconst [c]) ptr val mem)
	// cond: ((ssa.Is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVDstore [int32(c)] ptr val mem)
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		val := v_2
		mem := v_3
		if !((ssa.Is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
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
		if r.Op != ssaop.OpPPC64BRD {
			break
		}
		val := r.Args[0]
		mem := v_3
		if !(r.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDBRstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVDstoreidx ptr idx (Bswap64 val) mem)
	// result: (MOVDBRstoreidx ptr idx val mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpBswap64 {
			break
		}
		val := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVDBRstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVDstorezero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstorezero [off1] {sym} (ADDconst [off2] x) mem)
	// cond: ((ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)) || (ssa.Is16Bit(int64(off1)+off2)))
	// result: (MOVDstorezero [off1+int32(off2)] {sym} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !((ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)) || (ssa.Is16Bit(int64(off1) + off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVDstorezero [off1] {sym1} p:(MOVDaddr [off2] {sym2} x) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (x.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVDstorezero [off1+off2] {ssa.MergeSym(sym1,sym2)} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		x := p.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (x.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(x, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHBRstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHBRstore ptr (MOVHreg x) mem)
	// result: (MOVHBRstore ptr x mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVHBRstore)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHBRstore ptr (MOVHZreg x) mem)
	// result: (MOVHBRstore ptr x mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVHBRstore)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHBRstore ptr (MOVWreg x) mem)
	// result: (MOVHBRstore ptr x mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVHBRstore)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHBRstore ptr (MOVWZreg x) mem)
	// result: (MOVHBRstore ptr x mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVHBRstore)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHZload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHZload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVHZload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHZload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHZload [off1] {sym} (ADDconst [off2] x) mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (MOVHZload [off1+int32(off2)] {sym} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHZload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVHZload [0] {sym} p:(ADD ptr idx) mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVHZloadidx ptr idx mem)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		mem := v_1
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHZloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHZloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHZloadidx ptr (MOVDconst [c]) mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVHZload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHZload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHZloadidx (MOVDconst [c]) ptr mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVHZload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHZload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHZreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVHZreg y:(ANDconst [c] _))
	// cond: uint64(c) <= 0xFFFF
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(y.AuxInt)
		if !(uint64(c) <= 0xFFFF) {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVHZreg (SRWconst [c] (MOVBZreg x)))
	// result: (SRWconst [c] (MOVBZreg x))
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpPPC64MOVBZreg {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVHZreg (SRWconst [c] (MOVHZreg x)))
	// result: (SRWconst [c] (MOVHZreg x))
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVHZreg (SRWconst [c] x))
	// cond: x.Type.Size() <= 16
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(x.Type.Size() <= 16) {
			break
		}
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHZreg (SRDconst [c] x))
	// cond: c>=48
	// result: (SRDconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 48) {
			break
		}
		v.Reset(ssaop.OpPPC64SRDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHZreg (SRWconst [c] x))
	// cond: c>=16
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 16) {
			break
		}
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHZreg (RLWINM [r] y))
	// cond: mergePPC64AndRlwinm(0xFFFF,r) != 0
	// result: (RLWINM [mergePPC64AndRlwinm(0xFFFF,r)] y)
	for {
		if v_0.Op != ssaop.OpPPC64RLWINM {
			break
		}
		r := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_0.Args[0]
		if !(mergePPC64AndRlwinm(0xFFFF, r) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(mergePPC64AndRlwinm(0xFFFF, r))
		v.AddArg(y)
		return true
	}
	// match: (MOVHZreg y:(MOVHZreg _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVHZreg y:(MOVBZreg _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVBZreg {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVHZreg y:(MOVHBRload _ _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVHBRload {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVHZreg y:(MOVHreg x))
	// result: (MOVHZreg x)
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		x := y.Args[0]
		v.Reset(ssaop.OpPPC64MOVHZreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHZreg (OR <t> x (MOVWZreg y)))
	// result: (MOVHZreg (OR <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64OR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVHZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64OR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVHZreg (XOR <t> x (MOVWZreg y)))
	// result: (MOVHZreg (XOR <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64XOR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVHZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64XOR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVHZreg (AND <t> x (MOVWZreg y)))
	// result: (MOVHZreg (AND <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64AND {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVHZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64AND, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVHZreg (OR <t> x (MOVHZreg y)))
	// result: (MOVHZreg (OR <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64OR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVHZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVHZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64OR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVHZreg (XOR <t> x (MOVHZreg y)))
	// result: (MOVHZreg (XOR <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64XOR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVHZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVHZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64XOR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVHZreg (AND <t> x (MOVHZreg y)))
	// result: (MOVHZreg (AND <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64AND {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVHZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVHZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64AND, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVHZreg z:(ANDconst [c] (MOVBZload ptr x)))
	// result: z
	for {
		z := v_0
		if z.Op != ssaop.OpPPC64ANDconst {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != ssaop.OpPPC64MOVBZload {
			break
		}
		v.CopyOf(z)
		return true
	}
	// match: (MOVHZreg z:(AND y (MOVHZload ptr x)))
	// result: z
	for {
		z := v_0
		if z.Op != ssaop.OpPPC64AND {
			break
		}
		_ = z.Args[1]
		z_0 := z.Args[0]
		z_1 := z.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
			if z_1.Op != ssaop.OpPPC64MOVHZload {
				continue
			}
			v.CopyOf(z)
			return true
		}
		break
	}
	// match: (MOVHZreg z:(ANDconst [c] (MOVHZload ptr x)))
	// result: z
	for {
		z := v_0
		if z.Op != ssaop.OpPPC64ANDconst {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != ssaop.OpPPC64MOVHZload {
			break
		}
		v.CopyOf(z)
		return true
	}
	// match: (MOVHZreg x:(MOVBZload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVBZload {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVHZreg x:(MOVBZloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVBZloadidx {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVHZreg x:(MOVHZload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVHZload {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVHZreg x:(MOVHZloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVHZloadidx {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVHZreg x:(Arg <t>))
	// cond: (ssa.Is8BitInt(t) || ssa.Is16BitInt(t)) && !t.IsSigned()
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpArg {
			break
		}
		t := x.Type
		if !((ssa.Is8BitInt(t) || ssa.Is16BitInt(t)) && !t.IsSigned()) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVHZreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint16(c))])
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint16(c)))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVHload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off1] {sym} (ADDconst [off2] x) mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (MOVHload [off1+int32(off2)] {sym} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVHload [0] {sym} p:(ADD ptr idx) mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVHloadidx ptr idx mem)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		mem := v_1
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHloadidx ptr (MOVDconst [c]) mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVHload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHloadidx (MOVDconst [c]) ptr mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVHload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVHreg y:(ANDconst [c] _))
	// cond: uint64(c) <= 0x7FFF
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(y.AuxInt)
		if !(uint64(c) <= 0x7FFF) {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVHreg (SRAWconst [c] (MOVBreg x)))
	// result: (SRAWconst [c] (MOVBreg x))
	for {
		if v_0.Op != ssaop.OpPPC64SRAWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpPPC64MOVBreg {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVHreg (SRAWconst [c] (MOVHreg x)))
	// result: (SRAWconst [c] (MOVHreg x))
	for {
		if v_0.Op != ssaop.OpPPC64SRAWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVHreg (SRAWconst [c] x))
	// cond: x.Type.Size() <= 16
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRAWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(x.Type.Size() <= 16) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SRDconst [c] x))
	// cond: c>48
	// result: (SRDconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > 48) {
			break
		}
		v.Reset(ssaop.OpPPC64SRDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SRDconst [c] x))
	// cond: c==48
	// result: (SRADconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c == 48) {
			break
		}
		v.Reset(ssaop.OpPPC64SRADconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SRADconst [c] x))
	// cond: c>=48
	// result: (SRADconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRADconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 48) {
			break
		}
		v.Reset(ssaop.OpPPC64SRADconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SRWconst [c] x))
	// cond: c>16
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > 16) {
			break
		}
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SRAWconst [c] x))
	// cond: c>=16
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRAWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 16) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SRWconst [c] x))
	// cond: c==16
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c == 16) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg y:(MOVHreg _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVHreg y:(MOVBreg _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVBreg {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVHreg y:(MOVHZreg x))
	// result: (MOVHreg x)
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := y.Args[0]
		v.Reset(ssaop.OpPPC64MOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVHload {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVHreg x:(MOVHloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVHloadidx {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVHreg x:(Arg <t>))
	// cond: (ssa.Is8BitInt(t) || ssa.Is16BitInt(t)) && t.IsSigned()
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpArg {
			break
		}
		t := x.Type
		if !((ssa.Is8BitInt(t) || ssa.Is16BitInt(t)) && t.IsSigned()) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVHreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int16(c))])
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int16(c)))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVHstore [off1] {sym} (ADDconst [off2] x) val mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (MOVHstore [off1+int32(off2)] {sym} x val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(x, val, mem)
		return true
	}
	// match: (MOVHstore [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVHstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVHstorezero [off] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVHstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstore [0] {sym} p:(ADD ptr idx) val mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVHstoreidx ptr idx val mem)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHZreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWZreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr r:(BRH val) mem)
	// cond: r.Uses == 1
	// result: (MOVHBRstore (MOVDaddr <ptr.Type> [off] {sym} ptr) val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		r := v_1
		if r.Op != ssaop.OpPPC64BRH {
			break
		}
		val := r.Args[0]
		mem := v_2
		if !(r.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHBRstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDaddr, ptr.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg(ptr)
		v.AddArg3(v0, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (Bswap16 val) mem)
	// result: (MOVHBRstore (MOVDaddr <ptr.Type> [off] {sym} ptr) val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpBswap16 {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVHBRstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDaddr, ptr.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg(ptr)
		v.AddArg3(v0, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstoreidx ptr (MOVDconst [c]) val mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVHstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstoreidx (MOVDconst [c]) ptr val mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVHstore [int32(c)] ptr val mem)
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVHreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVHZreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVWreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVWZreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVHstoreidx)
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
		if r.Op != ssaop.OpPPC64BRH {
			break
		}
		val := r.Args[0]
		mem := v_3
		if !(r.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHBRstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (Bswap16 val) mem)
	// result: (MOVHBRstoreidx ptr idx val mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpBswap16 {
			break
		}
		val := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVHBRstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVHstorezero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstorezero [off1] {sym} (ADDconst [off2] x) mem)
	// cond: ((ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)) || (ssa.Is16Bit(int64(off1)+off2)))
	// result: (MOVHstorezero [off1+int32(off2)] {sym} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !((ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)) || (ssa.Is16Bit(int64(off1) + off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVHstorezero [off1] {sym1} p:(MOVDaddr [off2] {sym2} x) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (x.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVHstorezero [off1+off2] {ssa.MergeSym(sym1,sym2)} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		x := p.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (x.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(x, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWBRstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWBRstore ptr (MOVWreg x) mem)
	// result: (MOVWBRstore ptr x mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVWBRstore)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWBRstore ptr (MOVWZreg x) mem)
	// result: (MOVWBRstore ptr x mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVWBRstore)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWZload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWZload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVWZload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWZload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWZload [off1] {sym} (ADDconst [off2] x) mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (MOVWZload [off1+int32(off2)] {sym} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWZload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVWZload [0] {sym} p:(ADD ptr idx) mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVWZloadidx ptr idx mem)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		mem := v_1
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWZloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWZloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWZloadidx ptr (MOVDconst [c]) mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVWZload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWZload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWZloadidx (MOVDconst [c]) ptr mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVWZload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWZload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWZreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVWZreg y:(ANDconst [c] _))
	// cond: uint64(c) <= 0xFFFFFFFF
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(y.AuxInt)
		if !(uint64(c) <= 0xFFFFFFFF) {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVWZreg y:(AND (MOVDconst [c]) _))
	// cond: uint64(c) <= 0xFFFFFFFF
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64AND {
			break
		}
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			if y_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(y_0.AuxInt)
			if !(uint64(c) <= 0xFFFFFFFF) {
				continue
			}
			v.CopyOf(y)
			return true
		}
		break
	}
	// match: (MOVWZreg (SRWconst [c] (MOVBZreg x)))
	// result: (SRWconst [c] (MOVBZreg x))
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpPPC64MOVBZreg {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWZreg (SRWconst [c] (MOVHZreg x)))
	// result: (SRWconst [c] (MOVHZreg x))
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWZreg (SRWconst [c] (MOVWZreg x)))
	// result: (SRWconst [c] (MOVWZreg x))
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWZreg (SRWconst [c] x))
	// cond: x.Type.Size() <= 32
	// result: (SRWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(x.Type.Size() <= 32) {
			break
		}
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWZreg (SRDconst [c] x))
	// cond: c>=32
	// result: (SRDconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 32) {
			break
		}
		v.Reset(ssaop.OpPPC64SRDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWZreg (RLWINM [r] y))
	// cond: mergePPC64MovwzregRlwinm(r) != 0
	// result: (RLWINM [mergePPC64MovwzregRlwinm(r)] y)
	for {
		if v_0.Op != ssaop.OpPPC64RLWINM {
			break
		}
		r := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_0.Args[0]
		if !(mergePPC64MovwzregRlwinm(r) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(mergePPC64MovwzregRlwinm(r))
		v.AddArg(y)
		return true
	}
	// match: (MOVWZreg w:(SLWconst u))
	// result: w
	for {
		w := v_0
		if w.Op != ssaop.OpPPC64SLWconst {
			break
		}
		v.CopyOf(w)
		return true
	}
	// match: (MOVWZreg y:(MOVWZreg _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVWZreg y:(MOVHZreg _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVWZreg y:(MOVBZreg _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVBZreg {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVWZreg y:(MOVHBRload _ _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVHBRload {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVWZreg y:(MOVWBRload _ _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVWBRload {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVWZreg y:(MOVWreg x))
	// result: (MOVWZreg x)
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := y.Args[0]
		v.Reset(ssaop.OpPPC64MOVWZreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWZreg (OR <t> x (MOVWZreg y)))
	// result: (MOVWZreg (OR <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64OR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVWZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64OR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVWZreg (XOR <t> x (MOVWZreg y)))
	// result: (MOVWZreg (XOR <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64XOR {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVWZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64XOR, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVWZreg (AND <t> x (MOVWZreg y)))
	// result: (MOVWZreg (AND <t> x y))
	for {
		if v_0.Op != ssaop.OpPPC64AND {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpPPC64MOVWZreg {
				continue
			}
			y := v_0_1.Args[0]
			v.Reset(ssaop.OpPPC64MOVWZreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpPPC64AND, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MOVWZreg z:(ANDconst [c] (MOVBZload ptr x)))
	// result: z
	for {
		z := v_0
		if z.Op != ssaop.OpPPC64ANDconst {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != ssaop.OpPPC64MOVBZload {
			break
		}
		v.CopyOf(z)
		return true
	}
	// match: (MOVWZreg z:(AND y (MOVWZload ptr x)))
	// result: z
	for {
		z := v_0
		if z.Op != ssaop.OpPPC64AND {
			break
		}
		_ = z.Args[1]
		z_0 := z.Args[0]
		z_1 := z.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
			if z_1.Op != ssaop.OpPPC64MOVWZload {
				continue
			}
			v.CopyOf(z)
			return true
		}
		break
	}
	// match: (MOVWZreg z:(ANDconst [c] (MOVHZload ptr x)))
	// result: z
	for {
		z := v_0
		if z.Op != ssaop.OpPPC64ANDconst {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != ssaop.OpPPC64MOVHZload {
			break
		}
		v.CopyOf(z)
		return true
	}
	// match: (MOVWZreg z:(ANDconst [c] (MOVWZload ptr x)))
	// result: z
	for {
		z := v_0
		if z.Op != ssaop.OpPPC64ANDconst {
			break
		}
		z_0 := z.Args[0]
		if z_0.Op != ssaop.OpPPC64MOVWZload {
			break
		}
		v.CopyOf(z)
		return true
	}
	// match: (MOVWZreg x:(MOVBZload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVBZload {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWZreg x:(MOVBZloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVBZloadidx {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWZreg x:(MOVHZload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVHZload {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWZreg x:(MOVHZloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVHZloadidx {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWZreg x:(MOVWZload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVWZload {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWZreg x:(MOVWZloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVWZloadidx {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWZreg x:(Select0 (LoweredAtomicLoad32 _ _)))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpSelect0 {
			break
		}
		x_0 := x.Args[0]
		if x_0.Op != ssaop.OpPPC64LoweredAtomicLoad32 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWZreg x:(Arg <t>))
	// cond: (ssa.Is8BitInt(t) || ssa.Is16BitInt(t) || ssa.Is32BitInt(t)) && !t.IsSigned()
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpArg {
			break
		}
		t := x.Type
		if !((ssa.Is8BitInt(t) || ssa.Is16BitInt(t) || ssa.Is32BitInt(t)) && !t.IsSigned()) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWZreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint32(c))])
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint32(c)))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWload [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVWload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off1] {sym} (ADDconst [off2] x) mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (MOVWload [off1+int32(off2)] {sym} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVWload [0] {sym} p:(ADD ptr idx) mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVWloadidx ptr idx mem)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		mem := v_1
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWloadidx ptr (MOVDconst [c]) mem)
	// cond: ((ssa.Is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVWload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !((ssa.Is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWloadidx (MOVDconst [c]) ptr mem)
	// cond: ((ssa.Is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVWload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !((ssa.Is16Bit(c) && c%4 == 0) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVWreg y:(ANDconst [c] _))
	// cond: uint64(c) <= 0xFFFF
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(y.AuxInt)
		if !(uint64(c) <= 0xFFFF) {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVWreg y:(AND (MOVDconst [c]) _))
	// cond: uint64(c) <= 0x7FFFFFFF
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64AND {
			break
		}
		y_0 := y.Args[0]
		y_1 := y.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, y_0, y_1 = _i0+1, y_1, y_0 {
			if y_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(y_0.AuxInt)
			if !(uint64(c) <= 0x7FFFFFFF) {
				continue
			}
			v.CopyOf(y)
			return true
		}
		break
	}
	// match: (MOVWreg (SRAWconst [c] (MOVBreg x)))
	// result: (SRAWconst [c] (MOVBreg x))
	for {
		if v_0.Op != ssaop.OpPPC64SRAWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpPPC64MOVBreg {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWreg (SRAWconst [c] (MOVHreg x)))
	// result: (SRAWconst [c] (MOVHreg x))
	for {
		if v_0.Op != ssaop.OpPPC64SRAWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWreg (SRAWconst [c] (MOVWreg x)))
	// result: (SRAWconst [c] (MOVWreg x))
	for {
		if v_0.Op != ssaop.OpPPC64SRAWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWreg (SRAWconst [c] x))
	// cond: x.Type.Size() <= 32
	// result: (SRAWconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRAWconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(x.Type.Size() <= 32) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (SRDconst [c] x))
	// cond: c>32
	// result: (SRDconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c > 32) {
			break
		}
		v.Reset(ssaop.OpPPC64SRDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (SRADconst [c] x))
	// cond: c>=32
	// result: (SRADconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRADconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c >= 32) {
			break
		}
		v.Reset(ssaop.OpPPC64SRADconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (SRDconst [c] x))
	// cond: c==32
	// result: (SRADconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SRDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c == 32) {
			break
		}
		v.Reset(ssaop.OpPPC64SRADconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg y:(MOVWreg _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVWreg y:(MOVHreg _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVHreg {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVWreg y:(MOVBreg _))
	// result: y
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVBreg {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (MOVWreg y:(MOVWZreg x))
	// result: (MOVWreg x)
	for {
		y := v_0
		if y.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := y.Args[0]
		v.Reset(ssaop.OpPPC64MOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVHload {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWreg x:(MOVHloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVHloadidx {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWreg x:(MOVWload _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVWload {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWreg x:(MOVWloadidx _ _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVWloadidx {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWreg x:(Arg <t>))
	// cond: (ssa.Is8BitInt(t) || ssa.Is16BitInt(t) || ssa.Is32BitInt(t)) && t.IsSigned()
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpArg {
			break
		}
		t := x.Type
		if !((ssa.Is8BitInt(t) || ssa.Is16BitInt(t) || ssa.Is32BitInt(t)) && t.IsSigned()) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int32(c))])
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int32(c)))
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVWstore [off1] {sym} (ADDconst [off2] x) val mem)
	// cond: (ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)))
	// result: (MOVWstore [off1+int32(off2)] {sym} x val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is16Bit(int64(off1)+off2) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(x, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} p:(MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVWstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (ptr.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVDconst [0]) mem)
	// result: (MOVWstorezero [off] {sym} ptr mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVWstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstore [0] {sym} p:(ADD ptr idx) val mem)
	// cond: sym == nil && p.Uses == 1
	// result: (MOVWstoreidx ptr idx val mem)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64ADD {
			break
		}
		idx := p.Args[1]
		ptr := p.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil && p.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWZreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr r:(BRW val) mem)
	// cond: r.Uses == 1
	// result: (MOVWBRstore (MOVDaddr <ptr.Type> [off] {sym} ptr) val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		r := v_1
		if r.Op != ssaop.OpPPC64BRW {
			break
		}
		val := r.Args[0]
		mem := v_2
		if !(r.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWBRstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDaddr, ptr.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg(ptr)
		v.AddArg3(v0, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (Bswap32 val) mem)
	// result: (MOVWBRstore (MOVDaddr <ptr.Type> [off] {sym} ptr) val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpBswap32 {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpPPC64MOVWBRstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDaddr, ptr.Type)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg(ptr)
		v.AddArg3(v0, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreidx ptr (MOVDconst [c]) val mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVWstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstoreidx (MOVDconst [c]) ptr val mem)
	// cond: (ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c)))
	// result: (MOVWstore [int32(c)] ptr val mem)
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is16Bit(c) || (buildcfg.GOPPC64 >= 10 && ssa.Is32Bit(c))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVWreg x) mem)
	// result: (MOVWstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVWstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVWZreg x) mem)
	// result: (MOVWstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVWstoreidx)
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
		if r.Op != ssaop.OpPPC64BRW {
			break
		}
		val := r.Args[0]
		mem := v_3
		if !(r.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWBRstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (Bswap32 val) mem)
	// result: (MOVWBRstoreidx ptr idx val mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpBswap32 {
			break
		}
		val := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpPPC64MOVWBRstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MOVWstorezero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstorezero [off1] {sym} (ADDconst [off2] x) mem)
	// cond: ((ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)) || (ssa.Is16Bit(int64(off1)+off2)))
	// result: (MOVWstorezero [off1+int32(off2)] {sym} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		mem := v_1
		if !((ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1)+off2)) || (ssa.Is16Bit(int64(off1) + off2))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(x, mem)
		return true
	}
	// match: (MOVWstorezero [off1] {sym1} p:(MOVDaddr [off2] {sym2} x) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (x.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))
	// result: (MOVWstorezero [off1+off2] {ssa.MergeSym(sym1,sym2)} x mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		p := v_0
		if p.Op != ssaop.OpPPC64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(p.AuxInt)
		sym2 := ssa.AuxToSym(p.Aux)
		x := p.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ((ssa.Is16Bit(int64(off1+off2)) && (x.Op != ssaop.OpSB || p.Uses == 1)) || (ssa.SupportsPPC64PCRel() && ssa.Is32Bit(int64(off1+off2))))) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(x, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MTVSRD(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MTVSRD (MOVDconst [c]))
	// cond: !math.IsNaN(math.Float64frombits(uint64(c)))
	// result: (FMOVDconst [math.Float64frombits(uint64(c))])
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(!math.IsNaN(math.Float64frombits(uint64(c)))) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVDconst)
		v.AuxInt = ssa.Float64ToAuxInt(math.Float64frombits(uint64(c)))
		return true
	}
	// match: (MTVSRD x:(MOVDload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && ssa.Clobber(x)
	// result: @x.Block (FMOVDload [off] {sym} ptr mem)
	for {
		x := v_0
		if x.Op != ssaop.OpPPC64MOVDload {
			break
		}
		off := ssa.AuxIntToInt32(x.AuxInt)
		sym := ssa.AuxToSym(x.Aux)
		mem := x.Args[1]
		ptr := x.Args[0]
		if !(x.Uses == 1 && ssa.Clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(x.Pos, ssaop.OpPPC64FMOVDload, typ.Float64)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		v0.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64MULLD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULLD x (MOVDconst [c]))
	// cond: ssa.Is16Bit(c)
	// result: (MULLDconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.Is16Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpPPC64MULLDconst)
			v.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64MULLW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULLW x (MOVDconst [c]))
	// cond: ssa.Is16Bit(c)
	// result: (MULLWconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.Is16Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpPPC64MULLWconst)
			v.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64NEG(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NEG (ADDconst [c] x))
	// cond: ssa.Is32Bit(-c)
	// result: (SUBFCconst [-c] x)
	for {
		if v_0.Op != ssaop.OpPPC64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(-c)) {
			break
		}
		v.Reset(ssaop.OpPPC64SUBFCconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	// match: (NEG (SUBFCconst [c] x))
	// cond: ssa.Is32Bit(-c)
	// result: (ADDconst [-c] x)
	for {
		if v_0.Op != ssaop.OpPPC64SUBFCconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(-c)) {
			break
		}
		v.Reset(ssaop.OpPPC64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	// match: (NEG (SUB x y))
	// result: (SUB y x)
	for {
		if v_0.Op != ssaop.OpPPC64SUB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SUB)
		v.AddArg2(y, x)
		return true
	}
	// match: (NEG (NEG x))
	// result: x
	for {
		if v_0.Op != ssaop.OpPPC64NEG {
			break
		}
		x := v_0.Args[0]
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64NOR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NOR (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [^(c|d)])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			d := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpPPC64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(^(c | d))
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64NotEqual(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NotEqual (FlagEQ))
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (NotEqual (FlagLT))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (NotEqual (FlagGT))
	// result: (MOVDconst [1])
	for {
		if v_0.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (NotEqual (InvertFlags x))
	// result: (NotEqual x)
	for {
		if v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64NotEqual)
		v.AddArg(x)
		return true
	}
	// match: (NotEqual cmp)
	// result: (SETBCR [2] cmp)
	for {
		cmp := v_0
		v.Reset(ssaop.OpPPC64SETBCR)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v.AddArg(cmp)
		return true
	}
}
func rewriteValuePPC64_OpPPC64OR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (OR <t> x (MOVDconst [m]))
	// cond: t.IsUnsigned() && t.Size() == 1 && m != int64(uint8(m))
	// result: (ORconst [int64(uint8(m))] x)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(t.IsUnsigned() && t.Size() == 1 && m != int64(uint8(m))) {
				continue
			}
			v.Reset(ssaop.OpPPC64ORconst)
			v.AuxInt = ssa.Int64ToAuxInt(int64(uint8(m)))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (OR <t> x (MOVDconst [m]))
	// cond: t.IsUnsigned() && t.Size() == 2 && m != int64(uint16(m))
	// result: (ORconst [int64(uint16(m))] x)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(t.IsUnsigned() && t.Size() == 2 && m != int64(uint16(m))) {
				continue
			}
			v.Reset(ssaop.OpPPC64ORconst)
			v.AuxInt = ssa.Int64ToAuxInt(int64(uint16(m)))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (OR x (NOR y y))
	// result: (ORN x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64NOR {
				continue
			}
			y := v_1.Args[1]
			if y != v_1.Args[0] {
				continue
			}
			v.Reset(ssaop.OpPPC64ORN)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [c|d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			d := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpPPC64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c | d)
			return true
		}
		break
	}
	// match: (OR x (MOVDconst [c]))
	// cond: ssa.IsU32Bit(c)
	// result: (ORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.IsU32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpPPC64ORconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64ORN(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORN x (MOVDconst [-1]))
	// result: x
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ORN (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [c|^d])
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c | ^d)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64ORconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ORconst [c] (ORconst [d] x))
	// result: (ORconst [c|d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ORconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c | d)
		v.AddArg(x)
		return true
	}
	// match: (ORconst [-1] _)
	// result: (MOVDconst [-1])
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != -1 {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	// match: (ORconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64RLWINM(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (RLWINM [r] (MOVHZreg u))
	// cond: mergePPC64RlwinmAnd(r,0xFFFF) != 0
	// result: (RLWINM [mergePPC64RlwinmAnd(r,0xFFFF)] u)
	for {
		r := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		u := v_0.Args[0]
		if !(mergePPC64RlwinmAnd(r, 0xFFFF) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(mergePPC64RlwinmAnd(r, 0xFFFF))
		v.AddArg(u)
		return true
	}
	// match: (RLWINM [r] (ANDconst [a] u))
	// cond: mergePPC64RlwinmAnd(r,uint32(a)) != 0
	// result: (RLWINM [mergePPC64RlwinmAnd(r,uint32(a))] u)
	for {
		r := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ANDconst {
			break
		}
		a := ssa.AuxIntToInt64(v_0.AuxInt)
		u := v_0.Args[0]
		if !(mergePPC64RlwinmAnd(r, uint32(a)) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(mergePPC64RlwinmAnd(r, uint32(a)))
		v.AddArg(u)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64ROTL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROTL x (MOVDconst [c]))
	// result: (ROTLconst x [c&63])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpPPC64ROTLconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64ROTLW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROTLW x (MOVDconst [c]))
	// result: (ROTLWconst x [c&31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpPPC64ROTLWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64ROTLWconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ROTLWconst [r] (AND (MOVDconst [m]) x))
	// cond: ssa.IsPPC64WordRotateMask(m)
	// result: (RLWINM [ssa.EncodePPC64RotateMask(r,rotateLeft32(m,r),32)] x)
	for {
		r := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_0_0.AuxInt)
			x := v_0_1
			if !(ssa.IsPPC64WordRotateMask(m)) {
				continue
			}
			v.Reset(ssaop.OpPPC64RLWINM)
			v.AuxInt = ssa.Int64ToAuxInt(ssa.EncodePPC64RotateMask(r, rotateLeft32(m, r), 32))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ROTLWconst [r] (ANDconst [m] x))
	// cond: ssa.IsPPC64WordRotateMask(m)
	// result: (RLWINM [ssa.EncodePPC64RotateMask(r,rotateLeft32(m,r),32)] x)
	for {
		r := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ANDconst {
			break
		}
		m := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.IsPPC64WordRotateMask(m)) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.EncodePPC64RotateMask(r, rotateLeft32(m, r), 32))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SETBC(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETBC [0] (FlagLT))
	// result: (MOVDconst [1])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 || v_0.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SETBC [0] (FlagGT))
	// result: (MOVDconst [0])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 || v_0.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SETBC [0] (FlagEQ))
	// result: (MOVDconst [0])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 || v_0.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SETBC [1] (FlagGT))
	// result: (MOVDconst [1])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 1 || v_0.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SETBC [1] (FlagLT))
	// result: (MOVDconst [0])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 1 || v_0.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SETBC [1] (FlagEQ))
	// result: (MOVDconst [0])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 1 || v_0.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SETBC [2] (FlagEQ))
	// result: (MOVDconst [1])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SETBC [2] (FlagLT))
	// result: (MOVDconst [0])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SETBC [2] (FlagGT))
	// result: (MOVDconst [0])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SETBC [0] (InvertFlags bool))
	// result: (SETBC [1] bool)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 || v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v.AddArg(bool)
		return true
	}
	// match: (SETBC [1] (InvertFlags bool))
	// result: (SETBC [0] bool)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 1 || v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg(bool)
		return true
	}
	// match: (SETBC [2] (InvertFlags bool))
	// result: (SETBC [2] bool)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v.AddArg(bool)
		return true
	}
	// match: (SETBC [n] (InvertFlags bool))
	// result: (SETBCR [n] bool)
	for {
		n := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SETBCR)
		v.AuxInt = ssa.Int32ToAuxInt(n)
		v.AddArg(bool)
		return true
	}
	// match: (SETBC [2] (CMPconst [0] a:(ANDconst [1] _)))
	// result: (XORconst [1] a)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		a := v_0.Args[0]
		if a.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(a.AuxInt) != 1 {
			break
		}
		v.Reset(ssaop.OpPPC64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg(a)
		return true
	}
	// match: (SETBC [2] (CMPconst [0] a:(AND y z)))
	// cond: a.Uses == 1
	// result: (SETBC [2] (Select1 <types.TypeFlags> (ANDCC y z )))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		a := v_0.Args[0]
		if a.Op != ssaop.OpPPC64AND {
			break
		}
		z := a.Args[1]
		y := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
		v1.AddArg2(y, z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SETBC [2] (CMPconst [0] o:(OR y z)))
	// cond: o.Uses == 1
	// result: (SETBC [2] (Select1 <types.TypeFlags> (ORCC y z )))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		o := v_0.Args[0]
		if o.Op != ssaop.OpPPC64OR {
			break
		}
		z := o.Args[1]
		y := o.Args[0]
		if !(o.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AddArg2(y, z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SETBC [2] (CMPconst [0] a:(XOR y z)))
	// cond: a.Uses == 1
	// result: (SETBC [2] (Select1 <types.TypeFlags> (XORCC y z )))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		a := v_0.Args[0]
		if a.Op != ssaop.OpPPC64XOR {
			break
		}
		z := a.Args[1]
		y := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AddArg2(y, z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SETBCR(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SETBCR [0] (FlagLT))
	// result: (MOVDconst [0])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 || v_0.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SETBCR [0] (FlagGT))
	// result: (MOVDconst [1])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 || v_0.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SETBCR [0] (FlagEQ))
	// result: (MOVDconst [1])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 || v_0.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SETBCR [1] (FlagGT))
	// result: (MOVDconst [0])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 1 || v_0.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SETBCR [1] (FlagLT))
	// result: (MOVDconst [1])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 1 || v_0.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SETBCR [1] (FlagEQ))
	// result: (MOVDconst [1])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 1 || v_0.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SETBCR [2] (FlagEQ))
	// result: (MOVDconst [0])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64FlagEQ {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SETBCR [2] (FlagLT))
	// result: (MOVDconst [1])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64FlagLT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SETBCR [2] (FlagGT))
	// result: (MOVDconst [1])
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64FlagGT {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SETBCR [0] (InvertFlags bool))
	// result: (SETBCR [1] bool)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 0 || v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SETBCR)
		v.AuxInt = ssa.Int32ToAuxInt(1)
		v.AddArg(bool)
		return true
	}
	// match: (SETBCR [1] (InvertFlags bool))
	// result: (SETBCR [0] bool)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 1 || v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SETBCR)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v.AddArg(bool)
		return true
	}
	// match: (SETBCR [2] (InvertFlags bool))
	// result: (SETBCR [2] bool)
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SETBCR)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v.AddArg(bool)
		return true
	}
	// match: (SETBCR [n] (InvertFlags bool))
	// result: (SETBC [n] bool)
	for {
		n := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64InvertFlags {
			break
		}
		bool := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(n)
		v.AddArg(bool)
		return true
	}
	// match: (SETBCR [2] (CMPconst [0] a:(ANDconst [1] _)))
	// result: a
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		a := v_0.Args[0]
		if a.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(a.AuxInt) != 1 {
			break
		}
		v.CopyOf(a)
		return true
	}
	// match: (SETBCR [2] (CMPconst [0] a:(AND y z)))
	// cond: a.Uses == 1
	// result: (SETBCR [2] (Select1 <types.TypeFlags> (ANDCC y z )))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		a := v_0.Args[0]
		if a.Op != ssaop.OpPPC64AND {
			break
		}
		z := a.Args[1]
		y := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64SETBCR)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
		v1.AddArg2(y, z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SETBCR [2] (CMPconst [0] o:(OR y z)))
	// cond: o.Uses == 1
	// result: (SETBCR [2] (Select1 <types.TypeFlags> (ORCC y z )))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		o := v_0.Args[0]
		if o.Op != ssaop.OpPPC64OR {
			break
		}
		z := o.Args[1]
		y := o.Args[0]
		if !(o.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64SETBCR)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AddArg2(y, z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (SETBCR [2] (CMPconst [0] a:(XOR y z)))
	// cond: a.Uses == 1
	// result: (SETBCR [2] (Select1 <types.TypeFlags> (XORCC y z )))
	for {
		if ssa.AuxIntToInt32(v.AuxInt) != 2 || v_0.Op != ssaop.OpPPC64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		a := v_0.Args[0]
		if a.Op != ssaop.OpPPC64XOR {
			break
		}
		z := a.Args[1]
		y := a.Args[0]
		if !(a.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64SETBCR)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
		v1.AddArg2(y, z)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SLD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLD x (MOVDconst [c]))
	// result: (SLDconst [c&63 | (c>>6&1*63)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpPPC64SLDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c&63 | (c >> 6 & 1 * 63))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SLDconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SLDconst [l] (SRWconst [r] x))
	// cond: ssa.MergePPC64SldiSrw(l,r) != 0
	// result: (RLWINM [ssa.MergePPC64SldiSrw(l,r)] x)
	for {
		l := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64SRWconst {
			break
		}
		r := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.MergePPC64SldiSrw(l, r) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.MergePPC64SldiSrw(l, r))
		v.AddArg(x)
		return true
	}
	// match: (SLDconst [s] (RLWINM [r] y))
	// cond: mergePPC64SldiRlwinm(s,r) != 0
	// result: (RLWINM [mergePPC64SldiRlwinm(s,r)] y)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64RLWINM {
			break
		}
		r := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_0.Args[0]
		if !(mergePPC64SldiRlwinm(s, r) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(mergePPC64SldiRlwinm(s, r))
		v.AddArg(y)
		return true
	}
	// match: (SLDconst [c] z:(MOVBZreg x))
	// cond: c < 8 && z.Uses == 1
	// result: (CLRLSLDI [ssa.NewPPC64ShiftAuxInt(c,56,63,64)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != ssaop.OpPPC64MOVBZreg {
			break
		}
		x := z.Args[0]
		if !(c < 8 && z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64CLRLSLDI)
		v.AuxInt = ssa.Int32ToAuxInt(ssa.NewPPC64ShiftAuxInt(c, 56, 63, 64))
		v.AddArg(x)
		return true
	}
	// match: (SLDconst [c] z:(MOVHZreg x))
	// cond: c < 16 && z.Uses == 1
	// result: (CLRLSLDI [ssa.NewPPC64ShiftAuxInt(c,48,63,64)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := z.Args[0]
		if !(c < 16 && z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64CLRLSLDI)
		v.AuxInt = ssa.Int32ToAuxInt(ssa.NewPPC64ShiftAuxInt(c, 48, 63, 64))
		v.AddArg(x)
		return true
	}
	// match: (SLDconst [c] z:(MOVWZreg x))
	// cond: c < 32 && z.Uses == 1
	// result: (CLRLSLDI [ssa.NewPPC64ShiftAuxInt(c,32,63,64)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		x := z.Args[0]
		if !(c < 32 && z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64CLRLSLDI)
		v.AuxInt = ssa.Int32ToAuxInt(ssa.NewPPC64ShiftAuxInt(c, 32, 63, 64))
		v.AddArg(x)
		return true
	}
	// match: (SLDconst [c] z:(ANDconst [d] x))
	// cond: z.Uses == 1 && ssa.IsPPC64ValidShiftMask(d) && c <= (64-getPPC64ShiftMaskLength(d))
	// result: (CLRLSLDI [ssa.NewPPC64ShiftAuxInt(c,64-getPPC64ShiftMaskLength(d),63,64)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != ssaop.OpPPC64ANDconst {
			break
		}
		d := ssa.AuxIntToInt64(z.AuxInt)
		x := z.Args[0]
		if !(z.Uses == 1 && ssa.IsPPC64ValidShiftMask(d) && c <= (64-getPPC64ShiftMaskLength(d))) {
			break
		}
		v.Reset(ssaop.OpPPC64CLRLSLDI)
		v.AuxInt = ssa.Int32ToAuxInt(ssa.NewPPC64ShiftAuxInt(c, 64-getPPC64ShiftMaskLength(d), 63, 64))
		v.AddArg(x)
		return true
	}
	// match: (SLDconst [c] z:(AND (MOVDconst [d]) x))
	// cond: z.Uses == 1 && ssa.IsPPC64ValidShiftMask(d) && c<=(64-getPPC64ShiftMaskLength(d))
	// result: (CLRLSLDI [ssa.NewPPC64ShiftAuxInt(c,64-getPPC64ShiftMaskLength(d),63,64)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != ssaop.OpPPC64AND {
			break
		}
		_ = z.Args[1]
		z_0 := z.Args[0]
		z_1 := z.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
			if z_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			d := ssa.AuxIntToInt64(z_0.AuxInt)
			x := z_1
			if !(z.Uses == 1 && ssa.IsPPC64ValidShiftMask(d) && c <= (64-getPPC64ShiftMaskLength(d))) {
				continue
			}
			v.Reset(ssaop.OpPPC64CLRLSLDI)
			v.AuxInt = ssa.Int32ToAuxInt(ssa.NewPPC64ShiftAuxInt(c, 64-getPPC64ShiftMaskLength(d), 63, 64))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (SLDconst [c] (ADD x x))
	// cond: c < 63
	// result: (SLDconst [c+1] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ADD {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || !(c < 63) {
			break
		}
		v.Reset(ssaop.OpPPC64SLDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c + 1)
		v.AddArg(x)
		return true
	}
	// match: (SLDconst [c] z:(MOVWreg x))
	// cond: c < 32 && buildcfg.GOPPC64 >= 9
	// result: (EXTSWSLconst [c] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := z.Args[0]
		if !(c < 32 && buildcfg.GOPPC64 >= 9) {
			break
		}
		v.Reset(ssaop.OpPPC64EXTSWSLconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SLW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLW x (MOVDconst [c]))
	// result: (SLWconst [c&31 | (c>>5&1*31)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpPPC64SLWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c&31 | (c >> 5 & 1 * 31))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SLWconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SLWconst [s] (MOVWZreg w))
	// result: (SLWconst [s] w)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64MOVWZreg {
			break
		}
		w := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SLWconst)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg(w)
		return true
	}
	// match: (SLWconst [c] z:(MOVBZreg x))
	// cond: z.Uses == 1 && c < 8
	// result: (CLRLSLWI [ssa.NewPPC64ShiftAuxInt(c,24,31,32)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != ssaop.OpPPC64MOVBZreg {
			break
		}
		x := z.Args[0]
		if !(z.Uses == 1 && c < 8) {
			break
		}
		v.Reset(ssaop.OpPPC64CLRLSLWI)
		v.AuxInt = ssa.Int32ToAuxInt(ssa.NewPPC64ShiftAuxInt(c, 24, 31, 32))
		v.AddArg(x)
		return true
	}
	// match: (SLWconst [c] z:(MOVHZreg x))
	// cond: z.Uses == 1 && c < 16
	// result: (CLRLSLWI [ssa.NewPPC64ShiftAuxInt(c,16,31,32)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != ssaop.OpPPC64MOVHZreg {
			break
		}
		x := z.Args[0]
		if !(z.Uses == 1 && c < 16) {
			break
		}
		v.Reset(ssaop.OpPPC64CLRLSLWI)
		v.AuxInt = ssa.Int32ToAuxInt(ssa.NewPPC64ShiftAuxInt(c, 16, 31, 32))
		v.AddArg(x)
		return true
	}
	// match: (SLWconst [c] z:(ANDconst [d] x))
	// cond: z.Uses == 1 && ssa.IsPPC64ValidShiftMask(d) && c<=(32-getPPC64ShiftMaskLength(d))
	// result: (CLRLSLWI [ssa.NewPPC64ShiftAuxInt(c,32-getPPC64ShiftMaskLength(d),31,32)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != ssaop.OpPPC64ANDconst {
			break
		}
		d := ssa.AuxIntToInt64(z.AuxInt)
		x := z.Args[0]
		if !(z.Uses == 1 && ssa.IsPPC64ValidShiftMask(d) && c <= (32-getPPC64ShiftMaskLength(d))) {
			break
		}
		v.Reset(ssaop.OpPPC64CLRLSLWI)
		v.AuxInt = ssa.Int32ToAuxInt(ssa.NewPPC64ShiftAuxInt(c, 32-getPPC64ShiftMaskLength(d), 31, 32))
		v.AddArg(x)
		return true
	}
	// match: (SLWconst [c] z:(AND (MOVDconst [d]) x))
	// cond: z.Uses == 1 && ssa.IsPPC64ValidShiftMask(d) && c<=(32-getPPC64ShiftMaskLength(d))
	// result: (CLRLSLWI [ssa.NewPPC64ShiftAuxInt(c,32-getPPC64ShiftMaskLength(d),31,32)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != ssaop.OpPPC64AND {
			break
		}
		_ = z.Args[1]
		z_0 := z.Args[0]
		z_1 := z.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, z_0, z_1 = _i0+1, z_1, z_0 {
			if z_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			d := ssa.AuxIntToInt64(z_0.AuxInt)
			x := z_1
			if !(z.Uses == 1 && ssa.IsPPC64ValidShiftMask(d) && c <= (32-getPPC64ShiftMaskLength(d))) {
				continue
			}
			v.Reset(ssaop.OpPPC64CLRLSLWI)
			v.AuxInt = ssa.Int32ToAuxInt(ssa.NewPPC64ShiftAuxInt(c, 32-getPPC64ShiftMaskLength(d), 31, 32))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (SLWconst [c] (ADD x x))
	// cond: c < 31
	// result: (SLWconst [c+1] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ADD {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || !(c < 31) {
			break
		}
		v.Reset(ssaop.OpPPC64SLWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c + 1)
		v.AddArg(x)
		return true
	}
	// match: (SLWconst [c] z:(MOVWreg x))
	// cond: c < 32 && buildcfg.GOPPC64 >= 9
	// result: (EXTSWSLconst [c] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		z := v_0
		if z.Op != ssaop.OpPPC64MOVWreg {
			break
		}
		x := z.Args[0]
		if !(c < 32 && buildcfg.GOPPC64 >= 9) {
			break
		}
		v.Reset(ssaop.OpPPC64EXTSWSLconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SRAD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRAD x (MOVDconst [c]))
	// result: (SRADconst [c&63 | (c>>6&1*63)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpPPC64SRADconst)
		v.AuxInt = ssa.Int64ToAuxInt(c&63 | (c >> 6 & 1 * 63))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SRAW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRAW x (MOVDconst [c]))
	// result: (SRAWconst [c&31 | (c>>5&1*31)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c&31 | (c >> 5 & 1 * 31))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SRD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRD x (MOVDconst [c]))
	// result: (SRDconst [c&63 | (c>>6&1*63)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpPPC64SRDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c&63 | (c >> 6 & 1 * 63))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SRW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRW x (MOVDconst [c]))
	// result: (SRWconst [c&31 | (c>>5&1*31)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c&31 | (c >> 5 & 1 * 31))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SRWconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SRWconst (ANDconst [m] x) [s])
	// cond: ssa.MergePPC64RShiftMask(m>>uint(s),s,32) == 0
	// result: (MOVDconst [0])
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ANDconst {
			break
		}
		m := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(ssa.MergePPC64RShiftMask(m>>uint(s), s, 32) == 0) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SRWconst (ANDconst [m] x) [s])
	// cond: ssa.MergePPC64AndSrwi(m>>uint(s),s) != 0
	// result: (RLWINM [ssa.MergePPC64AndSrwi(m>>uint(s),s)] x)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64ANDconst {
			break
		}
		m := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.MergePPC64AndSrwi(m>>uint(s), s) != 0) {
			break
		}
		v.Reset(ssaop.OpPPC64RLWINM)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.MergePPC64AndSrwi(m>>uint(s), s))
		v.AddArg(x)
		return true
	}
	// match: (SRWconst (AND (MOVDconst [m]) x) [s])
	// cond: ssa.MergePPC64RShiftMask(m>>uint(s),s,32) == 0
	// result: (MOVDconst [0])
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_0_0.AuxInt)
			if !(ssa.MergePPC64RShiftMask(m>>uint(s), s, 32) == 0) {
				continue
			}
			v.Reset(ssaop.OpPPC64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (SRWconst (AND (MOVDconst [m]) x) [s])
	// cond: ssa.MergePPC64AndSrwi(m>>uint(s),s) != 0
	// result: (RLWINM [ssa.MergePPC64AndSrwi(m>>uint(s),s)] x)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64AND {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_0_0.AuxInt)
			x := v_0_1
			if !(ssa.MergePPC64AndSrwi(m>>uint(s), s) != 0) {
				continue
			}
			v.Reset(ssaop.OpPPC64RLWINM)
			v.AuxInt = ssa.Int64ToAuxInt(ssa.MergePPC64AndSrwi(m>>uint(s), s))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64SUB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUB x (MOVDconst [c]))
	// cond: ssa.Is32Bit(-c)
	// result: (ADDconst [-c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.Is32Bit(-c)) {
			break
		}
		v.Reset(ssaop.OpPPC64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	// match: (SUB (MOVDconst [c]) x)
	// cond: ssa.Is32Bit(c)
	// result: (SUBFCconst [c] x)
	for {
		if v_0.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpPPC64SUBFCconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SUBE(v *ssa.Value) bool {
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
		if v_2.Op != ssaop.OpSelect1 || v_2.Type != typ.UInt64 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpPPC64SUBCconst || ssa.AuxIntToInt64(v_2_0.AuxInt) != 0 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != ssaop.OpPPC64MOVDconst || ssa.AuxIntToInt64(v_2_0_0.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpPPC64SUBC)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64SUBFCconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SUBFCconst [c] (NEG x))
	// result: (ADDconst [c] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64NEG {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUBFCconst [c] (SUBFCconst [d] x))
	// cond: ssa.Is32Bit(c-d)
	// result: (ADDconst [c-d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64SUBFCconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(c - d)) {
			break
		}
		v.Reset(ssaop.OpPPC64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	// match: (SUBFCconst [0] x)
	// result: (NEG x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.Reset(ssaop.OpPPC64NEG)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPPC64XOR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XOR <t> x (MOVDconst [m]))
	// cond: t.IsUnsigned() && t.Size() == 1 && m != int64(uint8(m))
	// result: (XORconst [int64(uint8(m))] x)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(t.IsUnsigned() && t.Size() == 1 && m != int64(uint8(m))) {
				continue
			}
			v.Reset(ssaop.OpPPC64XORconst)
			v.AuxInt = ssa.Int64ToAuxInt(int64(uint8(m)))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XOR <t> x (MOVDconst [m]))
	// cond: t.IsUnsigned() && t.Size() == 2 && m != int64(uint16(m))
	// result: (XORconst [int64(uint16(m))] x)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			m := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(t.IsUnsigned() && t.Size() == 2 && m != int64(uint16(m))) {
				continue
			}
			v.Reset(ssaop.OpPPC64XORconst)
			v.AuxInt = ssa.Int64ToAuxInt(int64(uint16(m)))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XOR (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [c^d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			d := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpPPC64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c ^ d)
			return true
		}
		break
	}
	// match: (XOR x (MOVDconst [c]))
	// cond: ssa.IsU32Bit(c)
	// result: (XORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpPPC64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.IsU32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpPPC64XORconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValuePPC64_OpPPC64XORconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (XORconst [c] (XORconst [d] x))
	// result: (XORconst [c^d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpPPC64XORconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c ^ d)
		v.AddArg(x)
		return true
	}
	// match: (XORconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (XORconst [1] (SETBCR [n] cmp))
	// result: (SETBC [n] cmp)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 || v_0.Op != ssaop.OpPPC64SETBCR {
			break
		}
		n := ssa.AuxIntToInt32(v_0.AuxInt)
		cmp := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SETBC)
		v.AuxInt = ssa.Int32ToAuxInt(n)
		v.AddArg(cmp)
		return true
	}
	// match: (XORconst [1] (SETBC [n] cmp))
	// result: (SETBCR [n] cmp)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 || v_0.Op != ssaop.OpPPC64SETBC {
			break
		}
		n := ssa.AuxIntToInt32(v_0.AuxInt)
		cmp := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SETBCR)
		v.AuxInt = ssa.Int32ToAuxInt(n)
		v.AddArg(cmp)
		return true
	}
	return false
}
func rewriteValuePPC64_OpPopCount16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount16 x)
	// result: (POPCNTW (MOVHZreg x))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64POPCNTW)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpPopCount32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount32 x)
	// result: (POPCNTW (MOVWZreg x))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64POPCNTW)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpPopCount8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount8 x)
	// result: (POPCNTB (MOVBZreg x))
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64POPCNTB)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpPrefetchCache(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PrefetchCache ptr mem)
	// result: (DCBT ptr mem [0])
	for {
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64DCBT)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpPrefetchCacheStreamed(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PrefetchCacheStreamed ptr mem)
	// result: (DCBT ptr mem [16])
	for {
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64DCBT)
		v.AuxInt = ssa.Int64ToAuxInt(16)
		v.AddArg2(ptr, mem)
		return true
	}
}
func rewriteValuePPC64_OpRotateLeft16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVDconst [c]))
	// result: (Or16 (Lsh16x64 <t> x (MOVDconst [c&15])) (Rsh16Ux64 <t> x (MOVDconst [-c&15])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpOr16)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh16x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(c & 15)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh16Ux64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v3.AuxInt = ssa.Int64ToAuxInt(-c & 15)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValuePPC64_OpRotateLeft8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVDconst [c]))
	// result: (Or8 (Lsh8x64 <t> x (MOVDconst [c&7])) (Rsh8Ux64 <t> x (MOVDconst [-c&7])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpOr8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh8x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(c & 7)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh8Ux64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v3.AuxInt = ssa.Int64ToAuxInt(-c & 7)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValuePPC64_OpRsh16Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRD (MOVHZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux16 <t> x y)
	// result: (ISEL [2] (SRD <t> (MOVHZreg x) y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0xFFF0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v4.AuxInt = ssa.Int64ToAuxInt(0xFFF0)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRD (MOVHZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
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
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v3.AuxInt = ssa.Int32ToAuxInt(16)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 x (MOVDconst [c]))
	// cond: uint64(c) < 16
	// result: (SRWconst (ZeroExt16to32 x) [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16Ux64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRD (MOVHZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
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
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(16)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRD (MOVHZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux8 <t> x y)
	// result: (ISEL [2] (SRD <t> (MOVHZreg x) y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0x00F0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v4.AuxInt = ssa.Int64ToAuxInt(0x00F0)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAD (MOVHreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x16 <t> x y)
	// result: (ISEL [2] (SRAD <t> (MOVHreg x) y) (SRADconst <t> (MOVHreg x) [15]) (CMPconst [0] (ANDconst [0xFFF0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64SRADconst, t)
		v2.AuxInt = ssa.Int64ToAuxInt(15)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v4.AuxInt = ssa.Int64ToAuxInt(0xFFF0)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAD (MOVHreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHreg, typ.Int64)
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
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64SRADconst, t)
		v2.AuxInt = ssa.Int64ToAuxInt(15)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v3.AuxInt = ssa.Int32ToAuxInt(16)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 x (MOVDconst [c]))
	// cond: uint64(c) >= 16
	// result: (SRAWconst (SignExt16to32 x) [63])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x64 x (MOVDconst [c]))
	// cond: uint64(c) < 16
	// result: (SRAWconst (SignExt16to32 x) [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAD (MOVHreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHreg, typ.Int64)
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
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64SRADconst, t)
		v2.AuxInt = ssa.Int64ToAuxInt(15)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(16)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAD (MOVHreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x8 <t> x y)
	// result: (ISEL [2] (SRAD <t> (MOVHreg x) y) (SRADconst <t> (MOVHreg x) [15]) (CMPconst [0] (ANDconst [0x00F0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64SRADconst, t)
		v2.AuxInt = ssa.Int64ToAuxInt(15)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v4.AuxInt = ssa.Int64ToAuxInt(0x00F0)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh32Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32Ux16 <t> x y)
	// result: (ISEL [2] (SRW <t> x y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0xFFE0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v3.AuxInt = ssa.Int64ToAuxInt(0xFFE0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32Ux32 <t> x y)
	// result: (ISEL [0] (SRW <t> x y) (MOVDconst [0]) (CMPWUconst y [32]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 x (MOVDconst [c]))
	// cond: uint64(c) < 32
	// result: (SRWconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Rsh32Ux64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32Ux64 <t> x y)
	// result: (ISEL [0] (SRW <t> x y) (MOVDconst [0]) (CMPUconst y [32]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32Ux8 <t> x y)
	// result: (ISEL [2] (SRW <t> x y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0x00E0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v3.AuxInt = ssa.Int64ToAuxInt(0x00E0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32x16 <t> x y)
	// result: (ISEL [2] (SRAW <t> x y) (SRAWconst <t> x [31]) (CMPconst [0] (ANDconst [0xFFE0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAWconst, t)
		v1.AuxInt = ssa.Int64ToAuxInt(31)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v3.AuxInt = ssa.Int64ToAuxInt(0xFFE0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32x32 <t> x y)
	// result: (ISEL [0] (SRAW <t> x y) (SRAWconst <t> x [31]) (CMPWUconst y [32]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAWconst, t)
		v1.AuxInt = ssa.Int64ToAuxInt(31)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x64 x (MOVDconst [c]))
	// cond: uint64(c) >= 32
	// result: (SRAWconst x [63])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v.AddArg(x)
		return true
	}
	// match: (Rsh32x64 x (MOVDconst [c]))
	// cond: uint64(c) < 32
	// result: (SRAWconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Rsh32x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32x64 <t> x y)
	// result: (ISEL [0] (SRAW <t> x y) (SRAWconst <t> x [31]) (CMPUconst y [32]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAWconst, t)
		v1.AuxInt = ssa.Int64ToAuxInt(31)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAW x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAW)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32x8 <t> x y)
	// result: (ISEL [2] (SRAW <t> x y) (SRAWconst <t> x [31]) (CMPconst [0] (ANDconst [0x00E0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAW, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAWconst, t)
		v1.AuxInt = ssa.Int64ToAuxInt(31)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v3.AuxInt = ssa.Int64ToAuxInt(0x00E0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux16 <t> x y)
	// result: (ISEL [2] (SRD <t> x y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0xFFC0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v3.AuxInt = ssa.Int64ToAuxInt(0xFFC0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux32 <t> x y)
	// result: (ISEL [0] (SRD <t> x y) (MOVDconst [0]) (CMPWUconst y [64]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux64 x (MOVDconst [c]))
	// cond: uint64(c) < 64
	// result: (SRDconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpPPC64SRDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Rsh64Ux64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux64 <t> x y)
	// result: (ISEL [0] (SRD <t> x y) (MOVDconst [0]) (CMPUconst y [64]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux8 <t> x y)
	// result: (ISEL [2] (SRD <t> x y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0x00C0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v3.AuxInt = ssa.Int64ToAuxInt(0x00C0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x16 <t> x y)
	// result: (ISEL [2] (SRAD <t> x y) (SRADconst <t> x [63]) (CMPconst [0] (ANDconst [0xFFC0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64SRADconst, t)
		v1.AuxInt = ssa.Int64ToAuxInt(63)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v3.AuxInt = ssa.Int64ToAuxInt(0xFFC0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x32 <t> x y)
	// result: (ISEL [0] (SRAD <t> x y) (SRADconst <t> x [63]) (CMPWUconst y [64]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64SRADconst, t)
		v1.AuxInt = ssa.Int64ToAuxInt(63)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x64 x (MOVDconst [c]))
	// cond: uint64(c) >= 64
	// result: (SRADconst x [63])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.Reset(ssaop.OpPPC64SRADconst)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v.AddArg(x)
		return true
	}
	// match: (Rsh64x64 x (MOVDconst [c]))
	// cond: uint64(c) < 64
	// result: (SRADconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 64) {
			break
		}
		v.Reset(ssaop.OpPPC64SRADconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (Rsh64x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x64 <t> x y)
	// result: (ISEL [0] (SRAD <t> x y) (SRADconst <t> x [63]) (CMPUconst y [64]))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64SRADconst, t)
		v1.AuxInt = ssa.Int64ToAuxInt(63)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh64x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAD x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x8 <t> x y)
	// result: (ISEL [2] (SRAD <t> x y) (SRADconst <t> x [63]) (CMPconst [0] (ANDconst [0x00C0] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAD, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64SRADconst, t)
		v1.AuxInt = ssa.Int64ToAuxInt(63)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v3.AuxInt = ssa.Int64ToAuxInt(0x00C0)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValuePPC64_OpRsh8Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRD (MOVBZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux16 <t> x y)
	// result: (ISEL [2] (SRD <t> (MOVBZreg x) y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0xFFF8] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v4.AuxInt = ssa.Int64ToAuxInt(0xFFF8)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRD (MOVBZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
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
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v3.AuxInt = ssa.Int32ToAuxInt(8)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 x (MOVDconst [c]))
	// cond: uint64(c) < 8
	// result: (SRWconst (ZeroExt8to32 x) [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpPPC64SRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8Ux64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRD (MOVBZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
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
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(8)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRD (MOVBZreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux8 <t> x y)
	// result: (ISEL [2] (SRD <t> (MOVBZreg x) y) (MOVDconst [0]) (CMPconst [0] (ANDconst [0x00F8] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBZreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDconst, typ.Int64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v4.AuxInt = ssa.Int64ToAuxInt(0x00F8)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAD (MOVBreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x16 <t> x y)
	// result: (ISEL [2] (SRAD <t> (MOVBreg x) y) (SRADconst <t> (MOVBreg x) [7]) (CMPconst [0] (ANDconst [0xFFF8] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64SRADconst, t)
		v2.AuxInt = ssa.Int64ToAuxInt(7)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v4.AuxInt = ssa.Int64ToAuxInt(0xFFF8)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAD (MOVBreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBreg, typ.Int64)
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
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64SRADconst, t)
		v2.AuxInt = ssa.Int64ToAuxInt(7)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPWUconst, types.TypeFlags)
		v3.AuxInt = ssa.Int32ToAuxInt(8)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 x (MOVDconst [c]))
	// cond: uint64(c) >= 8
	// result: (SRAWconst (SignExt8to32 x) [63])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8x64 x (MOVDconst [c]))
	// cond: uint64(c) < 8
	// result: (SRAWconst (SignExt8to32 x) [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAD (MOVBreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBreg, typ.Int64)
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
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64SRADconst, t)
		v2.AuxInt = ssa.Int64ToAuxInt(7)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPUconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(8)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpRsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAD (MOVBreg x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpPPC64SRAD)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBreg, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x8 <t> x y)
	// result: (ISEL [2] (SRAD <t> (MOVBreg x) y) (SRADconst <t> (MOVBreg x) [7]) (CMPconst [0] (ANDconst [0x00F8] y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpPPC64ISEL)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SRAD, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVBreg, typ.Int64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64SRADconst, t)
		v2.AuxInt = ssa.Int64ToAuxInt(7)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64ANDconst, typ.Int)
		v4.AuxInt = ssa.Int64ToAuxInt(0x00F8)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValuePPC64_OpSelect0(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Mul64uhilo x y))
	// result: (MULHDU x y)
	for {
		if v_0.Op != ssaop.OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64MULHDU)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select0 (Mul64uover x y))
	// result: (MULLD x y)
	for {
		if v_0.Op != ssaop.OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64MULLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select0 (Add64carry x y c))
	// result: (Select0 <typ.UInt64> (ADDE x y (Select1 <typ.UInt64> (ADDCconst c [-1]))))
	for {
		if v_0.Op != ssaop.OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64ADDE, types.NewTuple(typ.UInt64, typ.UInt64))
		v1 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64ADDCconst, types.NewTuple(typ.UInt64, typ.UInt64))
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v2.AddArg(c)
		v1.AddArg(v2)
		v0.AddArg3(x, y, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 (Sub64borrow x y c))
	// result: (Select0 <typ.UInt64> (SUBE x y (Select1 <typ.UInt64> (SUBCconst c [0]))))
	for {
		if v_0.Op != ssaop.OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SUBE, types.NewTuple(typ.UInt64, typ.UInt64))
		v1 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64SUBCconst, types.NewTuple(typ.UInt64, typ.UInt64))
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v2.AddArg(c)
		v1.AddArg(v2)
		v0.AddArg3(x, y, v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuePPC64_OpSelect1(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Mul64uhilo x y))
	// result: (MULLD x y)
	for {
		if v_0.Op != ssaop.OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64MULLD)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select1 (Mul64uover x y))
	// result: (SETBCR [2] (CMPconst [0] (MULHDU <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpPPC64SETBCR)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MULHDU, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select1 (Add64carry x y c))
	// result: (ADDZEzero (Select1 <typ.UInt64> (ADDE x y (Select1 <typ.UInt64> (ADDCconst c [-1])))))
	for {
		if v_0.Op != ssaop.OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpPPC64ADDZEzero)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64ADDE, types.NewTuple(typ.UInt64, typ.UInt64))
		v2 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.UInt64)
		v3 := b.NewValue0(v.Pos, ssaop.OpPPC64ADDCconst, types.NewTuple(typ.UInt64, typ.UInt64))
		v3.AuxInt = ssa.Int64ToAuxInt(-1)
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
		if v_0.Op != ssaop.OpPPC64ADDCconst || ssa.AuxIntToInt64(v_0.AuxInt) != -1 {
			break
		}
		n := v_0.Args[0]
		if n.Op != ssaop.OpPPC64ADDZEzero {
			break
		}
		x := n.Args[0]
		if !(n.Uses <= 2) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (Select1 (Sub64borrow x y c))
	// result: (NEG (SUBZEzero (Select1 <typ.UInt64> (SUBE x y (Select1 <typ.UInt64> (SUBCconst c [0]))))))
	for {
		if v_0.Op != ssaop.OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpPPC64NEG)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64SUBZEzero, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64SUBE, types.NewTuple(typ.UInt64, typ.UInt64))
		v3 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.UInt64)
		v4 := b.NewValue0(v.Pos, ssaop.OpPPC64SUBCconst, types.NewTuple(typ.UInt64, typ.UInt64))
		v4.AuxInt = ssa.Int64ToAuxInt(0)
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
		if v_0.Op != ssaop.OpPPC64SUBCconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		n := v_0.Args[0]
		if n.Op != ssaop.OpPPC64NEG {
			break
		}
		n_0 := n.Args[0]
		if n_0.Op != ssaop.OpPPC64SUBZEzero {
			break
		}
		x := n_0.Args[0]
		if !(n.Uses <= 2) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValuePPC64_OpSelectN(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SelectN [0] call:(CALLstatic {sym} s1:(MOVDstore _ (MOVDconst [sz]) s2:(MOVDstore _ src s3:(MOVDstore {t} _ dst mem)))))
	// cond: sz >= 0 && ssa.IsSameCall(sym, "runtime.memmove") && s1.Uses == 1 && s2.Uses == 1 && s3.Uses == 1 && ssa.IsInlinableMemmove(dst, src, sz, config) && ssa.Clobber(s1, s2, s3, call)
	// result: (Move [sz] dst src mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != ssaop.OpPPC64CALLstatic || len(call.Args) != 1 {
			break
		}
		sym := ssa.AuxToCall(call.Aux)
		s1 := call.Args[0]
		if s1.Op != ssaop.OpPPC64MOVDstore {
			break
		}
		_ = s1.Args[2]
		s1_1 := s1.Args[1]
		if s1_1.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		sz := ssa.AuxIntToInt64(s1_1.AuxInt)
		s2 := s1.Args[2]
		if s2.Op != ssaop.OpPPC64MOVDstore {
			break
		}
		_ = s2.Args[2]
		src := s2.Args[1]
		s3 := s2.Args[2]
		if s3.Op != ssaop.OpPPC64MOVDstore {
			break
		}
		mem := s3.Args[2]
		dst := s3.Args[1]
		if !(sz >= 0 && ssa.IsSameCall(sym, "runtime.memmove") && s1.Uses == 1 && s2.Uses == 1 && s3.Uses == 1 && ssa.IsInlinableMemmove(dst, src, sz, config) && ssa.Clobber(s1, s2, s3, call)) {
			break
		}
		v.Reset(ssaop.OpMove)
		v.AuxInt = ssa.Int64ToAuxInt(sz)
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (SelectN [0] call:(CALLstatic {sym} dst src (MOVDconst [sz]) mem))
	// cond: sz >= 0 && ssa.IsSameCall(sym, "runtime.memmove") && call.Uses == 1 && ssa.IsInlinableMemmove(dst, src, sz, config) && ssa.Clobber(call)
	// result: (Move [sz] dst src mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != ssaop.OpPPC64CALLstatic || len(call.Args) != 4 {
			break
		}
		sym := ssa.AuxToCall(call.Aux)
		mem := call.Args[3]
		dst := call.Args[0]
		src := call.Args[1]
		call_2 := call.Args[2]
		if call_2.Op != ssaop.OpPPC64MOVDconst {
			break
		}
		sz := ssa.AuxIntToInt64(call_2.AuxInt)
		if !(sz >= 0 && ssa.IsSameCall(sym, "runtime.memmove") && call.Uses == 1 && ssa.IsInlinableMemmove(dst, src, sz, config) && ssa.Clobber(call)) {
			break
		}
		v.Reset(ssaop.OpMove)
		v.AuxInt = ssa.Int64ToAuxInt(sz)
		v.AddArg3(dst, src, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpSlicemask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SRADconst (NEG <t> x) [63])
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpPPC64SRADconst)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64NEG, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValuePPC64_OpStore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && t.IsFloat()
	// result: (FMOVDstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && t.IsFloat()
	// result: (FMOVSstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpPPC64FMOVSstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && !t.IsFloat()
	// result: (MOVDstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && !t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && !t.IsFloat()
	// result: (MOVWstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && !t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 2
	// result: (MOVHstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 2) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVHstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 1
	// result: (MOVBstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 1) {
			break
		}
		v.Reset(ssaop.OpPPC64MOVBstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValuePPC64_OpTrunc16to8(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64MOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (Trunc16to8 x)
	// result: (MOVBZreg x)
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64MOVBZreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpTrunc32to16(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64MOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to16 x)
	// result: (MOVHZreg x)
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64MOVHZreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpTrunc32to8(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64MOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to8 x)
	// result: (MOVBZreg x)
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64MOVBZreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpTrunc64to16(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64MOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to16 x)
	// result: (MOVHZreg x)
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64MOVHZreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpTrunc64to32(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64MOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 x)
	// result: (MOVWZreg x)
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64MOVWZreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpTrunc64to8(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpPPC64MOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to8 x)
	// result: (MOVBZreg x)
	for {
		x := v_0
		v.Reset(ssaop.OpPPC64MOVBZreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValuePPC64_OpZero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Zero [0] _ mem)
	// result: mem
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		mem := v_1
		v.CopyOf(mem)
		return true
	}
	// match: (Zero [1] destptr mem)
	// result: (MOVBstorezero destptr mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64MOVBstorezero)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [2] destptr mem)
	// result: (MOVHstorezero destptr mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64MOVHstorezero)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [3] destptr mem)
	// result: (MOVBstorezero [2] destptr (MOVHstorezero destptr mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64MOVBstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHstorezero, types.TypeMem)
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [4] destptr mem)
	// result: (MOVWstorezero destptr mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64MOVWstorezero)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [5] destptr mem)
	// result: (MOVBstorezero [4] destptr (MOVWstorezero destptr mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 5 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64MOVBstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWstorezero, types.TypeMem)
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [6] destptr mem)
	// result: (MOVHstorezero [4] destptr (MOVWstorezero destptr mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64MOVHstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWstorezero, types.TypeMem)
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [7] destptr mem)
	// result: (MOVBstorezero [6] destptr (MOVHstorezero [4] destptr (MOVWstorezero destptr mem)))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 7 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64MOVBstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVHstorezero, types.TypeMem)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVWstorezero, types.TypeMem)
		v1.AddArg2(destptr, mem)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [8] {t} destptr mem)
	// result: (MOVDstorezero destptr mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64MOVDstorezero)
		v.AddArg2(destptr, mem)
		return true
	}
	// match: (Zero [12] {t} destptr mem)
	// result: (MOVWstorezero [8] destptr (MOVDstorezero [0] destptr mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 12 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64MOVWstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDstorezero, types.TypeMem)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [16] {t} destptr mem)
	// result: (MOVDstorezero [8] destptr (MOVDstorezero [0] destptr mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 16 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64MOVDstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDstorezero, types.TypeMem)
		v0.AuxInt = ssa.Int32ToAuxInt(0)
		v0.AddArg2(destptr, mem)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [24] {t} destptr mem)
	// result: (MOVDstorezero [16] destptr (MOVDstorezero [8] destptr (MOVDstorezero [0] destptr mem)))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 24 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64MOVDstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(16)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDstorezero, types.TypeMem)
		v0.AuxInt = ssa.Int32ToAuxInt(8)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDstorezero, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(0)
		v1.AddArg2(destptr, mem)
		v0.AddArg2(destptr, v1)
		v.AddArg2(destptr, v0)
		return true
	}
	// match: (Zero [32] {t} destptr mem)
	// result: (MOVDstorezero [24] destptr (MOVDstorezero [16] destptr (MOVDstorezero [8] destptr (MOVDstorezero [0] destptr mem))))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 32 {
			break
		}
		destptr := v_0
		mem := v_1
		v.Reset(ssaop.OpPPC64MOVDstorezero)
		v.AuxInt = ssa.Int32ToAuxInt(24)
		v0 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDstorezero, types.TypeMem)
		v0.AuxInt = ssa.Int32ToAuxInt(16)
		v1 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDstorezero, types.TypeMem)
		v1.AuxInt = ssa.Int32ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, ssaop.OpPPC64MOVDstorezero, types.TypeMem)
		v2.AuxInt = ssa.Int32ToAuxInt(0)
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
		s := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(buildcfg.GOPPC64 <= 8 && s < 64) {
			break
		}
		v.Reset(ssaop.OpPPC64LoweredZeroShort)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: buildcfg.GOPPC64 <= 8
	// result: (LoweredZero [s] ptr mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(buildcfg.GOPPC64 <= 8) {
			break
		}
		v.Reset(ssaop.OpPPC64LoweredZero)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s < 128 && buildcfg.GOPPC64 >= 9
	// result: (LoweredQuadZeroShort [s] ptr mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s < 128 && buildcfg.GOPPC64 >= 9) {
			break
		}
		v.Reset(ssaop.OpPPC64LoweredQuadZeroShort)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: buildcfg.GOPPC64 >= 9
	// result: (LoweredQuadZero [s] ptr mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(buildcfg.GOPPC64 >= 9) {
			break
		}
		v.Reset(ssaop.OpPPC64LoweredQuadZero)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteBlockPPC64(b *ssa.Block) bool {
	typ := &b.Func.Config.Types
	switch b.Kind {
	case block.BlockPPC64EQ:
		// match: (EQ (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FlagEQ {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (EQ (FlagLT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpPPC64FlagLT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (EQ (FlagGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpPPC64FlagGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (EQ (InvertFlags cmp) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpPPC64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64EQ, cmp)
			return true
		}
		// match: (EQ (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (Select1 <types.TypeFlags> (ANDCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPconst [0] z:(OR x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (Select1 <types.TypeFlags> (ORCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64OR {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPconst [0] z:(XOR x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (Select1 <types.TypeFlags> (XORCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64XOR {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64EQ, v0)
				return true
			}
			break
		}
	case block.BlockPPC64GE:
		// match: (GE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FlagEQ {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GE (FlagLT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpPPC64FlagLT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (GE (FlagGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FlagGT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GE (InvertFlags cmp) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == ssaop.OpPPC64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64LE, cmp)
			return true
		}
		// match: (GE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (Select1 <types.TypeFlags> (ANDCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64GE, v0)
				return true
			}
			break
		}
		// match: (GE (CMPconst [0] z:(OR x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (Select1 <types.TypeFlags> (ORCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64OR {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64GE, v0)
				return true
			}
			break
		}
		// match: (GE (CMPconst [0] z:(XOR x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (Select1 <types.TypeFlags> (XORCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64XOR {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64GE, v0)
				return true
			}
			break
		}
	case block.BlockPPC64GT:
		// match: (GT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpPPC64FlagEQ {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (GT (FlagLT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpPPC64FlagLT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (GT (FlagGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FlagGT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GT (InvertFlags cmp) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == ssaop.OpPPC64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64LT, cmp)
			return true
		}
		// match: (GT (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (Select1 <types.TypeFlags> (ANDCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64GT, v0)
				return true
			}
			break
		}
		// match: (GT (CMPconst [0] z:(OR x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (Select1 <types.TypeFlags> (ORCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64OR {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64GT, v0)
				return true
			}
			break
		}
		// match: (GT (CMPconst [0] z:(XOR x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (Select1 <types.TypeFlags> (XORCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64XOR {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64GT, v0)
				return true
			}
			break
		}
	case block.BlockIf:
		// match: (If (Equal cc) yes no)
		// result: (EQ cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64Equal {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64EQ, cc)
			return true
		}
		// match: (If (NotEqual cc) yes no)
		// result: (NE cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64NotEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64NE, cc)
			return true
		}
		// match: (If (LessThan cc) yes no)
		// result: (LT cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64LessThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64LT, cc)
			return true
		}
		// match: (If (LessEqual cc) yes no)
		// result: (LE cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64LessEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64LE, cc)
			return true
		}
		// match: (If (GreaterThan cc) yes no)
		// result: (GT cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64GreaterThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64GT, cc)
			return true
		}
		// match: (If (GreaterEqual cc) yes no)
		// result: (GE cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64GreaterEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64GE, cc)
			return true
		}
		// match: (If (FLessThan cc) yes no)
		// result: (FLT cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FLessThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64FLT, cc)
			return true
		}
		// match: (If (FLessEqual cc) yes no)
		// result: (FLE cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FLessEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64FLE, cc)
			return true
		}
		// match: (If (FGreaterThan cc) yes no)
		// result: (FGT cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FGreaterThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64FGT, cc)
			return true
		}
		// match: (If (FGreaterEqual cc) yes no)
		// result: (FGE cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FGreaterEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64FGE, cc)
			return true
		}
		// match: (If cond yes no)
		// result: (NE (CMPconst [0] (ANDconst [1] cond)) yes no)
		for {
			cond := b.Controls[0]
			v0 := b.NewValue0(cond.Pos, ssaop.OpPPC64CMPconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(0)
			v1 := b.NewValue0(cond.Pos, ssaop.OpPPC64ANDconst, typ.Int)
			v1.AuxInt = ssa.Int64ToAuxInt(1)
			v1.AddArg(cond)
			v0.AddArg(v1)
			b.ResetWithControl(block.BlockPPC64NE, v0)
			return true
		}
	case block.BlockPPC64LE:
		// match: (LE (FlagEQ) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FlagEQ {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LE (FlagLT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FlagLT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LE (FlagGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpPPC64FlagGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (LE (InvertFlags cmp) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == ssaop.OpPPC64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64GE, cmp)
			return true
		}
		// match: (LE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (Select1 <types.TypeFlags> (ANDCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64LE, v0)
				return true
			}
			break
		}
		// match: (LE (CMPconst [0] z:(OR x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (Select1 <types.TypeFlags> (ORCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64OR {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64LE, v0)
				return true
			}
			break
		}
		// match: (LE (CMPconst [0] z:(XOR x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (Select1 <types.TypeFlags> (XORCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64XOR {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64LE, v0)
				return true
			}
			break
		}
	case block.BlockPPC64LT:
		// match: (LT (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpPPC64FlagEQ {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (LT (FlagLT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FlagLT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LT (FlagGT) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpPPC64FlagGT {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (LT (InvertFlags cmp) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == ssaop.OpPPC64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64GT, cmp)
			return true
		}
		// match: (LT (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (Select1 <types.TypeFlags> (ANDCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64LT, v0)
				return true
			}
			break
		}
		// match: (LT (CMPconst [0] z:(OR x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (Select1 <types.TypeFlags> (ORCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64OR {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64LT, v0)
				return true
			}
			break
		}
		// match: (LT (CMPconst [0] z:(XOR x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (Select1 <types.TypeFlags> (XORCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64XOR {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64LT, v0)
				return true
			}
			break
		}
	case block.BlockPPC64NE:
		// match: (NE (CMPconst [0] (ANDconst [1] (Equal cc))) yes no)
		// result: (EQ cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(v_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpPPC64Equal {
				break
			}
			cc := v_0_0_0.Args[0]
			b.ResetWithControl(block.BlockPPC64EQ, cc)
			return true
		}
		// match: (NE (CMPconst [0] (ANDconst [1] (NotEqual cc))) yes no)
		// result: (NE cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(v_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpPPC64NotEqual {
				break
			}
			cc := v_0_0_0.Args[0]
			b.ResetWithControl(block.BlockPPC64NE, cc)
			return true
		}
		// match: (NE (CMPconst [0] (ANDconst [1] (LessThan cc))) yes no)
		// result: (LT cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(v_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpPPC64LessThan {
				break
			}
			cc := v_0_0_0.Args[0]
			b.ResetWithControl(block.BlockPPC64LT, cc)
			return true
		}
		// match: (NE (CMPconst [0] (ANDconst [1] (LessEqual cc))) yes no)
		// result: (LE cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(v_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpPPC64LessEqual {
				break
			}
			cc := v_0_0_0.Args[0]
			b.ResetWithControl(block.BlockPPC64LE, cc)
			return true
		}
		// match: (NE (CMPconst [0] (ANDconst [1] (GreaterThan cc))) yes no)
		// result: (GT cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(v_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpPPC64GreaterThan {
				break
			}
			cc := v_0_0_0.Args[0]
			b.ResetWithControl(block.BlockPPC64GT, cc)
			return true
		}
		// match: (NE (CMPconst [0] (ANDconst [1] (GreaterEqual cc))) yes no)
		// result: (GE cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(v_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpPPC64GreaterEqual {
				break
			}
			cc := v_0_0_0.Args[0]
			b.ResetWithControl(block.BlockPPC64GE, cc)
			return true
		}
		// match: (NE (CMPconst [0] (ANDconst [1] (FLessThan cc))) yes no)
		// result: (FLT cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(v_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpPPC64FLessThan {
				break
			}
			cc := v_0_0_0.Args[0]
			b.ResetWithControl(block.BlockPPC64FLT, cc)
			return true
		}
		// match: (NE (CMPconst [0] (ANDconst [1] (FLessEqual cc))) yes no)
		// result: (FLE cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(v_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpPPC64FLessEqual {
				break
			}
			cc := v_0_0_0.Args[0]
			b.ResetWithControl(block.BlockPPC64FLE, cc)
			return true
		}
		// match: (NE (CMPconst [0] (ANDconst [1] (FGreaterThan cc))) yes no)
		// result: (FGT cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(v_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpPPC64FGreaterThan {
				break
			}
			cc := v_0_0_0.Args[0]
			b.ResetWithControl(block.BlockPPC64FGT, cc)
			return true
		}
		// match: (NE (CMPconst [0] (ANDconst [1] (FGreaterEqual cc))) yes no)
		// result: (FGE cc yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpPPC64ANDconst || ssa.AuxIntToInt64(v_0_0.AuxInt) != 1 {
				break
			}
			v_0_0_0 := v_0_0.Args[0]
			if v_0_0_0.Op != ssaop.OpPPC64FGreaterEqual {
				break
			}
			cc := v_0_0_0.Args[0]
			b.ResetWithControl(block.BlockPPC64FGE, cc)
			return true
		}
		// match: (NE (FlagEQ) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpPPC64FlagEQ {
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (NE (FlagLT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FlagLT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (NE (FlagGT) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpPPC64FlagGT {
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (NE (InvertFlags cmp) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpPPC64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockPPC64NE, cmp)
			return true
		}
		// match: (NE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (Select1 <types.TypeFlags> (ANDCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64ANDCC, types.NewTuple(typ.Int64, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64NE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPconst [0] z:(OR x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (Select1 <types.TypeFlags> (ORCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64OR {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64ORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64NE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPconst [0] z:(XOR x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (Select1 <types.TypeFlags> (XORCC x y)) yes no)
		for b.Controls[0].Op == ssaop.OpPPC64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpPPC64XOR {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpSelect1, types.TypeFlags)
				v1 := b.NewValue0(v_0.Pos, ssaop.OpPPC64XORCC, types.NewTuple(typ.Int, types.TypeFlags))
				v1.AddArg2(x, y)
				v0.AddArg(v1)
				b.ResetWithControl(block.BlockPPC64NE, v0)
				return true
			}
			break
		}
	}
	return false
}
