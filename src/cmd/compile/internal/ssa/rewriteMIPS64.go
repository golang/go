// Code generated from _gen/MIPS64.rules using 'go generate'; DO NOT EDIT.

package ssa

import "cmd/compile/internal/types"
import "cmd/compile/internal/ssa/block"
import "cmd/compile/internal/ssa/ssaop"

func rewriteValueMIPS64(v *Value) bool {
	switch v.Op {
	case ssaop.OpAbs:
		v.Op = ssaop.OpMIPS64ABSD
		return true
	case ssaop.OpAdd16:
		v.Op = ssaop.OpMIPS64ADDV
		return true
	case ssaop.OpAdd32:
		v.Op = ssaop.OpMIPS64ADDV
		return true
	case ssaop.OpAdd32F:
		v.Op = ssaop.OpMIPS64ADDF
		return true
	case ssaop.OpAdd64:
		v.Op = ssaop.OpMIPS64ADDV
		return true
	case ssaop.OpAdd64F:
		v.Op = ssaop.OpMIPS64ADDD
		return true
	case ssaop.OpAdd8:
		v.Op = ssaop.OpMIPS64ADDV
		return true
	case ssaop.OpAddPtr:
		v.Op = ssaop.OpMIPS64ADDV
		return true
	case ssaop.OpAddr:
		return rewriteValueMIPS64_OpAddr(v)
	case ssaop.OpAnd16:
		v.Op = ssaop.OpMIPS64AND
		return true
	case ssaop.OpAnd32:
		v.Op = ssaop.OpMIPS64AND
		return true
	case ssaop.OpAnd64:
		v.Op = ssaop.OpMIPS64AND
		return true
	case ssaop.OpAnd8:
		v.Op = ssaop.OpMIPS64AND
		return true
	case ssaop.OpAndB:
		v.Op = ssaop.OpMIPS64AND
		return true
	case ssaop.OpAtomicAdd32:
		v.Op = ssaop.OpMIPS64LoweredAtomicAdd32
		return true
	case ssaop.OpAtomicAdd64:
		v.Op = ssaop.OpMIPS64LoweredAtomicAdd64
		return true
	case ssaop.OpAtomicAnd32:
		v.Op = ssaop.OpMIPS64LoweredAtomicAnd32
		return true
	case ssaop.OpAtomicAnd8:
		return rewriteValueMIPS64_OpAtomicAnd8(v)
	case ssaop.OpAtomicCompareAndSwap32:
		return rewriteValueMIPS64_OpAtomicCompareAndSwap32(v)
	case ssaop.OpAtomicCompareAndSwap64:
		v.Op = ssaop.OpMIPS64LoweredAtomicCas64
		return true
	case ssaop.OpAtomicExchange32:
		v.Op = ssaop.OpMIPS64LoweredAtomicExchange32
		return true
	case ssaop.OpAtomicExchange64:
		v.Op = ssaop.OpMIPS64LoweredAtomicExchange64
		return true
	case ssaop.OpAtomicLoad32:
		v.Op = ssaop.OpMIPS64LoweredAtomicLoad32
		return true
	case ssaop.OpAtomicLoad64:
		v.Op = ssaop.OpMIPS64LoweredAtomicLoad64
		return true
	case ssaop.OpAtomicLoad8:
		v.Op = ssaop.OpMIPS64LoweredAtomicLoad8
		return true
	case ssaop.OpAtomicLoadPtr:
		v.Op = ssaop.OpMIPS64LoweredAtomicLoad64
		return true
	case ssaop.OpAtomicOr32:
		v.Op = ssaop.OpMIPS64LoweredAtomicOr32
		return true
	case ssaop.OpAtomicOr8:
		return rewriteValueMIPS64_OpAtomicOr8(v)
	case ssaop.OpAtomicStore32:
		v.Op = ssaop.OpMIPS64LoweredAtomicStore32
		return true
	case ssaop.OpAtomicStore64:
		v.Op = ssaop.OpMIPS64LoweredAtomicStore64
		return true
	case ssaop.OpAtomicStore8:
		v.Op = ssaop.OpMIPS64LoweredAtomicStore8
		return true
	case ssaop.OpAtomicStorePtrNoWB:
		v.Op = ssaop.OpMIPS64LoweredAtomicStore64
		return true
	case ssaop.OpAvg64u:
		return rewriteValueMIPS64_OpAvg64u(v)
	case ssaop.OpClosureCall:
		v.Op = ssaop.OpMIPS64CALLclosure
		return true
	case ssaop.OpCom16:
		return rewriteValueMIPS64_OpCom16(v)
	case ssaop.OpCom32:
		return rewriteValueMIPS64_OpCom32(v)
	case ssaop.OpCom64:
		return rewriteValueMIPS64_OpCom64(v)
	case ssaop.OpCom8:
		return rewriteValueMIPS64_OpCom8(v)
	case ssaop.OpConst16:
		return rewriteValueMIPS64_OpConst16(v)
	case ssaop.OpConst32:
		return rewriteValueMIPS64_OpConst32(v)
	case ssaop.OpConst32F:
		return rewriteValueMIPS64_OpConst32F(v)
	case ssaop.OpConst64:
		return rewriteValueMIPS64_OpConst64(v)
	case ssaop.OpConst64F:
		return rewriteValueMIPS64_OpConst64F(v)
	case ssaop.OpConst8:
		return rewriteValueMIPS64_OpConst8(v)
	case ssaop.OpConstBool:
		return rewriteValueMIPS64_OpConstBool(v)
	case ssaop.OpConstNil:
		return rewriteValueMIPS64_OpConstNil(v)
	case ssaop.OpCvt32Fto32:
		v.Op = ssaop.OpMIPS64TRUNCFW
		return true
	case ssaop.OpCvt32Fto64:
		v.Op = ssaop.OpMIPS64TRUNCFV
		return true
	case ssaop.OpCvt32Fto64F:
		v.Op = ssaop.OpMIPS64MOVFD
		return true
	case ssaop.OpCvt32to32F:
		v.Op = ssaop.OpMIPS64MOVWF
		return true
	case ssaop.OpCvt32to64F:
		v.Op = ssaop.OpMIPS64MOVWD
		return true
	case ssaop.OpCvt64Fto32:
		v.Op = ssaop.OpMIPS64TRUNCDW
		return true
	case ssaop.OpCvt64Fto32F:
		v.Op = ssaop.OpMIPS64MOVDF
		return true
	case ssaop.OpCvt64Fto64:
		v.Op = ssaop.OpMIPS64TRUNCDV
		return true
	case ssaop.OpCvt64to32F:
		v.Op = ssaop.OpMIPS64MOVVF
		return true
	case ssaop.OpCvt64to64F:
		v.Op = ssaop.OpMIPS64MOVVD
		return true
	case ssaop.OpCvtBoolToUint8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpDiv16:
		return rewriteValueMIPS64_OpDiv16(v)
	case ssaop.OpDiv16u:
		return rewriteValueMIPS64_OpDiv16u(v)
	case ssaop.OpDiv32:
		return rewriteValueMIPS64_OpDiv32(v)
	case ssaop.OpDiv32F:
		v.Op = ssaop.OpMIPS64DIVF
		return true
	case ssaop.OpDiv32u:
		return rewriteValueMIPS64_OpDiv32u(v)
	case ssaop.OpDiv64:
		return rewriteValueMIPS64_OpDiv64(v)
	case ssaop.OpDiv64F:
		v.Op = ssaop.OpMIPS64DIVD
		return true
	case ssaop.OpDiv64u:
		return rewriteValueMIPS64_OpDiv64u(v)
	case ssaop.OpDiv8:
		return rewriteValueMIPS64_OpDiv8(v)
	case ssaop.OpDiv8u:
		return rewriteValueMIPS64_OpDiv8u(v)
	case ssaop.OpEq16:
		return rewriteValueMIPS64_OpEq16(v)
	case ssaop.OpEq32:
		return rewriteValueMIPS64_OpEq32(v)
	case ssaop.OpEq32F:
		return rewriteValueMIPS64_OpEq32F(v)
	case ssaop.OpEq64:
		return rewriteValueMIPS64_OpEq64(v)
	case ssaop.OpEq64F:
		return rewriteValueMIPS64_OpEq64F(v)
	case ssaop.OpEq8:
		return rewriteValueMIPS64_OpEq8(v)
	case ssaop.OpEqB:
		return rewriteValueMIPS64_OpEqB(v)
	case ssaop.OpEqPtr:
		return rewriteValueMIPS64_OpEqPtr(v)
	case ssaop.OpGetCallerPC:
		v.Op = ssaop.OpMIPS64LoweredGetCallerPC
		return true
	case ssaop.OpGetCallerSP:
		v.Op = ssaop.OpMIPS64LoweredGetCallerSP
		return true
	case ssaop.OpGetClosurePtr:
		v.Op = ssaop.OpMIPS64LoweredGetClosurePtr
		return true
	case ssaop.OpHmul32:
		return rewriteValueMIPS64_OpHmul32(v)
	case ssaop.OpHmul32u:
		return rewriteValueMIPS64_OpHmul32u(v)
	case ssaop.OpHmul64:
		return rewriteValueMIPS64_OpHmul64(v)
	case ssaop.OpHmul64u:
		return rewriteValueMIPS64_OpHmul64u(v)
	case ssaop.OpInterCall:
		v.Op = ssaop.OpMIPS64CALLinter
		return true
	case ssaop.OpIsInBounds:
		return rewriteValueMIPS64_OpIsInBounds(v)
	case ssaop.OpIsNonNil:
		return rewriteValueMIPS64_OpIsNonNil(v)
	case ssaop.OpIsSliceInBounds:
		return rewriteValueMIPS64_OpIsSliceInBounds(v)
	case ssaop.OpLeq16:
		return rewriteValueMIPS64_OpLeq16(v)
	case ssaop.OpLeq16U:
		return rewriteValueMIPS64_OpLeq16U(v)
	case ssaop.OpLeq32:
		return rewriteValueMIPS64_OpLeq32(v)
	case ssaop.OpLeq32F:
		return rewriteValueMIPS64_OpLeq32F(v)
	case ssaop.OpLeq32U:
		return rewriteValueMIPS64_OpLeq32U(v)
	case ssaop.OpLeq64:
		return rewriteValueMIPS64_OpLeq64(v)
	case ssaop.OpLeq64F:
		return rewriteValueMIPS64_OpLeq64F(v)
	case ssaop.OpLeq64U:
		return rewriteValueMIPS64_OpLeq64U(v)
	case ssaop.OpLeq8:
		return rewriteValueMIPS64_OpLeq8(v)
	case ssaop.OpLeq8U:
		return rewriteValueMIPS64_OpLeq8U(v)
	case ssaop.OpLess16:
		return rewriteValueMIPS64_OpLess16(v)
	case ssaop.OpLess16U:
		return rewriteValueMIPS64_OpLess16U(v)
	case ssaop.OpLess32:
		return rewriteValueMIPS64_OpLess32(v)
	case ssaop.OpLess32F:
		return rewriteValueMIPS64_OpLess32F(v)
	case ssaop.OpLess32U:
		return rewriteValueMIPS64_OpLess32U(v)
	case ssaop.OpLess64:
		return rewriteValueMIPS64_OpLess64(v)
	case ssaop.OpLess64F:
		return rewriteValueMIPS64_OpLess64F(v)
	case ssaop.OpLess64U:
		return rewriteValueMIPS64_OpLess64U(v)
	case ssaop.OpLess8:
		return rewriteValueMIPS64_OpLess8(v)
	case ssaop.OpLess8U:
		return rewriteValueMIPS64_OpLess8U(v)
	case ssaop.OpLoad:
		return rewriteValueMIPS64_OpLoad(v)
	case ssaop.OpLocalAddr:
		return rewriteValueMIPS64_OpLocalAddr(v)
	case ssaop.OpLsh16x16:
		return rewriteValueMIPS64_OpLsh16x16(v)
	case ssaop.OpLsh16x32:
		return rewriteValueMIPS64_OpLsh16x32(v)
	case ssaop.OpLsh16x64:
		return rewriteValueMIPS64_OpLsh16x64(v)
	case ssaop.OpLsh16x8:
		return rewriteValueMIPS64_OpLsh16x8(v)
	case ssaop.OpLsh32x16:
		return rewriteValueMIPS64_OpLsh32x16(v)
	case ssaop.OpLsh32x32:
		return rewriteValueMIPS64_OpLsh32x32(v)
	case ssaop.OpLsh32x64:
		return rewriteValueMIPS64_OpLsh32x64(v)
	case ssaop.OpLsh32x8:
		return rewriteValueMIPS64_OpLsh32x8(v)
	case ssaop.OpLsh64x16:
		return rewriteValueMIPS64_OpLsh64x16(v)
	case ssaop.OpLsh64x32:
		return rewriteValueMIPS64_OpLsh64x32(v)
	case ssaop.OpLsh64x64:
		return rewriteValueMIPS64_OpLsh64x64(v)
	case ssaop.OpLsh64x8:
		return rewriteValueMIPS64_OpLsh64x8(v)
	case ssaop.OpLsh8x16:
		return rewriteValueMIPS64_OpLsh8x16(v)
	case ssaop.OpLsh8x32:
		return rewriteValueMIPS64_OpLsh8x32(v)
	case ssaop.OpLsh8x64:
		return rewriteValueMIPS64_OpLsh8x64(v)
	case ssaop.OpLsh8x8:
		return rewriteValueMIPS64_OpLsh8x8(v)
	case ssaop.OpMIPS64ADDV:
		return rewriteValueMIPS64_OpMIPS64ADDV(v)
	case ssaop.OpMIPS64ADDVconst:
		return rewriteValueMIPS64_OpMIPS64ADDVconst(v)
	case ssaop.OpMIPS64AND:
		return rewriteValueMIPS64_OpMIPS64AND(v)
	case ssaop.OpMIPS64ANDconst:
		return rewriteValueMIPS64_OpMIPS64ANDconst(v)
	case ssaop.OpMIPS64LoweredAtomicAdd32:
		return rewriteValueMIPS64_OpMIPS64LoweredAtomicAdd32(v)
	case ssaop.OpMIPS64LoweredAtomicAdd64:
		return rewriteValueMIPS64_OpMIPS64LoweredAtomicAdd64(v)
	case ssaop.OpMIPS64LoweredAtomicStore32:
		return rewriteValueMIPS64_OpMIPS64LoweredAtomicStore32(v)
	case ssaop.OpMIPS64LoweredAtomicStore64:
		return rewriteValueMIPS64_OpMIPS64LoweredAtomicStore64(v)
	case ssaop.OpMIPS64LoweredPanicBoundsCR:
		return rewriteValueMIPS64_OpMIPS64LoweredPanicBoundsCR(v)
	case ssaop.OpMIPS64LoweredPanicBoundsRC:
		return rewriteValueMIPS64_OpMIPS64LoweredPanicBoundsRC(v)
	case ssaop.OpMIPS64LoweredPanicBoundsRR:
		return rewriteValueMIPS64_OpMIPS64LoweredPanicBoundsRR(v)
	case ssaop.OpMIPS64MOVBUload:
		return rewriteValueMIPS64_OpMIPS64MOVBUload(v)
	case ssaop.OpMIPS64MOVBUreg:
		return rewriteValueMIPS64_OpMIPS64MOVBUreg(v)
	case ssaop.OpMIPS64MOVBload:
		return rewriteValueMIPS64_OpMIPS64MOVBload(v)
	case ssaop.OpMIPS64MOVBreg:
		return rewriteValueMIPS64_OpMIPS64MOVBreg(v)
	case ssaop.OpMIPS64MOVBstore:
		return rewriteValueMIPS64_OpMIPS64MOVBstore(v)
	case ssaop.OpMIPS64MOVDF:
		return rewriteValueMIPS64_OpMIPS64MOVDF(v)
	case ssaop.OpMIPS64MOVDload:
		return rewriteValueMIPS64_OpMIPS64MOVDload(v)
	case ssaop.OpMIPS64MOVDstore:
		return rewriteValueMIPS64_OpMIPS64MOVDstore(v)
	case ssaop.OpMIPS64MOVFload:
		return rewriteValueMIPS64_OpMIPS64MOVFload(v)
	case ssaop.OpMIPS64MOVFstore:
		return rewriteValueMIPS64_OpMIPS64MOVFstore(v)
	case ssaop.OpMIPS64MOVHUload:
		return rewriteValueMIPS64_OpMIPS64MOVHUload(v)
	case ssaop.OpMIPS64MOVHUreg:
		return rewriteValueMIPS64_OpMIPS64MOVHUreg(v)
	case ssaop.OpMIPS64MOVHload:
		return rewriteValueMIPS64_OpMIPS64MOVHload(v)
	case ssaop.OpMIPS64MOVHreg:
		return rewriteValueMIPS64_OpMIPS64MOVHreg(v)
	case ssaop.OpMIPS64MOVHstore:
		return rewriteValueMIPS64_OpMIPS64MOVHstore(v)
	case ssaop.OpMIPS64MOVVload:
		return rewriteValueMIPS64_OpMIPS64MOVVload(v)
	case ssaop.OpMIPS64MOVVnop:
		return rewriteValueMIPS64_OpMIPS64MOVVnop(v)
	case ssaop.OpMIPS64MOVVreg:
		return rewriteValueMIPS64_OpMIPS64MOVVreg(v)
	case ssaop.OpMIPS64MOVVstore:
		return rewriteValueMIPS64_OpMIPS64MOVVstore(v)
	case ssaop.OpMIPS64MOVWUload:
		return rewriteValueMIPS64_OpMIPS64MOVWUload(v)
	case ssaop.OpMIPS64MOVWUreg:
		return rewriteValueMIPS64_OpMIPS64MOVWUreg(v)
	case ssaop.OpMIPS64MOVWload:
		return rewriteValueMIPS64_OpMIPS64MOVWload(v)
	case ssaop.OpMIPS64MOVWreg:
		return rewriteValueMIPS64_OpMIPS64MOVWreg(v)
	case ssaop.OpMIPS64MOVWstore:
		return rewriteValueMIPS64_OpMIPS64MOVWstore(v)
	case ssaop.OpMIPS64NEGV:
		return rewriteValueMIPS64_OpMIPS64NEGV(v)
	case ssaop.OpMIPS64OR:
		return rewriteValueMIPS64_OpMIPS64OR(v)
	case ssaop.OpMIPS64ORconst:
		return rewriteValueMIPS64_OpMIPS64ORconst(v)
	case ssaop.OpMIPS64SGT:
		return rewriteValueMIPS64_OpMIPS64SGT(v)
	case ssaop.OpMIPS64SGTU:
		return rewriteValueMIPS64_OpMIPS64SGTU(v)
	case ssaop.OpMIPS64SGTUconst:
		return rewriteValueMIPS64_OpMIPS64SGTUconst(v)
	case ssaop.OpMIPS64SGTconst:
		return rewriteValueMIPS64_OpMIPS64SGTconst(v)
	case ssaop.OpMIPS64SLLV:
		return rewriteValueMIPS64_OpMIPS64SLLV(v)
	case ssaop.OpMIPS64SLLVconst:
		return rewriteValueMIPS64_OpMIPS64SLLVconst(v)
	case ssaop.OpMIPS64SRAV:
		return rewriteValueMIPS64_OpMIPS64SRAV(v)
	case ssaop.OpMIPS64SRAVconst:
		return rewriteValueMIPS64_OpMIPS64SRAVconst(v)
	case ssaop.OpMIPS64SRLV:
		return rewriteValueMIPS64_OpMIPS64SRLV(v)
	case ssaop.OpMIPS64SRLVconst:
		return rewriteValueMIPS64_OpMIPS64SRLVconst(v)
	case ssaop.OpMIPS64SUBV:
		return rewriteValueMIPS64_OpMIPS64SUBV(v)
	case ssaop.OpMIPS64SUBVconst:
		return rewriteValueMIPS64_OpMIPS64SUBVconst(v)
	case ssaop.OpMIPS64XOR:
		return rewriteValueMIPS64_OpMIPS64XOR(v)
	case ssaop.OpMIPS64XORconst:
		return rewriteValueMIPS64_OpMIPS64XORconst(v)
	case ssaop.OpMod16:
		return rewriteValueMIPS64_OpMod16(v)
	case ssaop.OpMod16u:
		return rewriteValueMIPS64_OpMod16u(v)
	case ssaop.OpMod32:
		return rewriteValueMIPS64_OpMod32(v)
	case ssaop.OpMod32u:
		return rewriteValueMIPS64_OpMod32u(v)
	case ssaop.OpMod64:
		return rewriteValueMIPS64_OpMod64(v)
	case ssaop.OpMod64u:
		return rewriteValueMIPS64_OpMod64u(v)
	case ssaop.OpMod8:
		return rewriteValueMIPS64_OpMod8(v)
	case ssaop.OpMod8u:
		return rewriteValueMIPS64_OpMod8u(v)
	case ssaop.OpMove:
		return rewriteValueMIPS64_OpMove(v)
	case ssaop.OpMul16:
		return rewriteValueMIPS64_OpMul16(v)
	case ssaop.OpMul32:
		return rewriteValueMIPS64_OpMul32(v)
	case ssaop.OpMul32F:
		v.Op = ssaop.OpMIPS64MULF
		return true
	case ssaop.OpMul64:
		return rewriteValueMIPS64_OpMul64(v)
	case ssaop.OpMul64F:
		v.Op = ssaop.OpMIPS64MULD
		return true
	case ssaop.OpMul64uhilo:
		v.Op = ssaop.OpMIPS64MULVU
		return true
	case ssaop.OpMul8:
		return rewriteValueMIPS64_OpMul8(v)
	case ssaop.OpNeg16:
		v.Op = ssaop.OpMIPS64NEGV
		return true
	case ssaop.OpNeg32:
		v.Op = ssaop.OpMIPS64NEGV
		return true
	case ssaop.OpNeg32F:
		v.Op = ssaop.OpMIPS64NEGF
		return true
	case ssaop.OpNeg64:
		v.Op = ssaop.OpMIPS64NEGV
		return true
	case ssaop.OpNeg64F:
		v.Op = ssaop.OpMIPS64NEGD
		return true
	case ssaop.OpNeg8:
		v.Op = ssaop.OpMIPS64NEGV
		return true
	case ssaop.OpNeq16:
		return rewriteValueMIPS64_OpNeq16(v)
	case ssaop.OpNeq32:
		return rewriteValueMIPS64_OpNeq32(v)
	case ssaop.OpNeq32F:
		return rewriteValueMIPS64_OpNeq32F(v)
	case ssaop.OpNeq64:
		return rewriteValueMIPS64_OpNeq64(v)
	case ssaop.OpNeq64F:
		return rewriteValueMIPS64_OpNeq64F(v)
	case ssaop.OpNeq8:
		return rewriteValueMIPS64_OpNeq8(v)
	case ssaop.OpNeqB:
		v.Op = ssaop.OpMIPS64XOR
		return true
	case ssaop.OpNeqPtr:
		return rewriteValueMIPS64_OpNeqPtr(v)
	case ssaop.OpNilCheck:
		v.Op = ssaop.OpMIPS64LoweredNilCheck
		return true
	case ssaop.OpNot:
		return rewriteValueMIPS64_OpNot(v)
	case ssaop.OpOffPtr:
		return rewriteValueMIPS64_OpOffPtr(v)
	case ssaop.OpOr16:
		v.Op = ssaop.OpMIPS64OR
		return true
	case ssaop.OpOr32:
		v.Op = ssaop.OpMIPS64OR
		return true
	case ssaop.OpOr64:
		v.Op = ssaop.OpMIPS64OR
		return true
	case ssaop.OpOr8:
		v.Op = ssaop.OpMIPS64OR
		return true
	case ssaop.OpOrB:
		v.Op = ssaop.OpMIPS64OR
		return true
	case ssaop.OpPanicBounds:
		v.Op = ssaop.OpMIPS64LoweredPanicBoundsRR
		return true
	case ssaop.OpPubBarrier:
		v.Op = ssaop.OpMIPS64LoweredPubBarrier
		return true
	case ssaop.OpRotateLeft16:
		return rewriteValueMIPS64_OpRotateLeft16(v)
	case ssaop.OpRotateLeft32:
		return rewriteValueMIPS64_OpRotateLeft32(v)
	case ssaop.OpRotateLeft64:
		return rewriteValueMIPS64_OpRotateLeft64(v)
	case ssaop.OpRotateLeft8:
		return rewriteValueMIPS64_OpRotateLeft8(v)
	case ssaop.OpRound32F:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpRound64F:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpRsh16Ux16:
		return rewriteValueMIPS64_OpRsh16Ux16(v)
	case ssaop.OpRsh16Ux32:
		return rewriteValueMIPS64_OpRsh16Ux32(v)
	case ssaop.OpRsh16Ux64:
		return rewriteValueMIPS64_OpRsh16Ux64(v)
	case ssaop.OpRsh16Ux8:
		return rewriteValueMIPS64_OpRsh16Ux8(v)
	case ssaop.OpRsh16x16:
		return rewriteValueMIPS64_OpRsh16x16(v)
	case ssaop.OpRsh16x32:
		return rewriteValueMIPS64_OpRsh16x32(v)
	case ssaop.OpRsh16x64:
		return rewriteValueMIPS64_OpRsh16x64(v)
	case ssaop.OpRsh16x8:
		return rewriteValueMIPS64_OpRsh16x8(v)
	case ssaop.OpRsh32Ux16:
		return rewriteValueMIPS64_OpRsh32Ux16(v)
	case ssaop.OpRsh32Ux32:
		return rewriteValueMIPS64_OpRsh32Ux32(v)
	case ssaop.OpRsh32Ux64:
		return rewriteValueMIPS64_OpRsh32Ux64(v)
	case ssaop.OpRsh32Ux8:
		return rewriteValueMIPS64_OpRsh32Ux8(v)
	case ssaop.OpRsh32x16:
		return rewriteValueMIPS64_OpRsh32x16(v)
	case ssaop.OpRsh32x32:
		return rewriteValueMIPS64_OpRsh32x32(v)
	case ssaop.OpRsh32x64:
		return rewriteValueMIPS64_OpRsh32x64(v)
	case ssaop.OpRsh32x8:
		return rewriteValueMIPS64_OpRsh32x8(v)
	case ssaop.OpRsh64Ux16:
		return rewriteValueMIPS64_OpRsh64Ux16(v)
	case ssaop.OpRsh64Ux32:
		return rewriteValueMIPS64_OpRsh64Ux32(v)
	case ssaop.OpRsh64Ux64:
		return rewriteValueMIPS64_OpRsh64Ux64(v)
	case ssaop.OpRsh64Ux8:
		return rewriteValueMIPS64_OpRsh64Ux8(v)
	case ssaop.OpRsh64x16:
		return rewriteValueMIPS64_OpRsh64x16(v)
	case ssaop.OpRsh64x32:
		return rewriteValueMIPS64_OpRsh64x32(v)
	case ssaop.OpRsh64x64:
		return rewriteValueMIPS64_OpRsh64x64(v)
	case ssaop.OpRsh64x8:
		return rewriteValueMIPS64_OpRsh64x8(v)
	case ssaop.OpRsh8Ux16:
		return rewriteValueMIPS64_OpRsh8Ux16(v)
	case ssaop.OpRsh8Ux32:
		return rewriteValueMIPS64_OpRsh8Ux32(v)
	case ssaop.OpRsh8Ux64:
		return rewriteValueMIPS64_OpRsh8Ux64(v)
	case ssaop.OpRsh8Ux8:
		return rewriteValueMIPS64_OpRsh8Ux8(v)
	case ssaop.OpRsh8x16:
		return rewriteValueMIPS64_OpRsh8x16(v)
	case ssaop.OpRsh8x32:
		return rewriteValueMIPS64_OpRsh8x32(v)
	case ssaop.OpRsh8x64:
		return rewriteValueMIPS64_OpRsh8x64(v)
	case ssaop.OpRsh8x8:
		return rewriteValueMIPS64_OpRsh8x8(v)
	case ssaop.OpSelect0:
		return rewriteValueMIPS64_OpSelect0(v)
	case ssaop.OpSelect1:
		return rewriteValueMIPS64_OpSelect1(v)
	case ssaop.OpSignExt16to32:
		v.Op = ssaop.OpMIPS64MOVHreg
		return true
	case ssaop.OpSignExt16to64:
		v.Op = ssaop.OpMIPS64MOVHreg
		return true
	case ssaop.OpSignExt32to64:
		v.Op = ssaop.OpMIPS64MOVWreg
		return true
	case ssaop.OpSignExt8to16:
		v.Op = ssaop.OpMIPS64MOVBreg
		return true
	case ssaop.OpSignExt8to32:
		v.Op = ssaop.OpMIPS64MOVBreg
		return true
	case ssaop.OpSignExt8to64:
		v.Op = ssaop.OpMIPS64MOVBreg
		return true
	case ssaop.OpSlicemask:
		return rewriteValueMIPS64_OpSlicemask(v)
	case ssaop.OpSqrt:
		v.Op = ssaop.OpMIPS64SQRTD
		return true
	case ssaop.OpSqrt32:
		v.Op = ssaop.OpMIPS64SQRTF
		return true
	case ssaop.OpStaticCall:
		v.Op = ssaop.OpMIPS64CALLstatic
		return true
	case ssaop.OpStore:
		return rewriteValueMIPS64_OpStore(v)
	case ssaop.OpSub16:
		v.Op = ssaop.OpMIPS64SUBV
		return true
	case ssaop.OpSub32:
		v.Op = ssaop.OpMIPS64SUBV
		return true
	case ssaop.OpSub32F:
		v.Op = ssaop.OpMIPS64SUBF
		return true
	case ssaop.OpSub64:
		v.Op = ssaop.OpMIPS64SUBV
		return true
	case ssaop.OpSub64F:
		v.Op = ssaop.OpMIPS64SUBD
		return true
	case ssaop.OpSub8:
		v.Op = ssaop.OpMIPS64SUBV
		return true
	case ssaop.OpSubPtr:
		v.Op = ssaop.OpMIPS64SUBV
		return true
	case ssaop.OpTailCall:
		v.Op = ssaop.OpMIPS64CALLtail
		return true
	case ssaop.OpTailCallInter:
		v.Op = ssaop.OpMIPS64CALLtailinter
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
	case ssaop.OpWB:
		v.Op = ssaop.OpMIPS64LoweredWB
		return true
	case ssaop.OpXor16:
		v.Op = ssaop.OpMIPS64XOR
		return true
	case ssaop.OpXor32:
		v.Op = ssaop.OpMIPS64XOR
		return true
	case ssaop.OpXor64:
		v.Op = ssaop.OpMIPS64XOR
		return true
	case ssaop.OpXor8:
		v.Op = ssaop.OpMIPS64XOR
		return true
	case ssaop.OpZero:
		return rewriteValueMIPS64_OpZero(v)
	case ssaop.OpZeroExt16to32:
		v.Op = ssaop.OpMIPS64MOVHUreg
		return true
	case ssaop.OpZeroExt16to64:
		v.Op = ssaop.OpMIPS64MOVHUreg
		return true
	case ssaop.OpZeroExt32to64:
		v.Op = ssaop.OpMIPS64MOVWUreg
		return true
	case ssaop.OpZeroExt8to16:
		v.Op = ssaop.OpMIPS64MOVBUreg
		return true
	case ssaop.OpZeroExt8to32:
		v.Op = ssaop.OpMIPS64MOVBUreg
		return true
	case ssaop.OpZeroExt8to64:
		v.Op = ssaop.OpMIPS64MOVBUreg
		return true
	}
	return false
}
func rewriteValueMIPS64_OpAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (MOVVaddr {sym} base)
	for {
		sym := AuxToSym(v.Aux)
		base := v_0
		v.Reset(ssaop.OpMIPS64MOVVaddr)
		v.Aux = SymToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValueMIPS64_OpAtomicAnd8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (AtomicAnd8 ptr val mem)
	// cond: !config.BigEndian
	// result: (LoweredAtomicAnd32 (AND <typ.UInt32Ptr> (MOVVconst [^3]) ptr) (OR <typ.UInt64> (SLLV <typ.UInt32> (ZeroExt8to32 val) (SLLVconst <typ.UInt64> [3] (ANDconst <typ.UInt64> [3] ptr))) (NOR (MOVVconst [0]) <typ.UInt64> (SLLV <typ.UInt64> (MOVVconst [0xff]) (SLLVconst <typ.UInt64> [3] (ANDconst <typ.UInt64> [3] ptr))))) mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		if !(!config.BigEndian) {
			break
		}
		v.Reset(ssaop.OpMIPS64LoweredAtomicAnd32)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64AND, typ.UInt32Ptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v1.AuxInt = Int64ToAuxInt(^3)
		v0.AddArg2(v1, ptr)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, typ.UInt64)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, typ.UInt32)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v4.AddArg(val)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLVconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(3)
		v6 := b.NewValue0(v.Pos, ssaop.OpMIPS64ANDconst, typ.UInt64)
		v6.AuxInt = Int64ToAuxInt(3)
		v6.AddArg(ptr)
		v5.AddArg(v6)
		v3.AddArg2(v4, v5)
		v7 := b.NewValue0(v.Pos, ssaop.OpMIPS64NOR, typ.UInt64)
		v8 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v8.AuxInt = Int64ToAuxInt(0)
		v9 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, typ.UInt64)
		v10 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v10.AuxInt = Int64ToAuxInt(0xff)
		v9.AddArg2(v10, v5)
		v7.AddArg2(v8, v9)
		v2.AddArg2(v3, v7)
		v.AddArg3(v0, v2, mem)
		return true
	}
	// match: (AtomicAnd8 ptr val mem)
	// cond: config.BigEndian
	// result: (LoweredAtomicAnd32 (AND <typ.UInt32Ptr> (MOVVconst [^3]) ptr) (OR <typ.UInt64> (SLLV <typ.UInt32> (ZeroExt8to32 val) (SLLVconst <typ.UInt64> [3] (ANDconst <typ.UInt64> [3] (XORconst <typ.UInt64> [3] ptr)))) (NOR (MOVVconst [0]) <typ.UInt64> (SLLV <typ.UInt64> (MOVVconst [0xff]) (SLLVconst <typ.UInt64> [3] (ANDconst <typ.UInt64> [3] (XORconst <typ.UInt64> [3] ptr)))))) mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		if !(config.BigEndian) {
			break
		}
		v.Reset(ssaop.OpMIPS64LoweredAtomicAnd32)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64AND, typ.UInt32Ptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v1.AuxInt = Int64ToAuxInt(^3)
		v0.AddArg2(v1, ptr)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, typ.UInt64)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, typ.UInt32)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v4.AddArg(val)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLVconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(3)
		v6 := b.NewValue0(v.Pos, ssaop.OpMIPS64ANDconst, typ.UInt64)
		v6.AuxInt = Int64ToAuxInt(3)
		v7 := b.NewValue0(v.Pos, ssaop.OpMIPS64XORconst, typ.UInt64)
		v7.AuxInt = Int64ToAuxInt(3)
		v7.AddArg(ptr)
		v6.AddArg(v7)
		v5.AddArg(v6)
		v3.AddArg2(v4, v5)
		v8 := b.NewValue0(v.Pos, ssaop.OpMIPS64NOR, typ.UInt64)
		v9 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v9.AuxInt = Int64ToAuxInt(0)
		v10 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, typ.UInt64)
		v11 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v11.AuxInt = Int64ToAuxInt(0xff)
		v10.AddArg2(v11, v5)
		v8.AddArg2(v9, v10)
		v2.AddArg2(v3, v8)
		v.AddArg3(v0, v2, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpAtomicCompareAndSwap32(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicCompareAndSwap32 ptr old new mem)
	// result: (LoweredAtomicCas32 ptr (SignExt32to64 old) new mem)
	for {
		ptr := v_0
		old := v_1
		new := v_2
		mem := v_3
		v.Reset(ssaop.OpMIPS64LoweredAtomicCas32)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(old)
		v.AddArg4(ptr, v0, new, mem)
		return true
	}
}
func rewriteValueMIPS64_OpAtomicOr8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (AtomicOr8 ptr val mem)
	// cond: !config.BigEndian
	// result: (LoweredAtomicOr32 (AND <typ.UInt32Ptr> (MOVVconst [^3]) ptr) (SLLV <typ.UInt32> (ZeroExt8to32 val) (SLLVconst <typ.UInt64> [3] (ANDconst <typ.UInt64> [3] ptr))) mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		if !(!config.BigEndian) {
			break
		}
		v.Reset(ssaop.OpMIPS64LoweredAtomicOr32)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64AND, typ.UInt32Ptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v1.AuxInt = Int64ToAuxInt(^3)
		v0.AddArg2(v1, ptr)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, typ.UInt32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v3.AddArg(val)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLVconst, typ.UInt64)
		v4.AuxInt = Int64ToAuxInt(3)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64ANDconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(3)
		v5.AddArg(ptr)
		v4.AddArg(v5)
		v2.AddArg2(v3, v4)
		v.AddArg3(v0, v2, mem)
		return true
	}
	// match: (AtomicOr8 ptr val mem)
	// cond: config.BigEndian
	// result: (LoweredAtomicOr32 (AND <typ.UInt32Ptr> (MOVVconst [^3]) ptr) (SLLV <typ.UInt32> (ZeroExt8to32 val) (SLLVconst <typ.UInt64> [3] (ANDconst <typ.UInt64> [3] (XORconst <typ.UInt64> [3] ptr)))) mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		if !(config.BigEndian) {
			break
		}
		v.Reset(ssaop.OpMIPS64LoweredAtomicOr32)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64AND, typ.UInt32Ptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v1.AuxInt = Int64ToAuxInt(^3)
		v0.AddArg2(v1, ptr)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, typ.UInt32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v3.AddArg(val)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLVconst, typ.UInt64)
		v4.AuxInt = Int64ToAuxInt(3)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64ANDconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(3)
		v6 := b.NewValue0(v.Pos, ssaop.OpMIPS64XORconst, typ.UInt64)
		v6.AuxInt = Int64ToAuxInt(3)
		v6.AddArg(ptr)
		v5.AddArg(v6)
		v4.AddArg(v5)
		v2.AddArg2(v3, v4)
		v.AddArg3(v0, v2, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpAvg64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Avg64u <t> x y)
	// result: (ADDV (SRLVconst <t> (SUBV <t> x y) [1]) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64ADDV)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLVconst, t)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SUBV, t)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValueMIPS64_OpCom16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com16 x)
	// result: (NOR (MOVVconst [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpMIPS64NOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueMIPS64_OpCom32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com32 x)
	// result: (NOR (MOVVconst [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpMIPS64NOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueMIPS64_OpCom64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com64 x)
	// result: (NOR (MOVVconst [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpMIPS64NOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueMIPS64_OpCom8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com8 x)
	// result: (NOR (MOVVconst [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpMIPS64NOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueMIPS64_OpConst16(v *Value) bool {
	// match: (Const16 [val])
	// result: (MOVVconst [int64(val)])
	for {
		val := AuxIntToInt16(v.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueMIPS64_OpConst32(v *Value) bool {
	// match: (Const32 [val])
	// result: (MOVVconst [int64(val)])
	for {
		val := AuxIntToInt32(v.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueMIPS64_OpConst32F(v *Value) bool {
	// match: (Const32F [val])
	// result: (MOVFconst [float64(val)])
	for {
		val := AuxIntToFloat32(v.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVFconst)
		v.AuxInt = Float64ToAuxInt(float64(val))
		return true
	}
}
func rewriteValueMIPS64_OpConst64(v *Value) bool {
	// match: (Const64 [val])
	// result: (MOVVconst [int64(val)])
	for {
		val := AuxIntToInt64(v.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueMIPS64_OpConst64F(v *Value) bool {
	// match: (Const64F [val])
	// result: (MOVDconst [float64(val)])
	for {
		val := AuxIntToFloat64(v.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVDconst)
		v.AuxInt = Float64ToAuxInt(float64(val))
		return true
	}
}
func rewriteValueMIPS64_OpConst8(v *Value) bool {
	// match: (Const8 [val])
	// result: (MOVVconst [int64(val)])
	for {
		val := AuxIntToInt8(v.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueMIPS64_OpConstBool(v *Value) bool {
	// match: (ConstBool [t])
	// result: (MOVVconst [int64(B2i(t))])
	for {
		t := AuxIntToBool(v.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(B2i(t)))
		return true
	}
}
func rewriteValueMIPS64_OpConstNil(v *Value) bool {
	// match: (ConstNil)
	// result: (MOVVconst [0])
	for {
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
}
func rewriteValueMIPS64_OpDiv16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 x y)
	// result: (Select1 (DIVV (SignExt16to64 x) (SignExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVV, types.NewTuple(typ.Int64, typ.Int64))
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpDiv16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (Select1 (DIVVU (ZeroExt16to64 x) (ZeroExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpDiv32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32 x y)
	// result: (Select1 (DIVV (SignExt32to64 x) (SignExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVV, types.NewTuple(typ.Int64, typ.Int64))
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpDiv32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32u x y)
	// result: (Select1 (DIVVU (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpDiv64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div64 x y)
	// result: (Select1 (DIVV x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVV, types.NewTuple(typ.Int64, typ.Int64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpDiv64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div64u x y)
	// result: (Select1 (DIVVU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpDiv8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (Select1 (DIVV (SignExt8to64 x) (SignExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVV, types.NewTuple(typ.Int64, typ.Int64))
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpDiv8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (Select1 (DIVVU (ZeroExt8to64 x) (ZeroExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpEq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// result: (SGTU (MOVVconst [1]) (XOR (ZeroExt16to64 x) (ZeroExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64XOR, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpEq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq32 x y)
	// result: (SGTU (MOVVconst [1]) (XOR (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64XOR, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpEq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32F x y)
	// result: (FPFlagTrue (CMPEQF x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64CMPEQF, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpEq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq64 x y)
	// result: (SGTU (MOVVconst [1]) (XOR x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64XOR, typ.UInt64)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpEq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64F x y)
	// result: (FPFlagTrue (CMPEQD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64CMPEQD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpEq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// result: (SGTU (MOVVconst [1]) (XOR (ZeroExt8to64 x) (ZeroExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64XOR, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpEqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// result: (XOR (MOVVconst [1]) (XOR <typ.Bool> x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64XOR, typ.Bool)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpEqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqPtr x y)
	// result: (SGTU (MOVVconst [1]) (XOR x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64XOR, typ.UInt64)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpHmul32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32 x y)
	// result: (SRAVconst (Select1 <typ.Int64> (MULV (SignExt32to64 x) (SignExt32to64 y))) [32])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAVconst)
		v.AuxInt = Int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.Int64)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MULV, types.NewTuple(typ.Int64, typ.Int64))
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpHmul32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32u x y)
	// result: (SRLVconst (Select1 <typ.UInt64> (MULVU (ZeroExt32to64 x) (ZeroExt32to64 y))) [32])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRLVconst)
		v.AuxInt = Int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MULVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpHmul64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul64 x y)
	// result: (Select0 (MULV x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MULV, types.NewTuple(typ.Int64, typ.Int64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpHmul64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul64u x y)
	// result: (Select0 (MULVU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MULVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpIsInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsInBounds idx len)
	// result: (SGTU len idx)
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v.AddArg2(len, idx)
		return true
	}
}
func rewriteValueMIPS64_OpIsNonNil(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsNonNil ptr)
	// result: (SGTU ptr (MOVVconst [0]))
	for {
		ptr := v_0
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(ptr, v0)
		return true
	}
}
func rewriteValueMIPS64_OpIsSliceInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsSliceInBounds idx len)
	// result: (XOR (MOVVconst [1]) (SGTU idx len))
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.OpMIPS64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v1.AddArg2(idx, len)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (XOR (MOVVconst [1]) (SGT (SignExt16to64 x) (SignExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGT, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLeq16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// result: (XOR (MOVVconst [1]) (SGTU (ZeroExt16to64 x) (ZeroExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32 x y)
	// result: (XOR (MOVVconst [1]) (SGT (SignExt32to64 x) (SignExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGT, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32F x y)
	// result: (FPFlagTrue (CMPGEF y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64CMPGEF, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpLeq32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32U x y)
	// result: (XOR (MOVVconst [1]) (SGTU (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64 x y)
	// result: (XOR (MOVVconst [1]) (SGT x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGT, typ.Bool)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64F x y)
	// result: (FPFlagTrue (CMPGED y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64CMPGED, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpLeq64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64U x y)
	// result: (XOR (MOVVconst [1]) (SGTU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (XOR (MOVVconst [1]) (SGT (SignExt8to64 x) (SignExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGT, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLeq8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// result: (XOR (MOVVconst [1]) (SGTU (ZeroExt8to64 x) (ZeroExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLess16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (SGT (SignExt16to64 y) (SignExt16to64 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGT)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLess16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// result: (SGTU (ZeroExt16to64 y) (ZeroExt16to64 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLess32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32 x y)
	// result: (SGT (SignExt32to64 y) (SignExt32to64 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGT)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLess32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32F x y)
	// result: (FPFlagTrue (CMPGTF y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64CMPGTF, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpLess32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32U x y)
	// result: (SGTU (ZeroExt32to64 y) (ZeroExt32to64 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLess64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64 x y)
	// result: (SGT y x)
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGT)
		v.AddArg2(y, x)
		return true
	}
}
func rewriteValueMIPS64_OpLess64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64F x y)
	// result: (FPFlagTrue (CMPGTD y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64CMPGTD, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpLess64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64U x y)
	// result: (SGTU y x)
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v.AddArg2(y, x)
		return true
	}
}
func rewriteValueMIPS64_OpLess8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (SGT (SignExt8to64 y) (SignExt8to64 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGT)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLess8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// result: (SGTU (ZeroExt8to64 y) (ZeroExt8to64 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpLoad(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Load <t> ptr mem)
	// cond: t.IsBoolean()
	// result: (MOVBUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.IsBoolean()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (Is8BitInt(t) && t.IsSigned())
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(Is8BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVBload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (Is8BitInt(t) && !t.IsSigned())
	// result: (MOVBUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(Is8BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (Is16BitInt(t) && t.IsSigned())
	// result: (MOVHload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(Is16BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (Is16BitInt(t) && !t.IsSigned())
	// result: (MOVHUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(Is16BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (Is32BitInt(t) && t.IsSigned())
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(Is32BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (Is32BitInt(t) && !t.IsSigned())
	// result: (MOVWUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(Is32BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (Is64BitInt(t) || IsPtr(t))
	// result: (MOVVload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(Is64BitInt(t) || IsPtr(t)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: Is32BitFloat(t)
	// result: (MOVFload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(Is32BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVFload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: Is64BitFloat(t)
	// result: (MOVDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(Is64BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpLocalAddr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LocalAddr <t> {sym} base mem)
	// cond: t.Elem().HasPointers()
	// result: (MOVVaddr {sym} (SPanchored base mem))
	for {
		t := v.Type
		sym := AuxToSym(v.Aux)
		base := v_0
		mem := v_1
		if !(t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVaddr)
		v.Aux = SymToAux(sym)
		v0 := b.NewValue0(v.Pos, ssaop.OpSPanchored, typ.Uintptr)
		v0.AddArg2(base, mem)
		v.AddArg(v0)
		return true
	}
	// match: (LocalAddr <t> {sym} base _)
	// cond: !t.Elem().HasPointers()
	// result: (MOVVaddr {sym} base)
	for {
		t := v.Type
		sym := AuxToSym(v.Aux)
		base := v_0
		if !(!t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVaddr)
		v.Aux = SymToAux(sym)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpLsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y))) (SLLV <t> x (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpLsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y))) (SLLV <t> x (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpLsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x64 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) y)) (SLLV <t> x y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v3.AddArg2(x, y)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueMIPS64_OpLsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y))) (SLLV <t> x (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpLsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y))) (SLLV <t> x (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpLsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y))) (SLLV <t> x (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpLsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x64 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) y)) (SLLV <t> x y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v3.AddArg2(x, y)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueMIPS64_OpLsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y))) (SLLV <t> x (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpLsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y))) (SLLV <t> x (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpLsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y))) (SLLV <t> x (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpLsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x64 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) y)) (SLLV <t> x y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v3.AddArg2(x, y)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueMIPS64_OpLsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y))) (SLLV <t> x (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpLsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y))) (SLLV <t> x (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpLsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y))) (SLLV <t> x (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpLsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x64 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) y)) (SLLV <t> x y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v3.AddArg2(x, y)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueMIPS64_OpLsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y))) (SLLV <t> x (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SLLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpMIPS64ADDV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDV x (MOVVconst <t> [c]))
	// cond: Is32Bit(c) && !t.IsPtr()
	// result: (ADDVconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMIPS64MOVVconst {
				continue
			}
			t := v_1.Type
			c := AuxIntToInt64(v_1.AuxInt)
			if !(Is32Bit(c) && !t.IsPtr()) {
				continue
			}
			v.Reset(ssaop.OpMIPS64ADDVconst)
			v.AuxInt = Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADDV x (NEGV y))
	// result: (SUBV x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMIPS64NEGV {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpMIPS64SUBV)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64ADDVconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ADDVconst [off1] (MOVVaddr [off2] {sym} ptr))
	// cond: Is32Bit(off1+int64(off2))
	// result: (MOVVaddr [int32(off1)+int32(off2)] {sym} ptr)
	for {
		off1 := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		if !(Is32Bit(off1 + int64(off2))) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVaddr)
		v.AuxInt = Int32ToAuxInt(int32(off1) + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg(ptr)
		return true
	}
	// match: (ADDVconst [0] x)
	// result: x
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ADDVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [c+d])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(c + d)
		return true
	}
	// match: (ADDVconst [c] (ADDVconst [d] x))
	// cond: Is32Bit(c+d)
	// result: (ADDVconst [c+d] x)
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(Is32Bit(c + d)) {
			break
		}
		v.Reset(ssaop.OpMIPS64ADDVconst)
		v.AuxInt = Int64ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (ADDVconst [c] (SUBVconst [d] x))
	// cond: Is32Bit(c-d)
	// result: (ADDVconst [c-d] x)
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64SUBVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(Is32Bit(c - d)) {
			break
		}
		v.Reset(ssaop.OpMIPS64ADDVconst)
		v.AuxInt = Int64ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64AND(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AND x (MOVVconst [c]))
	// cond: Is32Bit(c)
	// result: (ANDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMIPS64MOVVconst {
				continue
			}
			c := AuxIntToInt64(v_1.AuxInt)
			if !(Is32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpMIPS64ANDconst)
			v.AuxInt = Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64ANDconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ANDconst [0] _)
	// result: (MOVVconst [0])
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (ANDconst [-1] x)
	// result: x
	for {
		if AuxIntToInt64(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ANDconst [c] (MOVVconst [d]))
	// result: (MOVVconst [c&d])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(c & d)
		return true
	}
	// match: (ANDconst [c] (ANDconst [d] x))
	// result: (ANDconst [c&d] x)
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64ANDconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPS64ANDconst)
		v.AuxInt = Int64ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64LoweredAtomicAdd32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredAtomicAdd32 ptr (MOVVconst [c]) mem)
	// cond: Is32Bit(c)
	// result: (LoweredAtomicAddconst32 [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpMIPS64LoweredAtomicAddconst32)
		v.AuxInt = Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64LoweredAtomicAdd64(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredAtomicAdd64 ptr (MOVVconst [c]) mem)
	// cond: Is32Bit(c)
	// result: (LoweredAtomicAddconst64 [c] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpMIPS64LoweredAtomicAddconst64)
		v.AuxInt = Int64ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64LoweredAtomicStore32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredAtomicStore32 ptr (MOVVconst [0]) mem)
	// result: (LoweredAtomicStorezero32 ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst || AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpMIPS64LoweredAtomicStorezero32)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64LoweredAtomicStore64(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredAtomicStore64 ptr (MOVVconst [0]) mem)
	// result: (LoweredAtomicStorezero64 ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst || AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpMIPS64LoweredAtomicStorezero64)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64LoweredPanicBoundsCR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsCR [kind] {p} (MOVVconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {PanicBoundsCC{Cx:p.C, Cy:c}} mem)
	for {
		kind := AuxIntToInt64(v.AuxInt)
		p := AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.OpMIPS64LoweredPanicBoundsCC)
		v.AuxInt = Int64ToAuxInt(kind)
		v.Aux = PanicBoundsCCToAux(PanicBoundsCC{Cx: p.C, Cy: c})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64LoweredPanicBoundsRC(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRC [kind] {p} (MOVVconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {PanicBoundsCC{Cx:c, Cy:p.C}} mem)
	for {
		kind := AuxIntToInt64(v.AuxInt)
		p := AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.OpMIPS64LoweredPanicBoundsCC)
		v.AuxInt = Int64ToAuxInt(kind)
		v.Aux = PanicBoundsCCToAux(PanicBoundsCC{Cx: c, Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64LoweredPanicBoundsRR(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRR [kind] x (MOVVconst [c]) mem)
	// result: (LoweredPanicBoundsRC [kind] x {PanicBoundsC{C:c}} mem)
	for {
		kind := AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpMIPS64LoweredPanicBoundsRC)
		v.AuxInt = Int64ToAuxInt(kind)
		v.Aux = PanicBoundsCToAux(PanicBoundsC{C: c})
		v.AddArg2(x, mem)
		return true
	}
	// match: (LoweredPanicBoundsRR [kind] (MOVVconst [c]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {PanicBoundsC{C:c}} y mem)
	for {
		kind := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		y := v_1
		mem := v_2
		v.Reset(ssaop.OpMIPS64LoweredPanicBoundsCR)
		v.AuxInt = Int64ToAuxInt(kind)
		v.Aux = PanicBoundsCToAux(PanicBoundsC{C: c})
		v.AddArg2(y, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVBUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBUload [off] {sym} ptr (MOVBstore [off] {sym} ptr x _))
	// result: (MOVBUreg x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVBstore || AuxIntToInt32(v_1.AuxInt) != off || AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVBUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVBUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVBUload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVBUload [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVBUload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} (SB) _)
	// cond: SymIsRO(sym)
	// result: (MOVVconst [int64(Read8(sym, int64(off)))])
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(Read8(sym, int64(off))))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVBUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBUreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBUload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (MOVVconst [c]))
	// result: (MOVVconst [int64(uint8(c))])
	for {
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(uint8(c)))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVBload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBload [off] {sym} ptr (MOVBstore [off] {sym} ptr x _))
	// result: (MOVBreg x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVBstore || AuxIntToInt32(v_1.AuxInt) != off || AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVBload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVBload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVBload [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVBload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off] {sym} (SB) _)
	// cond: SymIsRO(sym)
	// result: (MOVVconst [int64(int8(Read8(sym, int64(off))))])
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(int8(Read8(sym, int64(off)))))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVBreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBreg x:(MOVBload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (MOVVconst [c]))
	// result: (MOVVconst [int64(int8(c))])
	for {
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(int8(c)))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVBstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVBstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVBstore [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVBreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVBUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVDF(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDF (ABSD (MOVFD x)))
	// result: (ABSF x)
	for {
		if v_0.Op != ssaop.OpMIPS64ABSD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpMIPS64MOVFD {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpMIPS64ABSF)
		v.AddArg(x)
		return true
	}
	// match: (MOVDF (SQRTD (MOVFD x)))
	// result: (SQRTF x)
	for {
		if v_0.Op != ssaop.OpMIPS64SQRTD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpMIPS64MOVFD {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpMIPS64SQRTF)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVDload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVDload [off] {sym} ptr (MOVVstore [off] {sym} ptr val _))
	// result: (MOVVgpfp val)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVstore || AuxIntToInt32(v_1.AuxInt) != off || AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVgpfp)
		v.AddArg(val)
		return true
	}
	// match: (MOVDload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVDload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVDload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVDload [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVDload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVDstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVDstore [off] {sym} ptr (MOVVgpfp val) mem)
	// result: (MOVVstore [off] {sym} ptr val mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVgpfp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVVstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVDstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVDstore)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVDstore [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVDstore)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVFload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVFload [off] {sym} ptr (MOVWstore [off] {sym} ptr val _))
	// result: (MOVWgpfp val)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVWstore || AuxIntToInt32(v_1.AuxInt) != off || AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWgpfp)
		v.AddArg(val)
		return true
	}
	// match: (MOVFload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVFload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVFload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVFload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVFload [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVFload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVFstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVFstore [off] {sym} ptr (MOVWgpfp val) mem)
	// result: (MOVWstore [off] {sym} ptr val mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVWgpfp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVWstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVFstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVFstore)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVFstore [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVFstore)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVHUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHUload [off] {sym} ptr (MOVHstore [off] {sym} ptr x _))
	// result: (MOVHUreg x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVHstore || AuxIntToInt32(v_1.AuxInt) != off || AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVHUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHUload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVHUload [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHUload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (SB) _)
	// cond: SymIsRO(sym)
	// result: (MOVVconst [int64(Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVHUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHUreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBUload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVHUload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVHUreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (MOVVconst [c]))
	// result: (MOVVconst [int64(uint16(c))])
	for {
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(uint16(c)))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVHload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHload [off] {sym} ptr (MOVHstore [off] {sym} ptr x _))
	// result: (MOVHreg x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVHstore || AuxIntToInt32(v_1.AuxInt) != off || AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVHload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVHload [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off] {sym} (SB) _)
	// cond: SymIsRO(sym)
	// result: (MOVVconst [int64(int16(Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder)))])
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(int16(Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder))))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVHreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHreg x:(MOVBload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBUload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVHload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVHreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (MOVVconst [c]))
	// result: (MOVVconst [int64(int16(c))])
	for {
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(int16(c)))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVHstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVHstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVHstore [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHUreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVVload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVVload [off] {sym} ptr (MOVDstore [off] {sym} ptr val _))
	// result: (MOVVfpgp val)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVDstore || AuxIntToInt32(v_1.AuxInt) != off || AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVfpgp)
		v.AddArg(val)
		return true
	}
	// match: (MOVVload [off] {sym} ptr (MOVVstore [off] {sym} ptr x _))
	// result: (MOVVreg x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVstore || AuxIntToInt32(v_1.AuxInt) != off || AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVVload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVVload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVVload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVVload [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVVload [off] {sym} (SB) _)
	// cond: SymIsRO(sym)
	// result: (MOVVconst [int64(Read64(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(Read64(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVVnop(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVVnop (MOVVconst [c]))
	// result: (MOVVconst [c])
	for {
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVVreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVVreg x)
	// cond: x.Uses == 1
	// result: (MOVVnop x)
	for {
		x := v_0
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVnop)
		v.AddArg(x)
		return true
	}
	// match: (MOVVreg (MOVVconst [c]))
	// result: (MOVVconst [c])
	for {
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVVstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVVstore [off] {sym} ptr (MOVVfpgp val) mem)
	// result: (MOVDstore [off] {sym} ptr val mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVfpgp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVDstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVVstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVVstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVstore)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVVstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVVstore [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVstore)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVWUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVWUload [off] {sym} ptr (MOVFstore [off] {sym} ptr val _))
	// result: (ZeroExt32to64 (MOVWfpgp <typ.Float32> val))
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVFstore || AuxIntToInt32(v_1.AuxInt) != off || AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpZeroExt32to64)
		v0 := b.NewValue0(v_1.Pos, ssaop.OpMIPS64MOVWfpgp, typ.Float32)
		v0.AddArg(val)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWUload [off] {sym} ptr (MOVWstore [off] {sym} ptr x _))
	// result: (MOVWUreg x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVWstore || AuxIntToInt32(v_1.AuxInt) != off || AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVWUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWUload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVWUload [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWUload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUload [off] {sym} (SB) _)
	// cond: SymIsRO(sym)
	// result: (MOVVconst [int64(Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVWUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWUreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBUload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVHUload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVWUload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVHUreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVWUreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (MOVVconst [c]))
	// result: (MOVVconst [int64(uint32(c))])
	for {
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(uint32(c)))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVWload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWload [off] {sym} ptr (MOVWstore [off] {sym} ptr x _))
	// result: (MOVWreg x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVWstore || AuxIntToInt32(v_1.AuxInt) != off || AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVWload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVWload [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWload)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (SB) _)
	// cond: SymIsRO(sym)
	// result: (MOVVconst [int64(int32(Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder)))])
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(int32(Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder))))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVWreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWreg x:(MOVBload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBUload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVHload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVHUload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVWload {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVHreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpMIPS64MOVWreg {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (MOVVconst [c]))
	// result: (MOVVconst [int64(int32(c))])
	for {
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(int32(c)))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64MOVWstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstore [off] {sym} ptr (MOVWfpgp val) mem)
	// result: (MOVFstore [off] {sym} ptr val mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVWfpgp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVFstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVWstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		off2 := AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWstore)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: CanMergeSym(sym1,sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)
	// result: (MOVWstore [off1+int32(off2)] {MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpMIPS64MOVVaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(CanMergeSym(sym1, sym2) && Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_shared)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWstore)
		v.AuxInt = Int32ToAuxInt(off1 + int32(off2))
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVWstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpMIPS64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVWstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64NEGV(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NEGV (SUBV x y))
	// result: (SUBV y x)
	for {
		if v_0.Op != ssaop.OpMIPS64SUBV {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPS64SUBV)
		v.AddArg2(y, x)
		return true
	}
	// match: (NEGV (NEGV x))
	// result: x
	for {
		if v_0.Op != ssaop.OpMIPS64NEGV {
			break
		}
		x := v_0.Args[0]
		v.CopyOf(x)
		return true
	}
	// match: (NEGV (MOVVconst [c]))
	// result: (MOVVconst [-c])
	for {
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(-c)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64OR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (OR x (MOVVconst [c]))
	// cond: Is32Bit(c)
	// result: (ORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMIPS64MOVVconst {
				continue
			}
			c := AuxIntToInt64(v_1.AuxInt)
			if !(Is32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpMIPS64ORconst)
			v.AuxInt = Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (OR x x)
	// result: x
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64ORconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ORconst [0] x)
	// result: x
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ORconst [-1] _)
	// result: (MOVVconst [-1])
	for {
		if AuxIntToInt64(v.AuxInt) != -1 {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(-1)
		return true
	}
	// match: (ORconst [c] (MOVVconst [d]))
	// result: (MOVVconst [c|d])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(c | d)
		return true
	}
	// match: (ORconst [c] (ORconst [d] x))
	// cond: Is32Bit(c|d)
	// result: (ORconst [c|d] x)
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64ORconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(Is32Bit(c | d)) {
			break
		}
		v.Reset(ssaop.OpMIPS64ORconst)
		v.AuxInt = Int64ToAuxInt(c | d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64SGT(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SGT (MOVVconst [c]) x)
	// cond: Is32Bit(c)
	// result: (SGTconst [c] x)
	for {
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		x := v_1
		if !(Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpMIPS64SGTconst)
		v.AuxInt = Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SGT x x)
	// result: (MOVVconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64SGTU(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SGTU (MOVVconst [c]) x)
	// cond: Is32Bit(c)
	// result: (SGTUconst [c] x)
	for {
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0.AuxInt)
		x := v_1
		if !(Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpMIPS64SGTUconst)
		v.AuxInt = Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SGTU x x)
	// result: (MOVVconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64SGTUconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SGTUconst [c] (MOVVconst [d]))
	// cond: uint64(c)>uint64(d)
	// result: (MOVVconst [1])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		if !(uint64(c) > uint64(d)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (MOVVconst [d]))
	// cond: uint64(c)<=uint64(d)
	// result: (MOVVconst [0])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		if !(uint64(c) <= uint64(d)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (SGTUconst [c] (MOVBUreg _))
	// cond: 0xff < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVBUreg || !(0xff < uint64(c)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (MOVHUreg _))
	// cond: 0xffff < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVHUreg || !(0xffff < uint64(c)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (ANDconst [m] _))
	// cond: uint64(m) < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64ANDconst {
			break
		}
		m := AuxIntToInt64(v_0.AuxInt)
		if !(uint64(m) < uint64(c)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (SRLVconst _ [d]))
	// cond: 0 < d && d <= 63 && 0xffffffffffffffff>>uint64(d) < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64SRLVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		if !(0 < d && d <= 63 && 0xffffffffffffffff>>uint64(d) < uint64(c)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64SGTconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SGTconst [c] (MOVVconst [d]))
	// cond: c>d
	// result: (MOVVconst [1])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		if !(c > d) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVVconst [d]))
	// cond: c<=d
	// result: (MOVVconst [0])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		if !(c <= d) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVBreg _))
	// cond: 0x7f < c
	// result: (MOVVconst [1])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVBreg || !(0x7f < c) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVBreg _))
	// cond: c <= -0x80
	// result: (MOVVconst [0])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVBreg || !(c <= -0x80) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVBUreg _))
	// cond: 0xff < c
	// result: (MOVVconst [1])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVBUreg || !(0xff < c) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVBUreg _))
	// cond: c < 0
	// result: (MOVVconst [0])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVBUreg || !(c < 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVHreg _))
	// cond: 0x7fff < c
	// result: (MOVVconst [1])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVHreg || !(0x7fff < c) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVHreg _))
	// cond: c <= -0x8000
	// result: (MOVVconst [0])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVHreg || !(c <= -0x8000) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVHUreg _))
	// cond: 0xffff < c
	// result: (MOVVconst [1])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVHUreg || !(0xffff < c) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVHUreg _))
	// cond: c < 0
	// result: (MOVVconst [0])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVHUreg || !(c < 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVWUreg _))
	// cond: c < 0
	// result: (MOVVconst [0])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVWUreg || !(c < 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (ANDconst [m] _))
	// cond: 0 <= m && m < c
	// result: (MOVVconst [1])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64ANDconst {
			break
		}
		m := AuxIntToInt64(v_0.AuxInt)
		if !(0 <= m && m < c) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (SRLVconst _ [d]))
	// cond: 0 <= c && 0 < d && d <= 63 && 0xffffffffffffffff>>uint64(d) < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64SRLVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		if !(0 <= c && 0 < d && d <= 63 && 0xffffffffffffffff>>uint64(d) < uint64(c)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64SLLV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLLV _ (MOVVconst [c]))
	// cond: uint64(c)>=64
	// result: (MOVVconst [0])
	for {
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (SLLV x (MOVVconst [c]))
	// result: (SLLVconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpMIPS64SLLVconst)
		v.AuxInt = Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64SLLVconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SLLVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [d<<uint64(c)])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(d << uint64(c))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64SRAV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRAV x (MOVVconst [c]))
	// cond: uint64(c)>=64
	// result: (SRAVconst x [63])
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.Reset(ssaop.OpMIPS64SRAVconst)
		v.AuxInt = Int64ToAuxInt(63)
		v.AddArg(x)
		return true
	}
	// match: (SRAV x (MOVVconst [c]))
	// result: (SRAVconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpMIPS64SRAVconst)
		v.AuxInt = Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64SRAVconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SRAVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [d>>uint64(c)])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(d >> uint64(c))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64SRLV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRLV _ (MOVVconst [c]))
	// cond: uint64(c)>=64
	// result: (MOVVconst [0])
	for {
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (SRLV x (MOVVconst [c]))
	// result: (SRLVconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpMIPS64SRLVconst)
		v.AuxInt = Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64SRLVconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SRLVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [int64(uint64(d)>>uint64(c))])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(uint64(d) >> uint64(c)))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64SUBV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBV x (MOVVconst [c]))
	// cond: Is32Bit(c)
	// result: (SUBVconst [c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpMIPS64SUBVconst)
		v.AuxInt = Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUBV x (NEGV y))
	// result: (ADDV x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpMIPS64NEGV {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpMIPS64ADDV)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUBV x x)
	// result: (MOVVconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (SUBV (MOVVconst [0]) x)
	// result: (NEGV x)
	for {
		if v_0.Op != ssaop.OpMIPS64MOVVconst || AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpMIPS64NEGV)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64SUBVconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SUBVconst [0] x)
	// result: x
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (SUBVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [d-c])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(d - c)
		return true
	}
	// match: (SUBVconst [c] (SUBVconst [d] x))
	// cond: Is32Bit(-c-d)
	// result: (ADDVconst [-c-d] x)
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64SUBVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(Is32Bit(-c - d)) {
			break
		}
		v.Reset(ssaop.OpMIPS64ADDVconst)
		v.AuxInt = Int64ToAuxInt(-c - d)
		v.AddArg(x)
		return true
	}
	// match: (SUBVconst [c] (ADDVconst [d] x))
	// cond: Is32Bit(-c+d)
	// result: (ADDVconst [-c+d] x)
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64ADDVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(Is32Bit(-c + d)) {
			break
		}
		v.Reset(ssaop.OpMIPS64ADDVconst)
		v.AuxInt = Int64ToAuxInt(-c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64XOR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XOR x (MOVVconst [c]))
	// cond: Is32Bit(c)
	// result: (XORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpMIPS64MOVVconst {
				continue
			}
			c := AuxIntToInt64(v_1.AuxInt)
			if !(Is32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpMIPS64XORconst)
			v.AuxInt = Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XOR x x)
	// result: (MOVVconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMIPS64XORconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (XORconst [0] x)
	// result: x
	for {
		if AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (XORconst [c] (MOVVconst [d]))
	// result: (MOVVconst [c^d])
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(c ^ d)
		return true
	}
	// match: (XORconst [c] (XORconst [d] x))
	// cond: Is32Bit(c^d)
	// result: (XORconst [c^d] x)
	for {
		c := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpMIPS64XORconst {
			break
		}
		d := AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(Is32Bit(c ^ d)) {
			break
		}
		v.Reset(ssaop.OpMIPS64XORconst)
		v.AuxInt = Int64ToAuxInt(c ^ d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMod16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// result: (Select0 (DIVV (SignExt16to64 x) (SignExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVV, types.NewTuple(typ.Int64, typ.Int64))
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpMod16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (Select0 (DIVVU (ZeroExt16to64 x) (ZeroExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpMod32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32 x y)
	// result: (Select0 (DIVV (SignExt32to64 x) (SignExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVV, types.NewTuple(typ.Int64, typ.Int64))
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpMod32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32u x y)
	// result: (Select0 (DIVVU (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpMod64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod64 x y)
	// result: (Select0 (DIVV x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVV, types.NewTuple(typ.Int64, typ.Int64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpMod64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod64u x y)
	// result: (Select0 (DIVVU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpMod8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (Select0 (DIVV (SignExt8to64 x) (SignExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVV, types.NewTuple(typ.Int64, typ.Int64))
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpMod8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (Select0 (DIVVU (ZeroExt8to64 x) (ZeroExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64DIVVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpMove(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
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
	// result: (MOVBstore dst (MOVBload src mem) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBload, typ.Int8)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore dst (MOVHload src mem) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHload, typ.Int16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (MOVBstore [1] dst (MOVBload [1] src mem) (MOVBstore dst (MOVBload src mem) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBload, typ.Int8)
		v0.AuxInt = Int32ToAuxInt(1)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBload, typ.Int8)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [4] {t} dst src mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore dst (MOVWload src mem) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVWload, typ.Int32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [4] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [2] dst (MOVHload [2] src mem) (MOVHstore dst (MOVHload src mem) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AuxInt = Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHload, typ.Int16)
		v0.AuxInt = Int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHload, typ.Int16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (MOVBstore [3] dst (MOVBload [3] src mem) (MOVBstore [2] dst (MOVBload [2] src mem) (MOVBstore [1] dst (MOVBload [1] src mem) (MOVBstore dst (MOVBload src mem) mem))))
	for {
		if AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBload, typ.Int8)
		v0.AuxInt = Int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBload, typ.Int8)
		v2.AuxInt = Int32ToAuxInt(2)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBstore, types.TypeMem)
		v3.AuxInt = Int32ToAuxInt(1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBload, typ.Int8)
		v4.AuxInt = Int32ToAuxInt(1)
		v4.AddArg2(src, mem)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBstore, types.TypeMem)
		v6 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBload, typ.Int8)
		v6.AddArg2(src, mem)
		v5.AddArg3(dst, v6, mem)
		v3.AddArg3(dst, v4, v5)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [8] {t} dst src mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVVstore dst (MOVVload src mem) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVload, typ.UInt64)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [8] {t} dst src mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [4] dst (MOVWload [4] src mem) (MOVWstore dst (MOVWload src mem) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWstore)
		v.AuxInt = Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVWload, typ.Int32)
		v0.AuxInt = Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVWload, typ.Int32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [8] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [6] dst (MOVHload [6] src mem) (MOVHstore [4] dst (MOVHload [4] src mem) (MOVHstore [2] dst (MOVHload [2] src mem) (MOVHstore dst (MOVHload src mem) mem))))
	for {
		if AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AuxInt = Int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHload, typ.Int16)
		v0.AuxInt = Int32ToAuxInt(6)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHload, typ.Int16)
		v2.AuxInt = Int32ToAuxInt(4)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHstore, types.TypeMem)
		v3.AuxInt = Int32ToAuxInt(2)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHload, typ.Int16)
		v4.AuxInt = Int32ToAuxInt(2)
		v4.AddArg2(src, mem)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHstore, types.TypeMem)
		v6 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHload, typ.Int16)
		v6.AddArg2(src, mem)
		v5.AddArg3(dst, v6, mem)
		v3.AddArg3(dst, v4, v5)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [3] dst src mem)
	// result: (MOVBstore [2] dst (MOVBload [2] src mem) (MOVBstore [1] dst (MOVBload [1] src mem) (MOVBstore dst (MOVBload src mem) mem)))
	for {
		if AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBload, typ.Int8)
		v0.AuxInt = Int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBload, typ.Int8)
		v2.AuxInt = Int32ToAuxInt(1)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBload, typ.Int8)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [6] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [4] dst (MOVHload [4] src mem) (MOVHstore [2] dst (MOVHload [2] src mem) (MOVHstore dst (MOVHload src mem) mem)))
	for {
		if AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AuxInt = Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHload, typ.Int16)
		v0.AuxInt = Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHload, typ.Int16)
		v2.AuxInt = Int32ToAuxInt(2)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHload, typ.Int16)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [12] {t} dst src mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [8] dst (MOVWload [8] src mem) (MOVWstore [4] dst (MOVWload [4] src mem) (MOVWstore dst (MOVWload src mem) mem)))
	for {
		if AuxIntToInt64(v.AuxInt) != 12 {
			break
		}
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWstore)
		v.AuxInt = Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVWload, typ.Int32)
		v0.AuxInt = Int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVWstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVWload, typ.Int32)
		v2.AuxInt = Int32ToAuxInt(4)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVWstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVWload, typ.Int32)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [16] {t} dst src mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVVstore [8] dst (MOVVload [8] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 16 {
			break
		}
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVstore)
		v.AuxInt = Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVload, typ.UInt64)
		v0.AuxInt = Int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [24] {t} dst src mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVVstore [16] dst (MOVVload [16] src mem) (MOVVstore [8] dst (MOVVload [8] src mem) (MOVVstore dst (MOVVload src mem) mem)))
	for {
		if AuxIntToInt64(v.AuxInt) != 24 {
			break
		}
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVstore)
		v.AuxInt = Int32ToAuxInt(16)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVload, typ.UInt64)
		v0.AuxInt = Int32ToAuxInt(16)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVload, typ.UInt64)
		v2.AuxInt = Int32ToAuxInt(8)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVload, typ.UInt64)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] {t} dst src mem)
	// cond: s%8 == 0 && s >= 24 && s <= 8*128 && t.Alignment()%8 == 0 && LogLargeCopyValue(v, s)
	// result: (DUFFCOPY [16 * (128 - s/8)] dst src mem)
	for {
		s := AuxIntToInt64(v.AuxInt)
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s%8 == 0 && s >= 24 && s <= 8*128 && t.Alignment()%8 == 0 && LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.OpMIPS64DUFFCOPY)
		v.AuxInt = Int64ToAuxInt(16 * (128 - s/8))
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] {t} dst src mem)
	// cond: s > 24 && LogLargeCopyValue(v, s) || t.Alignment()%8 != 0
	// result: (LoweredMove [t.Alignment()] dst src (ADDVconst <src.Type> src [s-MoveSize(t.Alignment(), config)]) mem)
	for {
		s := AuxIntToInt64(v.AuxInt)
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 24 && LogLargeCopyValue(v, s) || t.Alignment()%8 != 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64LoweredMove)
		v.AuxInt = Int64ToAuxInt(t.Alignment())
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64ADDVconst, src.Type)
		v0.AuxInt = Int64ToAuxInt(s - MoveSize(t.Alignment(), config))
		v0.AddArg(src)
		v.AddArg4(dst, src, v0, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpMul16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul16 x y)
	// result: (Select1 (MULVU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MULVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpMul32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul32 x y)
	// result: (Select1 (MULVU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MULVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpMul64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul64 x y)
	// result: (Select1 (MULVU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MULVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpMul8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul8 x y)
	// result: (Select1 (MULVU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MULVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpNeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// result: (SGTU (XOR (ZeroExt16to32 x) (ZeroExt16to64 y)) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64XOR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v3.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueMIPS64_OpNeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq32 x y)
	// result: (SGTU (XOR (ZeroExt32to64 x) (ZeroExt32to64 y)) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64XOR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v3.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueMIPS64_OpNeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32F x y)
	// result: (FPFlagFalse (CMPEQF x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64FPFlagFalse)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64CMPEQF, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpNeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq64 x y)
	// result: (SGTU (XOR x y) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64XOR, typ.UInt64)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v1.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpNeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64F x y)
	// result: (FPFlagFalse (CMPEQD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64FPFlagFalse)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64CMPEQD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpNeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// result: (SGTU (XOR (ZeroExt8to64 x) (ZeroExt8to64 y)) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64XOR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v3.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueMIPS64_OpNeqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NeqPtr x y)
	// result: (SGTU (XOR x y) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64XOR, typ.UInt64)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v1.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpNot(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Not x)
	// result: (XORconst [1] x)
	for {
		x := v_0
		v.Reset(ssaop.OpMIPS64XORconst)
		v.AuxInt = Int64ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueMIPS64_OpOffPtr(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OffPtr [off] ptr:(SP))
	// cond: Is32Bit(off)
	// result: (MOVVaddr [int32(off)] ptr)
	for {
		off := AuxIntToInt64(v.AuxInt)
		ptr := v_0
		if ptr.Op != ssaop.OpSP || !(Is32Bit(off)) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVaddr)
		v.AuxInt = Int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// cond: Is32Bit(off)
	// result: (ADDVconst [off] ptr)
	for {
		off := AuxIntToInt64(v.AuxInt)
		ptr := v_0
		if !(Is32Bit(off)) {
			break
		}
		v.Reset(ssaop.OpMIPS64ADDVconst)
		v.AuxInt = Int64ToAuxInt(off)
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// cond: !Is32Bit(off)
	// result: (ADDV ptr (MOVVconst <typ.UInt64> [off]))
	for {
		off := AuxIntToInt64(v.AuxInt)
		ptr := v_0
		if !(!Is32Bit(off)) {
			break
		}
		v.Reset(ssaop.OpMIPS64ADDV)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(off)
		v.AddArg2(ptr, v0)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpRotateLeft16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVVconst [c]))
	// result: (Or16 (Lsh16x64 <t> x (MOVVconst [c&15])) (Rsh16Ux64 <t> x (MOVVconst [-c&15])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpOr16)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh16x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v1.AuxInt = Int64ToAuxInt(c & 15)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh16Ux64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v3.AuxInt = Int64ToAuxInt(-c & 15)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpRotateLeft32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft32 <t> x (MOVVconst [c]))
	// result: (Or32 (Lsh32x64 <t> x (MOVVconst [c&31])) (Rsh32Ux64 <t> x (MOVVconst [-c&31])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpOr32)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh32x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v1.AuxInt = Int64ToAuxInt(c & 31)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh32Ux64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v3.AuxInt = Int64ToAuxInt(-c & 31)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpRotateLeft64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft64 <t> x (MOVVconst [c]))
	// result: (Or64 (Lsh64x64 <t> x (MOVVconst [c&63])) (Rsh64Ux64 <t> x (MOVVconst [-c&63])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpOr64)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh64x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v1.AuxInt = Int64ToAuxInt(c & 63)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh64Ux64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v3.AuxInt = Int64ToAuxInt(-c & 63)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpRotateLeft8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVVconst [c]))
	// result: (Or8 (Lsh8x64 <t> x (MOVVconst [c&7])) (Rsh8Ux64 <t> x (MOVVconst [-c&7])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpOr8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh8x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v1.AuxInt = Int64ToAuxInt(c & 7)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh8Ux64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v3.AuxInt = Int64ToAuxInt(-c & 7)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpRsh16Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y))) (SRLV <t> (ZeroExt16to64 x) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v5 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v5.AddArg(x)
		v4.AddArg2(v5, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpRsh16Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y))) (SRLV <t> (ZeroExt16to64 x) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v5 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v5.AddArg(x)
		v4.AddArg2(v5, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpRsh16Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) y)) (SRLV <t> (ZeroExt16to64 x) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v4.AddArg(x)
		v3.AddArg2(v4, y)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueMIPS64_OpRsh16Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y))) (SRLV <t> (ZeroExt16to64 x) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v5 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v5.AddArg(x)
		v4.AddArg2(v5, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpRsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 <t> x y)
	// result: (SRAV (SignExt16to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt16to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpRsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 <t> x y)
	// result: (SRAV (SignExt16to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt32to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpRsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 <t> x y)
	// result: (SRAV (SignExt16to64 x) (OR <t> (NEGV <t> (SGTU y (MOVVconst <typ.UInt64> [63]))) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v4.AuxInt = Int64ToAuxInt(63)
		v3.AddArg2(y, v4)
		v2.AddArg(v3)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpRsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 <t> x y)
	// result: (SRAV (SignExt16to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt8to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpRsh32Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y))) (SRLV <t> (ZeroExt32to64 x) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v5 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v5.AddArg(x)
		v4.AddArg2(v5, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpRsh32Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y))) (SRLV <t> (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v5 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v5.AddArg(x)
		v4.AddArg2(v5, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpRsh32Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) y)) (SRLV <t> (ZeroExt32to64 x) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v4.AddArg(x)
		v3.AddArg2(v4, y)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueMIPS64_OpRsh32Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y))) (SRLV <t> (ZeroExt32to64 x) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v5 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v5.AddArg(x)
		v4.AddArg2(v5, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpRsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 <t> x y)
	// result: (SRAV (SignExt32to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt16to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpRsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 <t> x y)
	// result: (SRAV (SignExt32to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt32to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpRsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x64 <t> x y)
	// result: (SRAV (SignExt32to64 x) (OR <t> (NEGV <t> (SGTU y (MOVVconst <typ.UInt64> [63]))) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v4.AuxInt = Int64ToAuxInt(63)
		v3.AddArg2(y, v4)
		v2.AddArg(v3)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpRsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 <t> x y)
	// result: (SRAV (SignExt32to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt8to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpRsh64Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y))) (SRLV <t> x (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpRsh64Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y))) (SRLV <t> x (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpRsh64Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux64 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) y)) (SRLV <t> x y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v3.AddArg2(x, y)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueMIPS64_OpRsh64Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y))) (SRLV <t> x (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v4.AddArg2(x, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpRsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 <t> x y)
	// result: (SRAV x (OR <t> (NEGV <t> (SGTU (ZeroExt16to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v4.AuxInt = Int64ToAuxInt(63)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg2(v1, v3)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueMIPS64_OpRsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x32 <t> x y)
	// result: (SRAV x (OR <t> (NEGV <t> (SGTU (ZeroExt32to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v4.AuxInt = Int64ToAuxInt(63)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg2(v1, v3)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueMIPS64_OpRsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x64 <t> x y)
	// result: (SRAV x (OR <t> (NEGV <t> (SGTU y (MOVVconst <typ.UInt64> [63]))) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v3.AuxInt = Int64ToAuxInt(63)
		v2.AddArg2(y, v3)
		v1.AddArg(v2)
		v0.AddArg2(v1, y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueMIPS64_OpRsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 <t> x y)
	// result: (SRAV x (OR <t> (NEGV <t> (SGTU (ZeroExt8to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v4.AuxInt = Int64ToAuxInt(63)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg2(v1, v3)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueMIPS64_OpRsh8Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y))) (SRLV <t> (ZeroExt8to64 x) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v5 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v5.AddArg(x)
		v4.AddArg2(v5, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpRsh8Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y))) (SRLV <t> (ZeroExt8to64 x) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v5 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v5.AddArg(x)
		v4.AddArg2(v5, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpRsh8Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) y)) (SRLV <t> (ZeroExt8to64 x) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v4.AddArg(x)
		v3.AddArg2(v4, y)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueMIPS64_OpRsh8Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 <t> x y)
	// result: (AND (NEGV <t> (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y))) (SRLV <t> (ZeroExt8to64 x) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64AND)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64SRLV, t)
		v5 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v5.AddArg(x)
		v4.AddArg2(v5, v3)
		v.AddArg2(v0, v4)
		return true
	}
}
func rewriteValueMIPS64_OpRsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 <t> x y)
	// result: (SRAV (SignExt8to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt16to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpRsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 <t> x y)
	// result: (SRAV (SignExt8to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt32to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpRsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 <t> x y)
	// result: (SRAV (SignExt8to64 x) (OR <t> (NEGV <t> (SGTU y (MOVVconst <typ.UInt64> [63]))) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v4.AuxInt = Int64ToAuxInt(63)
		v3.AddArg2(y, v4)
		v2.AddArg(v3)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpRsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 <t> x y)
	// result: (SRAV (SignExt8to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt8to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpMIPS64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v5.AuxInt = Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueMIPS64_OpSelect0(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Mul64uover x y))
	// result: (Select1 <typ.UInt64> (MULVU x y))
	for {
		if v_0.Op != ssaop.OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpSelect1)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MULVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 <t> (Add64carry x y c))
	// result: (ADDV (ADDV <t> x y) c)
	for {
		t := v.Type
		if v_0.Op != ssaop.OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpMIPS64ADDV)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64ADDV, t)
		v0.AddArg2(x, y)
		v.AddArg2(v0, c)
		return true
	}
	// match: (Select0 <t> (Sub64borrow x y c))
	// result: (SUBV (SUBV <t> x y) c)
	for {
		t := v.Type
		if v_0.Op != ssaop.OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpMIPS64SUBV)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64SUBV, t)
		v0.AddArg2(x, y)
		v.AddArg2(v0, c)
		return true
	}
	// match: (Select0 (DIVVU _ (MOVVconst [1])))
	// result: (MOVVconst [0])
	for {
		if v_0.Op != ssaop.OpMIPS64DIVVU {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpMIPS64MOVVconst || AuxIntToInt64(v_0_1.AuxInt) != 1 {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(0)
		return true
	}
	// match: (Select0 (DIVVU x (MOVVconst [c])))
	// cond: IsPowerOfTwo(c)
	// result: (ANDconst [c-1] x)
	for {
		if v_0.Op != ssaop.OpMIPS64DIVVU {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0_1.AuxInt)
		if !(IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpMIPS64ANDconst)
		v.AuxInt = Int64ToAuxInt(c - 1)
		v.AddArg(x)
		return true
	}
	// match: (Select0 (DIVV (MOVVconst [c]) (MOVVconst [d])))
	// cond: d != 0
	// result: (MOVVconst [c%d])
	for {
		if v_0.Op != ssaop.OpMIPS64DIVV {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0_0.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(c % d)
		return true
	}
	// match: (Select0 (DIVVU (MOVVconst [c]) (MOVVconst [d])))
	// cond: d != 0
	// result: (MOVVconst [int64(uint64(c)%uint64(d))])
	for {
		if v_0.Op != ssaop.OpMIPS64DIVVU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0_0.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(uint64(c) % uint64(d)))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpSelect1(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Mul64uover x y))
	// result: (SGTU <typ.Bool> (Select0 <typ.UInt64> (MULVU x y)) (MOVVconst <typ.UInt64> [0]))
	for {
		if v_0.Op != ssaop.OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpMIPS64SGTU)
		v.Type = typ.Bool
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MULVU, types.NewTuple(typ.UInt64, typ.UInt64))
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v2.AuxInt = Int64ToAuxInt(0)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Select1 <t> (Add64carry x y c))
	// result: (OR (SGTU <t> x s:(ADDV <t> x y)) (SGTU <t> s (ADDV <t> s c)))
	for {
		t := v.Type
		if v_0.Op != ssaop.OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpMIPS64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, t)
		s := b.NewValue0(v.Pos, ssaop.OpMIPS64ADDV, t)
		s.AddArg2(x, y)
		v0.AddArg2(x, s)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64ADDV, t)
		v3.AddArg2(s, c)
		v2.AddArg2(s, v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Select1 <t> (Sub64borrow x y c))
	// result: (OR (SGTU <t> s:(SUBV <t> x y) x) (SGTU <t> (SUBV <t> s c) s))
	for {
		t := v.Type
		if v_0.Op != ssaop.OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpMIPS64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, t)
		s := b.NewValue0(v.Pos, ssaop.OpMIPS64SUBV, t)
		s.AddArg2(x, y)
		v0.AddArg2(s, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64SGTU, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64SUBV, t)
		v3.AddArg2(s, c)
		v2.AddArg2(v3, s)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Select1 (MULVU x (MOVVconst [-1])))
	// result: (NEGV x)
	for {
		if v_0.Op != ssaop.OpMIPS64MULVU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpMIPS64MOVVconst || AuxIntToInt64(v_0_1.AuxInt) != -1 {
				continue
			}
			v.Reset(ssaop.OpMIPS64NEGV)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Select1 (MULVU _ (MOVVconst [0])))
	// result: (MOVVconst [0])
	for {
		if v_0.Op != ssaop.OpMIPS64MULVU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_1.Op != ssaop.OpMIPS64MOVVconst || AuxIntToInt64(v_0_1.AuxInt) != 0 {
				continue
			}
			v.Reset(ssaop.OpMIPS64MOVVconst)
			v.AuxInt = Int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (Select1 (MULVU x (MOVVconst [1])))
	// result: x
	for {
		if v_0.Op != ssaop.OpMIPS64MULVU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpMIPS64MOVVconst || AuxIntToInt64(v_0_1.AuxInt) != 1 {
				continue
			}
			v.CopyOf(x)
			return true
		}
		break
	}
	// match: (Select1 (MULVU x (MOVVconst [c])))
	// cond: IsPowerOfTwo(c)
	// result: (SLLVconst [Log64(c)] x)
	for {
		if v_0.Op != ssaop.OpMIPS64MULVU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			if v_0_1.Op != ssaop.OpMIPS64MOVVconst {
				continue
			}
			c := AuxIntToInt64(v_0_1.AuxInt)
			if !(IsPowerOfTwo(c)) {
				continue
			}
			v.Reset(ssaop.OpMIPS64SLLVconst)
			v.AuxInt = Int64ToAuxInt(Log64(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (Select1 (DIVVU x (MOVVconst [1])))
	// result: x
	for {
		if v_0.Op != ssaop.OpMIPS64DIVVU {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpMIPS64MOVVconst || AuxIntToInt64(v_0_1.AuxInt) != 1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (Select1 (DIVVU x (MOVVconst [c])))
	// cond: IsPowerOfTwo(c)
	// result: (SRLVconst [Log64(c)] x)
	for {
		if v_0.Op != ssaop.OpMIPS64DIVVU {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0_1.AuxInt)
		if !(IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpMIPS64SRLVconst)
		v.AuxInt = Int64ToAuxInt(Log64(c))
		v.AddArg(x)
		return true
	}
	// match: (Select1 (MULVU (MOVVconst [c]) (MOVVconst [d])))
	// result: (MOVVconst [c*d])
	for {
		if v_0.Op != ssaop.OpMIPS64MULVU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpMIPS64MOVVconst {
				continue
			}
			c := AuxIntToInt64(v_0_0.AuxInt)
			if v_0_1.Op != ssaop.OpMIPS64MOVVconst {
				continue
			}
			d := AuxIntToInt64(v_0_1.AuxInt)
			v.Reset(ssaop.OpMIPS64MOVVconst)
			v.AuxInt = Int64ToAuxInt(c * d)
			return true
		}
		break
	}
	// match: (Select1 (DIVV (MOVVconst [c]) (MOVVconst [d])))
	// cond: d != 0
	// result: (MOVVconst [c/d])
	for {
		if v_0.Op != ssaop.OpMIPS64DIVV {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0_0.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(c / d)
		return true
	}
	// match: (Select1 (DIVVU (MOVVconst [c]) (MOVVconst [d])))
	// cond: d != 0
	// result: (MOVVconst [int64(uint64(c)/uint64(d))])
	for {
		if v_0.Op != ssaop.OpMIPS64DIVVU {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		c := AuxIntToInt64(v_0_0.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpMIPS64MOVVconst {
			break
		}
		d := AuxIntToInt64(v_0_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVconst)
		v.AuxInt = Int64ToAuxInt(int64(uint64(c) / uint64(d)))
		return true
	}
	return false
}
func rewriteValueMIPS64_OpSlicemask(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SRAVconst (NEGV <t> x) [63])
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpMIPS64SRAVconst)
		v.AuxInt = Int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64NEGV, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueMIPS64_OpStore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 1
	// result: (MOVBstore ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 1) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 2
	// result: (MOVHstore ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 2) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && !t.IsFloat()
	// result: (MOVWstore ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && !t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && !t.IsFloat()
	// result: (MOVVstore ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && !t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && t.IsFloat()
	// result: (MOVFstore ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVFstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && t.IsFloat()
	// result: (MOVDstore ptr val mem)
	for {
		t := AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueMIPS64_OpZero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
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
	// match: (Zero [1] ptr mem)
	// result: (MOVBstore ptr (MOVVconst [0]) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore ptr (MOVVconst [0]) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] ptr mem)
	// result: (MOVBstore [1] ptr (MOVVconst [0]) (MOVBstore [0] ptr (MOVVconst [0]) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(0)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore ptr (MOVVconst [0]) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [2] ptr (MOVVconst [0]) (MOVHstore [0] ptr (MOVVconst [0]) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AuxInt = Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(0)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [4] ptr mem)
	// result: (MOVBstore [3] ptr (MOVVconst [0]) (MOVBstore [2] ptr (MOVVconst [0]) (MOVBstore [1] ptr (MOVVconst [0]) (MOVBstore [0] ptr (MOVVconst [0]) mem))))
	for {
		if AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBstore, types.TypeMem)
		v2.AuxInt = Int32ToAuxInt(1)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBstore, types.TypeMem)
		v3.AuxInt = Int32ToAuxInt(0)
		v3.AddArg3(ptr, v0, mem)
		v2.AddArg3(ptr, v0, v3)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [8] {t} ptr mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVVstore ptr (MOVVconst [0]) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [8] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [4] ptr (MOVVconst [0]) (MOVWstore [0] ptr (MOVVconst [0]) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWstore)
		v.AuxInt = Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVWstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(0)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [8] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [6] ptr (MOVVconst [0]) (MOVHstore [4] ptr (MOVVconst [0]) (MOVHstore [2] ptr (MOVVconst [0]) (MOVHstore [0] ptr (MOVVconst [0]) mem))))
	for {
		if AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AuxInt = Int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHstore, types.TypeMem)
		v2.AuxInt = Int32ToAuxInt(2)
		v3 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHstore, types.TypeMem)
		v3.AuxInt = Int32ToAuxInt(0)
		v3.AddArg3(ptr, v0, mem)
		v2.AddArg3(ptr, v0, v3)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [3] ptr mem)
	// result: (MOVBstore [2] ptr (MOVVconst [0]) (MOVBstore [1] ptr (MOVVconst [0]) (MOVBstore [0] ptr (MOVVconst [0]) mem)))
	for {
		if AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpMIPS64MOVBstore)
		v.AuxInt = Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVBstore, types.TypeMem)
		v2.AuxInt = Int32ToAuxInt(0)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [6] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [4] ptr (MOVVconst [0]) (MOVHstore [2] ptr (MOVVconst [0]) (MOVHstore [0] ptr (MOVVconst [0]) mem)))
	for {
		if AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%2 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVHstore)
		v.AuxInt = Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVHstore, types.TypeMem)
		v2.AuxInt = Int32ToAuxInt(0)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [12] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore [8] ptr (MOVVconst [0]) (MOVWstore [4] ptr (MOVVconst [0]) (MOVWstore [0] ptr (MOVVconst [0]) mem)))
	for {
		if AuxIntToInt64(v.AuxInt) != 12 {
			break
		}
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVWstore)
		v.AuxInt = Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVWstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVWstore, types.TypeMem)
		v2.AuxInt = Int32ToAuxInt(0)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [16] {t} ptr mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVVstore [8] ptr (MOVVconst [0]) (MOVVstore [0] ptr (MOVVconst [0]) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 16 {
			break
		}
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVstore)
		v.AuxInt = Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(0)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [24] {t} ptr mem)
	// cond: t.Alignment()%8 == 0
	// result: (MOVVstore [16] ptr (MOVVconst [0]) (MOVVstore [8] ptr (MOVVconst [0]) (MOVVstore [0] ptr (MOVVconst [0]) mem)))
	for {
		if AuxIntToInt64(v.AuxInt) != 24 {
			break
		}
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(t.Alignment()%8 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64MOVVstore)
		v.AuxInt = Int32ToAuxInt(16)
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVconst, typ.UInt64)
		v0.AuxInt = Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, ssaop.OpMIPS64MOVVstore, types.TypeMem)
		v2.AuxInt = Int32ToAuxInt(0)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [s] {t} ptr mem)
	// cond: s%8 == 0 && s > 24 && s <= 8*128 && t.Alignment()%8 == 0
	// result: (DUFFZERO [8 * (128 - s/8)] ptr mem)
	for {
		s := AuxIntToInt64(v.AuxInt)
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(s%8 == 0 && s > 24 && s <= 8*128 && t.Alignment()%8 == 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64DUFFZERO)
		v.AuxInt = Int64ToAuxInt(8 * (128 - s/8))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] {t} ptr mem)
	// cond: s > 8*128 || t.Alignment()%8 != 0
	// result: (LoweredZero [t.Alignment()] ptr (ADDVconst <ptr.Type> ptr [s-MoveSize(t.Alignment(), config)]) mem)
	for {
		s := AuxIntToInt64(v.AuxInt)
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(s > 8*128 || t.Alignment()%8 != 0) {
			break
		}
		v.Reset(ssaop.OpMIPS64LoweredZero)
		v.AuxInt = Int64ToAuxInt(t.Alignment())
		v0 := b.NewValue0(v.Pos, ssaop.OpMIPS64ADDVconst, ptr.Type)
		v0.AuxInt = Int64ToAuxInt(s - MoveSize(t.Alignment(), config))
		v0.AddArg(ptr)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteBlockMIPS64(b *Block) bool {
	switch b.Kind {
	case block.BlockMIPS64EQ:
		// match: (EQ (FPFlagTrue cmp) yes no)
		// result: (FPF cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64FPFlagTrue {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPS64FPF, cmp)
			return true
		}
		// match: (EQ (FPFlagFalse cmp) yes no)
		// result: (FPT cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64FPFlagFalse {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPS64FPT, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGT _ _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64XORconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPS64SGT {
				break
			}
			b.ResetWithControl(block.BlockMIPS64NE, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGTU _ _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64XORconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPS64SGTU {
				break
			}
			b.ResetWithControl(block.BlockMIPS64NE, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGTconst _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64XORconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPS64SGTconst {
				break
			}
			b.ResetWithControl(block.BlockMIPS64NE, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGTUconst _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64XORconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPS64SGTUconst {
				break
			}
			b.ResetWithControl(block.BlockMIPS64NE, cmp)
			return true
		}
		// match: (EQ (SGTUconst [1] x) yes no)
		// result: (NE x yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64SGTUconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPS64NE, x)
			return true
		}
		// match: (EQ (SGTU x (MOVVconst [0])) yes no)
		// result: (EQ x yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64SGTU {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.OpMIPS64MOVVconst || AuxIntToInt64(v_0_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockMIPS64EQ, x)
			return true
		}
		// match: (EQ (SGTconst [0] x) yes no)
		// result: (GEZ x yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64SGTconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPS64GEZ, x)
			return true
		}
		// match: (EQ (SGT x (MOVVconst [0])) yes no)
		// result: (LEZ x yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64SGT {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.OpMIPS64MOVVconst || AuxIntToInt64(v_0_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockMIPS64LEZ, x)
			return true
		}
		// match: (EQ (MOVVconst [0]) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64MOVVconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (EQ (MOVVconst [c]) yes no)
		// cond: c != 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpMIPS64MOVVconst {
			v_0 := b.Controls[0]
			c := AuxIntToInt64(v_0.AuxInt)
			if !(c != 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockMIPS64GEZ:
		// match: (GEZ (MOVVconst [c]) yes no)
		// cond: c >= 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64MOVVconst {
			v_0 := b.Controls[0]
			c := AuxIntToInt64(v_0.AuxInt)
			if !(c >= 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GEZ (MOVVconst [c]) yes no)
		// cond: c < 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpMIPS64MOVVconst {
			v_0 := b.Controls[0]
			c := AuxIntToInt64(v_0.AuxInt)
			if !(c < 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockMIPS64GTZ:
		// match: (GTZ (MOVVconst [c]) yes no)
		// cond: c > 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64MOVVconst {
			v_0 := b.Controls[0]
			c := AuxIntToInt64(v_0.AuxInt)
			if !(c > 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GTZ (MOVVconst [c]) yes no)
		// cond: c <= 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpMIPS64MOVVconst {
			v_0 := b.Controls[0]
			c := AuxIntToInt64(v_0.AuxInt)
			if !(c <= 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockIf:
		// match: (If cond yes no)
		// result: (NE cond yes no)
		for {
			cond := b.Controls[0]
			b.ResetWithControl(block.BlockMIPS64NE, cond)
			return true
		}
	case block.BlockMIPS64LEZ:
		// match: (LEZ (MOVVconst [c]) yes no)
		// cond: c <= 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64MOVVconst {
			v_0 := b.Controls[0]
			c := AuxIntToInt64(v_0.AuxInt)
			if !(c <= 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LEZ (MOVVconst [c]) yes no)
		// cond: c > 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpMIPS64MOVVconst {
			v_0 := b.Controls[0]
			c := AuxIntToInt64(v_0.AuxInt)
			if !(c > 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockMIPS64LTZ:
		// match: (LTZ (MOVVconst [c]) yes no)
		// cond: c < 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64MOVVconst {
			v_0 := b.Controls[0]
			c := AuxIntToInt64(v_0.AuxInt)
			if !(c < 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LTZ (MOVVconst [c]) yes no)
		// cond: c >= 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpMIPS64MOVVconst {
			v_0 := b.Controls[0]
			c := AuxIntToInt64(v_0.AuxInt)
			if !(c >= 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockMIPS64NE:
		// match: (NE (FPFlagTrue cmp) yes no)
		// result: (FPT cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64FPFlagTrue {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPS64FPT, cmp)
			return true
		}
		// match: (NE (FPFlagFalse cmp) yes no)
		// result: (FPF cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64FPFlagFalse {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPS64FPF, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGT _ _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64XORconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPS64SGT {
				break
			}
			b.ResetWithControl(block.BlockMIPS64EQ, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGTU _ _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64XORconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPS64SGTU {
				break
			}
			b.ResetWithControl(block.BlockMIPS64EQ, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGTconst _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64XORconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPS64SGTconst {
				break
			}
			b.ResetWithControl(block.BlockMIPS64EQ, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGTUconst _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64XORconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpMIPS64SGTUconst {
				break
			}
			b.ResetWithControl(block.BlockMIPS64EQ, cmp)
			return true
		}
		// match: (NE (SGTUconst [1] x) yes no)
		// result: (EQ x yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64SGTUconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPS64EQ, x)
			return true
		}
		// match: (NE (SGTU x (MOVVconst [0])) yes no)
		// result: (NE x yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64SGTU {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.OpMIPS64MOVVconst || AuxIntToInt64(v_0_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockMIPS64NE, x)
			return true
		}
		// match: (NE (SGTconst [0] x) yes no)
		// result: (LTZ x yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64SGTconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockMIPS64LTZ, x)
			return true
		}
		// match: (NE (SGT x (MOVVconst [0])) yes no)
		// result: (GTZ x yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64SGT {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.OpMIPS64MOVVconst || AuxIntToInt64(v_0_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockMIPS64GTZ, x)
			return true
		}
		// match: (NE (MOVVconst [0]) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpMIPS64MOVVconst {
			v_0 := b.Controls[0]
			if AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (NE (MOVVconst [c]) yes no)
		// cond: c != 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpMIPS64MOVVconst {
			v_0 := b.Controls[0]
			c := AuxIntToInt64(v_0.AuxInt)
			if !(c != 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
	}
	return false
}
