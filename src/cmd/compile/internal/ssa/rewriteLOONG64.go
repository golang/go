// Code generated from _gen/LOONG64.rules using 'go generate'; DO NOT EDIT.

package ssa

import "cmd/compile/internal/types"

func rewriteValueLOONG64(v *Value) bool {
	switch v.Op {
	case OpAbs:
		v.Op = OpLOONG64ABSD
		return true
	case OpAdd16:
		v.Op = OpLOONG64ADDV
		return true
	case OpAdd32:
		v.Op = OpLOONG64ADDV
		return true
	case OpAdd32F:
		v.Op = OpLOONG64ADDF
		return true
	case OpAdd64:
		v.Op = OpLOONG64ADDV
		return true
	case OpAdd64F:
		v.Op = OpLOONG64ADDD
		return true
	case OpAdd8:
		v.Op = OpLOONG64ADDV
		return true
	case OpAddPtr:
		v.Op = OpLOONG64ADDV
		return true
	case OpAddr:
		return rewriteValueLOONG64_OpAddr(v)
	case OpAnd16:
		v.Op = OpLOONG64AND
		return true
	case OpAnd32:
		v.Op = OpLOONG64AND
		return true
	case OpAnd64:
		v.Op = OpLOONG64AND
		return true
	case OpAnd8:
		v.Op = OpLOONG64AND
		return true
	case OpAndB:
		v.Op = OpLOONG64AND
		return true
	case OpAtomicAdd32:
		v.Op = OpLOONG64LoweredAtomicAdd32
		return true
	case OpAtomicAdd64:
		v.Op = OpLOONG64LoweredAtomicAdd64
		return true
	case OpAtomicCompareAndSwap32:
		return rewriteValueLOONG64_OpAtomicCompareAndSwap32(v)
	case OpAtomicCompareAndSwap64:
		v.Op = OpLOONG64LoweredAtomicCas64
		return true
	case OpAtomicExchange32:
		v.Op = OpLOONG64LoweredAtomicExchange32
		return true
	case OpAtomicExchange64:
		v.Op = OpLOONG64LoweredAtomicExchange64
		return true
	case OpAtomicLoad32:
		v.Op = OpLOONG64LoweredAtomicLoad32
		return true
	case OpAtomicLoad64:
		v.Op = OpLOONG64LoweredAtomicLoad64
		return true
	case OpAtomicLoad8:
		v.Op = OpLOONG64LoweredAtomicLoad8
		return true
	case OpAtomicLoadPtr:
		v.Op = OpLOONG64LoweredAtomicLoad64
		return true
	case OpAtomicStore32:
		v.Op = OpLOONG64LoweredAtomicStore32
		return true
	case OpAtomicStore64:
		v.Op = OpLOONG64LoweredAtomicStore64
		return true
	case OpAtomicStore8:
		v.Op = OpLOONG64LoweredAtomicStore8
		return true
	case OpAtomicStorePtrNoWB:
		v.Op = OpLOONG64LoweredAtomicStore64
		return true
	case OpAvg64u:
		return rewriteValueLOONG64_OpAvg64u(v)
	case OpClosureCall:
		v.Op = OpLOONG64CALLclosure
		return true
	case OpCom16:
		return rewriteValueLOONG64_OpCom16(v)
	case OpCom32:
		return rewriteValueLOONG64_OpCom32(v)
	case OpCom64:
		return rewriteValueLOONG64_OpCom64(v)
	case OpCom8:
		return rewriteValueLOONG64_OpCom8(v)
	case OpCondSelect:
		return rewriteValueLOONG64_OpCondSelect(v)
	case OpConst16:
		return rewriteValueLOONG64_OpConst16(v)
	case OpConst32:
		return rewriteValueLOONG64_OpConst32(v)
	case OpConst32F:
		return rewriteValueLOONG64_OpConst32F(v)
	case OpConst64:
		return rewriteValueLOONG64_OpConst64(v)
	case OpConst64F:
		return rewriteValueLOONG64_OpConst64F(v)
	case OpConst8:
		return rewriteValueLOONG64_OpConst8(v)
	case OpConstBool:
		return rewriteValueLOONG64_OpConstBool(v)
	case OpConstNil:
		return rewriteValueLOONG64_OpConstNil(v)
	case OpCopysign:
		v.Op = OpLOONG64FCOPYSGD
		return true
	case OpCvt32Fto32:
		v.Op = OpLOONG64TRUNCFW
		return true
	case OpCvt32Fto64:
		v.Op = OpLOONG64TRUNCFV
		return true
	case OpCvt32Fto64F:
		v.Op = OpLOONG64MOVFD
		return true
	case OpCvt32to32F:
		v.Op = OpLOONG64MOVWF
		return true
	case OpCvt32to64F:
		v.Op = OpLOONG64MOVWD
		return true
	case OpCvt64Fto32:
		v.Op = OpLOONG64TRUNCDW
		return true
	case OpCvt64Fto32F:
		v.Op = OpLOONG64MOVDF
		return true
	case OpCvt64Fto64:
		v.Op = OpLOONG64TRUNCDV
		return true
	case OpCvt64to32F:
		v.Op = OpLOONG64MOVVF
		return true
	case OpCvt64to64F:
		v.Op = OpLOONG64MOVVD
		return true
	case OpCvtBoolToUint8:
		v.Op = OpCopy
		return true
	case OpDiv16:
		return rewriteValueLOONG64_OpDiv16(v)
	case OpDiv16u:
		return rewriteValueLOONG64_OpDiv16u(v)
	case OpDiv32:
		return rewriteValueLOONG64_OpDiv32(v)
	case OpDiv32F:
		v.Op = OpLOONG64DIVF
		return true
	case OpDiv32u:
		return rewriteValueLOONG64_OpDiv32u(v)
	case OpDiv64:
		return rewriteValueLOONG64_OpDiv64(v)
	case OpDiv64F:
		v.Op = OpLOONG64DIVD
		return true
	case OpDiv64u:
		v.Op = OpLOONG64DIVVU
		return true
	case OpDiv8:
		return rewriteValueLOONG64_OpDiv8(v)
	case OpDiv8u:
		return rewriteValueLOONG64_OpDiv8u(v)
	case OpEq16:
		return rewriteValueLOONG64_OpEq16(v)
	case OpEq32:
		return rewriteValueLOONG64_OpEq32(v)
	case OpEq32F:
		return rewriteValueLOONG64_OpEq32F(v)
	case OpEq64:
		return rewriteValueLOONG64_OpEq64(v)
	case OpEq64F:
		return rewriteValueLOONG64_OpEq64F(v)
	case OpEq8:
		return rewriteValueLOONG64_OpEq8(v)
	case OpEqB:
		return rewriteValueLOONG64_OpEqB(v)
	case OpEqPtr:
		return rewriteValueLOONG64_OpEqPtr(v)
	case OpGetCallerPC:
		v.Op = OpLOONG64LoweredGetCallerPC
		return true
	case OpGetCallerSP:
		v.Op = OpLOONG64LoweredGetCallerSP
		return true
	case OpGetClosurePtr:
		v.Op = OpLOONG64LoweredGetClosurePtr
		return true
	case OpHmul32:
		return rewriteValueLOONG64_OpHmul32(v)
	case OpHmul32u:
		return rewriteValueLOONG64_OpHmul32u(v)
	case OpHmul64:
		v.Op = OpLOONG64MULHV
		return true
	case OpHmul64u:
		v.Op = OpLOONG64MULHVU
		return true
	case OpInterCall:
		v.Op = OpLOONG64CALLinter
		return true
	case OpIsInBounds:
		return rewriteValueLOONG64_OpIsInBounds(v)
	case OpIsNonNil:
		return rewriteValueLOONG64_OpIsNonNil(v)
	case OpIsSliceInBounds:
		return rewriteValueLOONG64_OpIsSliceInBounds(v)
	case OpLOONG64ADDV:
		return rewriteValueLOONG64_OpLOONG64ADDV(v)
	case OpLOONG64ADDVconst:
		return rewriteValueLOONG64_OpLOONG64ADDVconst(v)
	case OpLOONG64AND:
		return rewriteValueLOONG64_OpLOONG64AND(v)
	case OpLOONG64ANDconst:
		return rewriteValueLOONG64_OpLOONG64ANDconst(v)
	case OpLOONG64DIVV:
		return rewriteValueLOONG64_OpLOONG64DIVV(v)
	case OpLOONG64DIVVU:
		return rewriteValueLOONG64_OpLOONG64DIVVU(v)
	case OpLOONG64LoweredAtomicAdd32:
		return rewriteValueLOONG64_OpLOONG64LoweredAtomicAdd32(v)
	case OpLOONG64LoweredAtomicAdd64:
		return rewriteValueLOONG64_OpLOONG64LoweredAtomicAdd64(v)
	case OpLOONG64LoweredAtomicStore32:
		return rewriteValueLOONG64_OpLOONG64LoweredAtomicStore32(v)
	case OpLOONG64LoweredAtomicStore64:
		return rewriteValueLOONG64_OpLOONG64LoweredAtomicStore64(v)
	case OpLOONG64MASKEQZ:
		return rewriteValueLOONG64_OpLOONG64MASKEQZ(v)
	case OpLOONG64MASKNEZ:
		return rewriteValueLOONG64_OpLOONG64MASKNEZ(v)
	case OpLOONG64MOVBUload:
		return rewriteValueLOONG64_OpLOONG64MOVBUload(v)
	case OpLOONG64MOVBUloadidx:
		return rewriteValueLOONG64_OpLOONG64MOVBUloadidx(v)
	case OpLOONG64MOVBUreg:
		return rewriteValueLOONG64_OpLOONG64MOVBUreg(v)
	case OpLOONG64MOVBload:
		return rewriteValueLOONG64_OpLOONG64MOVBload(v)
	case OpLOONG64MOVBloadidx:
		return rewriteValueLOONG64_OpLOONG64MOVBloadidx(v)
	case OpLOONG64MOVBreg:
		return rewriteValueLOONG64_OpLOONG64MOVBreg(v)
	case OpLOONG64MOVBstore:
		return rewriteValueLOONG64_OpLOONG64MOVBstore(v)
	case OpLOONG64MOVBstoreidx:
		return rewriteValueLOONG64_OpLOONG64MOVBstoreidx(v)
	case OpLOONG64MOVBstorezero:
		return rewriteValueLOONG64_OpLOONG64MOVBstorezero(v)
	case OpLOONG64MOVBstorezeroidx:
		return rewriteValueLOONG64_OpLOONG64MOVBstorezeroidx(v)
	case OpLOONG64MOVDload:
		return rewriteValueLOONG64_OpLOONG64MOVDload(v)
	case OpLOONG64MOVDloadidx:
		return rewriteValueLOONG64_OpLOONG64MOVDloadidx(v)
	case OpLOONG64MOVDstore:
		return rewriteValueLOONG64_OpLOONG64MOVDstore(v)
	case OpLOONG64MOVDstoreidx:
		return rewriteValueLOONG64_OpLOONG64MOVDstoreidx(v)
	case OpLOONG64MOVFload:
		return rewriteValueLOONG64_OpLOONG64MOVFload(v)
	case OpLOONG64MOVFloadidx:
		return rewriteValueLOONG64_OpLOONG64MOVFloadidx(v)
	case OpLOONG64MOVFstore:
		return rewriteValueLOONG64_OpLOONG64MOVFstore(v)
	case OpLOONG64MOVFstoreidx:
		return rewriteValueLOONG64_OpLOONG64MOVFstoreidx(v)
	case OpLOONG64MOVHUload:
		return rewriteValueLOONG64_OpLOONG64MOVHUload(v)
	case OpLOONG64MOVHUloadidx:
		return rewriteValueLOONG64_OpLOONG64MOVHUloadidx(v)
	case OpLOONG64MOVHUreg:
		return rewriteValueLOONG64_OpLOONG64MOVHUreg(v)
	case OpLOONG64MOVHload:
		return rewriteValueLOONG64_OpLOONG64MOVHload(v)
	case OpLOONG64MOVHloadidx:
		return rewriteValueLOONG64_OpLOONG64MOVHloadidx(v)
	case OpLOONG64MOVHreg:
		return rewriteValueLOONG64_OpLOONG64MOVHreg(v)
	case OpLOONG64MOVHstore:
		return rewriteValueLOONG64_OpLOONG64MOVHstore(v)
	case OpLOONG64MOVHstoreidx:
		return rewriteValueLOONG64_OpLOONG64MOVHstoreidx(v)
	case OpLOONG64MOVHstorezero:
		return rewriteValueLOONG64_OpLOONG64MOVHstorezero(v)
	case OpLOONG64MOVHstorezeroidx:
		return rewriteValueLOONG64_OpLOONG64MOVHstorezeroidx(v)
	case OpLOONG64MOVVload:
		return rewriteValueLOONG64_OpLOONG64MOVVload(v)
	case OpLOONG64MOVVloadidx:
		return rewriteValueLOONG64_OpLOONG64MOVVloadidx(v)
	case OpLOONG64MOVVnop:
		return rewriteValueLOONG64_OpLOONG64MOVVnop(v)
	case OpLOONG64MOVVreg:
		return rewriteValueLOONG64_OpLOONG64MOVVreg(v)
	case OpLOONG64MOVVstore:
		return rewriteValueLOONG64_OpLOONG64MOVVstore(v)
	case OpLOONG64MOVVstoreidx:
		return rewriteValueLOONG64_OpLOONG64MOVVstoreidx(v)
	case OpLOONG64MOVVstorezero:
		return rewriteValueLOONG64_OpLOONG64MOVVstorezero(v)
	case OpLOONG64MOVVstorezeroidx:
		return rewriteValueLOONG64_OpLOONG64MOVVstorezeroidx(v)
	case OpLOONG64MOVWUload:
		return rewriteValueLOONG64_OpLOONG64MOVWUload(v)
	case OpLOONG64MOVWUloadidx:
		return rewriteValueLOONG64_OpLOONG64MOVWUloadidx(v)
	case OpLOONG64MOVWUreg:
		return rewriteValueLOONG64_OpLOONG64MOVWUreg(v)
	case OpLOONG64MOVWload:
		return rewriteValueLOONG64_OpLOONG64MOVWload(v)
	case OpLOONG64MOVWloadidx:
		return rewriteValueLOONG64_OpLOONG64MOVWloadidx(v)
	case OpLOONG64MOVWreg:
		return rewriteValueLOONG64_OpLOONG64MOVWreg(v)
	case OpLOONG64MOVWstore:
		return rewriteValueLOONG64_OpLOONG64MOVWstore(v)
	case OpLOONG64MOVWstoreidx:
		return rewriteValueLOONG64_OpLOONG64MOVWstoreidx(v)
	case OpLOONG64MOVWstorezero:
		return rewriteValueLOONG64_OpLOONG64MOVWstorezero(v)
	case OpLOONG64MOVWstorezeroidx:
		return rewriteValueLOONG64_OpLOONG64MOVWstorezeroidx(v)
	case OpLOONG64MULV:
		return rewriteValueLOONG64_OpLOONG64MULV(v)
	case OpLOONG64NEGV:
		return rewriteValueLOONG64_OpLOONG64NEGV(v)
	case OpLOONG64NOR:
		return rewriteValueLOONG64_OpLOONG64NOR(v)
	case OpLOONG64NORconst:
		return rewriteValueLOONG64_OpLOONG64NORconst(v)
	case OpLOONG64OR:
		return rewriteValueLOONG64_OpLOONG64OR(v)
	case OpLOONG64ORconst:
		return rewriteValueLOONG64_OpLOONG64ORconst(v)
	case OpLOONG64REMV:
		return rewriteValueLOONG64_OpLOONG64REMV(v)
	case OpLOONG64REMVU:
		return rewriteValueLOONG64_OpLOONG64REMVU(v)
	case OpLOONG64ROTR:
		return rewriteValueLOONG64_OpLOONG64ROTR(v)
	case OpLOONG64ROTRV:
		return rewriteValueLOONG64_OpLOONG64ROTRV(v)
	case OpLOONG64SGT:
		return rewriteValueLOONG64_OpLOONG64SGT(v)
	case OpLOONG64SGTU:
		return rewriteValueLOONG64_OpLOONG64SGTU(v)
	case OpLOONG64SGTUconst:
		return rewriteValueLOONG64_OpLOONG64SGTUconst(v)
	case OpLOONG64SGTconst:
		return rewriteValueLOONG64_OpLOONG64SGTconst(v)
	case OpLOONG64SLLV:
		return rewriteValueLOONG64_OpLOONG64SLLV(v)
	case OpLOONG64SLLVconst:
		return rewriteValueLOONG64_OpLOONG64SLLVconst(v)
	case OpLOONG64SRAV:
		return rewriteValueLOONG64_OpLOONG64SRAV(v)
	case OpLOONG64SRAVconst:
		return rewriteValueLOONG64_OpLOONG64SRAVconst(v)
	case OpLOONG64SRLV:
		return rewriteValueLOONG64_OpLOONG64SRLV(v)
	case OpLOONG64SRLVconst:
		return rewriteValueLOONG64_OpLOONG64SRLVconst(v)
	case OpLOONG64SUBV:
		return rewriteValueLOONG64_OpLOONG64SUBV(v)
	case OpLOONG64SUBVconst:
		return rewriteValueLOONG64_OpLOONG64SUBVconst(v)
	case OpLOONG64XOR:
		return rewriteValueLOONG64_OpLOONG64XOR(v)
	case OpLOONG64XORconst:
		return rewriteValueLOONG64_OpLOONG64XORconst(v)
	case OpLeq16:
		return rewriteValueLOONG64_OpLeq16(v)
	case OpLeq16U:
		return rewriteValueLOONG64_OpLeq16U(v)
	case OpLeq32:
		return rewriteValueLOONG64_OpLeq32(v)
	case OpLeq32F:
		return rewriteValueLOONG64_OpLeq32F(v)
	case OpLeq32U:
		return rewriteValueLOONG64_OpLeq32U(v)
	case OpLeq64:
		return rewriteValueLOONG64_OpLeq64(v)
	case OpLeq64F:
		return rewriteValueLOONG64_OpLeq64F(v)
	case OpLeq64U:
		return rewriteValueLOONG64_OpLeq64U(v)
	case OpLeq8:
		return rewriteValueLOONG64_OpLeq8(v)
	case OpLeq8U:
		return rewriteValueLOONG64_OpLeq8U(v)
	case OpLess16:
		return rewriteValueLOONG64_OpLess16(v)
	case OpLess16U:
		return rewriteValueLOONG64_OpLess16U(v)
	case OpLess32:
		return rewriteValueLOONG64_OpLess32(v)
	case OpLess32F:
		return rewriteValueLOONG64_OpLess32F(v)
	case OpLess32U:
		return rewriteValueLOONG64_OpLess32U(v)
	case OpLess64:
		return rewriteValueLOONG64_OpLess64(v)
	case OpLess64F:
		return rewriteValueLOONG64_OpLess64F(v)
	case OpLess64U:
		return rewriteValueLOONG64_OpLess64U(v)
	case OpLess8:
		return rewriteValueLOONG64_OpLess8(v)
	case OpLess8U:
		return rewriteValueLOONG64_OpLess8U(v)
	case OpLoad:
		return rewriteValueLOONG64_OpLoad(v)
	case OpLocalAddr:
		return rewriteValueLOONG64_OpLocalAddr(v)
	case OpLsh16x16:
		return rewriteValueLOONG64_OpLsh16x16(v)
	case OpLsh16x32:
		return rewriteValueLOONG64_OpLsh16x32(v)
	case OpLsh16x64:
		return rewriteValueLOONG64_OpLsh16x64(v)
	case OpLsh16x8:
		return rewriteValueLOONG64_OpLsh16x8(v)
	case OpLsh32x16:
		return rewriteValueLOONG64_OpLsh32x16(v)
	case OpLsh32x32:
		return rewriteValueLOONG64_OpLsh32x32(v)
	case OpLsh32x64:
		return rewriteValueLOONG64_OpLsh32x64(v)
	case OpLsh32x8:
		return rewriteValueLOONG64_OpLsh32x8(v)
	case OpLsh64x16:
		return rewriteValueLOONG64_OpLsh64x16(v)
	case OpLsh64x32:
		return rewriteValueLOONG64_OpLsh64x32(v)
	case OpLsh64x64:
		return rewriteValueLOONG64_OpLsh64x64(v)
	case OpLsh64x8:
		return rewriteValueLOONG64_OpLsh64x8(v)
	case OpLsh8x16:
		return rewriteValueLOONG64_OpLsh8x16(v)
	case OpLsh8x32:
		return rewriteValueLOONG64_OpLsh8x32(v)
	case OpLsh8x64:
		return rewriteValueLOONG64_OpLsh8x64(v)
	case OpLsh8x8:
		return rewriteValueLOONG64_OpLsh8x8(v)
	case OpMax32F:
		v.Op = OpLOONG64FMAXF
		return true
	case OpMax64F:
		v.Op = OpLOONG64FMAXD
		return true
	case OpMin32F:
		v.Op = OpLOONG64FMINF
		return true
	case OpMin64F:
		v.Op = OpLOONG64FMIND
		return true
	case OpMod16:
		return rewriteValueLOONG64_OpMod16(v)
	case OpMod16u:
		return rewriteValueLOONG64_OpMod16u(v)
	case OpMod32:
		return rewriteValueLOONG64_OpMod32(v)
	case OpMod32u:
		return rewriteValueLOONG64_OpMod32u(v)
	case OpMod64:
		return rewriteValueLOONG64_OpMod64(v)
	case OpMod64u:
		v.Op = OpLOONG64REMVU
		return true
	case OpMod8:
		return rewriteValueLOONG64_OpMod8(v)
	case OpMod8u:
		return rewriteValueLOONG64_OpMod8u(v)
	case OpMove:
		return rewriteValueLOONG64_OpMove(v)
	case OpMul16:
		v.Op = OpLOONG64MULV
		return true
	case OpMul32:
		v.Op = OpLOONG64MULV
		return true
	case OpMul32F:
		v.Op = OpLOONG64MULF
		return true
	case OpMul64:
		v.Op = OpLOONG64MULV
		return true
	case OpMul64F:
		v.Op = OpLOONG64MULD
		return true
	case OpMul8:
		v.Op = OpLOONG64MULV
		return true
	case OpNeg16:
		v.Op = OpLOONG64NEGV
		return true
	case OpNeg32:
		v.Op = OpLOONG64NEGV
		return true
	case OpNeg32F:
		v.Op = OpLOONG64NEGF
		return true
	case OpNeg64:
		v.Op = OpLOONG64NEGV
		return true
	case OpNeg64F:
		v.Op = OpLOONG64NEGD
		return true
	case OpNeg8:
		v.Op = OpLOONG64NEGV
		return true
	case OpNeq16:
		return rewriteValueLOONG64_OpNeq16(v)
	case OpNeq32:
		return rewriteValueLOONG64_OpNeq32(v)
	case OpNeq32F:
		return rewriteValueLOONG64_OpNeq32F(v)
	case OpNeq64:
		return rewriteValueLOONG64_OpNeq64(v)
	case OpNeq64F:
		return rewriteValueLOONG64_OpNeq64F(v)
	case OpNeq8:
		return rewriteValueLOONG64_OpNeq8(v)
	case OpNeqB:
		v.Op = OpLOONG64XOR
		return true
	case OpNeqPtr:
		return rewriteValueLOONG64_OpNeqPtr(v)
	case OpNilCheck:
		v.Op = OpLOONG64LoweredNilCheck
		return true
	case OpNot:
		return rewriteValueLOONG64_OpNot(v)
	case OpOffPtr:
		return rewriteValueLOONG64_OpOffPtr(v)
	case OpOr16:
		v.Op = OpLOONG64OR
		return true
	case OpOr32:
		v.Op = OpLOONG64OR
		return true
	case OpOr64:
		v.Op = OpLOONG64OR
		return true
	case OpOr8:
		v.Op = OpLOONG64OR
		return true
	case OpOrB:
		v.Op = OpLOONG64OR
		return true
	case OpPanicBounds:
		return rewriteValueLOONG64_OpPanicBounds(v)
	case OpRotateLeft16:
		return rewriteValueLOONG64_OpRotateLeft16(v)
	case OpRotateLeft32:
		return rewriteValueLOONG64_OpRotateLeft32(v)
	case OpRotateLeft64:
		return rewriteValueLOONG64_OpRotateLeft64(v)
	case OpRotateLeft8:
		return rewriteValueLOONG64_OpRotateLeft8(v)
	case OpRound32F:
		v.Op = OpCopy
		return true
	case OpRound64F:
		v.Op = OpCopy
		return true
	case OpRsh16Ux16:
		return rewriteValueLOONG64_OpRsh16Ux16(v)
	case OpRsh16Ux32:
		return rewriteValueLOONG64_OpRsh16Ux32(v)
	case OpRsh16Ux64:
		return rewriteValueLOONG64_OpRsh16Ux64(v)
	case OpRsh16Ux8:
		return rewriteValueLOONG64_OpRsh16Ux8(v)
	case OpRsh16x16:
		return rewriteValueLOONG64_OpRsh16x16(v)
	case OpRsh16x32:
		return rewriteValueLOONG64_OpRsh16x32(v)
	case OpRsh16x64:
		return rewriteValueLOONG64_OpRsh16x64(v)
	case OpRsh16x8:
		return rewriteValueLOONG64_OpRsh16x8(v)
	case OpRsh32Ux16:
		return rewriteValueLOONG64_OpRsh32Ux16(v)
	case OpRsh32Ux32:
		return rewriteValueLOONG64_OpRsh32Ux32(v)
	case OpRsh32Ux64:
		return rewriteValueLOONG64_OpRsh32Ux64(v)
	case OpRsh32Ux8:
		return rewriteValueLOONG64_OpRsh32Ux8(v)
	case OpRsh32x16:
		return rewriteValueLOONG64_OpRsh32x16(v)
	case OpRsh32x32:
		return rewriteValueLOONG64_OpRsh32x32(v)
	case OpRsh32x64:
		return rewriteValueLOONG64_OpRsh32x64(v)
	case OpRsh32x8:
		return rewriteValueLOONG64_OpRsh32x8(v)
	case OpRsh64Ux16:
		return rewriteValueLOONG64_OpRsh64Ux16(v)
	case OpRsh64Ux32:
		return rewriteValueLOONG64_OpRsh64Ux32(v)
	case OpRsh64Ux64:
		return rewriteValueLOONG64_OpRsh64Ux64(v)
	case OpRsh64Ux8:
		return rewriteValueLOONG64_OpRsh64Ux8(v)
	case OpRsh64x16:
		return rewriteValueLOONG64_OpRsh64x16(v)
	case OpRsh64x32:
		return rewriteValueLOONG64_OpRsh64x32(v)
	case OpRsh64x64:
		return rewriteValueLOONG64_OpRsh64x64(v)
	case OpRsh64x8:
		return rewriteValueLOONG64_OpRsh64x8(v)
	case OpRsh8Ux16:
		return rewriteValueLOONG64_OpRsh8Ux16(v)
	case OpRsh8Ux32:
		return rewriteValueLOONG64_OpRsh8Ux32(v)
	case OpRsh8Ux64:
		return rewriteValueLOONG64_OpRsh8Ux64(v)
	case OpRsh8Ux8:
		return rewriteValueLOONG64_OpRsh8Ux8(v)
	case OpRsh8x16:
		return rewriteValueLOONG64_OpRsh8x16(v)
	case OpRsh8x32:
		return rewriteValueLOONG64_OpRsh8x32(v)
	case OpRsh8x64:
		return rewriteValueLOONG64_OpRsh8x64(v)
	case OpRsh8x8:
		return rewriteValueLOONG64_OpRsh8x8(v)
	case OpSelect0:
		return rewriteValueLOONG64_OpSelect0(v)
	case OpSelect1:
		return rewriteValueLOONG64_OpSelect1(v)
	case OpSelectN:
		return rewriteValueLOONG64_OpSelectN(v)
	case OpSignExt16to32:
		v.Op = OpLOONG64MOVHreg
		return true
	case OpSignExt16to64:
		v.Op = OpLOONG64MOVHreg
		return true
	case OpSignExt32to64:
		v.Op = OpLOONG64MOVWreg
		return true
	case OpSignExt8to16:
		v.Op = OpLOONG64MOVBreg
		return true
	case OpSignExt8to32:
		v.Op = OpLOONG64MOVBreg
		return true
	case OpSignExt8to64:
		v.Op = OpLOONG64MOVBreg
		return true
	case OpSlicemask:
		return rewriteValueLOONG64_OpSlicemask(v)
	case OpSqrt:
		v.Op = OpLOONG64SQRTD
		return true
	case OpSqrt32:
		v.Op = OpLOONG64SQRTF
		return true
	case OpStaticCall:
		v.Op = OpLOONG64CALLstatic
		return true
	case OpStore:
		return rewriteValueLOONG64_OpStore(v)
	case OpSub16:
		v.Op = OpLOONG64SUBV
		return true
	case OpSub32:
		v.Op = OpLOONG64SUBV
		return true
	case OpSub32F:
		v.Op = OpLOONG64SUBF
		return true
	case OpSub64:
		v.Op = OpLOONG64SUBV
		return true
	case OpSub64F:
		v.Op = OpLOONG64SUBD
		return true
	case OpSub8:
		v.Op = OpLOONG64SUBV
		return true
	case OpSubPtr:
		v.Op = OpLOONG64SUBV
		return true
	case OpTailCall:
		v.Op = OpLOONG64CALLtail
		return true
	case OpTrunc16to8:
		v.Op = OpCopy
		return true
	case OpTrunc32to16:
		v.Op = OpCopy
		return true
	case OpTrunc32to8:
		v.Op = OpCopy
		return true
	case OpTrunc64to16:
		v.Op = OpCopy
		return true
	case OpTrunc64to32:
		v.Op = OpCopy
		return true
	case OpTrunc64to8:
		v.Op = OpCopy
		return true
	case OpWB:
		v.Op = OpLOONG64LoweredWB
		return true
	case OpXor16:
		v.Op = OpLOONG64XOR
		return true
	case OpXor32:
		v.Op = OpLOONG64XOR
		return true
	case OpXor64:
		v.Op = OpLOONG64XOR
		return true
	case OpXor8:
		v.Op = OpLOONG64XOR
		return true
	case OpZero:
		return rewriteValueLOONG64_OpZero(v)
	case OpZeroExt16to32:
		v.Op = OpLOONG64MOVHUreg
		return true
	case OpZeroExt16to64:
		v.Op = OpLOONG64MOVHUreg
		return true
	case OpZeroExt32to64:
		v.Op = OpLOONG64MOVWUreg
		return true
	case OpZeroExt8to16:
		v.Op = OpLOONG64MOVBUreg
		return true
	case OpZeroExt8to32:
		v.Op = OpLOONG64MOVBUreg
		return true
	case OpZeroExt8to64:
		v.Op = OpLOONG64MOVBUreg
		return true
	}
	return false
}
func rewriteValueLOONG64_OpAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (MOVVaddr {sym} base)
	for {
		sym := auxToSym(v.Aux)
		base := v_0
		v.reset(OpLOONG64MOVVaddr)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValueLOONG64_OpAtomicCompareAndSwap32(v *Value) bool {
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
		v.reset(OpLOONG64LoweredAtomicCas32)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(old)
		v.AddArg4(ptr, v0, new, mem)
		return true
	}
}
func rewriteValueLOONG64_OpAvg64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Avg64u <t> x y)
	// result: (ADDV (SRLVconst <t> (SUBV <t> x y) [1]) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64ADDV)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLVconst, t)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64SUBV, t)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValueLOONG64_OpCom16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com16 x)
	// result: (NOR (MOVVconst [0]) x)
	for {
		x := v_0
		v.reset(OpLOONG64NOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueLOONG64_OpCom32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com32 x)
	// result: (NOR (MOVVconst [0]) x)
	for {
		x := v_0
		v.reset(OpLOONG64NOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueLOONG64_OpCom64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com64 x)
	// result: (NOR (MOVVconst [0]) x)
	for {
		x := v_0
		v.reset(OpLOONG64NOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueLOONG64_OpCom8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com8 x)
	// result: (NOR (MOVVconst [0]) x)
	for {
		x := v_0
		v.reset(OpLOONG64NOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueLOONG64_OpCondSelect(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CondSelect <t> x y cond)
	// result: (OR (MASKEQZ <t> x cond) (MASKNEZ <t> y cond))
	for {
		t := v.Type
		x := v_0
		y := v_1
		cond := v_2
		v.reset(OpLOONG64OR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MASKEQZ, t)
		v0.AddArg2(x, cond)
		v1 := b.NewValue0(v.Pos, OpLOONG64MASKNEZ, t)
		v1.AddArg2(y, cond)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpConst16(v *Value) bool {
	// match: (Const16 [val])
	// result: (MOVVconst [int64(val)])
	for {
		val := auxIntToInt16(v.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueLOONG64_OpConst32(v *Value) bool {
	// match: (Const32 [val])
	// result: (MOVVconst [int64(val)])
	for {
		val := auxIntToInt32(v.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueLOONG64_OpConst32F(v *Value) bool {
	// match: (Const32F [val])
	// result: (MOVFconst [float64(val)])
	for {
		val := auxIntToFloat32(v.AuxInt)
		v.reset(OpLOONG64MOVFconst)
		v.AuxInt = float64ToAuxInt(float64(val))
		return true
	}
}
func rewriteValueLOONG64_OpConst64(v *Value) bool {
	// match: (Const64 [val])
	// result: (MOVVconst [int64(val)])
	for {
		val := auxIntToInt64(v.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueLOONG64_OpConst64F(v *Value) bool {
	// match: (Const64F [val])
	// result: (MOVDconst [float64(val)])
	for {
		val := auxIntToFloat64(v.AuxInt)
		v.reset(OpLOONG64MOVDconst)
		v.AuxInt = float64ToAuxInt(float64(val))
		return true
	}
}
func rewriteValueLOONG64_OpConst8(v *Value) bool {
	// match: (Const8 [val])
	// result: (MOVVconst [int64(val)])
	for {
		val := auxIntToInt8(v.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueLOONG64_OpConstBool(v *Value) bool {
	// match: (ConstBool [t])
	// result: (MOVVconst [int64(b2i(t))])
	for {
		t := auxIntToBool(v.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(b2i(t)))
		return true
	}
}
func rewriteValueLOONG64_OpConstNil(v *Value) bool {
	// match: (ConstNil)
	// result: (MOVVconst [0])
	for {
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
}
func rewriteValueLOONG64_OpDiv16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 x y)
	// result: (DIVV (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64DIVV)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpDiv16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (DIVVU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64DIVVU)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpDiv32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32 x y)
	// result: (DIVV (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64DIVV)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpDiv32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32u x y)
	// result: (DIVVU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64DIVVU)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpDiv64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64 x y)
	// result: (DIVV x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64DIVV)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueLOONG64_OpDiv8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (DIVV (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64DIVV)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpDiv8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (DIVVU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64DIVVU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpEq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// result: (SGTU (MOVVconst [1]) (XOR (ZeroExt16to64 x) (ZeroExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64XOR, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpEq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq32 x y)
	// result: (SGTU (MOVVconst [1]) (XOR (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64XOR, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpEq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32F x y)
	// result: (FPFlagTrue (CMPEQF x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, OpLOONG64CMPEQF, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueLOONG64_OpEq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq64 x y)
	// result: (SGTU (MOVVconst [1]) (XOR x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64XOR, typ.UInt64)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpEq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64F x y)
	// result: (FPFlagTrue (CMPEQD x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, OpLOONG64CMPEQD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueLOONG64_OpEq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// result: (SGTU (MOVVconst [1]) (XOR (ZeroExt8to64 x) (ZeroExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64XOR, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpEqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// result: (XOR (MOVVconst [1]) (XOR <typ.Bool> x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64XOR, typ.Bool)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpEqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqPtr x y)
	// result: (SGTU (MOVVconst [1]) (XOR x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64XOR, typ.UInt64)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpHmul32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32 x y)
	// result: (SRAVconst (MULV (SignExt32to64 x) (SignExt32to64 y)) [32])
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SRAVconst)
		v.AuxInt = int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, OpLOONG64MULV, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueLOONG64_OpHmul32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32u x y)
	// result: (SRLVconst (MULV (ZeroExt32to64 x) (ZeroExt32to64 y)) [32])
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SRLVconst)
		v.AuxInt = int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, OpLOONG64MULV, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueLOONG64_OpIsInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsInBounds idx len)
	// result: (SGTU len idx)
	for {
		idx := v_0
		len := v_1
		v.reset(OpLOONG64SGTU)
		v.AddArg2(len, idx)
		return true
	}
}
func rewriteValueLOONG64_OpIsNonNil(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsNonNil ptr)
	// result: (SGTU ptr (MOVVconst [0]))
	for {
		ptr := v_0
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(ptr, v0)
		return true
	}
}
func rewriteValueLOONG64_OpIsSliceInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsSliceInBounds idx len)
	// result: (XOR (MOVVconst [1]) (SGTU idx len))
	for {
		idx := v_0
		len := v_1
		v.reset(OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v1.AddArg2(idx, len)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLOONG64ADDV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDV x (MOVVconst <t> [c]))
	// cond: is32Bit(c) && !t.IsPtr()
	// result: (ADDVconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpLOONG64MOVVconst {
				continue
			}
			t := v_1.Type
			c := auxIntToInt64(v_1.AuxInt)
			if !(is32Bit(c) && !t.IsPtr()) {
				continue
			}
			v.reset(OpLOONG64ADDVconst)
			v.AuxInt = int64ToAuxInt(c)
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
			if v_1.Op != OpLOONG64NEGV {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpLOONG64SUBV)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64ADDVconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ADDVconst [off1] (MOVVaddr [off2] {sym} ptr))
	// cond: is32Bit(off1+int64(off2))
	// result: (MOVVaddr [int32(off1)+int32(off2)] {sym} ptr)
	for {
		off1 := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		if !(is32Bit(off1 + int64(off2))) {
			break
		}
		v.reset(OpLOONG64MOVVaddr)
		v.AuxInt = int32ToAuxInt(int32(off1) + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg(ptr)
		return true
	}
	// match: (ADDVconst [0] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (ADDVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [c+d])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(c + d)
		return true
	}
	// match: (ADDVconst [c] (ADDVconst [d] x))
	// cond: is32Bit(c+d)
	// result: (ADDVconst [c+d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpLOONG64ADDVconst)
		v.AuxInt = int64ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (ADDVconst [c] (SUBVconst [d] x))
	// cond: is32Bit(c-d)
	// result: (ADDVconst [c-d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64SUBVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(c - d)) {
			break
		}
		v.reset(OpLOONG64ADDVconst)
		v.AuxInt = int64ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64AND(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AND x (MOVVconst [c]))
	// cond: is32Bit(c)
	// result: (ANDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpLOONG64MOVVconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(is32Bit(c)) {
				continue
			}
			v.reset(OpLOONG64ANDconst)
			v.AuxInt = int64ToAuxInt(c)
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
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64ANDconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ANDconst [0] _)
	// result: (MOVVconst [0])
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (ANDconst [-1] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (ANDconst [c] (MOVVconst [d]))
	// result: (MOVVconst [c&d])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(c & d)
		return true
	}
	// match: (ANDconst [c] (ANDconst [d] x))
	// result: (ANDconst [c&d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64ANDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpLOONG64ANDconst)
		v.AuxInt = int64ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64DIVV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVV (MOVVconst [c]) (MOVVconst [d]))
	// cond: d != 0
	// result: (MOVVconst [c/d])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(c / d)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64DIVVU(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVVU x (MOVVconst [1]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (DIVVU x (MOVVconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (SRLVconst [log64(c)] x)
	for {
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpLOONG64SRLVconst)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg(x)
		return true
	}
	// match: (DIVVU (MOVVconst [c]) (MOVVconst [d]))
	// cond: d != 0
	// result: (MOVVconst [int64(uint64(c)/uint64(d))])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) / uint64(d)))
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64LoweredAtomicAdd32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredAtomicAdd32 ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (LoweredAtomicAddconst32 [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64LoweredAtomicAddconst32)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64LoweredAtomicAdd64(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredAtomicAdd64 ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (LoweredAtomicAddconst64 [c] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64LoweredAtomicAddconst64)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64LoweredAtomicStore32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredAtomicStore32 ptr (MOVVconst [0]) mem)
	// result: (LoweredAtomicStorezero32 ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpLOONG64LoweredAtomicStorezero32)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64LoweredAtomicStore64(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredAtomicStore64 ptr (MOVVconst [0]) mem)
	// result: (LoweredAtomicStorezero64 ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpLOONG64LoweredAtomicStorezero64)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MASKEQZ(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MASKEQZ (MOVVconst [0]) cond)
	// result: (MOVVconst [0])
	for {
		if v_0.Op != OpLOONG64MOVVconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MASKEQZ x (MOVVconst [c]))
	// cond: c == 0
	// result: (MOVVconst [0])
	for {
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(c == 0) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MASKEQZ x (MOVVconst [c]))
	// cond: c != 0
	// result: x
	for {
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(c != 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MASKNEZ(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MASKNEZ (MOVVconst [0]) cond)
	// result: (MOVVconst [0])
	for {
		if v_0.Op != OpLOONG64MOVVconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVBUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBUload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVBUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVBUload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVBUload [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVBUload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBUloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVBUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVBUloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBUloadidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVBUload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVBUload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUloadidx (MOVVconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVBUload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVBUload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVBUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBUreg (SRLVconst [rc] x))
	// cond: rc < 8
	// result: (BSTRPICKV [rc + (7+rc)<<6] x)
	for {
		if v_0.Op != OpLOONG64SRLVconst {
			break
		}
		rc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(rc < 8) {
			break
		}
		v.reset(OpLOONG64BSTRPICKV)
		v.AuxInt = int64ToAuxInt(rc + (7+rc)<<6)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(SGT _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpLOONG64SGT {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(SGTU _ _))
	// result: x
	for {
		x := v_0
		if x.Op != OpLOONG64SGTU {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg x:(XOR (MOVVconst [1]) (SGT _ _)))
	// result: x
	for {
		x := v_0
		if x.Op != OpLOONG64XOR {
			break
		}
		_ = x.Args[1]
		x_0 := x.Args[0]
		x_1 := x.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, x_0, x_1 = _i0+1, x_1, x_0 {
			if x_0.Op != OpLOONG64MOVVconst || auxIntToInt64(x_0.AuxInt) != 1 || x_1.Op != OpLOONG64SGT {
				continue
			}
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (MOVBUreg x:(XOR (MOVVconst [1]) (SGTU _ _)))
	// result: x
	for {
		x := v_0
		if x.Op != OpLOONG64XOR {
			break
		}
		_ = x.Args[1]
		x_0 := x.Args[0]
		x_1 := x.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, x_0, x_1 = _i0+1, x_1, x_0 {
			if x_0.Op != OpLOONG64MOVVconst || auxIntToInt64(x_0.AuxInt) != 1 || x_1.Op != OpLOONG64SGTU {
				continue
			}
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (MOVBUreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBUload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBUreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (SLLVconst [lc] x))
	// cond: lc >= 8
	// result: (MOVVconst [0])
	for {
		if v_0.Op != OpLOONG64SLLVconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		if !(lc >= 8) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MOVBUreg (MOVVconst [c]))
	// result: (MOVVconst [int64(uint8(c))])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(uint8(c)))
		return true
	}
	// match: (MOVBUreg (ANDconst [c] x))
	// result: (ANDconst [c&0xff] x)
	for {
		if v_0.Op != OpLOONG64ANDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpLOONG64ANDconst)
		v.AuxInt = int64ToAuxInt(c & 0xff)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVBload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVBload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVBload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVBload [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVBload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVBloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVBloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBloadidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVBload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVBload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBloadidx (MOVVconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVBload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVBload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVBreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBreg x:(MOVBload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (MOVVconst [c]))
	// result: (MOVVconst [int64(int8(c))])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(int8(c)))
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVBstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVBstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVBstore [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVBreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVBUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVBstore)
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
		if v_1.Op != OpLOONG64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVBstore)
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
		if v_1.Op != OpLOONG64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVVconst [0]) mem)
	// result: (MOVBstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpLOONG64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} (ADDV ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBstoreidx ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVBstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVBstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstoreidx ptr (MOVVconst [c]) val mem)
	// cond: is32Bit(c)
	// result: (MOVBstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstoreidx (MOVVconst [c]) idx val mem)
	// cond: is32Bit(c)
	// result: (MOVBstore [int32(c)] idx val mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVVconst [0]) mem)
	// result: (MOVBstorezeroidx ptr idx mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpLOONG64MOVVconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		mem := v_3
		v.reset(OpLOONG64MOVBstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVBstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstorezero [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVBstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstorezero [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVBstorezero [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstorezero [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBstorezeroidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVBstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVBstorezeroidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstorezeroidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVBstorezero [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstorezeroidx (MOVVconst [c]) idx mem)
	// cond: is32Bit(c)
	// result: (MOVBstorezero [int32(c)] idx mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVDload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVDload [off] {sym} ptr (MOVVstore [off] {sym} ptr val _))
	// result: (MOVVgpfp val)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpLOONG64MOVVgpfp)
		v.AddArg(val)
		return true
	}
	// match: (MOVDload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVDload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVDload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVDload [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVDload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVDloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVDloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDloadidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVDload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVDload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDloadidx (MOVVconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVDload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVDload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVDstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVDstore [off] {sym} ptr (MOVVgpfp val) mem)
	// result: (MOVVstore [off] {sym} ptr val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVgpfp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVDstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVDstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVDstore [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVDstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} (ADDV ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstoreidx ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVDstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVDstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstoreidx ptr (MOVVconst [c]) val mem)
	// cond: is32Bit(c)
	// result: (MOVDstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVDstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstoreidx (MOVVconst [c]) idx val mem)
	// cond: is32Bit(c)
	// result: (MOVDstore [int32(c)] idx val mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVDstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVFload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVFload [off] {sym} ptr (MOVWstore [off] {sym} ptr val _))
	// result: (MOVWgpfp val)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVWstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpLOONG64MOVWgpfp)
		v.AddArg(val)
		return true
	}
	// match: (MOVFload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVFload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVFload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVFload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVFload [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVFload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVFload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVFloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVFloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVFloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVFloadidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVFload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVFload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVFloadidx (MOVVconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVFload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVFload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVFstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVFstore [off] {sym} ptr (MOVWgpfp val) mem)
	// result: (MOVWstore [off] {sym} ptr val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVWgpfp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVFstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVFstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVFstore [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVFstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstore [off] {sym} (ADDV ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVFstoreidx ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVFstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVFstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVFstoreidx ptr (MOVVconst [c]) val mem)
	// cond: is32Bit(c)
	// result: (MOVFstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVFstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstoreidx (MOVVconst [c]) idx val mem)
	// cond: is32Bit(c)
	// result: (MOVFstore [int32(c)] idx val mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVFstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVHUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHUload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVHUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVHUload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVHUload [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVHUload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHUloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVHUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVHUloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHUloadidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVHUload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVHUload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUloadidx (MOVVconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVHUload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVHUload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVHUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHUreg (SRLVconst [rc] x))
	// cond: rc < 16
	// result: (BSTRPICKV [rc + (15+rc)<<6] x)
	for {
		if v_0.Op != OpLOONG64SRLVconst {
			break
		}
		rc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(rc < 16) {
			break
		}
		v.reset(OpLOONG64BSTRPICKV)
		v.AuxInt = int64ToAuxInt(rc + (15+rc)<<6)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBUload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVHUload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBUreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVHUreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (SLLVconst [lc] x))
	// cond: lc >= 16
	// result: (MOVVconst [0])
	for {
		if v_0.Op != OpLOONG64SLLVconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		if !(lc >= 16) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MOVHUreg (MOVVconst [c]))
	// result: (MOVVconst [int64(uint16(c))])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(uint16(c)))
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVHload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVHload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVHload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVHload [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVHload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVHloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVHloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHloadidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVHload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVHload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHloadidx (MOVVconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVHload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVHload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVHreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHreg x:(MOVBload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBUload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVHload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBUreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVHreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (MOVVconst [c]))
	// result: (MOVVconst [int64(int16(c))])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(int16(c)))
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVHstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVHstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVHstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVHstore [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVHstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVHstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHUreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVHstore)
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
		if v_1.Op != OpLOONG64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVHstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVHstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVVconst [0]) mem)
	// result: (MOVHstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpLOONG64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} (ADDV ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstoreidx ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVHstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVHstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstoreidx ptr (MOVVconst [c]) val mem)
	// cond: is32Bit(c)
	// result: (MOVHstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVHstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstoreidx (MOVVconst [c]) idx val mem)
	// cond: is32Bit(c)
	// result: (MOVHstore [int32(c)] idx val mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVHstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVVconst [0]) mem)
	// result: (MOVHstorezeroidx ptr idx mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpLOONG64MOVVconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		mem := v_3
		v.reset(OpLOONG64MOVHstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVHstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHstorezero [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVHstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstorezero [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVHstorezero [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstorezero [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstorezeroidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVHstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVHstorezeroidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstorezeroidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVHstorezero [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstorezeroidx (MOVVconst [c]) idx mem)
	// cond: is32Bit(c)
	// result: (MOVHstorezero [int32(c)] idx mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVVload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVVload [off] {sym} ptr (MOVDstore [off] {sym} ptr val _))
	// result: (MOVVfpgp val)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVDstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpLOONG64MOVVfpgp)
		v.AddArg(val)
		return true
	}
	// match: (MOVVload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVVload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVVload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVVload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVVload [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVVload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVVload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVVloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVVloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVVloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVVloadidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVVload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVVload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVVloadidx (MOVVconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVVload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVVload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVVnop(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVVnop (MOVVconst [c]))
	// result: (MOVVconst [c])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVVreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVVreg x)
	// cond: x.Uses == 1
	// result: (MOVVnop x)
	for {
		x := v_0
		if !(x.Uses == 1) {
			break
		}
		v.reset(OpLOONG64MOVVnop)
		v.AddArg(x)
		return true
	}
	// match: (MOVVreg (MOVVconst [c]))
	// result: (MOVVconst [c])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVVstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVVstore [off] {sym} ptr (MOVVfpgp val) mem)
	// result: (MOVDstore [off] {sym} ptr val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVfpgp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVDstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVVstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVVstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVVstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVVstore [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVVstore [off] {sym} ptr (MOVVconst [0]) mem)
	// result: (MOVVstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpLOONG64MOVVstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVVstore [off] {sym} (ADDV ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVVstoreidx ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVVstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVVstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVVstoreidx ptr (MOVVconst [c]) val mem)
	// cond: is32Bit(c)
	// result: (MOVVstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVVstoreidx (MOVVconst [c]) idx val mem)
	// cond: is32Bit(c)
	// result: (MOVVstore [int32(c)] idx val mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (MOVVstoreidx ptr idx (MOVVconst [0]) mem)
	// result: (MOVVstorezeroidx ptr idx mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpLOONG64MOVVconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		mem := v_3
		v.reset(OpLOONG64MOVVstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVVstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVVstorezero [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVVstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVVstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVVstorezero [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVVstorezero [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVVstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVVstorezero [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVVstorezeroidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVVstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVVstorezeroidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVVstorezeroidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVVstorezero [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVVstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVVstorezeroidx (MOVVconst [c]) idx mem)
	// cond: is32Bit(c)
	// result: (MOVVstorezero [int32(c)] idx mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVVstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVWUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVWUload [off] {sym} ptr (MOVFstore [off] {sym} ptr val _))
	// result: (ZeroExt32to64 (MOVWfpgp <typ.Float32> val))
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVFstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpZeroExt32to64)
		v0 := b.NewValue0(v_1.Pos, OpLOONG64MOVWfpgp, typ.Float32)
		v0.AddArg(val)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWUload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVWUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVWUload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVWUload [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVWUload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWUloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVWUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVWUloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWUloadidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVWUload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVWUload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUloadidx (MOVVconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVWUload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVWUload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVWUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWUreg (SRLVconst [rc] x))
	// cond: rc < 32
	// result: (BSTRPICKV [rc + (31+rc)<<6] x)
	for {
		if v_0.Op != OpLOONG64SRLVconst {
			break
		}
		rc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(rc < 32) {
			break
		}
		v.reset(OpLOONG64BSTRPICKV)
		v.AuxInt = int64ToAuxInt(rc + (31+rc)<<6)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBUload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVHUload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVWUload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBUreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVHUreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVWUreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (SLLVconst [lc] x))
	// cond: lc >= 32
	// result: (MOVVconst [0])
	for {
		if v_0.Op != OpLOONG64SLLVconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		if !(lc >= 32) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MOVWUreg (MOVVconst [c]))
	// result: (MOVVconst [int64(uint32(c))])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(uint32(c)))
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVWload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVWload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVWload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVWload [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVWload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVWloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVWloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWloadidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVWload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVWload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWloadidx (MOVVconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVWload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVWload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVWreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWreg x:(MOVBload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBUload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVHload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVHUload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVWload {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVBUreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVHreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != OpLOONG64MOVWreg {
			break
		}
		v.reset(OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (MOVVconst [c]))
	// result: (MOVVconst [int64(int32(c))])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(int32(c)))
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVWstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstore [off] {sym} ptr (MOVWfpgp val) mem)
	// result: (MOVFstore [off] {sym} ptr val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVWfpgp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVFstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVWstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVWstore [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVVconst [0]) mem)
	// result: (MOVWstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpLOONG64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} (ADDV ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstoreidx ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVWstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVWstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreidx ptr (MOVVconst [c]) val mem)
	// cond: is32Bit(c)
	// result: (MOVWstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstoreidx (MOVVconst [c]) idx val mem)
	// cond: is32Bit(c)
	// result: (MOVWstore [int32(c)] idx val mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVVconst [0]) mem)
	// result: (MOVWstorezeroidx ptr idx mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpLOONG64MOVVconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		mem := v_3
		v.reset(OpLOONG64MOVWstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVWstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstorezero [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVWstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstorezero [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (MOVWstorezero [off1+int32(off2)] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64MOVVaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpLOONG64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstorezero [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstorezeroidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpLOONG64MOVWstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MOVWstorezeroidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstorezeroidx ptr (MOVVconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVWstorezero [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstorezeroidx (MOVVconst [c]) idx mem)
	// cond: is32Bit(c)
	// result: (MOVWstorezero [int32(c)] idx mem)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(idx, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64MULV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULV x (MOVVconst [-1]))
	// result: (NEGV x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_1.AuxInt) != -1 {
				continue
			}
			v.reset(OpLOONG64NEGV)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MULV _ (MOVVconst [0]))
	// result: (MOVVconst [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_1.AuxInt) != 0 {
				continue
			}
			v.reset(OpLOONG64MOVVconst)
			v.AuxInt = int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (MULV x (MOVVconst [1]))
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_1.AuxInt) != 1 {
				continue
			}
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (MULV x (MOVVconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (SLLVconst [log64(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpLOONG64MOVVconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo(c)) {
				continue
			}
			v.reset(OpLOONG64SLLVconst)
			v.AuxInt = int64ToAuxInt(log64(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MULV (MOVVconst [c]) (MOVVconst [d]))
	// result: (MOVVconst [c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpLOONG64MOVVconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpLOONG64MOVVconst {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpLOONG64MOVVconst)
			v.AuxInt = int64ToAuxInt(c * d)
			return true
		}
		break
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64NEGV(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NEGV (MOVVconst [c]))
	// result: (MOVVconst [-c])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(-c)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64NOR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NOR x (MOVVconst [c]))
	// cond: is32Bit(c)
	// result: (NORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpLOONG64MOVVconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(is32Bit(c)) {
				continue
			}
			v.reset(OpLOONG64NORconst)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64NORconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NORconst [c] (MOVVconst [d]))
	// result: (MOVVconst [^(c|d)])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(^(c | d))
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64OR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (OR x (MOVVconst [c]))
	// cond: is32Bit(c)
	// result: (ORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpLOONG64MOVVconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(is32Bit(c)) {
				continue
			}
			v.reset(OpLOONG64ORconst)
			v.AuxInt = int64ToAuxInt(c)
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
		v.copyOf(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64ORconst(v *Value) bool {
	v_0 := v.Args[0]
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
	// match: (ORconst [-1] _)
	// result: (MOVVconst [-1])
	for {
		if auxIntToInt64(v.AuxInt) != -1 {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	// match: (ORconst [c] (MOVVconst [d]))
	// result: (MOVVconst [c|d])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(c | d)
		return true
	}
	// match: (ORconst [c] (ORconst [d] x))
	// cond: is32Bit(c|d)
	// result: (ORconst [c|d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64ORconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(c | d)) {
			break
		}
		v.reset(OpLOONG64ORconst)
		v.AuxInt = int64ToAuxInt(c | d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64REMV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (REMV (MOVVconst [c]) (MOVVconst [d]))
	// cond: d != 0
	// result: (MOVVconst [c%d])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(c % d)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64REMVU(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (REMVU _ (MOVVconst [1]))
	// result: (MOVVconst [0])
	for {
		if v_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (REMVU x (MOVVconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (ANDconst [c-1] x)
	for {
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpLOONG64ANDconst)
		v.AuxInt = int64ToAuxInt(c - 1)
		v.AddArg(x)
		return true
	}
	// match: (REMVU (MOVVconst [c]) (MOVVconst [d]))
	// cond: d != 0
	// result: (MOVVconst [int64(uint64(c)%uint64(d))])
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) % uint64(d)))
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64ROTR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROTR x (MOVVconst [c]))
	// result: (ROTRconst x [c&31])
	for {
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpLOONG64ROTRconst)
		v.AuxInt = int64ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64ROTRV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROTRV x (MOVVconst [c]))
	// result: (ROTRVconst x [c&63])
	for {
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpLOONG64ROTRVconst)
		v.AuxInt = int64ToAuxInt(c & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64SGT(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SGT (MOVVconst [c]) x)
	// cond: is32Bit(c)
	// result: (SGTconst [c] x)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64SGTconst)
		v.AuxInt = int64ToAuxInt(c)
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
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64SGTU(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SGTU (MOVVconst [c]) x)
	// cond: is32Bit(c)
	// result: (SGTUconst [c] x)
	for {
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64SGTUconst)
		v.AuxInt = int64ToAuxInt(c)
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
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64SGTUconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SGTUconst [c] (MOVVconst [d]))
	// cond: uint64(c)>uint64(d)
	// result: (MOVVconst [1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		if !(uint64(c) > uint64(d)) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (MOVVconst [d]))
	// cond: uint64(c)<=uint64(d)
	// result: (MOVVconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		if !(uint64(c) <= uint64(d)) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SGTUconst [c] (MOVBUreg _))
	// cond: 0xff < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVBUreg || !(0xff < uint64(c)) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (MOVHUreg _))
	// cond: 0xffff < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVHUreg || !(0xffff < uint64(c)) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (ANDconst [m] _))
	// cond: uint64(m) < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64ANDconst {
			break
		}
		m := auxIntToInt64(v_0.AuxInt)
		if !(uint64(m) < uint64(c)) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (SRLVconst _ [d]))
	// cond: 0 < d && d <= 63 && 0xffffffffffffffff>>uint64(d) < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64SRLVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		if !(0 < d && d <= 63 && 0xffffffffffffffff>>uint64(d) < uint64(c)) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64SGTconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SGTconst [c] (MOVVconst [d]))
	// cond: c>d
	// result: (MOVVconst [1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		if !(c > d) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVVconst [d]))
	// cond: c<=d
	// result: (MOVVconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		if !(c <= d) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVBreg _))
	// cond: 0x7f < c
	// result: (MOVVconst [1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVBreg || !(0x7f < c) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVBreg _))
	// cond: c <= -0x80
	// result: (MOVVconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVBreg || !(c <= -0x80) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVBUreg _))
	// cond: 0xff < c
	// result: (MOVVconst [1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVBUreg || !(0xff < c) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVBUreg _))
	// cond: c < 0
	// result: (MOVVconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVBUreg || !(c < 0) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVHreg _))
	// cond: 0x7fff < c
	// result: (MOVVconst [1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVHreg || !(0x7fff < c) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVHreg _))
	// cond: c <= -0x8000
	// result: (MOVVconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVHreg || !(c <= -0x8000) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVHUreg _))
	// cond: 0xffff < c
	// result: (MOVVconst [1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVHUreg || !(0xffff < c) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVHUreg _))
	// cond: c < 0
	// result: (MOVVconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVHUreg || !(c < 0) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVWUreg _))
	// cond: c < 0
	// result: (MOVVconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVWUreg || !(c < 0) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (ANDconst [m] _))
	// cond: 0 <= m && m < c
	// result: (MOVVconst [1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64ANDconst {
			break
		}
		m := auxIntToInt64(v_0.AuxInt)
		if !(0 <= m && m < c) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (SRLVconst _ [d]))
	// cond: 0 <= c && 0 < d && d <= 63 && 0xffffffffffffffff>>uint64(d) < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64SRLVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		if !(0 <= c && 0 < d && d <= 63 && 0xffffffffffffffff>>uint64(d) < uint64(c)) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64SLLV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLLV _ (MOVVconst [c]))
	// cond: uint64(c)>=64
	// result: (MOVVconst [0])
	for {
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SLLV x (MOVVconst [c]))
	// result: (SLLVconst x [c])
	for {
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpLOONG64SLLVconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64SLLVconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SLLVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [d<<uint64(c)])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(d << uint64(c))
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64SRAV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRAV x (MOVVconst [c]))
	// cond: uint64(c)>=64
	// result: (SRAVconst x [63])
	for {
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpLOONG64SRAVconst)
		v.AuxInt = int64ToAuxInt(63)
		v.AddArg(x)
		return true
	}
	// match: (SRAV x (MOVVconst [c]))
	// result: (SRAVconst x [c])
	for {
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpLOONG64SRAVconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64SRAVconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SRAVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [d>>uint64(c)])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(d >> uint64(c))
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64SRLV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRLV _ (MOVVconst [c]))
	// cond: uint64(c)>=64
	// result: (MOVVconst [0])
	for {
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SRLV x (MOVVconst [c]))
	// result: (SRLVconst x [c])
	for {
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpLOONG64SRLVconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64SRLVconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SRLVconst [rc] (SLLVconst [lc] x))
	// cond: lc <= rc
	// result: (BSTRPICKV [rc-lc + ((64-lc)-1)<<6] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64SLLVconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc <= rc) {
			break
		}
		v.reset(OpLOONG64BSTRPICKV)
		v.AuxInt = int64ToAuxInt(rc - lc + ((64-lc)-1)<<6)
		v.AddArg(x)
		return true
	}
	// match: (SRLVconst [rc] (MOVWUreg x))
	// cond: rc < 32
	// result: (BSTRPICKV [rc + 31<<6] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 32) {
			break
		}
		v.reset(OpLOONG64BSTRPICKV)
		v.AuxInt = int64ToAuxInt(rc + 31<<6)
		v.AddArg(x)
		return true
	}
	// match: (SRLVconst [rc] (MOVHUreg x))
	// cond: rc < 16
	// result: (BSTRPICKV [rc + 15<<6] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 16) {
			break
		}
		v.reset(OpLOONG64BSTRPICKV)
		v.AuxInt = int64ToAuxInt(rc + 15<<6)
		v.AddArg(x)
		return true
	}
	// match: (SRLVconst [rc] (MOVBUreg x))
	// cond: rc < 8
	// result: (BSTRPICKV [rc + 7<<6] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 8) {
			break
		}
		v.reset(OpLOONG64BSTRPICKV)
		v.AuxInt = int64ToAuxInt(rc + 7<<6)
		v.AddArg(x)
		return true
	}
	// match: (SRLVconst [rc] (MOVWUreg x))
	// cond: rc >= 32
	// result: (MOVVconst [0])
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVWUreg {
			break
		}
		if !(rc >= 32) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SRLVconst [rc] (MOVHUreg x))
	// cond: rc >= 16
	// result: (MOVVconst [0])
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVHUreg {
			break
		}
		if !(rc >= 16) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SRLVconst [rc] (MOVBUreg x))
	// cond: rc >= 8
	// result: (MOVVconst [0])
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVBUreg {
			break
		}
		if !(rc >= 8) {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SRLVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [int64(uint64(d)>>uint64(c))])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(d) >> uint64(c)))
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64SUBV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBV x (MOVVconst [c]))
	// cond: is32Bit(c)
	// result: (SUBVconst [c] x)
	for {
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpLOONG64SUBVconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUBV x x)
	// result: (MOVVconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SUBV (MOVVconst [0]) x)
	// result: (NEGV x)
	for {
		if v_0.Op != OpLOONG64MOVVconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.reset(OpLOONG64NEGV)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64SUBVconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SUBVconst [0] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (SUBVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [d-c])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(d - c)
		return true
	}
	// match: (SUBVconst [c] (SUBVconst [d] x))
	// cond: is32Bit(-c-d)
	// result: (ADDVconst [-c-d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64SUBVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(-c - d)) {
			break
		}
		v.reset(OpLOONG64ADDVconst)
		v.AuxInt = int64ToAuxInt(-c - d)
		v.AddArg(x)
		return true
	}
	// match: (SUBVconst [c] (ADDVconst [d] x))
	// cond: is32Bit(-c+d)
	// result: (ADDVconst [-c+d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64ADDVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(-c + d)) {
			break
		}
		v.reset(OpLOONG64ADDVconst)
		v.AuxInt = int64ToAuxInt(-c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64XOR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XOR x (MOVVconst [c]))
	// cond: is32Bit(c)
	// result: (XORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpLOONG64MOVVconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(is32Bit(c)) {
				continue
			}
			v.reset(OpLOONG64XORconst)
			v.AuxInt = int64ToAuxInt(c)
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
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLOONG64XORconst(v *Value) bool {
	v_0 := v.Args[0]
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
	// match: (XORconst [-1] x)
	// result: (NORconst [0] x)
	for {
		if auxIntToInt64(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.reset(OpLOONG64NORconst)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg(x)
		return true
	}
	// match: (XORconst [c] (MOVVconst [d]))
	// result: (MOVVconst [c^d])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64MOVVconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpLOONG64MOVVconst)
		v.AuxInt = int64ToAuxInt(c ^ d)
		return true
	}
	// match: (XORconst [c] (XORconst [d] x))
	// cond: is32Bit(c^d)
	// result: (XORconst [c^d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpLOONG64XORconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(is32Bit(c ^ d)) {
			break
		}
		v.reset(OpLOONG64XORconst)
		v.AuxInt = int64ToAuxInt(c ^ d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (XOR (MOVVconst [1]) (SGT (SignExt16to64 x) (SignExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGT, typ.Bool)
		v2 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLeq16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// result: (XOR (MOVVconst [1]) (SGTU (ZeroExt16to64 x) (ZeroExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32 x y)
	// result: (XOR (MOVVconst [1]) (SGT (SignExt32to64 x) (SignExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGT, typ.Bool)
		v2 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32F x y)
	// result: (FPFlagTrue (CMPGEF y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, OpLOONG64CMPGEF, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueLOONG64_OpLeq32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32U x y)
	// result: (XOR (MOVVconst [1]) (SGTU (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64 x y)
	// result: (XOR (MOVVconst [1]) (SGT x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGT, typ.Bool)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64F x y)
	// result: (FPFlagTrue (CMPGED y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, OpLOONG64CMPGED, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueLOONG64_OpLeq64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64U x y)
	// result: (XOR (MOVVconst [1]) (SGTU x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (XOR (MOVVconst [1]) (SGT (SignExt8to64 x) (SignExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGT, typ.Bool)
		v2 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLeq8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// result: (XOR (MOVVconst [1]) (SGTU (ZeroExt8to64 x) (ZeroExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLess16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (SGT (SignExt16to64 y) (SignExt16to64 x))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGT)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLess16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// result: (SGTU (ZeroExt16to64 y) (ZeroExt16to64 x))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLess32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32 x y)
	// result: (SGT (SignExt32to64 y) (SignExt32to64 x))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGT)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLess32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32F x y)
	// result: (FPFlagTrue (CMPGTF y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, OpLOONG64CMPGTF, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueLOONG64_OpLess32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32U x y)
	// result: (SGTU (ZeroExt32to64 y) (ZeroExt32to64 x))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLess64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64 x y)
	// result: (SGT y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGT)
		v.AddArg2(y, x)
		return true
	}
}
func rewriteValueLOONG64_OpLess64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64F x y)
	// result: (FPFlagTrue (CMPGTD y x))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, OpLOONG64CMPGTD, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueLOONG64_OpLess64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64U x y)
	// result: (SGTU y x)
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v.AddArg2(y, x)
		return true
	}
}
func rewriteValueLOONG64_OpLess8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (SGT (SignExt8to64 y) (SignExt8to64 x))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGT)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLess8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// result: (SGTU (ZeroExt8to64 y) (ZeroExt8to64 x))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLoad(v *Value) bool {
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
		v.reset(OpLOONG64MOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is8BitInt(t) && t.IsSigned())
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is8BitInt(t) && t.IsSigned()) {
			break
		}
		v.reset(OpLOONG64MOVBload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is8BitInt(t) && !t.IsSigned())
	// result: (MOVBUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is8BitInt(t) && !t.IsSigned()) {
			break
		}
		v.reset(OpLOONG64MOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is16BitInt(t) && t.IsSigned())
	// result: (MOVHload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is16BitInt(t) && t.IsSigned()) {
			break
		}
		v.reset(OpLOONG64MOVHload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is16BitInt(t) && !t.IsSigned())
	// result: (MOVHUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is16BitInt(t) && !t.IsSigned()) {
			break
		}
		v.reset(OpLOONG64MOVHUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is32BitInt(t) && t.IsSigned())
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitInt(t) && t.IsSigned()) {
			break
		}
		v.reset(OpLOONG64MOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is32BitInt(t) && !t.IsSigned())
	// result: (MOVWUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitInt(t) && !t.IsSigned()) {
			break
		}
		v.reset(OpLOONG64MOVWUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (MOVVload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpLOONG64MOVVload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is32BitFloat(t)
	// result: (MOVFload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitFloat(t)) {
			break
		}
		v.reset(OpLOONG64MOVFload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is64BitFloat(t)
	// result: (MOVDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is64BitFloat(t)) {
			break
		}
		v.reset(OpLOONG64MOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLocalAddr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LocalAddr <t> {sym} base mem)
	// cond: t.Elem().HasPointers()
	// result: (MOVVaddr {sym} (SPanchored base mem))
	for {
		t := v.Type
		sym := auxToSym(v.Aux)
		base := v_0
		mem := v_1
		if !(t.Elem().HasPointers()) {
			break
		}
		v.reset(OpLOONG64MOVVaddr)
		v.Aux = symToAux(sym)
		v0 := b.NewValue0(v.Pos, OpSPanchored, typ.Uintptr)
		v0.AddArg2(base, mem)
		v.AddArg(v0)
		return true
	}
	// match: (LocalAddr <t> {sym} base _)
	// cond: !t.Elem().HasPointers()
	// result: (MOVVaddr {sym} base)
	for {
		t := v.Type
		sym := auxToSym(v.Aux)
		base := v_0
		if !(!t.Elem().HasPointers()) {
			break
		}
		v.reset(OpLOONG64MOVVaddr)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpLsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 <t> x y)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpLsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 <t> x y)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpLsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x64 <t> x y)
	// result: (MASKEQZ (SLLV <t> x y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 <t> x y)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpLsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 <t> x y)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpLsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 <t> x y)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpLsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x64 <t> x y)
	// result: (MASKEQZ (SLLV <t> x y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 <t> x y)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpLsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 <t> x y)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpLsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 <t> x y)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpLsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x64 <t> x y)
	// result: (MASKEQZ (SLLV <t> x y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 <t> x y)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpLsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 <t> x y)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpLsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 <t> x y)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpLsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x64 <t> x y)
	// result: (MASKEQZ (SLLV <t> x y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpLsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 <t> x y)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpMod16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// result: (REMV (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64REMV)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpMod16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (REMVU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64REMVU)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpMod32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32 x y)
	// result: (REMV (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64REMV)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpMod32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32u x y)
	// result: (REMVU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64REMVU)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpMod64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod64 x y)
	// result: (REMV x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64REMV)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueLOONG64_OpMod8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (REMV (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64REMV)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpMod8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (REMVU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64REMVU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpMove(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
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
	// result: (MOVBstore dst (MOVBUload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVBstore)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVBUload, typ.UInt8)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (MOVHstore dst (MOVHUload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVHstore)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVHUload, typ.UInt16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [3] dst src mem)
	// result: (MOVBstore [2] dst (MOVBUload [2] src mem) (MOVHstore dst (MOVHUload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVBUload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVHstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVHUload, typ.UInt16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (MOVWstore dst (MOVWUload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVWstore)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVWUload, typ.UInt32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [5] dst src mem)
	// result: (MOVBstore [4] dst (MOVBUload [4] src mem) (MOVWstore dst (MOVWUload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 5 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVBUload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVWUload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [6] dst src mem)
	// result: (MOVHstore [4] dst (MOVHUload [4] src mem) (MOVWstore dst (MOVWUload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 6 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVHstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVHUload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVWUload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [7] dst src mem)
	// result: (MOVWstore [3] dst (MOVWUload [3] src mem) (MOVWstore dst (MOVWUload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 7 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVWUload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVWUload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [8] dst src mem)
	// result: (MOVVstore dst (MOVVload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVVstore)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [9] dst src mem)
	// result: (MOVBstore [8] dst (MOVBUload [8] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 9 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVBUload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [10] dst src mem)
	// result: (MOVHstore [8] dst (MOVHUload [8] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 10 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVHstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVHUload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [11] dst src mem)
	// result: (MOVWstore [7] dst (MOVWload [7] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 11 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVWload, typ.Int32)
		v0.AuxInt = int32ToAuxInt(7)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [12] dst src mem)
	// result: (MOVWstore [8] dst (MOVWUload [8] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 12 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVWUload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [13] dst src mem)
	// result: (MOVVstore [5] dst (MOVVload [5] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 13 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(5)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(5)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [14] dst src mem)
	// result: (MOVVstore [6] dst (MOVVload [6] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 14 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(6)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [15] dst src mem)
	// result: (MOVVstore [7] dst (MOVVload [7] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 15 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(7)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [16] dst src mem)
	// result: (MOVVstore [8] dst (MOVVload [8] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s%8 != 0 && s > 16
	// result: (Move [s%8] (OffPtr <dst.Type> dst [s-s%8]) (OffPtr <src.Type> src [s-s%8]) (Move [s-s%8] dst src mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s%8 != 0 && s > 16) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = int64ToAuxInt(s % 8)
		v0 := b.NewValue0(v.Pos, OpOffPtr, dst.Type)
		v0.AuxInt = int64ToAuxInt(s - s%8)
		v0.AddArg(dst)
		v1 := b.NewValue0(v.Pos, OpOffPtr, src.Type)
		v1.AuxInt = int64ToAuxInt(s - s%8)
		v1.AddArg(src)
		v2 := b.NewValue0(v.Pos, OpMove, types.TypeMem)
		v2.AuxInt = int64ToAuxInt(s - s%8)
		v2.AddArg3(dst, src, mem)
		v.AddArg3(v0, v1, v2)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s%8 == 0 && s > 16 && s <= 8*128 && !config.noDuffDevice && logLargeCopy(v, s)
	// result: (DUFFCOPY [16 * (128 - s/8)] dst src mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s%8 == 0 && s > 16 && s <= 8*128 && !config.noDuffDevice && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpLOONG64DUFFCOPY)
		v.AuxInt = int64ToAuxInt(16 * (128 - s/8))
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s%8 == 0 && s > 1024 && logLargeCopy(v, s)
	// result: (LoweredMove dst src (ADDVconst <src.Type> src [s-8]) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s%8 == 0 && s > 1024 && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpLOONG64LoweredMove)
		v0 := b.NewValue0(v.Pos, OpLOONG64ADDVconst, src.Type)
		v0.AuxInt = int64ToAuxInt(s - 8)
		v0.AddArg(src)
		v.AddArg4(dst, src, v0, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpNeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// result: (SGTU (XOR (ZeroExt16to32 x) (ZeroExt16to64 y)) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpLOONG64XOR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpNeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq32 x y)
	// result: (SGTU (XOR (ZeroExt32to64 x) (ZeroExt32to64 y)) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpLOONG64XOR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpNeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32F x y)
	// result: (FPFlagFalse (CMPEQF x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64FPFlagFalse)
		v0 := b.NewValue0(v.Pos, OpLOONG64CMPEQF, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueLOONG64_OpNeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq64 x y)
	// result: (SGTU (XOR x y) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpLOONG64XOR, typ.UInt64)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpNeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64F x y)
	// result: (FPFlagFalse (CMPEQD x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64FPFlagFalse)
		v0 := b.NewValue0(v.Pos, OpLOONG64CMPEQD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueLOONG64_OpNeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// result: (SGTU (XOR (ZeroExt8to64 x) (ZeroExt8to64 y)) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpLOONG64XOR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpNeqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NeqPtr x y)
	// result: (SGTU (XOR x y) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, OpLOONG64XOR, typ.UInt64)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpNot(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Not x)
	// result: (XORconst [1] x)
	for {
		x := v_0
		v.reset(OpLOONG64XORconst)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueLOONG64_OpOffPtr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (OffPtr [off] ptr:(SP))
	// result: (MOVVaddr [int32(off)] ptr)
	for {
		off := auxIntToInt64(v.AuxInt)
		ptr := v_0
		if ptr.Op != OpSP {
			break
		}
		v.reset(OpLOONG64MOVVaddr)
		v.AuxInt = int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// result: (ADDVconst [off] ptr)
	for {
		off := auxIntToInt64(v.AuxInt)
		ptr := v_0
		v.reset(OpLOONG64ADDVconst)
		v.AuxInt = int64ToAuxInt(off)
		v.AddArg(ptr)
		return true
	}
}
func rewriteValueLOONG64_OpPanicBounds(v *Value) bool {
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
		v.reset(OpLOONG64LoweredPanicBoundsA)
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
		v.reset(OpLOONG64LoweredPanicBoundsB)
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
		v.reset(OpLOONG64LoweredPanicBoundsC)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg3(x, y, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpRotateLeft16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVVconst [c]))
	// result: (Or16 (Lsh16x64 <t> x (MOVVconst [c&15])) (Rsh16Ux64 <t> x (MOVVconst [-c&15])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpLsh16x64, t)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(c & 15)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpRsh16Ux64, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(-c & 15)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (RotateLeft16 <t> x y)
	// result: (ROTR <t> (OR <typ.UInt32> (ZeroExt16to32 x) (SLLVconst <t> (ZeroExt16to32 x) [16])) (NEGV <typ.Int64> y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64ROTR)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpLOONG64OR, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpLOONG64SLLVconst, t)
		v2.AuxInt = int64ToAuxInt(16)
		v2.AddArg(v1)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64NEGV, typ.Int64)
		v3.AddArg(y)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpRotateLeft32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RotateLeft32 x y)
	// result: (ROTR x (NEGV <y.Type> y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64ROTR)
		v0 := b.NewValue0(v.Pos, OpLOONG64NEGV, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueLOONG64_OpRotateLeft64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RotateLeft64 x y)
	// result: (ROTRV x (NEGV <y.Type> y))
	for {
		x := v_0
		y := v_1
		v.reset(OpLOONG64ROTRV)
		v0 := b.NewValue0(v.Pos, OpLOONG64NEGV, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueLOONG64_OpRotateLeft8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVVconst [c]))
	// result: (Or8 (Lsh8x64 <t> x (MOVVconst [c&7])) (Rsh8Ux64 <t> x (MOVVconst [-c&7])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpLOONG64MOVVconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpLsh8x64, t)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(c & 7)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpRsh8Ux64, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(-c & 7)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (RotateLeft8 <t> x y)
	// result: (OR <t> (SLLV <t> x (ANDconst <typ.Int64> [7] y)) (SRLV <t> (ZeroExt8to64 x) (ANDconst <typ.Int64> [7] (NEGV <typ.Int64> y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64OR)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, OpLOONG64ANDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(7)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, OpLOONG64ANDconst, typ.Int64)
		v4.AuxInt = int64ToAuxInt(7)
		v5 := b.NewValue0(v.Pos, OpLOONG64NEGV, typ.Int64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v2.AddArg2(v3, v4)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpRsh16Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 <t> x y)
	// result: (MASKEQZ (SRLV <t> (ZeroExt16to64 x) (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpRsh16Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 <t> x y)
	// result: (MASKEQZ (SRLV <t> (ZeroExt16to64 x) (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpRsh16Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 <t> x y)
	// result: (MASKEQZ (SRLV <t> (ZeroExt16to64 x) y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, y)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpRsh16Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 <t> x y)
	// result: (MASKEQZ (SRLV <t> (ZeroExt16to64 x) (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpRsh16x16(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpRsh16x32(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpRsh16x64(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(63)
		v3.AddArg2(y, v4)
		v2.AddArg(v3)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpRsh16x8(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpRsh32Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 <t> x y)
	// result: (MASKEQZ (SRLV <t> (ZeroExt32to64 x) (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpRsh32Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 <t> x y)
	// result: (MASKEQZ (SRLV <t> (ZeroExt32to64 x) (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpRsh32Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 <t> x y)
	// result: (MASKEQZ (SRLV <t> (ZeroExt32to64 x) y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, y)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpRsh32Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 <t> x y)
	// result: (MASKEQZ (SRLV <t> (ZeroExt32to64 x) (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpRsh32x16(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpRsh32x32(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpRsh32x64(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(63)
		v3.AddArg2(y, v4)
		v2.AddArg(v3)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpRsh32x8(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpRsh64Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 <t> x y)
	// result: (MASKEQZ (SRLV <t> x (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpRsh64Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 <t> x y)
	// result: (MASKEQZ (SRLV <t> x (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpRsh64Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux64 <t> x y)
	// result: (MASKEQZ (SRLV <t> x y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v2.AuxInt = int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpRsh64Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 <t> x y)
	// result: (MASKEQZ (SRLV <t> x (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpRsh64x16(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v1 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(63)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg2(v1, v3)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueLOONG64_OpRsh64x32(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v1 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(63)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg2(v1, v3)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueLOONG64_OpRsh64x64(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v1 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(63)
		v2.AddArg2(y, v3)
		v1.AddArg(v2)
		v0.AddArg2(v1, y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueLOONG64_OpRsh64x8(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v1 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(63)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg2(v1, v3)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueLOONG64_OpRsh8Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 <t> x y)
	// result: (MASKEQZ (SRLV <t> (ZeroExt8to64 x) (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpRsh8Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 <t> x y)
	// result: (MASKEQZ (SRLV <t> (ZeroExt8to64 x) (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpRsh8Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 <t> x y)
	// result: (MASKEQZ (SRLV <t> (ZeroExt8to64 x) y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(64)
		v2.AddArg2(v3, y)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueLOONG64_OpRsh8Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 <t> x y)
	// result: (MASKEQZ (SRLV <t> (ZeroExt8to64 x) (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueLOONG64_OpRsh8x16(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpRsh8x32(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpRsh8x64(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = int64ToAuxInt(63)
		v3.AddArg2(y, v4)
		v2.AddArg(v3)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpRsh8x8(v *Value) bool {
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
		v.reset(OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueLOONG64_OpSelect0(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Select0 (Mul64uhilo x y))
	// result: (MULHVU x y)
	for {
		if v_0.Op != OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLOONG64MULHVU)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select0 (Mul64uover x y))
	// result: (MULV x y)
	for {
		if v_0.Op != OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLOONG64MULV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select0 <t> (Add64carry x y c))
	// result: (ADDV (ADDV <t> x y) c)
	for {
		t := v.Type
		if v_0.Op != OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLOONG64ADDV)
		v0 := b.NewValue0(v.Pos, OpLOONG64ADDV, t)
		v0.AddArg2(x, y)
		v.AddArg2(v0, c)
		return true
	}
	// match: (Select0 <t> (Sub64borrow x y c))
	// result: (SUBV (SUBV <t> x y) c)
	for {
		t := v.Type
		if v_0.Op != OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLOONG64SUBV)
		v0 := b.NewValue0(v.Pos, OpLOONG64SUBV, t)
		v0.AddArg2(x, y)
		v.AddArg2(v0, c)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpSelect1(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Mul64uhilo x y))
	// result: (MULV x y)
	for {
		if v_0.Op != OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLOONG64MULV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select1 (Mul64uover x y))
	// result: (SGTU <typ.Bool> (MULHVU x y) (MOVVconst <typ.UInt64> [0]))
	for {
		if v_0.Op != OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpLOONG64SGTU)
		v.Type = typ.Bool
		v0 := b.NewValue0(v.Pos, OpLOONG64MULHVU, typ.UInt64)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Select1 <t> (Add64carry x y c))
	// result: (OR (SGTU <t> x s:(ADDV <t> x y)) (SGTU <t> s (ADDV <t> s c)))
	for {
		t := v.Type
		if v_0.Op != OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLOONG64OR)
		v0 := b.NewValue0(v.Pos, OpLOONG64SGTU, t)
		s := b.NewValue0(v.Pos, OpLOONG64ADDV, t)
		s.AddArg2(x, y)
		v0.AddArg2(x, s)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64ADDV, t)
		v3.AddArg2(s, c)
		v2.AddArg2(s, v3)
		v.AddArg2(v0, v2)
		return true
	}
	// match: (Select1 <t> (Sub64borrow x y c))
	// result: (OR (SGTU <t> s:(SUBV <t> x y) x) (SGTU <t> (SUBV <t> s c) s))
	for {
		t := v.Type
		if v_0.Op != OpSub64borrow {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLOONG64OR)
		v0 := b.NewValue0(v.Pos, OpLOONG64SGTU, t)
		s := b.NewValue0(v.Pos, OpLOONG64SUBV, t)
		s.AddArg2(x, y)
		v0.AddArg2(s, x)
		v2 := b.NewValue0(v.Pos, OpLOONG64SGTU, t)
		v3 := b.NewValue0(v.Pos, OpLOONG64SUBV, t)
		v3.AddArg2(s, c)
		v2.AddArg2(v3, s)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpSelectN(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SelectN [0] call:(CALLstatic {sym} dst src (MOVVconst [sz]) mem))
	// cond: sz >= 0 && isSameCall(sym, "runtime.memmove") && call.Uses == 1 && isInlinableMemmove(dst, src, sz, config) && clobber(call)
	// result: (Move [sz] dst src mem)
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != OpLOONG64CALLstatic || len(call.Args) != 4 {
			break
		}
		sym := auxToCall(call.Aux)
		mem := call.Args[3]
		dst := call.Args[0]
		src := call.Args[1]
		call_2 := call.Args[2]
		if call_2.Op != OpLOONG64MOVVconst {
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
func rewriteValueLOONG64_OpSlicemask(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SRAVconst (NEGV <t> x) [63])
	for {
		t := v.Type
		x := v_0
		v.reset(OpLOONG64SRAVconst)
		v.AuxInt = int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, OpLOONG64NEGV, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueLOONG64_OpStore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
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
		v.reset(OpLOONG64MOVBstore)
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
		v.reset(OpLOONG64MOVHstore)
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
		v.reset(OpLOONG64MOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && !t.IsFloat()
	// result: (MOVVstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && !t.IsFloat()) {
			break
		}
		v.reset(OpLOONG64MOVVstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && t.IsFloat()
	// result: (MOVFstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && t.IsFloat()) {
			break
		}
		v.reset(OpLOONG64MOVFstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && t.IsFloat()
	// result: (MOVDstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && t.IsFloat()) {
			break
		}
		v.reset(OpLOONG64MOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueLOONG64_OpZero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
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
	// match: (Zero [1] ptr mem)
	// result: (MOVBstore ptr (MOVVconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 1 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVBstore)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] ptr mem)
	// result: (MOVHstore ptr (MOVVconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 2 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVHstore)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [3] ptr mem)
	// result: (MOVBstore [2] ptr (MOVVconst [0]) (MOVHstore ptr (MOVVconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 3 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVHstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// result: (MOVWstore ptr (MOVVconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVWstore)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [5] ptr mem)
	// result: (MOVBstore [4] ptr (MOVVconst [0]) (MOVWstore ptr (MOVVconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 5 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [6] ptr mem)
	// result: (MOVHstore [4] ptr (MOVVconst [0]) (MOVWstore ptr (MOVVconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 6 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVHstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [7] ptr mem)
	// result: (MOVWstore [3] ptr (MOVVconst [0]) (MOVWstore ptr (MOVVconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 7 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [8] {t} ptr mem)
	// result: (MOVVstore ptr (MOVVconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVVstore)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [9] ptr mem)
	// result: (MOVBstore [8] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 9 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVBstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [10] ptr mem)
	// result: (MOVHstore [8] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 10 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVHstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [11] ptr mem)
	// result: (MOVWstore [7] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 11 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [12] ptr mem)
	// result: (MOVWstore [8] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 12 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVWstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [13] ptr mem)
	// result: (MOVVstore [5] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 13 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(5)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [14] ptr mem)
	// result: (MOVVstore [6] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 14 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [15] ptr mem)
	// result: (MOVVstore [7] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 15 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [16] ptr mem)
	// result: (MOVVstore [8] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpLOONG64MOVVstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s%8 != 0 && s > 16
	// result: (Zero [s%8] (OffPtr <ptr.Type> ptr [s-s%8]) (Zero [s-s%8] ptr mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s%8 != 0 && s > 16) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(s % 8)
		v0 := b.NewValue0(v.Pos, OpOffPtr, ptr.Type)
		v0.AuxInt = int64ToAuxInt(s - s%8)
		v0.AddArg(ptr)
		v1 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v1.AuxInt = int64ToAuxInt(s - s%8)
		v1.AddArg2(ptr, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s%8 == 0 && s > 16 && s <= 8*128 && !config.noDuffDevice
	// result: (DUFFZERO [8 * (128 - s/8)] ptr mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s%8 == 0 && s > 16 && s <= 8*128 && !config.noDuffDevice) {
			break
		}
		v.reset(OpLOONG64DUFFZERO)
		v.AuxInt = int64ToAuxInt(8 * (128 - s/8))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s%8 == 0 && s > 8*128
	// result: (LoweredZero ptr (ADDVconst <ptr.Type> ptr [s-8]) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s%8 == 0 && s > 8*128) {
			break
		}
		v.reset(OpLOONG64LoweredZero)
		v0 := b.NewValue0(v.Pos, OpLOONG64ADDVconst, ptr.Type)
		v0.AuxInt = int64ToAuxInt(s - 8)
		v0.AddArg(ptr)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteBlockLOONG64(b *Block) bool {
	typ := &b.Func.Config.Types
	switch b.Kind {
	case BlockLOONG64EQ:
		// match: (EQ (FPFlagTrue cmp) yes no)
		// result: (FPF cmp yes no)
		for b.Controls[0].Op == OpLOONG64FPFlagTrue {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockLOONG64FPF, cmp)
			return true
		}
		// match: (EQ (FPFlagFalse cmp) yes no)
		// result: (FPT cmp yes no)
		for b.Controls[0].Op == OpLOONG64FPFlagFalse {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockLOONG64FPT, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGT _ _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != OpLOONG64SGT {
				break
			}
			b.resetWithControl(BlockLOONG64NE, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGTU _ _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != OpLOONG64SGTU {
				break
			}
			b.resetWithControl(BlockLOONG64NE, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGTconst _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != OpLOONG64SGTconst {
				break
			}
			b.resetWithControl(BlockLOONG64NE, cmp)
			return true
		}
		// match: (EQ (XORconst [1] cmp:(SGTUconst _)) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != OpLOONG64SGTUconst {
				break
			}
			b.resetWithControl(BlockLOONG64NE, cmp)
			return true
		}
		// match: (EQ (SGTUconst [1] x) yes no)
		// result: (NE x yes no)
		for b.Controls[0].Op == OpLOONG64SGTUconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			x := v_0.Args[0]
			b.resetWithControl(BlockLOONG64NE, x)
			return true
		}
		// match: (EQ (SGTU x (MOVVconst [0])) yes no)
		// result: (EQ x yes no)
		for b.Controls[0].Op == OpLOONG64SGTU {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_0_1.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockLOONG64EQ, x)
			return true
		}
		// match: (EQ (SGTconst [0] x) yes no)
		// result: (GEZ x yes no)
		for b.Controls[0].Op == OpLOONG64SGTconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.resetWithControl(BlockLOONG64GEZ, x)
			return true
		}
		// match: (EQ (SGT x (MOVVconst [0])) yes no)
		// result: (LEZ x yes no)
		for b.Controls[0].Op == OpLOONG64SGT {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_0_1.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockLOONG64LEZ, x)
			return true
		}
		// match: (EQ (SGTU (MOVVconst [c]) y) yes no)
		// cond: c >= -2048 && c <= 2047
		// result: (EQ (SGTUconst [c] y) yes no)
		for b.Controls[0].Op == OpLOONG64SGTU {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpLOONG64MOVVconst {
				break
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if !(c >= -2048 && c <= 2047) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpLOONG64SGTUconst, typ.Bool)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockLOONG64EQ, v0)
			return true
		}
		// match: (EQ (SUBV x y) yes no)
		// result: (BEQ x y yes no)
		for b.Controls[0].Op == OpLOONG64SUBV {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.resetWithControl2(BlockLOONG64BEQ, x, y)
			return true
		}
		// match: (EQ (SGT x y) yes no)
		// result: (BGE y x yes no)
		for b.Controls[0].Op == OpLOONG64SGT {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.resetWithControl2(BlockLOONG64BGE, y, x)
			return true
		}
		// match: (EQ (SGTU x y) yes no)
		// result: (BGEU y x yes no)
		for b.Controls[0].Op == OpLOONG64SGTU {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.resetWithControl2(BlockLOONG64BGEU, y, x)
			return true
		}
		// match: (EQ (MOVVconst [0]) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (EQ (MOVVconst [c]) yes no)
		// cond: c != 0
		// result: (First no yes)
		for b.Controls[0].Op == OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(c != 0) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case BlockLOONG64GEZ:
		// match: (GEZ (MOVVconst [c]) yes no)
		// cond: c >= 0
		// result: (First yes no)
		for b.Controls[0].Op == OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(c >= 0) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (GEZ (MOVVconst [c]) yes no)
		// cond: c < 0
		// result: (First no yes)
		for b.Controls[0].Op == OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(c < 0) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case BlockLOONG64GTZ:
		// match: (GTZ (MOVVconst [c]) yes no)
		// cond: c > 0
		// result: (First yes no)
		for b.Controls[0].Op == OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(c > 0) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (GTZ (MOVVconst [c]) yes no)
		// cond: c <= 0
		// result: (First no yes)
		for b.Controls[0].Op == OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(c <= 0) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case BlockIf:
		// match: (If cond yes no)
		// result: (NE (MOVBUreg <typ.UInt64> cond) yes no)
		for {
			cond := b.Controls[0]
			v0 := b.NewValue0(cond.Pos, OpLOONG64MOVBUreg, typ.UInt64)
			v0.AddArg(cond)
			b.resetWithControl(BlockLOONG64NE, v0)
			return true
		}
	case BlockLOONG64LEZ:
		// match: (LEZ (MOVVconst [c]) yes no)
		// cond: c <= 0
		// result: (First yes no)
		for b.Controls[0].Op == OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(c <= 0) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (LEZ (MOVVconst [c]) yes no)
		// cond: c > 0
		// result: (First no yes)
		for b.Controls[0].Op == OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(c > 0) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case BlockLOONG64LTZ:
		// match: (LTZ (MOVVconst [c]) yes no)
		// cond: c < 0
		// result: (First yes no)
		for b.Controls[0].Op == OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(c < 0) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (LTZ (MOVVconst [c]) yes no)
		// cond: c >= 0
		// result: (First no yes)
		for b.Controls[0].Op == OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(c >= 0) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case BlockLOONG64NE:
		// match: (NE (FPFlagTrue cmp) yes no)
		// result: (FPT cmp yes no)
		for b.Controls[0].Op == OpLOONG64FPFlagTrue {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockLOONG64FPT, cmp)
			return true
		}
		// match: (NE (FPFlagFalse cmp) yes no)
		// result: (FPF cmp yes no)
		for b.Controls[0].Op == OpLOONG64FPFlagFalse {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockLOONG64FPF, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGT _ _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != OpLOONG64SGT {
				break
			}
			b.resetWithControl(BlockLOONG64EQ, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGTU _ _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != OpLOONG64SGTU {
				break
			}
			b.resetWithControl(BlockLOONG64EQ, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGTconst _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != OpLOONG64SGTconst {
				break
			}
			b.resetWithControl(BlockLOONG64EQ, cmp)
			return true
		}
		// match: (NE (XORconst [1] cmp:(SGTUconst _)) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != OpLOONG64SGTUconst {
				break
			}
			b.resetWithControl(BlockLOONG64EQ, cmp)
			return true
		}
		// match: (NE (SGTUconst [1] x) yes no)
		// result: (EQ x yes no)
		for b.Controls[0].Op == OpLOONG64SGTUconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			x := v_0.Args[0]
			b.resetWithControl(BlockLOONG64EQ, x)
			return true
		}
		// match: (NE (SGTU x (MOVVconst [0])) yes no)
		// result: (NE x yes no)
		for b.Controls[0].Op == OpLOONG64SGTU {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_0_1.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockLOONG64NE, x)
			return true
		}
		// match: (NE (SGTconst [0] x) yes no)
		// result: (LTZ x yes no)
		for b.Controls[0].Op == OpLOONG64SGTconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.resetWithControl(BlockLOONG64LTZ, x)
			return true
		}
		// match: (NE (SGT x (MOVVconst [0])) yes no)
		// result: (GTZ x yes no)
		for b.Controls[0].Op == OpLOONG64SGT {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpLOONG64MOVVconst || auxIntToInt64(v_0_1.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockLOONG64GTZ, x)
			return true
		}
		// match: (NE (SGTU (MOVVconst [c]) y) yes no)
		// cond: c >= -2048 && c <= 2047
		// result: (NE (SGTUconst [c] y) yes no)
		for b.Controls[0].Op == OpLOONG64SGTU {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpLOONG64MOVVconst {
				break
			}
			c := auxIntToInt64(v_0_0.AuxInt)
			if !(c >= -2048 && c <= 2047) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpLOONG64SGTUconst, typ.Bool)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockLOONG64NE, v0)
			return true
		}
		// match: (NE (SUBV x y) yes no)
		// result: (BNE x y yes no)
		for b.Controls[0].Op == OpLOONG64SUBV {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.resetWithControl2(BlockLOONG64BNE, x, y)
			return true
		}
		// match: (NE (SGT x y) yes no)
		// result: (BLT y x yes no)
		for b.Controls[0].Op == OpLOONG64SGT {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.resetWithControl2(BlockLOONG64BLT, y, x)
			return true
		}
		// match: (NE (SGTU x y) yes no)
		// result: (BLTU y x yes no)
		for b.Controls[0].Op == OpLOONG64SGTU {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.resetWithControl2(BlockLOONG64BLTU, y, x)
			return true
		}
		// match: (NE (MOVVconst [0]) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (NE (MOVVconst [c]) yes no)
		// cond: c != 0
		// result: (First yes no)
		for b.Controls[0].Op == OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(c != 0) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
	}
	return false
}
