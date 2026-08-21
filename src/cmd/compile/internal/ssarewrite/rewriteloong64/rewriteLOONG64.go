// Code generated from _gen/LOONG64.rules using 'go generate'; DO NOT EDIT.

package rewriteloong64

import "cmd/compile/internal/types"
import "cmd/compile/internal/ssa/block"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func RewriteValue(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpAbs:
		v.Op = ssaop.OpLOONG64ABSD
		return true
	case ssaop.OpAdd16:
		v.Op = ssaop.OpLOONG64ADDV
		return true
	case ssaop.OpAdd32:
		v.Op = ssaop.OpLOONG64ADDV
		return true
	case ssaop.OpAdd32F:
		v.Op = ssaop.OpLOONG64ADDF
		return true
	case ssaop.OpAdd64:
		v.Op = ssaop.OpLOONG64ADDV
		return true
	case ssaop.OpAdd64F:
		v.Op = ssaop.OpLOONG64ADDD
		return true
	case ssaop.OpAdd8:
		v.Op = ssaop.OpLOONG64ADDV
		return true
	case ssaop.OpAddPtr:
		v.Op = ssaop.OpLOONG64ADDV
		return true
	case ssaop.OpAddr:
		return rewriteValue_OpAddr(v)
	case ssaop.OpAnd16:
		v.Op = ssaop.OpLOONG64AND
		return true
	case ssaop.OpAnd32:
		v.Op = ssaop.OpLOONG64AND
		return true
	case ssaop.OpAnd64:
		v.Op = ssaop.OpLOONG64AND
		return true
	case ssaop.OpAnd8:
		v.Op = ssaop.OpLOONG64AND
		return true
	case ssaop.OpAndB:
		v.Op = ssaop.OpLOONG64AND
		return true
	case ssaop.OpAtomicAdd32:
		v.Op = ssaop.OpLOONG64LoweredAtomicAdd32
		return true
	case ssaop.OpAtomicAdd64:
		v.Op = ssaop.OpLOONG64LoweredAtomicAdd64
		return true
	case ssaop.OpAtomicAnd32:
		v.Op = ssaop.OpLOONG64LoweredAtomicAnd32
		return true
	case ssaop.OpAtomicAnd32value:
		v.Op = ssaop.OpLOONG64LoweredAtomicAnd32value
		return true
	case ssaop.OpAtomicAnd64value:
		v.Op = ssaop.OpLOONG64LoweredAtomicAnd64value
		return true
	case ssaop.OpAtomicAnd8:
		return rewriteValue_OpAtomicAnd8(v)
	case ssaop.OpAtomicCompareAndSwap32:
		return rewriteValue_OpAtomicCompareAndSwap32(v)
	case ssaop.OpAtomicCompareAndSwap32Variant:
		return rewriteValue_OpAtomicCompareAndSwap32Variant(v)
	case ssaop.OpAtomicCompareAndSwap64:
		v.Op = ssaop.OpLOONG64LoweredAtomicCas64
		return true
	case ssaop.OpAtomicCompareAndSwap64Variant:
		v.Op = ssaop.OpLOONG64LoweredAtomicCas64Variant
		return true
	case ssaop.OpAtomicExchange32:
		v.Op = ssaop.OpLOONG64LoweredAtomicExchange32
		return true
	case ssaop.OpAtomicExchange64:
		v.Op = ssaop.OpLOONG64LoweredAtomicExchange64
		return true
	case ssaop.OpAtomicExchange8Variant:
		v.Op = ssaop.OpLOONG64LoweredAtomicExchange8Variant
		return true
	case ssaop.OpAtomicLoad32:
		v.Op = ssaop.OpLOONG64LoweredAtomicLoad32
		return true
	case ssaop.OpAtomicLoad64:
		v.Op = ssaop.OpLOONG64LoweredAtomicLoad64
		return true
	case ssaop.OpAtomicLoad8:
		v.Op = ssaop.OpLOONG64LoweredAtomicLoad8
		return true
	case ssaop.OpAtomicLoadPtr:
		v.Op = ssaop.OpLOONG64LoweredAtomicLoad64
		return true
	case ssaop.OpAtomicOr32:
		v.Op = ssaop.OpLOONG64LoweredAtomicOr32
		return true
	case ssaop.OpAtomicOr32value:
		v.Op = ssaop.OpLOONG64LoweredAtomicOr32value
		return true
	case ssaop.OpAtomicOr64value:
		v.Op = ssaop.OpLOONG64LoweredAtomicOr64value
		return true
	case ssaop.OpAtomicOr8:
		return rewriteValue_OpAtomicOr8(v)
	case ssaop.OpAtomicStore32:
		v.Op = ssaop.OpLOONG64LoweredAtomicStore32
		return true
	case ssaop.OpAtomicStore32Variant:
		v.Op = ssaop.OpLOONG64LoweredAtomicStore32Variant
		return true
	case ssaop.OpAtomicStore64:
		v.Op = ssaop.OpLOONG64LoweredAtomicStore64
		return true
	case ssaop.OpAtomicStore64Variant:
		v.Op = ssaop.OpLOONG64LoweredAtomicStore64Variant
		return true
	case ssaop.OpAtomicStore8:
		v.Op = ssaop.OpLOONG64LoweredAtomicStore8
		return true
	case ssaop.OpAtomicStore8Variant:
		v.Op = ssaop.OpLOONG64LoweredAtomicStore8Variant
		return true
	case ssaop.OpAtomicStorePtrNoWB:
		v.Op = ssaop.OpLOONG64LoweredAtomicStore64
		return true
	case ssaop.OpAvg64u:
		return rewriteValue_OpAvg64u(v)
	case ssaop.OpBitLen16:
		return rewriteValue_OpBitLen16(v)
	case ssaop.OpBitLen32:
		return rewriteValue_OpBitLen32(v)
	case ssaop.OpBitLen64:
		return rewriteValue_OpBitLen64(v)
	case ssaop.OpBitLen8:
		return rewriteValue_OpBitLen8(v)
	case ssaop.OpBitRev16:
		return rewriteValue_OpBitRev16(v)
	case ssaop.OpBitRev32:
		v.Op = ssaop.OpLOONG64BITREVW
		return true
	case ssaop.OpBitRev64:
		v.Op = ssaop.OpLOONG64BITREVV
		return true
	case ssaop.OpBitRev8:
		v.Op = ssaop.OpLOONG64BITREV4B
		return true
	case ssaop.OpBswap16:
		v.Op = ssaop.OpLOONG64REVB2H
		return true
	case ssaop.OpBswap32:
		v.Op = ssaop.OpLOONG64REVB2W
		return true
	case ssaop.OpBswap64:
		v.Op = ssaop.OpLOONG64REVBV
		return true
	case ssaop.OpCeil:
		v.Op = ssaop.OpLOONG64FRINTPD
		return true
	case ssaop.OpClosureCall:
		v.Op = ssaop.OpLOONG64CALLclosure
		return true
	case ssaop.OpCom16:
		return rewriteValue_OpCom16(v)
	case ssaop.OpCom32:
		return rewriteValue_OpCom32(v)
	case ssaop.OpCom64:
		return rewriteValue_OpCom64(v)
	case ssaop.OpCom8:
		return rewriteValue_OpCom8(v)
	case ssaop.OpCondSelect:
		return rewriteValue_OpCondSelect(v)
	case ssaop.OpConst16:
		return rewriteValue_OpConst16(v)
	case ssaop.OpConst32:
		return rewriteValue_OpConst32(v)
	case ssaop.OpConst32F:
		return rewriteValue_OpConst32F(v)
	case ssaop.OpConst64:
		return rewriteValue_OpConst64(v)
	case ssaop.OpConst64F:
		return rewriteValue_OpConst64F(v)
	case ssaop.OpConst8:
		return rewriteValue_OpConst8(v)
	case ssaop.OpConstBool:
		return rewriteValue_OpConstBool(v)
	case ssaop.OpConstNil:
		return rewriteValue_OpConstNil(v)
	case ssaop.OpCopysign:
		v.Op = ssaop.OpLOONG64FCOPYSGD
		return true
	case ssaop.OpCtz16:
		return rewriteValue_OpCtz16(v)
	case ssaop.OpCtz16NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCtz32:
		v.Op = ssaop.OpLOONG64CTZW
		return true
	case ssaop.OpCtz32NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCtz64:
		v.Op = ssaop.OpLOONG64CTZV
		return true
	case ssaop.OpCtz64NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCtz8:
		return rewriteValue_OpCtz8(v)
	case ssaop.OpCtz8NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCvt32Fto32:
		v.Op = ssaop.OpLOONG64TRUNCFW
		return true
	case ssaop.OpCvt32Fto64:
		v.Op = ssaop.OpLOONG64TRUNCFV
		return true
	case ssaop.OpCvt32Fto64F:
		v.Op = ssaop.OpLOONG64MOVFD
		return true
	case ssaop.OpCvt32to32F:
		v.Op = ssaop.OpLOONG64MOVWF
		return true
	case ssaop.OpCvt32to64F:
		v.Op = ssaop.OpLOONG64MOVWD
		return true
	case ssaop.OpCvt64Fto32:
		v.Op = ssaop.OpLOONG64TRUNCDW
		return true
	case ssaop.OpCvt64Fto32F:
		v.Op = ssaop.OpLOONG64MOVDF
		return true
	case ssaop.OpCvt64Fto64:
		v.Op = ssaop.OpLOONG64TRUNCDV
		return true
	case ssaop.OpCvt64to32F:
		v.Op = ssaop.OpLOONG64MOVVF
		return true
	case ssaop.OpCvt64to64F:
		v.Op = ssaop.OpLOONG64MOVVD
		return true
	case ssaop.OpCvtBoolToUint8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpDiv16:
		return rewriteValue_OpDiv16(v)
	case ssaop.OpDiv16u:
		return rewriteValue_OpDiv16u(v)
	case ssaop.OpDiv32:
		return rewriteValue_OpDiv32(v)
	case ssaop.OpDiv32F:
		v.Op = ssaop.OpLOONG64DIVF
		return true
	case ssaop.OpDiv32u:
		return rewriteValue_OpDiv32u(v)
	case ssaop.OpDiv64:
		return rewriteValue_OpDiv64(v)
	case ssaop.OpDiv64F:
		v.Op = ssaop.OpLOONG64DIVD
		return true
	case ssaop.OpDiv64u:
		v.Op = ssaop.OpLOONG64DIVVU
		return true
	case ssaop.OpDiv8:
		return rewriteValue_OpDiv8(v)
	case ssaop.OpDiv8u:
		return rewriteValue_OpDiv8u(v)
	case ssaop.OpEq16:
		return rewriteValue_OpEq16(v)
	case ssaop.OpEq32:
		return rewriteValue_OpEq32(v)
	case ssaop.OpEq32F:
		return rewriteValue_OpEq32F(v)
	case ssaop.OpEq64:
		return rewriteValue_OpEq64(v)
	case ssaop.OpEq64F:
		return rewriteValue_OpEq64F(v)
	case ssaop.OpEq8:
		return rewriteValue_OpEq8(v)
	case ssaop.OpEqB:
		return rewriteValue_OpEqB(v)
	case ssaop.OpEqPtr:
		return rewriteValue_OpEqPtr(v)
	case ssaop.OpFMA:
		v.Op = ssaop.OpLOONG64FMADDD
		return true
	case ssaop.OpFloor:
		v.Op = ssaop.OpLOONG64FRINTMD
		return true
	case ssaop.OpGetCallerPC:
		v.Op = ssaop.OpLOONG64LoweredGetCallerPC
		return true
	case ssaop.OpGetCallerSP:
		v.Op = ssaop.OpLOONG64LoweredGetCallerSP
		return true
	case ssaop.OpGetClosurePtr:
		v.Op = ssaop.OpLOONG64LoweredGetClosurePtr
		return true
	case ssaop.OpHmul32:
		v.Op = ssaop.OpLOONG64MULH
		return true
	case ssaop.OpHmul32u:
		v.Op = ssaop.OpLOONG64MULHU
		return true
	case ssaop.OpHmul64:
		v.Op = ssaop.OpLOONG64MULHV
		return true
	case ssaop.OpHmul64u:
		v.Op = ssaop.OpLOONG64MULHVU
		return true
	case ssaop.OpInterCall:
		v.Op = ssaop.OpLOONG64CALLinter
		return true
	case ssaop.OpIsInBounds:
		return rewriteValue_OpIsInBounds(v)
	case ssaop.OpIsNonNil:
		return rewriteValue_OpIsNonNil(v)
	case ssaop.OpIsSliceInBounds:
		return rewriteValue_OpIsSliceInBounds(v)
	case ssaop.OpLOONG64ADDD:
		return rewriteValue_OpLOONG64ADDD(v)
	case ssaop.OpLOONG64ADDF:
		return rewriteValue_OpLOONG64ADDF(v)
	case ssaop.OpLOONG64ADDV:
		return rewriteValue_OpLOONG64ADDV(v)
	case ssaop.OpLOONG64ADDVconst:
		return rewriteValue_OpLOONG64ADDVconst(v)
	case ssaop.OpLOONG64ADDshiftLLV:
		return rewriteValue_OpLOONG64ADDshiftLLV(v)
	case ssaop.OpLOONG64AND:
		return rewriteValue_OpLOONG64AND(v)
	case ssaop.OpLOONG64ANDconst:
		return rewriteValue_OpLOONG64ANDconst(v)
	case ssaop.OpLOONG64DIVV:
		return rewriteValue_OpLOONG64DIVV(v)
	case ssaop.OpLOONG64DIVVU:
		return rewriteValue_OpLOONG64DIVVU(v)
	case ssaop.OpLOONG64LoweredPanicBoundsCR:
		return rewriteValue_OpLOONG64LoweredPanicBoundsCR(v)
	case ssaop.OpLOONG64LoweredPanicBoundsRC:
		return rewriteValue_OpLOONG64LoweredPanicBoundsRC(v)
	case ssaop.OpLOONG64LoweredPanicBoundsRR:
		return rewriteValue_OpLOONG64LoweredPanicBoundsRR(v)
	case ssaop.OpLOONG64MASKEQZ:
		return rewriteValue_OpLOONG64MASKEQZ(v)
	case ssaop.OpLOONG64MASKNEZ:
		return rewriteValue_OpLOONG64MASKNEZ(v)
	case ssaop.OpLOONG64MOVBUload:
		return rewriteValue_OpLOONG64MOVBUload(v)
	case ssaop.OpLOONG64MOVBUloadidx:
		return rewriteValue_OpLOONG64MOVBUloadidx(v)
	case ssaop.OpLOONG64MOVBUreg:
		return rewriteValue_OpLOONG64MOVBUreg(v)
	case ssaop.OpLOONG64MOVBload:
		return rewriteValue_OpLOONG64MOVBload(v)
	case ssaop.OpLOONG64MOVBloadidx:
		return rewriteValue_OpLOONG64MOVBloadidx(v)
	case ssaop.OpLOONG64MOVBreg:
		return rewriteValue_OpLOONG64MOVBreg(v)
	case ssaop.OpLOONG64MOVBstore:
		return rewriteValue_OpLOONG64MOVBstore(v)
	case ssaop.OpLOONG64MOVBstoreidx:
		return rewriteValue_OpLOONG64MOVBstoreidx(v)
	case ssaop.OpLOONG64MOVDF:
		return rewriteValue_OpLOONG64MOVDF(v)
	case ssaop.OpLOONG64MOVDload:
		return rewriteValue_OpLOONG64MOVDload(v)
	case ssaop.OpLOONG64MOVDloadidx:
		return rewriteValue_OpLOONG64MOVDloadidx(v)
	case ssaop.OpLOONG64MOVDstore:
		return rewriteValue_OpLOONG64MOVDstore(v)
	case ssaop.OpLOONG64MOVDstoreidx:
		return rewriteValue_OpLOONG64MOVDstoreidx(v)
	case ssaop.OpLOONG64MOVFload:
		return rewriteValue_OpLOONG64MOVFload(v)
	case ssaop.OpLOONG64MOVFloadidx:
		return rewriteValue_OpLOONG64MOVFloadidx(v)
	case ssaop.OpLOONG64MOVFstore:
		return rewriteValue_OpLOONG64MOVFstore(v)
	case ssaop.OpLOONG64MOVFstoreidx:
		return rewriteValue_OpLOONG64MOVFstoreidx(v)
	case ssaop.OpLOONG64MOVHUload:
		return rewriteValue_OpLOONG64MOVHUload(v)
	case ssaop.OpLOONG64MOVHUloadidx:
		return rewriteValue_OpLOONG64MOVHUloadidx(v)
	case ssaop.OpLOONG64MOVHUreg:
		return rewriteValue_OpLOONG64MOVHUreg(v)
	case ssaop.OpLOONG64MOVHload:
		return rewriteValue_OpLOONG64MOVHload(v)
	case ssaop.OpLOONG64MOVHloadidx:
		return rewriteValue_OpLOONG64MOVHloadidx(v)
	case ssaop.OpLOONG64MOVHreg:
		return rewriteValue_OpLOONG64MOVHreg(v)
	case ssaop.OpLOONG64MOVHstore:
		return rewriteValue_OpLOONG64MOVHstore(v)
	case ssaop.OpLOONG64MOVHstoreidx:
		return rewriteValue_OpLOONG64MOVHstoreidx(v)
	case ssaop.OpLOONG64MOVVload:
		return rewriteValue_OpLOONG64MOVVload(v)
	case ssaop.OpLOONG64MOVVloadidx:
		return rewriteValue_OpLOONG64MOVVloadidx(v)
	case ssaop.OpLOONG64MOVVnop:
		return rewriteValue_OpLOONG64MOVVnop(v)
	case ssaop.OpLOONG64MOVVreg:
		return rewriteValue_OpLOONG64MOVVreg(v)
	case ssaop.OpLOONG64MOVVstore:
		return rewriteValue_OpLOONG64MOVVstore(v)
	case ssaop.OpLOONG64MOVVstoreidx:
		return rewriteValue_OpLOONG64MOVVstoreidx(v)
	case ssaop.OpLOONG64MOVWUload:
		return rewriteValue_OpLOONG64MOVWUload(v)
	case ssaop.OpLOONG64MOVWUloadidx:
		return rewriteValue_OpLOONG64MOVWUloadidx(v)
	case ssaop.OpLOONG64MOVWUreg:
		return rewriteValue_OpLOONG64MOVWUreg(v)
	case ssaop.OpLOONG64MOVWload:
		return rewriteValue_OpLOONG64MOVWload(v)
	case ssaop.OpLOONG64MOVWloadidx:
		return rewriteValue_OpLOONG64MOVWloadidx(v)
	case ssaop.OpLOONG64MOVWreg:
		return rewriteValue_OpLOONG64MOVWreg(v)
	case ssaop.OpLOONG64MOVWstore:
		return rewriteValue_OpLOONG64MOVWstore(v)
	case ssaop.OpLOONG64MOVWstoreidx:
		return rewriteValue_OpLOONG64MOVWstoreidx(v)
	case ssaop.OpLOONG64MULV:
		return rewriteValue_OpLOONG64MULV(v)
	case ssaop.OpLOONG64NEGV:
		return rewriteValue_OpLOONG64NEGV(v)
	case ssaop.OpLOONG64NOR:
		return rewriteValue_OpLOONG64NOR(v)
	case ssaop.OpLOONG64NORconst:
		return rewriteValue_OpLOONG64NORconst(v)
	case ssaop.OpLOONG64OR:
		return rewriteValue_OpLOONG64OR(v)
	case ssaop.OpLOONG64ORN:
		return rewriteValue_OpLOONG64ORN(v)
	case ssaop.OpLOONG64ORconst:
		return rewriteValue_OpLOONG64ORconst(v)
	case ssaop.OpLOONG64REMV:
		return rewriteValue_OpLOONG64REMV(v)
	case ssaop.OpLOONG64REMVU:
		return rewriteValue_OpLOONG64REMVU(v)
	case ssaop.OpLOONG64ROTR:
		return rewriteValue_OpLOONG64ROTR(v)
	case ssaop.OpLOONG64ROTRV:
		return rewriteValue_OpLOONG64ROTRV(v)
	case ssaop.OpLOONG64SGT:
		return rewriteValue_OpLOONG64SGT(v)
	case ssaop.OpLOONG64SGTU:
		return rewriteValue_OpLOONG64SGTU(v)
	case ssaop.OpLOONG64SGTUconst:
		return rewriteValue_OpLOONG64SGTUconst(v)
	case ssaop.OpLOONG64SGTconst:
		return rewriteValue_OpLOONG64SGTconst(v)
	case ssaop.OpLOONG64SLL:
		return rewriteValue_OpLOONG64SLL(v)
	case ssaop.OpLOONG64SLLV:
		return rewriteValue_OpLOONG64SLLV(v)
	case ssaop.OpLOONG64SLLVconst:
		return rewriteValue_OpLOONG64SLLVconst(v)
	case ssaop.OpLOONG64SLLconst:
		return rewriteValue_OpLOONG64SLLconst(v)
	case ssaop.OpLOONG64SRA:
		return rewriteValue_OpLOONG64SRA(v)
	case ssaop.OpLOONG64SRAV:
		return rewriteValue_OpLOONG64SRAV(v)
	case ssaop.OpLOONG64SRAVconst:
		return rewriteValue_OpLOONG64SRAVconst(v)
	case ssaop.OpLOONG64SRL:
		return rewriteValue_OpLOONG64SRL(v)
	case ssaop.OpLOONG64SRLV:
		return rewriteValue_OpLOONG64SRLV(v)
	case ssaop.OpLOONG64SRLVconst:
		return rewriteValue_OpLOONG64SRLVconst(v)
	case ssaop.OpLOONG64SUBD:
		return rewriteValue_OpLOONG64SUBD(v)
	case ssaop.OpLOONG64SUBF:
		return rewriteValue_OpLOONG64SUBF(v)
	case ssaop.OpLOONG64SUBV:
		return rewriteValue_OpLOONG64SUBV(v)
	case ssaop.OpLOONG64SUBVconst:
		return rewriteValue_OpLOONG64SUBVconst(v)
	case ssaop.OpLOONG64XOR:
		return rewriteValue_OpLOONG64XOR(v)
	case ssaop.OpLOONG64XORconst:
		return rewriteValue_OpLOONG64XORconst(v)
	case ssaop.OpLeq16:
		return rewriteValue_OpLeq16(v)
	case ssaop.OpLeq16U:
		return rewriteValue_OpLeq16U(v)
	case ssaop.OpLeq32:
		return rewriteValue_OpLeq32(v)
	case ssaop.OpLeq32F:
		return rewriteValue_OpLeq32F(v)
	case ssaop.OpLeq32U:
		return rewriteValue_OpLeq32U(v)
	case ssaop.OpLeq64:
		return rewriteValue_OpLeq64(v)
	case ssaop.OpLeq64F:
		return rewriteValue_OpLeq64F(v)
	case ssaop.OpLeq64U:
		return rewriteValue_OpLeq64U(v)
	case ssaop.OpLeq8:
		return rewriteValue_OpLeq8(v)
	case ssaop.OpLeq8U:
		return rewriteValue_OpLeq8U(v)
	case ssaop.OpLess16:
		return rewriteValue_OpLess16(v)
	case ssaop.OpLess16U:
		return rewriteValue_OpLess16U(v)
	case ssaop.OpLess32:
		return rewriteValue_OpLess32(v)
	case ssaop.OpLess32F:
		return rewriteValue_OpLess32F(v)
	case ssaop.OpLess32U:
		return rewriteValue_OpLess32U(v)
	case ssaop.OpLess64:
		return rewriteValue_OpLess64(v)
	case ssaop.OpLess64F:
		return rewriteValue_OpLess64F(v)
	case ssaop.OpLess64U:
		return rewriteValue_OpLess64U(v)
	case ssaop.OpLess8:
		return rewriteValue_OpLess8(v)
	case ssaop.OpLess8U:
		return rewriteValue_OpLess8U(v)
	case ssaop.OpLoad:
		return rewriteValue_OpLoad(v)
	case ssaop.OpLocalAddr:
		return rewriteValue_OpLocalAddr(v)
	case ssaop.OpLsh16x16:
		return rewriteValue_OpLsh16x16(v)
	case ssaop.OpLsh16x32:
		return rewriteValue_OpLsh16x32(v)
	case ssaop.OpLsh16x64:
		return rewriteValue_OpLsh16x64(v)
	case ssaop.OpLsh16x8:
		return rewriteValue_OpLsh16x8(v)
	case ssaop.OpLsh32x16:
		return rewriteValue_OpLsh32x16(v)
	case ssaop.OpLsh32x32:
		return rewriteValue_OpLsh32x32(v)
	case ssaop.OpLsh32x64:
		return rewriteValue_OpLsh32x64(v)
	case ssaop.OpLsh32x8:
		return rewriteValue_OpLsh32x8(v)
	case ssaop.OpLsh64x16:
		return rewriteValue_OpLsh64x16(v)
	case ssaop.OpLsh64x32:
		return rewriteValue_OpLsh64x32(v)
	case ssaop.OpLsh64x64:
		return rewriteValue_OpLsh64x64(v)
	case ssaop.OpLsh64x8:
		return rewriteValue_OpLsh64x8(v)
	case ssaop.OpLsh8x16:
		return rewriteValue_OpLsh8x16(v)
	case ssaop.OpLsh8x32:
		return rewriteValue_OpLsh8x32(v)
	case ssaop.OpLsh8x64:
		return rewriteValue_OpLsh8x64(v)
	case ssaop.OpLsh8x8:
		return rewriteValue_OpLsh8x8(v)
	case ssaop.OpMax32F:
		v.Op = ssaop.OpLOONG64FMAXF
		return true
	case ssaop.OpMax64F:
		v.Op = ssaop.OpLOONG64FMAXD
		return true
	case ssaop.OpMin32F:
		v.Op = ssaop.OpLOONG64FMINF
		return true
	case ssaop.OpMin64F:
		v.Op = ssaop.OpLOONG64FMIND
		return true
	case ssaop.OpMod16:
		return rewriteValue_OpMod16(v)
	case ssaop.OpMod16u:
		return rewriteValue_OpMod16u(v)
	case ssaop.OpMod32:
		return rewriteValue_OpMod32(v)
	case ssaop.OpMod32u:
		return rewriteValue_OpMod32u(v)
	case ssaop.OpMod64:
		return rewriteValue_OpMod64(v)
	case ssaop.OpMod64u:
		v.Op = ssaop.OpLOONG64REMVU
		return true
	case ssaop.OpMod8:
		return rewriteValue_OpMod8(v)
	case ssaop.OpMod8u:
		return rewriteValue_OpMod8u(v)
	case ssaop.OpMove:
		return rewriteValue_OpMove(v)
	case ssaop.OpMul16:
		v.Op = ssaop.OpLOONG64MULV
		return true
	case ssaop.OpMul32:
		v.Op = ssaop.OpLOONG64MULV
		return true
	case ssaop.OpMul32F:
		v.Op = ssaop.OpLOONG64MULF
		return true
	case ssaop.OpMul64:
		v.Op = ssaop.OpLOONG64MULV
		return true
	case ssaop.OpMul64F:
		v.Op = ssaop.OpLOONG64MULD
		return true
	case ssaop.OpMul8:
		v.Op = ssaop.OpLOONG64MULV
		return true
	case ssaop.OpNeg16:
		v.Op = ssaop.OpLOONG64NEGV
		return true
	case ssaop.OpNeg32:
		v.Op = ssaop.OpLOONG64NEGV
		return true
	case ssaop.OpNeg32F:
		v.Op = ssaop.OpLOONG64NEGF
		return true
	case ssaop.OpNeg64:
		v.Op = ssaop.OpLOONG64NEGV
		return true
	case ssaop.OpNeg64F:
		v.Op = ssaop.OpLOONG64NEGD
		return true
	case ssaop.OpNeg8:
		v.Op = ssaop.OpLOONG64NEGV
		return true
	case ssaop.OpNeq16:
		return rewriteValue_OpNeq16(v)
	case ssaop.OpNeq32:
		return rewriteValue_OpNeq32(v)
	case ssaop.OpNeq32F:
		return rewriteValue_OpNeq32F(v)
	case ssaop.OpNeq64:
		return rewriteValue_OpNeq64(v)
	case ssaop.OpNeq64F:
		return rewriteValue_OpNeq64F(v)
	case ssaop.OpNeq8:
		return rewriteValue_OpNeq8(v)
	case ssaop.OpNeqB:
		v.Op = ssaop.OpLOONG64XOR
		return true
	case ssaop.OpNeqPtr:
		return rewriteValue_OpNeqPtr(v)
	case ssaop.OpNilCheck:
		v.Op = ssaop.OpLOONG64LoweredNilCheck
		return true
	case ssaop.OpNot:
		return rewriteValue_OpNot(v)
	case ssaop.OpOffPtr:
		return rewriteValue_OpOffPtr(v)
	case ssaop.OpOr16:
		v.Op = ssaop.OpLOONG64OR
		return true
	case ssaop.OpOr32:
		v.Op = ssaop.OpLOONG64OR
		return true
	case ssaop.OpOr64:
		v.Op = ssaop.OpLOONG64OR
		return true
	case ssaop.OpOr8:
		v.Op = ssaop.OpLOONG64OR
		return true
	case ssaop.OpOrB:
		v.Op = ssaop.OpLOONG64OR
		return true
	case ssaop.OpPanicBounds:
		v.Op = ssaop.OpLOONG64LoweredPanicBoundsRR
		return true
	case ssaop.OpPopCount16:
		return rewriteValue_OpPopCount16(v)
	case ssaop.OpPopCount32:
		return rewriteValue_OpPopCount32(v)
	case ssaop.OpPopCount64:
		return rewriteValue_OpPopCount64(v)
	case ssaop.OpPrefetchCache:
		return rewriteValue_OpPrefetchCache(v)
	case ssaop.OpPrefetchCacheStreamed:
		return rewriteValue_OpPrefetchCacheStreamed(v)
	case ssaop.OpPubBarrier:
		v.Op = ssaop.OpLOONG64LoweredPubBarrier
		return true
	case ssaop.OpRotateLeft16:
		return rewriteValue_OpRotateLeft16(v)
	case ssaop.OpRotateLeft32:
		return rewriteValue_OpRotateLeft32(v)
	case ssaop.OpRotateLeft64:
		return rewriteValue_OpRotateLeft64(v)
	case ssaop.OpRotateLeft8:
		return rewriteValue_OpRotateLeft8(v)
	case ssaop.OpRound32F:
		v.Op = ssaop.OpLOONG64LoweredRound32F
		return true
	case ssaop.OpRound64F:
		v.Op = ssaop.OpLOONG64LoweredRound64F
		return true
	case ssaop.OpRoundToEven:
		v.Op = ssaop.OpLOONG64FRINTND
		return true
	case ssaop.OpRsh16Ux16:
		return rewriteValue_OpRsh16Ux16(v)
	case ssaop.OpRsh16Ux32:
		return rewriteValue_OpRsh16Ux32(v)
	case ssaop.OpRsh16Ux64:
		return rewriteValue_OpRsh16Ux64(v)
	case ssaop.OpRsh16Ux8:
		return rewriteValue_OpRsh16Ux8(v)
	case ssaop.OpRsh16x16:
		return rewriteValue_OpRsh16x16(v)
	case ssaop.OpRsh16x32:
		return rewriteValue_OpRsh16x32(v)
	case ssaop.OpRsh16x64:
		return rewriteValue_OpRsh16x64(v)
	case ssaop.OpRsh16x8:
		return rewriteValue_OpRsh16x8(v)
	case ssaop.OpRsh32Ux16:
		return rewriteValue_OpRsh32Ux16(v)
	case ssaop.OpRsh32Ux32:
		return rewriteValue_OpRsh32Ux32(v)
	case ssaop.OpRsh32Ux64:
		return rewriteValue_OpRsh32Ux64(v)
	case ssaop.OpRsh32Ux8:
		return rewriteValue_OpRsh32Ux8(v)
	case ssaop.OpRsh32x16:
		return rewriteValue_OpRsh32x16(v)
	case ssaop.OpRsh32x32:
		return rewriteValue_OpRsh32x32(v)
	case ssaop.OpRsh32x64:
		return rewriteValue_OpRsh32x64(v)
	case ssaop.OpRsh32x8:
		return rewriteValue_OpRsh32x8(v)
	case ssaop.OpRsh64Ux16:
		return rewriteValue_OpRsh64Ux16(v)
	case ssaop.OpRsh64Ux32:
		return rewriteValue_OpRsh64Ux32(v)
	case ssaop.OpRsh64Ux64:
		return rewriteValue_OpRsh64Ux64(v)
	case ssaop.OpRsh64Ux8:
		return rewriteValue_OpRsh64Ux8(v)
	case ssaop.OpRsh64x16:
		return rewriteValue_OpRsh64x16(v)
	case ssaop.OpRsh64x32:
		return rewriteValue_OpRsh64x32(v)
	case ssaop.OpRsh64x64:
		return rewriteValue_OpRsh64x64(v)
	case ssaop.OpRsh64x8:
		return rewriteValue_OpRsh64x8(v)
	case ssaop.OpRsh8Ux16:
		return rewriteValue_OpRsh8Ux16(v)
	case ssaop.OpRsh8Ux32:
		return rewriteValue_OpRsh8Ux32(v)
	case ssaop.OpRsh8Ux64:
		return rewriteValue_OpRsh8Ux64(v)
	case ssaop.OpRsh8Ux8:
		return rewriteValue_OpRsh8Ux8(v)
	case ssaop.OpRsh8x16:
		return rewriteValue_OpRsh8x16(v)
	case ssaop.OpRsh8x32:
		return rewriteValue_OpRsh8x32(v)
	case ssaop.OpRsh8x64:
		return rewriteValue_OpRsh8x64(v)
	case ssaop.OpRsh8x8:
		return rewriteValue_OpRsh8x8(v)
	case ssaop.OpSelect0:
		return rewriteValue_OpSelect0(v)
	case ssaop.OpSelect1:
		return rewriteValue_OpSelect1(v)
	case ssaop.OpSelectN:
		return rewriteValue_OpSelectN(v)
	case ssaop.OpSignExt16to32:
		v.Op = ssaop.OpLOONG64MOVHreg
		return true
	case ssaop.OpSignExt16to64:
		v.Op = ssaop.OpLOONG64MOVHreg
		return true
	case ssaop.OpSignExt32to64:
		v.Op = ssaop.OpLOONG64MOVWreg
		return true
	case ssaop.OpSignExt8to16:
		v.Op = ssaop.OpLOONG64MOVBreg
		return true
	case ssaop.OpSignExt8to32:
		v.Op = ssaop.OpLOONG64MOVBreg
		return true
	case ssaop.OpSignExt8to64:
		v.Op = ssaop.OpLOONG64MOVBreg
		return true
	case ssaop.OpSlicemask:
		return rewriteValue_OpSlicemask(v)
	case ssaop.OpSqrt:
		v.Op = ssaop.OpLOONG64SQRTD
		return true
	case ssaop.OpSqrt32:
		v.Op = ssaop.OpLOONG64SQRTF
		return true
	case ssaop.OpStaticCall:
		v.Op = ssaop.OpLOONG64CALLstatic
		return true
	case ssaop.OpStore:
		return rewriteValue_OpStore(v)
	case ssaop.OpSub16:
		v.Op = ssaop.OpLOONG64SUBV
		return true
	case ssaop.OpSub32:
		v.Op = ssaop.OpLOONG64SUBV
		return true
	case ssaop.OpSub32F:
		v.Op = ssaop.OpLOONG64SUBF
		return true
	case ssaop.OpSub64:
		v.Op = ssaop.OpLOONG64SUBV
		return true
	case ssaop.OpSub64F:
		v.Op = ssaop.OpLOONG64SUBD
		return true
	case ssaop.OpSub8:
		v.Op = ssaop.OpLOONG64SUBV
		return true
	case ssaop.OpSubPtr:
		v.Op = ssaop.OpLOONG64SUBV
		return true
	case ssaop.OpTailCall:
		v.Op = ssaop.OpLOONG64CALLtail
		return true
	case ssaop.OpTailCallInter:
		v.Op = ssaop.OpLOONG64CALLtailinter
		return true
	case ssaop.OpTrunc:
		v.Op = ssaop.OpLOONG64FRINTZD
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
		v.Op = ssaop.OpLOONG64LoweredWB
		return true
	case ssaop.OpXor16:
		v.Op = ssaop.OpLOONG64XOR
		return true
	case ssaop.OpXor32:
		v.Op = ssaop.OpLOONG64XOR
		return true
	case ssaop.OpXor64:
		v.Op = ssaop.OpLOONG64XOR
		return true
	case ssaop.OpXor8:
		v.Op = ssaop.OpLOONG64XOR
		return true
	case ssaop.OpZero:
		return rewriteValue_OpZero(v)
	case ssaop.OpZeroExt16to32:
		v.Op = ssaop.OpLOONG64MOVHUreg
		return true
	case ssaop.OpZeroExt16to64:
		v.Op = ssaop.OpLOONG64MOVHUreg
		return true
	case ssaop.OpZeroExt32to64:
		v.Op = ssaop.OpLOONG64MOVWUreg
		return true
	case ssaop.OpZeroExt8to16:
		v.Op = ssaop.OpLOONG64MOVBUreg
		return true
	case ssaop.OpZeroExt8to32:
		v.Op = ssaop.OpLOONG64MOVBUreg
		return true
	case ssaop.OpZeroExt8to64:
		v.Op = ssaop.OpLOONG64MOVBUreg
		return true
	}
	return false
}
func rewriteValue_OpAddr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (MOVVaddr {sym} base)
	for {
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		v.Reset(ssaop.OpLOONG64MOVVaddr)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValue_OpAtomicAnd8(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicAnd8 ptr val mem)
	// result: (LoweredAtomicAnd32 (AND <typ.Uintptr> (MOVVconst [^3]) ptr) (NORconst [0] <typ.UInt32> (SLLV <typ.UInt32> (XORconst <typ.UInt32> [0xff] (ZeroExt8to32 val)) (SLLVconst <typ.UInt64> [3] (ANDconst <typ.UInt64> [3] ptr)))) mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64LoweredAtomicAnd32)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64AND, typ.Uintptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(^3)
		v0.AddArg2(v1, ptr)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64NORconst, typ.UInt32)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, typ.UInt32)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64XORconst, typ.UInt32)
		v4.AuxInt = ssa.Int64ToAuxInt(0xff)
		v5 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v5.AddArg(val)
		v4.AddArg(v5)
		v6 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.UInt64)
		v6.AuxInt = ssa.Int64ToAuxInt(3)
		v7 := b.NewValue0(v.Pos, ssaop.OpLOONG64ANDconst, typ.UInt64)
		v7.AuxInt = ssa.Int64ToAuxInt(3)
		v7.AddArg(ptr)
		v6.AddArg(v7)
		v3.AddArg2(v4, v6)
		v2.AddArg(v3)
		v.AddArg3(v0, v2, mem)
		return true
	}
}
func rewriteValue_OpAtomicCompareAndSwap32(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpLOONG64LoweredAtomicCas32)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(old)
		v.AddArg4(ptr, v0, new, mem)
		return true
	}
}
func rewriteValue_OpAtomicCompareAndSwap32Variant(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicCompareAndSwap32Variant ptr old new mem)
	// result: (LoweredAtomicCas32Variant ptr (SignExt32to64 old) new mem)
	for {
		ptr := v_0
		old := v_1
		new := v_2
		mem := v_3
		v.Reset(ssaop.OpLOONG64LoweredAtomicCas32Variant)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(old)
		v.AddArg4(ptr, v0, new, mem)
		return true
	}
}
func rewriteValue_OpAtomicOr8(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicOr8 ptr val mem)
	// result: (LoweredAtomicOr32 (AND <typ.Uintptr> (MOVVconst [^3]) ptr) (SLLV <typ.UInt32> (ZeroExt8to32 val) (SLLVconst <typ.UInt64> [3] (ANDconst <typ.UInt64> [3] ptr))) mem)
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64LoweredAtomicOr32)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64AND, typ.Uintptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(^3)
		v0.AddArg2(v1, ptr)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, typ.UInt32)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v3.AddArg(val)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(3)
		v5 := b.NewValue0(v.Pos, ssaop.OpLOONG64ANDconst, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(3)
		v5.AddArg(ptr)
		v4.AddArg(v5)
		v2.AddArg2(v3, v4)
		v.AddArg3(v0, v2, mem)
		return true
	}
}
func rewriteValue_OpAvg64u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Avg64u <t> x y)
	// result: (ADDV (SRLVconst <t> (SUBV <t> x y) [1]) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64ADDV)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLVconst, t)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SUBV, t)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpBitLen16(v *ssa.Value) bool {
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
func rewriteValue_OpBitLen32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (BitLen32 <t> x)
	// result: (NEGV <t> (SUBVconst <t> [32] (CLZW <t> x)))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpLOONG64NEGV)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SUBVconst, t)
		v0.AuxInt = ssa.Int64ToAuxInt(32)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64CLZW, t)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpBitLen64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (BitLen64 <t> x)
	// result: (NEGV <t> (SUBVconst <t> [64] (CLZV <t> x)))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpLOONG64NEGV)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SUBVconst, t)
		v0.AuxInt = ssa.Int64ToAuxInt(64)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64CLZV, t)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpBitLen8(v *ssa.Value) bool {
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
func rewriteValue_OpBitRev16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (BitRev16 <t> x)
	// result: (REVB2H (BITREV4B <t> x))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpLOONG64REVB2H)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64BITREV4B, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpCom16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com16 x)
	// result: (NOR (MOVVconst [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpLOONG64NOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValue_OpCom32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com32 x)
	// result: (NOR (MOVVconst [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpLOONG64NOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValue_OpCom64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com64 x)
	// result: (NOR (MOVVconst [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpLOONG64NOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValue_OpCom8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Com8 x)
	// result: (NOR (MOVVconst [0]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpLOONG64NOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValue_OpCondSelect(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpLOONG64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MASKEQZ, t)
		v0.AddArg2(x, cond)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MASKNEZ, t)
		v1.AddArg2(y, cond)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpConst16(v *ssa.Value) bool {
	// match: (Const16 [val])
	// result: (MOVVconst [int64(val)])
	for {
		val := ssa.AuxIntToInt16(v.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValue_OpConst32(v *ssa.Value) bool {
	// match: (Const32 [val])
	// result: (MOVVconst [int64(val)])
	for {
		val := ssa.AuxIntToInt32(v.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValue_OpConst32F(v *ssa.Value) bool {
	// match: (Const32F [val])
	// result: (MOVFconst [float64(val)])
	for {
		val := ssa.AuxIntToFloat32(v.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVFconst)
		v.AuxInt = ssa.Float64ToAuxInt(float64(val))
		return true
	}
}
func rewriteValue_OpConst64(v *ssa.Value) bool {
	// match: (Const64 [val])
	// result: (MOVVconst [int64(val)])
	for {
		val := ssa.AuxIntToInt64(v.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValue_OpConst64F(v *ssa.Value) bool {
	// match: (Const64F [val])
	// result: (MOVDconst [float64(val)])
	for {
		val := ssa.AuxIntToFloat64(v.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVDconst)
		v.AuxInt = ssa.Float64ToAuxInt(float64(val))
		return true
	}
}
func rewriteValue_OpConst8(v *ssa.Value) bool {
	// match: (Const8 [val])
	// result: (MOVVconst [int64(val)])
	for {
		val := ssa.AuxIntToInt8(v.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValue_OpConstBool(v *ssa.Value) bool {
	// match: (ConstBool [t])
	// result: (MOVVconst [int64(ssa.B2i(t))])
	for {
		t := ssa.AuxIntToBool(v.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(ssa.B2i(t)))
		return true
	}
}
func rewriteValue_OpConstNil(v *ssa.Value) bool {
	// match: (ConstNil)
	// result: (MOVVconst [0])
	for {
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
}
func rewriteValue_OpCtz16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 x)
	// result: (CTZV (OR <typ.UInt64> x (MOVVconst [1<<16])))
	for {
		x := v_0
		v.Reset(ssaop.OpLOONG64CTZV)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(1 << 16)
		v0.AddArg2(x, v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpCtz8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 x)
	// result: (CTZV (OR <typ.UInt64> x (MOVVconst [1<<8])))
	for {
		x := v_0
		v.Reset(ssaop.OpLOONG64CTZV)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(1 << 8)
		v0.AddArg2(x, v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpDiv16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 x y)
	// result: (DIVV (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64DIVV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpDiv16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (DIVVU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64DIVVU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpDiv32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32 x y)
	// result: (DIVV (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64DIVV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpDiv32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32u x y)
	// result: (DIVVU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64DIVVU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpDiv64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64 x y)
	// result: (DIVV x y)
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64DIVV)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValue_OpDiv8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (DIVV (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64DIVV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpDiv8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (DIVVU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64DIVVU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpEq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// result: (SGTU (MOVVconst [1]) (XOR (ZeroExt16to64 x) (ZeroExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64XOR, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpEq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq32 x y)
	// result: (SGTU (MOVVconst [1]) (XOR (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64XOR, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpEq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32F x y)
	// result: (FPFlagTrue (CMPEQF x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64CMPEQF, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq64 x y)
	// result: (SGTU (MOVVconst [1]) (XOR x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64XOR, typ.UInt64)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpEq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64F x y)
	// result: (FPFlagTrue (CMPEQD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64CMPEQD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// result: (SGTU (MOVVconst [1]) (XOR (ZeroExt8to64 x) (ZeroExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64XOR, typ.UInt64)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpEqB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// result: (XOR (MOVVconst [1]) (XOR <typ.Bool> x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64XOR, typ.Bool)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpEqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqPtr x y)
	// result: (SGTU (MOVVconst [1]) (XOR x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64XOR, typ.UInt64)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpIsInBounds(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IsInBounds idx len)
	// result: (SGTU len idx)
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v.AddArg2(len, idx)
		return true
	}
}
func rewriteValue_OpIsNonNil(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsNonNil ptr)
	// result: (SGTU ptr (MOVVconst [0]))
	for {
		ptr := v_0
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(ptr, v0)
		return true
	}
}
func rewriteValue_OpIsSliceInBounds(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsSliceInBounds idx len)
	// result: (XOR (MOVVconst [1]) (SGTU idx len))
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v1.AddArg2(idx, len)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLOONG64ADDD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDD (MULD x y) z)
	// cond: z.Block.Func.UseFMA(v)
	// result: (FMADDD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64MULD {
				continue
			}
			y := v_0.Args[1]
			x := v_0.Args[0]
			z := v_1
			if !(z.Block.Func.UseFMA(v)) {
				continue
			}
			v.Reset(ssaop.OpLOONG64FMADDD)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (ADDD z (NEGD (MULD x y)))
	// cond: z.Block.Func.UseFMA(v)
	// result: (FNMSUBD x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			z := v_0
			if v_1.Op != ssaop.OpLOONG64NEGD {
				continue
			}
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpLOONG64MULD {
				continue
			}
			y := v_1_0.Args[1]
			x := v_1_0.Args[0]
			if !(z.Block.Func.UseFMA(v)) {
				continue
			}
			v.Reset(ssaop.OpLOONG64FNMSUBD)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpLOONG64ADDF(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDF (MULF x y) z)
	// cond: z.Block.Func.UseFMA(v)
	// result: (FMADDF x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64MULF {
				continue
			}
			y := v_0.Args[1]
			x := v_0.Args[0]
			z := v_1
			if !(z.Block.Func.UseFMA(v)) {
				continue
			}
			v.Reset(ssaop.OpLOONG64FMADDF)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (ADDF z (NEGF (MULF x y)))
	// cond: z.Block.Func.UseFMA(v)
	// result: (FNMSUBF x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			z := v_0
			if v_1.Op != ssaop.OpLOONG64NEGF {
				continue
			}
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpLOONG64MULF {
				continue
			}
			y := v_1_0.Args[1]
			x := v_1_0.Args[0]
			if !(z.Block.Func.UseFMA(v)) {
				continue
			}
			v.Reset(ssaop.OpLOONG64FNMSUBF)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpLOONG64ADDV(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDV <typ.UInt16> (SRLVconst [8] <typ.UInt16> x) (SLLVconst [8] <typ.UInt16> x))
	// result: (REVB2H x)
	for {
		if v.Type != typ.UInt16 {
			break
		}
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64SRLVconst || v_0.Type != typ.UInt16 || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpLOONG64SLLVconst || v_1.Type != typ.UInt16 || ssa.AuxIntToInt64(v_1.AuxInt) != 8 || x != v_1.Args[0] {
				continue
			}
			v.Reset(ssaop.OpLOONG64REVB2H)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADDV (SRLconst [8] (ANDconst [c1] x)) (SLLconst [8] (ANDconst [c2] x)))
	// cond: uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff
	// result: (REVB2H x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpLOONG64ANDconst {
				continue
			}
			c1 := ssa.AuxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != ssaop.OpLOONG64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 8 {
				continue
			}
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpLOONG64ANDconst {
				continue
			}
			c2 := ssa.AuxIntToInt64(v_1_0.AuxInt)
			if x != v_1_0.Args[0] || !(uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff) {
				continue
			}
			v.Reset(ssaop.OpLOONG64REVB2H)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADDV (SRLVconst [8] (AND (MOVVconst [c1]) x)) (SLLVconst [8] (AND (MOVVconst [c2]) x)))
	// cond: uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff
	// result: (REVB4H x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64SRLVconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpLOONG64AND {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				if v_0_0_0.Op != ssaop.OpLOONG64MOVVconst {
					continue
				}
				c1 := ssa.AuxIntToInt64(v_0_0_0.AuxInt)
				x := v_0_0_1
				if v_1.Op != ssaop.OpLOONG64SLLVconst || ssa.AuxIntToInt64(v_1.AuxInt) != 8 {
					continue
				}
				v_1_0 := v_1.Args[0]
				if v_1_0.Op != ssaop.OpLOONG64AND {
					continue
				}
				_ = v_1_0.Args[1]
				v_1_0_0 := v_1_0.Args[0]
				v_1_0_1 := v_1_0.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, v_1_0_0, v_1_0_1 = _i2+1, v_1_0_1, v_1_0_0 {
					if v_1_0_0.Op != ssaop.OpLOONG64MOVVconst {
						continue
					}
					c2 := ssa.AuxIntToInt64(v_1_0_0.AuxInt)
					if x != v_1_0_1 || !(uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff) {
						continue
					}
					v.Reset(ssaop.OpLOONG64REVB4H)
					v.AddArg(x)
					return true
				}
			}
		}
		break
	}
	// match: (ADDV (SRLVconst [8] (AND (MOVVconst [c1]) x)) (SLLVconst [8] (ANDconst [c2] x)))
	// cond: uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff
	// result: (REVB4H (ANDconst <x.Type> [0xffffffff] x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64SRLVconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpLOONG64AND {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				if v_0_0_0.Op != ssaop.OpLOONG64MOVVconst {
					continue
				}
				c1 := ssa.AuxIntToInt64(v_0_0_0.AuxInt)
				x := v_0_0_1
				if v_1.Op != ssaop.OpLOONG64SLLVconst || ssa.AuxIntToInt64(v_1.AuxInt) != 8 {
					continue
				}
				v_1_0 := v_1.Args[0]
				if v_1_0.Op != ssaop.OpLOONG64ANDconst {
					continue
				}
				c2 := ssa.AuxIntToInt64(v_1_0.AuxInt)
				if x != v_1_0.Args[0] || !(uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff) {
					continue
				}
				v.Reset(ssaop.OpLOONG64REVB4H)
				v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64ANDconst, x.Type)
				v0.AuxInt = ssa.Int64ToAuxInt(0xffffffff)
				v0.AddArg(x)
				v.AddArg(v0)
				return true
			}
		}
		break
	}
	// match: (ADDV x (MOVVconst <t> [c]))
	// cond: ssa.Is32Bit(c) && !t.IsPtr()
	// result: (ADDVconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpLOONG64MOVVconst {
				continue
			}
			t := v_1.Type
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.Is32Bit(c) && !t.IsPtr()) {
				continue
			}
			v.Reset(ssaop.OpLOONG64ADDVconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADDV x0 x1:(SLLVconst [c] y))
	// cond: x1.Uses == 1 && c > 0 && c <= 4
	// result: (ADDshiftLLV x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpLOONG64SLLVconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(x1.Uses == 1 && c > 0 && c <= 4) {
				continue
			}
			v.Reset(ssaop.OpLOONG64ADDshiftLLV)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (ADDV x (NEGV y))
	// result: (SUBV x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpLOONG64NEGV {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpLOONG64SUBV)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpLOONG64ADDVconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ADDVconst [off1] (MOVVaddr [off2] {sym} ptr))
	// cond: ssa.Is32Bit(off1+int64(off2))
	// result: (MOVVaddr [int32(off1)+int32(off2)] {sym} ptr)
	for {
		off1 := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		if !(ssa.Is32Bit(off1 + int64(off2))) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVaddr)
		v.AuxInt = ssa.Int32ToAuxInt(int32(off1) + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(ptr)
		return true
	}
	// match: (ADDVconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ADDVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [c+d])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c + d)
		return true
	}
	// match: (ADDVconst [c] (ADDVconst [d] x))
	// cond: ssa.Is32Bit(c+d)
	// result: (ADDVconst [c+d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(c + d)) {
			break
		}
		v.Reset(ssaop.OpLOONG64ADDVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (ADDVconst [c] (SUBVconst [d] x))
	// cond: ssa.Is32Bit(c-d)
	// result: (ADDVconst [c-d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64SUBVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(c - d)) {
			break
		}
		v.Reset(ssaop.OpLOONG64ADDVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	// match: (ADDVconst [c] x)
	// cond: ssa.Is32Bit(c) && c&0xffff == 0 && c != 0
	// result: (ADDV16const [c] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if !(ssa.Is32Bit(c) && c&0xffff == 0 && c != 0) {
			break
		}
		v.Reset(ssaop.OpLOONG64ADDV16const)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64ADDshiftLLV(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDshiftLLV x (MOVVconst [c]) [d])
	// cond: ssa.Is12Bit(c<<d)
	// result: (ADDVconst x [c<<d])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.Is12Bit(c << d)) {
			break
		}
		v.Reset(ssaop.OpLOONG64ADDVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c << d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64AND(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AND x (MOVVconst [c]))
	// cond: ssa.Is32Bit(c)
	// result: (ANDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpLOONG64MOVVconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.Is32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpLOONG64ANDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
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
	// match: (AND x (NORconst [0] y))
	// result: (ANDN x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpLOONG64NORconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpLOONG64ANDN)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpLOONG64ANDconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ANDconst [0] _)
	// result: (MOVVconst [0])
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
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
	// match: (ANDconst [0xffff] x)
	// result: (MOVHUreg x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0xffff {
			break
		}
		x := v_0
		v.Reset(ssaop.OpLOONG64MOVHUreg)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [0xffffffff] x)
	// result: (MOVWUreg x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0xffffffff {
			break
		}
		x := v_0
		v.Reset(ssaop.OpLOONG64MOVWUreg)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVVconst [d]))
	// result: (MOVVconst [c&d])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & d)
		return true
	}
	// match: (ANDconst [c] (ANDconst [d] x))
	// result: (ANDconst [c&d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64ANDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpLOONG64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64DIVV(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVV (MOVVconst [c]) (MOVVconst [d]))
	// cond: d != 0
	// result: (MOVVconst [c/d])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c / d)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64DIVVU(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVVU x (MOVVconst [1]))
	// result: x
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (DIVVU x (MOVVconst [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (SRLVconst [ssa.Log64(c)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLVconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
		v.AddArg(x)
		return true
	}
	// match: (DIVVU (MOVVconst [c]) (MOVVconst [d]))
	// cond: d != 0
	// result: (MOVVconst [int64(uint64(c)/uint64(d))])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) / uint64(d)))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64LoweredPanicBoundsCR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsCR [kind] {p} (MOVVconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:p.C, Cy:c}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		p := ssa.AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.OpLOONG64LoweredPanicBoundsCC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: p.C, Cy: c})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64LoweredPanicBoundsRC(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRC [kind] {p} (MOVVconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:c, Cy:p.C}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		p := ssa.AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.OpLOONG64LoweredPanicBoundsCC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: c, Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64LoweredPanicBoundsRR(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRR [kind] x (MOVVconst [c]) mem)
	// result: (LoweredPanicBoundsRC [kind] x {ssa.PanicBoundsC{C:c}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpLOONG64LoweredPanicBoundsRC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: c})
		v.AddArg2(x, mem)
		return true
	}
	// match: (LoweredPanicBoundsRR [kind] (MOVVconst [c]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {ssa.PanicBoundsC{C:c}} y mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64LoweredPanicBoundsCR)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: c})
		v.AddArg2(y, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MASKEQZ(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MASKEQZ (MOVVconst [0]) cond)
	// result: (MOVVconst [0])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (MASKEQZ x (MOVVconst [c]))
	// cond: c == 0
	// result: (MOVVconst [0])
	for {
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(c == 0) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (MASKEQZ x (MOVVconst [c]))
	// cond: c != 0
	// result: x
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(c != 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MASKNEZ(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MASKNEZ (MOVVconst [0]) cond)
	// result: (MOVVconst [0])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVBUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVBUload [off] {sym} ptr (MOVBstore [off] {sym} ptr x _))
	// result: (MOVBUreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVBstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBUload [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBUloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} (ADDshiftLLV [shift] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBUloadidx ptr (SLLVconst <typ.Int64> [shift] idx) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBUloadidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVVconst [int64(ssa.Read8(sym, int64(off)))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Read8(sym, int64(off))))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVBUloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBUloadidx ptr (MOVVconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVBUload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUloadidx (MOVVconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVBUload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVBUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBUreg (SRLVconst [rc] x))
	// cond: rc < 8
	// result: (BSTRPICKV [rc + (7+rc)<<6] x)
	for {
		if v_0.Op != ssaop.OpLOONG64SRLVconst {
			break
		}
		rc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(rc < 8) {
			break
		}
		v.Reset(ssaop.OpLOONG64BSTRPICKV)
		v.AuxInt = ssa.Int64ToAuxInt(rc + (7+rc)<<6)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(SGT _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64SGT {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(SGTU _ _))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64SGTU {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(XOR (MOVVconst [1]) (SGT _ _)))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64XOR {
			break
		}
		_ = x.Args[1]
		x_0 := x.Args[0]
		x_1 := x.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, x_0, x_1 = _i0+1, x_1, x_0 {
			if x_0.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(x_0.AuxInt) != 1 || x_1.Op != ssaop.OpLOONG64SGT {
				continue
			}
			v.CopyOf(x)
			return true
		}
		break
	}
	// match: (MOVBUreg x:(XOR (MOVVconst [1]) (SGTU _ _)))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64XOR {
			break
		}
		_ = x.Args[1]
		x_0 := x.Args[0]
		x_1 := x.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, x_0, x_1 = _i0+1, x_1, x_0 {
			if x_0.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(x_0.AuxInt) != 1 || x_1.Op != ssaop.OpLOONG64SGTU {
				continue
			}
			v.CopyOf(x)
			return true
		}
		break
	}
	// match: (MOVBUreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (SLLVconst [lc] x))
	// cond: lc >= 8
	// result: (MOVVconst [0])
	for {
		if v_0.Op != ssaop.OpLOONG64SLLVconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(lc >= 8) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (MOVBUreg (MOVVconst [c]))
	// result: (MOVVconst [int64(uint8(c))])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint8(c)))
		return true
	}
	// match: (MOVBUreg (ANDconst [c] x))
	// result: (ANDconst [c&0xff] x)
	for {
		if v_0.Op != ssaop.OpLOONG64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpLOONG64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 0xff)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(SRLconst [c] y))
	// cond: c >= 24
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64SRLconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 24) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg x:(ANDconst [c] y))
	// cond: c >= 0 && int64(uint8(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(uint8(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVBload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVBload [off] {sym} ptr (MOVBstore [off] {sym} ptr x _))
	// result: (MOVBreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVBstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBload [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVBload [off] {sym} (ADDshiftLLV [shift] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBloadidx ptr (SLLVconst <typ.Int64> [shift] idx) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBloadidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVVconst [int64(int8(ssa.Read8(sym, int64(off))))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int8(ssa.Read8(sym, int64(off)))))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVBloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBloadidx ptr (MOVVconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVBload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBloadidx (MOVVconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVBload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVBreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBreg x:(MOVBload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (MOVVconst [c]))
	// result: (MOVVconst [int64(int8(c))])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int8(c)))
		return true
	}
	// match: (MOVBreg x:(ANDconst [c] y))
	// cond: c >= 0 && int64(int8(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(int8(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVBstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVBstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBstore [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVBreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVBstore)
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
		if v_1.Op != ssaop.OpLOONG64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVBstore)
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
		if v_1.Op != ssaop.OpLOONG64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} (ADDV ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBstoreidx ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} (ADDshiftLLV [shift] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBstoreidx ptr (SLLVconst <typ.Int64> [shift] idx) val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBstoreidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg4(ptr, v0, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVBstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstoreidx ptr (MOVVconst [c]) val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVBstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstoreidx (MOVVconst [c]) idx val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVBstore [int32(c)] idx val mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVDF(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDF (ABSD (MOVFD x)))
	// result: (ABSF x)
	for {
		if v_0.Op != ssaop.OpLOONG64ABSD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpLOONG64MOVFD {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpLOONG64ABSF)
		v.AddArg(x)
		return true
	}
	// match: (MOVDF (SQRTD (MOVFD x)))
	// result: (SQRTF x)
	for {
		if v_0.Op != ssaop.OpLOONG64SQRTD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpLOONG64MOVFD {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpLOONG64SQRTF)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVDload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVDload [off] {sym} ptr (MOVVstore [off] {sym} ptr val _))
	// result: (MOVVgpfp val)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVgpfp)
		v.AddArg(val)
		return true
	}
	// match: (MOVDload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVDload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVDload [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVDload [off] {sym} (ADDshiftLLV [shift] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDloadidx ptr (SLLVconst <typ.Int64> [shift] idx) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDloadidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVDloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDloadidx ptr (MOVVconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVDload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDloadidx (MOVVconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVDload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVDstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVDstore [off] {sym} ptr (MOVVgpfp val) mem)
	// result: (MOVVstore [off] {sym} ptr val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVgpfp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVDstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVDstore [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} (ADDV ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstoreidx ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} (ADDshiftLLV [shift] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstoreidx ptr (SLLVconst <typ.Int64> [shift] idx) val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDstoreidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg4(ptr, v0, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVDstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstoreidx ptr (MOVVconst [c]) val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVDstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstoreidx (MOVVconst [c]) idx val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVDstore [int32(c)] idx val mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVFload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVFload [off] {sym} ptr (MOVWstore [off] {sym} ptr val _))
	// result: (MOVWgpfp val)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVWstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWgpfp)
		v.AddArg(val)
		return true
	}
	// match: (MOVFload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVFload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVFload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVFload [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVFload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVFloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVFload [off] {sym} (ADDshiftLLV [shift] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVFloadidx ptr (SLLVconst <typ.Int64> [shift] idx) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFloadidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVFloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVFloadidx ptr (MOVVconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVFload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVFloadidx (MOVVconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVFload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVFstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVFstore [off] {sym} ptr (MOVWgpfp val) mem)
	// result: (MOVWstore [off] {sym} ptr val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVWgpfp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVFstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVFstore [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstore [off] {sym} (ADDV ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVFstoreidx ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVFstore [off] {sym} (ADDshiftLLV [shift] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVFstoreidx ptr (SLLVconst <typ.Int64> [shift] idx) val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFstoreidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg4(ptr, v0, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVFstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVFstoreidx ptr (MOVVconst [c]) val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVFstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstoreidx (MOVVconst [c]) idx val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVFstore [int32(c)] idx val mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVHUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVHUload [off] {sym} ptr (MOVHstore [off] {sym} ptr x _))
	// result: (MOVHUreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVHstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHUload [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHUloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (ADDshiftLLV [shift] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHUloadidx ptr (SLLVconst <typ.Int64> [shift] idx) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHUloadidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVVconst [int64(ssa.Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVHUloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHUloadidx ptr (MOVVconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVHUload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUloadidx (MOVVconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVHUload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVHUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHUreg (SRLVconst [rc] x))
	// cond: rc < 16
	// result: (BSTRPICKV [rc + (15+rc)<<6] x)
	for {
		if v_0.Op != ssaop.OpLOONG64SRLVconst {
			break
		}
		rc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(rc < 16) {
			break
		}
		v.Reset(ssaop.OpLOONG64BSTRPICKV)
		v.AuxInt = ssa.Int64ToAuxInt(rc + (15+rc)<<6)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (SLLVconst [lc] x))
	// cond: lc >= 16
	// result: (MOVVconst [0])
	for {
		if v_0.Op != ssaop.OpLOONG64SLLVconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(lc >= 16) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (MOVHUreg (MOVVconst [c]))
	// result: (MOVVconst [int64(uint16(c))])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint16(c)))
		return true
	}
	// match: (MOVHUreg x:(SRLconst [c] y))
	// cond: c >= 16
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64SRLconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 16) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVHUreg x:(ANDconst [c] y))
	// cond: c >= 0 && int64(uint16(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(uint16(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVHload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVHload [off] {sym} ptr (MOVHstore [off] {sym} ptr x _))
	// result: (MOVHreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVHstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHload [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHload [off] {sym} (ADDshiftLLV [shift] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHloadidx ptr (SLLVconst <typ.Int64> [shift] idx) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHloadidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVHload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVVconst [int64(int16(ssa.Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder)))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int16(ssa.Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder))))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVHloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHloadidx ptr (MOVVconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVHload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHloadidx (MOVVconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVHload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVHreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHreg x:(MOVBload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (MOVVconst [c]))
	// result: (MOVVconst [int64(int16(c))])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int16(c)))
		return true
	}
	// match: (MOVHreg x:(ANDconst [c] y))
	// cond: c >= 0 && int64(int16(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(int16(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVHstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVHstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHstore [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHUreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVHstore)
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
		if v_1.Op != ssaop.OpLOONG64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} (ADDV ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstoreidx ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} (ADDshiftLLV [shift] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstoreidx ptr (SLLVconst <typ.Int64> [shift] idx) val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHstoreidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg4(ptr, v0, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVHstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstoreidx ptr (MOVVconst [c]) val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVHstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstoreidx (MOVVconst [c]) idx val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVHstore [int32(c)] idx val mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVVload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVVload [off] {sym} ptr (MOVDstore [off] {sym} ptr val _))
	// result: (MOVVfpgp val)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVDstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVfpgp)
		v.AddArg(val)
		return true
	}
	// match: (MOVVload [off] {sym} ptr (MOVVstore [off] {sym} ptr x _))
	// result: (MOVVreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVVload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVVload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVVload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVVload [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVVload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVVloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVVload [off] {sym} (ADDshiftLLV [shift] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVVloadidx ptr (SLLVconst <typ.Int64> [shift] idx) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVloadidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVVload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVVconst [int64(ssa.Read64(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Read64(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVVloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVVloadidx ptr (MOVVconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVVload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVVloadidx (MOVVconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVVload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVVnop(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVVnop (MOVVconst [c]))
	// result: (MOVVconst [c])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVVreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVVreg x)
	// cond: x.Uses == 1
	// result: (MOVVnop x)
	for {
		x := v_0
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVnop)
		v.AddArg(x)
		return true
	}
	// match: (MOVVreg (MOVVconst [c]))
	// result: (MOVVconst [c])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVVstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVVstore [off] {sym} ptr (MOVVfpgp val) mem)
	// result: (MOVDstore [off] {sym} ptr val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVfpgp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVVstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVVstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVVstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVVstore [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVVstore [off] {sym} (ADDV ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVVstoreidx ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVVstore [off] {sym} (ADDshiftLLV [shift] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVVstoreidx ptr (SLLVconst <typ.Int64> [shift] idx) val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVstoreidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg4(ptr, v0, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVVstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVVstoreidx ptr (MOVVconst [c]) val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVVstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVVstoreidx (MOVVconst [c]) idx val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVVstore [int32(c)] idx val mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVWUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVWUload [off] {sym} ptr (MOVFstore [off] {sym} ptr val _))
	// result: (ZeroExt32to64 (MOVWfpgp <typ.Float32> val))
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVFstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpZeroExt32to64)
		v0 := b.NewValue0(v_1.Pos, ssaop.OpLOONG64MOVWfpgp, typ.Float32)
		v0.AddArg(val)
		v.AddArg(v0)
		return true
	}
	// match: (MOVWUload [off] {sym} ptr (MOVWstore [off] {sym} ptr x _))
	// result: (MOVWUreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVWstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWUload [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWUloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWUload [off] {sym} (ADDshiftLLV [shift] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWUloadidx ptr (SLLVconst <typ.Int64> [shift] idx) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWUloadidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVWUload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVVconst [int64(ssa.Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVWUloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWUloadidx ptr (MOVVconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVWUload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUloadidx (MOVVconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVWUload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVWUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWUreg (SRLVconst [rc] x))
	// cond: rc < 32
	// result: (BSTRPICKV [rc + (31+rc)<<6] x)
	for {
		if v_0.Op != ssaop.OpLOONG64SRLVconst {
			break
		}
		rc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(rc < 32) {
			break
		}
		v.Reset(ssaop.OpLOONG64BSTRPICKV)
		v.AuxInt = ssa.Int64ToAuxInt(rc + (31+rc)<<6)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVWUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVWUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVWUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (SLLVconst [lc] x))
	// cond: lc >= 32
	// result: (MOVVconst [0])
	for {
		if v_0.Op != ssaop.OpLOONG64SLLVconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(lc >= 32) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (MOVWUreg (MOVVconst [c]))
	// result: (MOVVconst [int64(uint32(c))])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint32(c)))
		return true
	}
	// match: (MOVWUreg x:(SRLconst [c] y))
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64SRLconst {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWUreg x:(ANDconst [c] y))
	// cond: c >= 0 && int64(uint32(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(uint32(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVWload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVWload [off] {sym} ptr (MOVWstore [off] {sym} ptr x _))
	// result: (MOVWreg x)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVWstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		x := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWload [off1] {sym} (ADDVconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWload [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (ADDV ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (ADDshiftLLV [shift] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWloadidx ptr (SLLVconst <typ.Int64> [shift] idx) mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWloadidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVVconst [int64(int32(ssa.Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder)))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int32(ssa.Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder))))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVWloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWloadidx ptr (MOVVconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVWload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWloadidx (MOVVconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVWload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVWreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWreg x:(MOVBload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWload _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVWload {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHUloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWloadidx _ _ _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVWloadidx {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVHreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWreg _))
	// result: (MOVVreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64MOVWreg {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (MOVVconst [c]))
	// result: (MOVVconst [int64(int32(c))])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int32(c)))
		return true
	}
	// match: (MOVWreg x:(ANDconst [c] y))
	// cond: c >= 0 && int64(int32(c)) == c
	// result: x
	for {
		x := v_0
		if x.Op != ssaop.OpLOONG64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		if !(c >= 0 && int64(int32(c)) == c) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVWstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (MOVWstore [off] {sym} ptr (MOVWfpgp val) mem)
	// result: (MOVFstore [off] {sym} ptr val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVWfpgp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVFstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDVconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (MOVVaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWstore [off1+int32(off2)] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64MOVVaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWUreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} (ADDV ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstoreidx ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} (ADDshiftLLV [shift] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstoreidx ptr (SLLVconst <typ.Int64> [shift] idx) val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpLOONG64ADDshiftLLV {
			break
		}
		shift := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWstoreidx)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, typ.Int64)
		v0.AuxInt = ssa.Int64ToAuxInt(shift)
		v0.AddArg(idx)
		v.AddArg4(ptr, v0, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MOVWstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreidx ptr (MOVVconst [c]) val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVWstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstoreidx (MOVVconst [c]) idx val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVWstore [int32(c)] idx val mem)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64MULV(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MULV r:(MOVWUreg x) s:(MOVWUreg y))
	// cond: r.Uses == 1 && s.Uses == 1
	// result: (MULWVWU x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			r := v_0
			if r.Op != ssaop.OpLOONG64MOVWUreg {
				continue
			}
			x := r.Args[0]
			s := v_1
			if s.Op != ssaop.OpLOONG64MOVWUreg {
				continue
			}
			y := s.Args[0]
			if !(r.Uses == 1 && s.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpLOONG64MULWVWU)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (MULV r:(MOVWreg x) s:(MOVWreg y))
	// cond: r.Uses == 1 && s.Uses == 1
	// result: (MULWVW x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			r := v_0
			if r.Op != ssaop.OpLOONG64MOVWreg {
				continue
			}
			x := r.Args[0]
			s := v_1
			if s.Op != ssaop.OpLOONG64MOVWreg {
				continue
			}
			y := s.Args[0]
			if !(r.Uses == 1 && s.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpLOONG64MULWVW)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (MULV _ (MOVVconst [0]))
	// result: (MOVVconst [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_1.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				continue
			}
			v.Reset(ssaop.OpLOONG64MOVVconst)
			v.AuxInt = ssa.Int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (MULV x (MOVVconst [1]))
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
				continue
			}
			v.CopyOf(x)
			return true
		}
		break
	}
	// match: (MULV x (MOVVconst [c]))
	// cond: ssa.CanMulStrengthReduce(config, c)
	// result: {ssa.MulStrengthReduce(v, x, c)}
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpLOONG64MOVVconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.CanMulStrengthReduce(config, c)) {
				continue
			}
			v.CopyOf(ssa.MulStrengthReduce(v, x, c))
			return true
		}
		break
	}
	// match: (MULV (MOVVconst [c]) (MOVVconst [d]))
	// result: (MOVVconst [c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64MOVVconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpLOONG64MOVVconst {
				continue
			}
			d := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpLOONG64MOVVconst)
			v.AuxInt = ssa.Int64ToAuxInt(c * d)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpLOONG64NEGV(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (NEGV (SUBV x y))
	// result: (SUBV y x)
	for {
		if v_0.Op != ssaop.OpLOONG64SUBV {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpLOONG64SUBV)
		v.AddArg2(y, x)
		return true
	}
	// match: (NEGV <t> s:(ADDVconst [c] (SUBV x y)))
	// cond: s.Uses == 1 && ssa.Is12Bit(-c)
	// result: (ADDVconst [-c] (SUBV <t> y x))
	for {
		t := v.Type
		s := v_0
		if s.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		c := ssa.AuxIntToInt64(s.AuxInt)
		s_0 := s.Args[0]
		if s_0.Op != ssaop.OpLOONG64SUBV {
			break
		}
		y := s_0.Args[1]
		x := s_0.Args[0]
		if !(s.Uses == 1 && ssa.Is12Bit(-c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64ADDVconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SUBV, t)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (NEGV (NEGV x))
	// result: x
	for {
		if v_0.Op != ssaop.OpLOONG64NEGV {
			break
		}
		x := v_0.Args[0]
		v.CopyOf(x)
		return true
	}
	// match: (NEGV <t> s:(ADDVconst [c] (NEGV x)))
	// cond: s.Uses == 1 && ssa.Is12Bit(-c)
	// result: (ADDVconst [-c] x)
	for {
		s := v_0
		if s.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		c := ssa.AuxIntToInt64(s.AuxInt)
		s_0 := s.Args[0]
		if s_0.Op != ssaop.OpLOONG64NEGV {
			break
		}
		x := s_0.Args[0]
		if !(s.Uses == 1 && ssa.Is12Bit(-c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64ADDVconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	// match: (NEGV (MOVVconst [c]))
	// result: (MOVVconst [-c])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64NOR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NOR x (MOVVconst [c]))
	// cond: ssa.Is32Bit(c)
	// result: (NORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpLOONG64MOVVconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.Is32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpLOONG64NORconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpLOONG64NORconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NORconst [c] (MOVVconst [d]))
	// result: (MOVVconst [^(c|d)])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(^(c | d))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64OR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OR <typ.UInt16> (SRLVconst [8] <typ.UInt16> x) (SLLVconst [8] <typ.UInt16> x))
	// result: (REVB2H x)
	for {
		if v.Type != typ.UInt16 {
			break
		}
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64SRLVconst || v_0.Type != typ.UInt16 || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpLOONG64SLLVconst || v_1.Type != typ.UInt16 || ssa.AuxIntToInt64(v_1.AuxInt) != 8 || x != v_1.Args[0] {
				continue
			}
			v.Reset(ssaop.OpLOONG64REVB2H)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (OR (SRLconst [8] (ANDconst [c1] x)) (SLLconst [8] (ANDconst [c2] x)))
	// cond: uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff
	// result: (REVB2H x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpLOONG64ANDconst {
				continue
			}
			c1 := ssa.AuxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != ssaop.OpLOONG64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 8 {
				continue
			}
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpLOONG64ANDconst {
				continue
			}
			c2 := ssa.AuxIntToInt64(v_1_0.AuxInt)
			if x != v_1_0.Args[0] || !(uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff) {
				continue
			}
			v.Reset(ssaop.OpLOONG64REVB2H)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (OR (SRLVconst [8] (AND (MOVVconst [c1]) x)) (SLLVconst [8] (AND (MOVVconst [c2]) x)))
	// cond: uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff
	// result: (REVB4H x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64SRLVconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpLOONG64AND {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				if v_0_0_0.Op != ssaop.OpLOONG64MOVVconst {
					continue
				}
				c1 := ssa.AuxIntToInt64(v_0_0_0.AuxInt)
				x := v_0_0_1
				if v_1.Op != ssaop.OpLOONG64SLLVconst || ssa.AuxIntToInt64(v_1.AuxInt) != 8 {
					continue
				}
				v_1_0 := v_1.Args[0]
				if v_1_0.Op != ssaop.OpLOONG64AND {
					continue
				}
				_ = v_1_0.Args[1]
				v_1_0_0 := v_1_0.Args[0]
				v_1_0_1 := v_1_0.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, v_1_0_0, v_1_0_1 = _i2+1, v_1_0_1, v_1_0_0 {
					if v_1_0_0.Op != ssaop.OpLOONG64MOVVconst {
						continue
					}
					c2 := ssa.AuxIntToInt64(v_1_0_0.AuxInt)
					if x != v_1_0_1 || !(uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff) {
						continue
					}
					v.Reset(ssaop.OpLOONG64REVB4H)
					v.AddArg(x)
					return true
				}
			}
		}
		break
	}
	// match: (OR (SRLVconst [8] (AND (MOVVconst [c1]) x)) (SLLVconst [8] (ANDconst [c2] x)))
	// cond: uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff
	// result: (REVB4H (ANDconst <x.Type> [0xffffffff] x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64SRLVconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpLOONG64AND {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				if v_0_0_0.Op != ssaop.OpLOONG64MOVVconst {
					continue
				}
				c1 := ssa.AuxIntToInt64(v_0_0_0.AuxInt)
				x := v_0_0_1
				if v_1.Op != ssaop.OpLOONG64SLLVconst || ssa.AuxIntToInt64(v_1.AuxInt) != 8 {
					continue
				}
				v_1_0 := v_1.Args[0]
				if v_1_0.Op != ssaop.OpLOONG64ANDconst {
					continue
				}
				c2 := ssa.AuxIntToInt64(v_1_0.AuxInt)
				if x != v_1_0.Args[0] || !(uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff) {
					continue
				}
				v.Reset(ssaop.OpLOONG64REVB4H)
				v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64ANDconst, x.Type)
				v0.AuxInt = ssa.Int64ToAuxInt(0xffffffff)
				v0.AddArg(x)
				v.AddArg(v0)
				return true
			}
		}
		break
	}
	// match: (OR x (MOVVconst [c]))
	// cond: ssa.Is32Bit(c)
	// result: (ORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpLOONG64MOVVconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.Is32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpLOONG64ORconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
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
	// match: (OR x (NORconst [0] y))
	// result: (ORN x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpLOONG64NORconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpLOONG64ORN)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpLOONG64ORN(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORN x (MOVVconst [-1]))
	// result: x
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64ORconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
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
	// match: (ORconst [-1] _)
	// result: (MOVVconst [-1])
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != -1 {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	// match: (ORconst [c] (MOVVconst [d]))
	// result: (MOVVconst [c|d])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c | d)
		return true
	}
	// match: (ORconst [c] (ORconst [d] x))
	// cond: ssa.Is32Bit(c|d)
	// result: (ORconst [c|d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64ORconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(c | d)) {
			break
		}
		v.Reset(ssaop.OpLOONG64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c | d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64REMV(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (REMV (MOVVconst [c]) (MOVVconst [d]))
	// cond: d != 0
	// result: (MOVVconst [c%d])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c % d)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64REMVU(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (REMVU _ (MOVVconst [1]))
	// result: (MOVVconst [0])
	for {
		if v_1.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (REMVU x (MOVVconst [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (ANDconst [c-1] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c - 1)
		v.AddArg(x)
		return true
	}
	// match: (REMVU (MOVVconst [c]) (MOVVconst [d]))
	// cond: d != 0
	// result: (MOVVconst [int64(uint64(c)%uint64(d))])
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) % uint64(d)))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64ROTR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROTR x (MOVVconst [c]))
	// result: (ROTRconst x [c&31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpLOONG64ROTRconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64ROTRV(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROTRV x (MOVVconst [c]))
	// result: (ROTRVconst x [c&63])
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpLOONG64ROTRVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SGT(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SGT (MOVVconst [c]) (NEGV (SUBVconst [d] x)))
	// cond: ssa.Is32Bit(d-c)
	// result: (SGT x (MOVVconst [d-c]))
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpLOONG64NEGV {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpLOONG64SUBVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1_0.AuxInt)
		x := v_1_0.Args[0]
		if !(ssa.Is32Bit(d - c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SGT)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(d - c)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SGT (MOVVconst [c]) x)
	// cond: ssa.Is32Bit(c)
	// result: (SGTconst [c] x)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SGTconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
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
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SGTU(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SGTU (MOVVconst [c]) x)
	// cond: ssa.Is32Bit(c)
	// result: (SGTUconst [c] x)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SGTUconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
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
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SGTUconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SGTUconst [c] (MOVVconst [d]))
	// cond: uint64(c)>uint64(d)
	// result: (MOVVconst [1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(uint64(c) > uint64(d)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (MOVVconst [d]))
	// cond: uint64(c)<=uint64(d)
	// result: (MOVVconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(uint64(c) <= uint64(d)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SGTUconst [c] (MOVBUreg _))
	// cond: 0xff < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVBUreg || !(0xff < uint64(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (MOVHUreg _))
	// cond: 0xffff < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVHUreg || !(0xffff < uint64(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (ANDconst [m] _))
	// cond: uint64(m) < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64ANDconst {
			break
		}
		m := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(uint64(m) < uint64(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SGTUconst [c] (SRLVconst _ [d]))
	// cond: 0 < d && d <= 63 && 0xffffffffffffffff>>uint64(d) < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64SRLVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(0 < d && d <= 63 && 0xffffffffffffffff>>uint64(d) < uint64(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SGTconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SGTconst [c] (MOVVconst [d]))
	// cond: c>d
	// result: (MOVVconst [1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(c > d) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVVconst [d]))
	// cond: c<=d
	// result: (MOVVconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(c <= d) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVBreg _))
	// cond: 0x7f < c
	// result: (MOVVconst [1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVBreg || !(0x7f < c) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVBreg _))
	// cond: c <= -0x80
	// result: (MOVVconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVBreg || !(c <= -0x80) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVBUreg _))
	// cond: 0xff < c
	// result: (MOVVconst [1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVBUreg || !(0xff < c) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVBUreg _))
	// cond: c < 0
	// result: (MOVVconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVBUreg || !(c < 0) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVHreg _))
	// cond: 0x7fff < c
	// result: (MOVVconst [1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVHreg || !(0x7fff < c) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVHreg _))
	// cond: c <= -0x8000
	// result: (MOVVconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVHreg || !(c <= -0x8000) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVHUreg _))
	// cond: 0xffff < c
	// result: (MOVVconst [1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVHUreg || !(0xffff < c) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (MOVHUreg _))
	// cond: c < 0
	// result: (MOVVconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVHUreg || !(c < 0) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (MOVWUreg _))
	// cond: c < 0
	// result: (MOVVconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVWUreg || !(c < 0) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SGTconst [c] (ANDconst [m] _))
	// cond: 0 <= m && m < c
	// result: (MOVVconst [1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64ANDconst {
			break
		}
		m := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(0 <= m && m < c) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	// match: (SGTconst [c] (SRLVconst _ [d]))
	// cond: 0 <= c && 0 < d && d <= 63 && 0xffffffffffffffff>>uint64(d) < uint64(c)
	// result: (MOVVconst [1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64SRLVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(0 <= c && 0 < d && d <= 63 && 0xffffffffffffffff>>uint64(d) < uint64(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLL _ (MOVVconst [c]))
	// cond: uint64(c)>=32
	// result: (MOVVconst [0])
	for {
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SLL x (MOVVconst [c]))
	// cond: uint64(c) >=0 && uint64(c) <=31
	// result: (SLLconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 0 && uint64(c) <= 31) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SLL x (ANDconst [31] y))
	// result: (SLL x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64ANDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 31 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpLOONG64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SLLV(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLLV _ (MOVVconst [c]))
	// cond: uint64(c)>=64
	// result: (MOVVconst [0])
	for {
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SLLV x (MOVVconst [c]))
	// result: (SLLVconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpLOONG64SLLVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SLLV x (ANDconst [63] y))
	// result: (SLLV x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64ANDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 63 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SLLVconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SLLVconst <t> [c] (ADDV x x))
	// cond: c < t.Size() * 8 - 1
	// result: (SLLVconst [c+1] x)
	for {
		t := v.Type
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || !(c < t.Size()*8-1) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c + 1)
		v.AddArg(x)
		return true
	}
	// match: (SLLVconst <t> [c] (ADDV x x))
	// cond: c >= t.Size() * 8 - 1
	// result: (MOVVconst [0])
	for {
		t := v.Type
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || !(c >= t.Size()*8-1) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SLLVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [d<<uint64(c)])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(d << uint64(c))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SLLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SLLconst <t> [c] (ADDV x x))
	// cond: c < t.Size() * 8 - 1
	// result: (SLLconst [c+1] x)
	for {
		t := v.Type
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || !(c < t.Size()*8-1) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLconst)
		v.AuxInt = ssa.Int64ToAuxInt(c + 1)
		v.AddArg(x)
		return true
	}
	// match: (SLLconst <t> [c] (ADDV x x))
	// cond: c >= t.Size() * 8 - 1
	// result: (MOVVconst [0])
	for {
		t := v.Type
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64ADDV {
			break
		}
		x := v_0.Args[1]
		if x != v_0.Args[0] || !(c >= t.Size()*8-1) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRA x (MOVVconst [c]))
	// cond: uint64(c)>=32
	// result: (SRAconst x [31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAconst)
		v.AuxInt = ssa.Int64ToAuxInt(31)
		v.AddArg(x)
		return true
	}
	// match: (SRA x (MOVVconst [c]))
	// cond: uint64(c) >=0 && uint64(c) <=31
	// result: (SRAconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 0 && uint64(c) <= 31) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SRA x (ANDconst [31] y))
	// result: (SRA x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64ANDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 31 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpLOONG64SRA)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SRAV(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRAV x (MOVVconst [c]))
	// cond: uint64(c)>=64
	// result: (SRAVconst x [63])
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAVconst)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v.AddArg(x)
		return true
	}
	// match: (SRAV x (MOVVconst [c]))
	// result: (SRAVconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpLOONG64SRAVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SRAV x (ANDconst [63] y))
	// result: (SRAV x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64ANDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 63 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpLOONG64SRAV)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SRAVconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (SRAVconst [rc] (MOVWreg y))
	// cond: rc >= 0 && rc <= 31
	// result: (SRAconst [int64(rc)] y)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVWreg {
			break
		}
		y := v_0.Args[0]
		if !(rc >= 0 && rc <= 31) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(rc))
		v.AddArg(y)
		return true
	}
	// match: (SRAVconst <t> [rc] (MOVBreg y))
	// cond: rc >= 8
	// result: (SRAVconst [63] (SLLVconst <t> [56] y))
	for {
		t := v.Type
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVBreg {
			break
		}
		y := v_0.Args[0]
		if !(rc >= 8) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAVconst)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, t)
		v0.AuxInt = ssa.Int64ToAuxInt(56)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SRAVconst <t> [rc] (MOVHreg y))
	// cond: rc >= 16
	// result: (SRAVconst [63] (SLLVconst <t> [48] y))
	for {
		t := v.Type
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVHreg {
			break
		}
		y := v_0.Args[0]
		if !(rc >= 16) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAVconst)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, t)
		v0.AuxInt = ssa.Int64ToAuxInt(48)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SRAVconst <t> [rc] (MOVWreg y))
	// cond: rc >= 32
	// result: (SRAconst [31] y)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVWreg {
			break
		}
		y := v_0.Args[0]
		if !(rc >= 32) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAconst)
		v.AuxInt = ssa.Int64ToAuxInt(31)
		v.AddArg(y)
		return true
	}
	// match: (SRAVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [d>>uint64(c)])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(d >> uint64(c))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRL _ (MOVVconst [c]))
	// cond: uint64(c)>=32
	// result: (MOVVconst [0])
	for {
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SRL x (MOVVconst [c]))
	// cond: uint64(c) >=0 && uint64(c) <=31
	// result: (SRLconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 0 && uint64(c) <= 31) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SRL x (ANDconst [31] y))
	// result: (SRL x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64ANDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 31 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpLOONG64SRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SRLV(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRLV _ (MOVVconst [c]))
	// cond: uint64(c)>=64
	// result: (MOVVconst [0])
	for {
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 64) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SRLV x (MOVVconst [c]))
	// result: (SRLVconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpLOONG64SRLVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SRLV x (ANDconst [63] y))
	// result: (SRLV x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64ANDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 63 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpLOONG64SRLV)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SRLVconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SRLVconst [rc] (SLLVconst [lc] x))
	// cond: lc <= rc
	// result: (BSTRPICKV [rc-lc + ((64-lc)-1)<<6] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64SLLVconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc <= rc) {
			break
		}
		v.Reset(ssaop.OpLOONG64BSTRPICKV)
		v.AuxInt = ssa.Int64ToAuxInt(rc - lc + ((64-lc)-1)<<6)
		v.AddArg(x)
		return true
	}
	// match: (SRLVconst [rc] (MOVWUreg x))
	// cond: rc < 32
	// result: (BSTRPICKV [rc + 31<<6] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 32) {
			break
		}
		v.Reset(ssaop.OpLOONG64BSTRPICKV)
		v.AuxInt = ssa.Int64ToAuxInt(rc + 31<<6)
		v.AddArg(x)
		return true
	}
	// match: (SRLVconst [rc] (MOVHUreg x))
	// cond: rc < 16
	// result: (BSTRPICKV [rc + 15<<6] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 16) {
			break
		}
		v.Reset(ssaop.OpLOONG64BSTRPICKV)
		v.AuxInt = ssa.Int64ToAuxInt(rc + 15<<6)
		v.AddArg(x)
		return true
	}
	// match: (SRLVconst [rc] (MOVBUreg x))
	// cond: rc < 8
	// result: (BSTRPICKV [rc + 7<<6] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 8) {
			break
		}
		v.Reset(ssaop.OpLOONG64BSTRPICKV)
		v.AuxInt = ssa.Int64ToAuxInt(rc + 7<<6)
		v.AddArg(x)
		return true
	}
	// match: (SRLVconst [rc] (MOVWUreg y))
	// cond: rc >= 0 && rc <= 31
	// result: (SRLconst [int64(rc)] y)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVWUreg {
			break
		}
		y := v_0.Args[0]
		if !(rc >= 0 && rc <= 31) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(rc))
		v.AddArg(y)
		return true
	}
	// match: (SRLVconst [rc] (MOVWUreg x))
	// cond: rc >= 32
	// result: (MOVVconst [0])
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVWUreg {
			break
		}
		if !(rc >= 32) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SRLVconst [rc] (MOVHUreg x))
	// cond: rc >= 16
	// result: (MOVVconst [0])
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVHUreg {
			break
		}
		if !(rc >= 16) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SRLVconst [rc] (MOVBUreg x))
	// cond: rc >= 8
	// result: (MOVVconst [0])
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVBUreg {
			break
		}
		if !(rc >= 8) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SRLVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [int64(uint64(d)>>uint64(c))])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(d) >> uint64(c)))
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SUBD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBD (MULD x y) z)
	// cond: z.Block.Func.UseFMA(v)
	// result: (FMSUBD x y z)
	for {
		if v_0.Op != ssaop.OpLOONG64MULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		if !(z.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64FMSUBD)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUBD z (MULD x y))
	// cond: z.Block.Func.UseFMA(v)
	// result: (FNMSUBD x y z)
	for {
		z := v_0
		if v_1.Op != ssaop.OpLOONG64MULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(z.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64FNMSUBD)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUBD z (NEGD (MULD x y)))
	// cond: z.Block.Func.UseFMA(v)
	// result: (FMADDD x y z)
	for {
		z := v_0
		if v_1.Op != ssaop.OpLOONG64NEGD {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpLOONG64MULD {
			break
		}
		y := v_1_0.Args[1]
		x := v_1_0.Args[0]
		if !(z.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64FMADDD)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUBD (NEGD (MULD x y)) z)
	// cond: z.Block.Func.UseFMA(v)
	// result: (FNMADDD x y z)
	for {
		if v_0.Op != ssaop.OpLOONG64NEGD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpLOONG64MULD {
			break
		}
		y := v_0_0.Args[1]
		x := v_0_0.Args[0]
		z := v_1
		if !(z.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64FNMADDD)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SUBF(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBF (MULF x y) z)
	// cond: z.Block.Func.UseFMA(v)
	// result: (FMSUBF x y z)
	for {
		if v_0.Op != ssaop.OpLOONG64MULF {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		if !(z.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64FMSUBF)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUBF z (MULF x y))
	// cond: z.Block.Func.UseFMA(v)
	// result: (FNMSUBF x y z)
	for {
		z := v_0
		if v_1.Op != ssaop.OpLOONG64MULF {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(z.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64FNMSUBF)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUBF z (NEGF (MULF x y)))
	// cond: z.Block.Func.UseFMA(v)
	// result: (FMADDF x y z)
	for {
		z := v_0
		if v_1.Op != ssaop.OpLOONG64NEGF {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpLOONG64MULF {
			break
		}
		y := v_1_0.Args[1]
		x := v_1_0.Args[0]
		if !(z.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64FMADDF)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUBF (NEGF (MULF x y)) z)
	// cond: z.Block.Func.UseFMA(v)
	// result: (FNMADDF x y z)
	for {
		if v_0.Op != ssaop.OpLOONG64NEGF {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpLOONG64MULF {
			break
		}
		y := v_0_0.Args[1]
		x := v_0_0.Args[0]
		z := v_1
		if !(z.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64FNMADDF)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SUBV(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBV x (MOVVconst [c]))
	// cond: ssa.Is32Bit(c)
	// result: (SUBVconst [c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SUBVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUBV x (NEGV y))
	// result: (ADDV x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpLOONG64NEGV {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpLOONG64ADDV)
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
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SUBV (MOVVconst [0]) x)
	// result: (NEGV x)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpLOONG64NEGV)
		v.AddArg(x)
		return true
	}
	// match: (SUBV (MOVVconst [c]) (NEGV (SUBVconst [d] x)))
	// result: (ADDVconst [c-d] x)
	for {
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpLOONG64NEGV {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpLOONG64SUBVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1_0.AuxInt)
		x := v_1_0.Args[0]
		v.Reset(ssaop.OpLOONG64ADDVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64SUBVconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SUBVconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (SUBVconst [c] (MOVVconst [d]))
	// result: (MOVVconst [d-c])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(d - c)
		return true
	}
	// match: (SUBVconst [c] (SUBVconst [d] x))
	// cond: ssa.Is32Bit(-c-d)
	// result: (ADDVconst [-c-d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64SUBVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(-c - d)) {
			break
		}
		v.Reset(ssaop.OpLOONG64ADDVconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c - d)
		v.AddArg(x)
		return true
	}
	// match: (SUBVconst [c] (ADDVconst [d] x))
	// cond: ssa.Is32Bit(-c+d)
	// result: (ADDVconst [-c+d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64ADDVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(-c + d)) {
			break
		}
		v.Reset(ssaop.OpLOONG64ADDVconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64XOR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (XOR <typ.UInt16> (SRLVconst [8] <typ.UInt16> x) (SLLVconst [8] <typ.UInt16> x))
	// result: (REVB2H x)
	for {
		if v.Type != typ.UInt16 {
			break
		}
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64SRLVconst || v_0.Type != typ.UInt16 || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpLOONG64SLLVconst || v_1.Type != typ.UInt16 || ssa.AuxIntToInt64(v_1.AuxInt) != 8 || x != v_1.Args[0] {
				continue
			}
			v.Reset(ssaop.OpLOONG64REVB2H)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XOR (SRLconst [8] (ANDconst [c1] x)) (SLLconst [8] (ANDconst [c2] x)))
	// cond: uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff
	// result: (REVB2H x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpLOONG64ANDconst {
				continue
			}
			c1 := ssa.AuxIntToInt64(v_0_0.AuxInt)
			x := v_0_0.Args[0]
			if v_1.Op != ssaop.OpLOONG64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 8 {
				continue
			}
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != ssaop.OpLOONG64ANDconst {
				continue
			}
			c2 := ssa.AuxIntToInt64(v_1_0.AuxInt)
			if x != v_1_0.Args[0] || !(uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff) {
				continue
			}
			v.Reset(ssaop.OpLOONG64REVB2H)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XOR (SRLVconst [8] (AND (MOVVconst [c1]) x)) (SLLVconst [8] (AND (MOVVconst [c2]) x)))
	// cond: uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff
	// result: (REVB4H x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64SRLVconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpLOONG64AND {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				if v_0_0_0.Op != ssaop.OpLOONG64MOVVconst {
					continue
				}
				c1 := ssa.AuxIntToInt64(v_0_0_0.AuxInt)
				x := v_0_0_1
				if v_1.Op != ssaop.OpLOONG64SLLVconst || ssa.AuxIntToInt64(v_1.AuxInt) != 8 {
					continue
				}
				v_1_0 := v_1.Args[0]
				if v_1_0.Op != ssaop.OpLOONG64AND {
					continue
				}
				_ = v_1_0.Args[1]
				v_1_0_0 := v_1_0.Args[0]
				v_1_0_1 := v_1_0.Args[1]
				for _i2 := 0; _i2 <= 1; _i2, v_1_0_0, v_1_0_1 = _i2+1, v_1_0_1, v_1_0_0 {
					if v_1_0_0.Op != ssaop.OpLOONG64MOVVconst {
						continue
					}
					c2 := ssa.AuxIntToInt64(v_1_0_0.AuxInt)
					if x != v_1_0_1 || !(uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff) {
						continue
					}
					v.Reset(ssaop.OpLOONG64REVB4H)
					v.AddArg(x)
					return true
				}
			}
		}
		break
	}
	// match: (XOR (SRLVconst [8] (AND (MOVVconst [c1]) x)) (SLLVconst [8] (ANDconst [c2] x)))
	// cond: uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff
	// result: (REVB4H (ANDconst <x.Type> [0xffffffff] x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpLOONG64SRLVconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
				continue
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpLOONG64AND {
				continue
			}
			_ = v_0_0.Args[1]
			v_0_0_0 := v_0_0.Args[0]
			v_0_0_1 := v_0_0.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, v_0_0_0, v_0_0_1 = _i1+1, v_0_0_1, v_0_0_0 {
				if v_0_0_0.Op != ssaop.OpLOONG64MOVVconst {
					continue
				}
				c1 := ssa.AuxIntToInt64(v_0_0_0.AuxInt)
				x := v_0_0_1
				if v_1.Op != ssaop.OpLOONG64SLLVconst || ssa.AuxIntToInt64(v_1.AuxInt) != 8 {
					continue
				}
				v_1_0 := v_1.Args[0]
				if v_1_0.Op != ssaop.OpLOONG64ANDconst {
					continue
				}
				c2 := ssa.AuxIntToInt64(v_1_0.AuxInt)
				if x != v_1_0.Args[0] || !(uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff) {
					continue
				}
				v.Reset(ssaop.OpLOONG64REVB4H)
				v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64ANDconst, x.Type)
				v0.AuxInt = ssa.Int64ToAuxInt(0xffffffff)
				v0.AddArg(x)
				v.AddArg(v0)
				return true
			}
		}
		break
	}
	// match: (XOR x (MOVVconst [c]))
	// cond: ssa.Is32Bit(c)
	// result: (XORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpLOONG64MOVVconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.Is32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpLOONG64XORconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
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
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpLOONG64XORconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
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
	// match: (XORconst [-1] x)
	// result: (NORconst [0] x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.Reset(ssaop.OpLOONG64NORconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg(x)
		return true
	}
	// match: (XORconst [c] (MOVVconst [d]))
	// result: (MOVVconst [c^d])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpLOONG64MOVVconst)
		v.AuxInt = ssa.Int64ToAuxInt(c ^ d)
		return true
	}
	// match: (XORconst [c] (XORconst [d] x))
	// cond: ssa.Is32Bit(c^d)
	// result: (XORconst [c^d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpLOONG64XORconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(ssa.Is32Bit(c ^ d)) {
			break
		}
		v.Reset(ssaop.OpLOONG64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c ^ d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpLeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (XOR (MOVVconst [1]) (SGT (SignExt16to64 x) (SignExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGT, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLeq16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// result: (XOR (MOVVconst [1]) (SGTU (ZeroExt16to64 x) (ZeroExt16to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32 x y)
	// result: (XOR (MOVVconst [1]) (SGT (SignExt32to64 x) (SignExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGT, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLeq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32F x y)
	// result: (FPFlagTrue (CMPGEF y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64CMPGEF, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32U x y)
	// result: (XOR (MOVVconst [1]) (SGTU (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLeq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64 x y)
	// result: (XOR (MOVVconst [1]) (SGT x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGT, typ.Bool)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLeq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64F x y)
	// result: (FPFlagTrue (CMPGED y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64CMPGED, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq64U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64U x y)
	// result: (XOR (MOVVconst [1]) (SGTU x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLeq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (XOR (MOVVconst [1]) (SGT (SignExt8to64 x) (SignExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGT, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLeq8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// result: (XOR (MOVVconst [1]) (SGTU (ZeroExt8to64 x) (ZeroExt8to64 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v1.AddArg2(v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLess16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (SGT (SignExt16to64 y) (SignExt16to64 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGT)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLess16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// result: (SGTU (ZeroExt16to64 y) (ZeroExt16to64 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLess32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32 x y)
	// result: (SGT (SignExt32to64 y) (SignExt32to64 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGT)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLess32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32F x y)
	// result: (FPFlagTrue (CMPGTF y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64CMPGTF, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32U x y)
	// result: (SGTU (ZeroExt32to64 y) (ZeroExt32to64 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLess64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64 x y)
	// result: (SGT y x)
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGT)
		v.AddArg2(y, x)
		return true
	}
}
func rewriteValue_OpLess64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64F x y)
	// result: (FPFlagTrue (CMPGTD y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64FPFlagTrue)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64CMPGTD, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess64U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Less64U x y)
	// result: (SGTU y x)
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v.AddArg2(y, x)
		return true
	}
}
func rewriteValue_OpLess8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (SGT (SignExt8to64 y) (SignExt8to64 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGT)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLess8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// result: (SGTU (ZeroExt8to64 y) (ZeroExt8to64 x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpLoad(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpLOONG64MOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is8BitInt(t) && t.IsSigned())
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is8BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is8BitInt(t) && !t.IsSigned())
	// result: (MOVBUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is8BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is16BitInt(t) && t.IsSigned())
	// result: (MOVHload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is16BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is16BitInt(t) && !t.IsSigned())
	// result: (MOVHUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is16BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVHUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is32BitInt(t) && t.IsSigned())
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitInt(t) && t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is32BitInt(t) && !t.IsSigned())
	// result: (MOVWUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitInt(t) && !t.IsSigned()) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVWUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (ssa.Is64BitInt(t) ||ssa.IsPtr(t))
	// result: (MOVVload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is64BitInt(t) || ssa.IsPtr(t)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is32BitFloat(t)
	// result: (MOVFload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is32BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ssa.Is64BitFloat(t)
	// result: (MOVDload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(ssa.Is64BitFloat(t)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpLocalAddr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LocalAddr <t> {sym} base mem)
	// cond: t.Elem().HasPointers()
	// result: (MOVVaddr {sym} (SPanchored base mem))
	for {
		t := v.Type
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		mem := v_1
		if !(t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVaddr)
		v.Aux = ssa.SymToAux(sym)
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
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		if !(!t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVaddr)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValue_OpLsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLLV <t> x y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpLsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLL)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLL <t> x (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [32]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(32)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLL)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLL <t> x (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [32]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(32)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLL)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLL <t> x y) (SGTU (MOVVconst <typ.UInt64> [32]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpLsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLL)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLL <t> x (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [32]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(32)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh64x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh64x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh64x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLLV <t> x y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpLsh64x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLLV <t> x y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpLsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SLLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SLLV <t> x (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpMod16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// result: (REMV (SignExt16to64 x) (SignExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64REMV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpMod16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (REMVU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64REMVU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpMod32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32 x y)
	// result: (REMV (SignExt32to64 x) (SignExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64REMV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpMod32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32u x y)
	// result: (REMVU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64REMVU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpMod64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod64 x y)
	// result: (REMV x y)
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64REMV)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValue_OpMod8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (REMV (SignExt8to64 x) (SignExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64REMV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpMod8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (REMVU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64REMVU)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpMove(v *ssa.Value) bool {
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
	// result: (MOVBstore dst (MOVBUload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVBUload, typ.UInt8)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (MOVHstore dst (MOVHUload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVHUload, typ.UInt16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [3] dst src mem)
	// result: (MOVBstore [2] dst (MOVBUload [2] src mem) (MOVHstore dst (MOVHUload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVBUload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVHstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVHUload, typ.UInt16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (MOVWstore dst (MOVWUload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWUload, typ.UInt32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [5] dst src mem)
	// result: (MOVBstore [4] dst (MOVBUload [4] src mem) (MOVWstore dst (MOVWUload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 5 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVBUload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWUload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [6] dst src mem)
	// result: (MOVHstore [4] dst (MOVHUload [4] src mem) (MOVWstore dst (MOVWUload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVHUload, typ.UInt16)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWUload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [7] dst src mem)
	// result: (MOVWstore [3] dst (MOVWUload [3] src mem) (MOVWstore dst (MOVWUload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 7 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWUload, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWUload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [8] dst src mem)
	// result: (MOVVstore dst (MOVVload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [9] dst src mem)
	// result: (MOVBstore [8] dst (MOVBUload [8] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 9 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVBUload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [10] dst src mem)
	// result: (MOVHstore [8] dst (MOVHUload [8] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 10 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVHUload, typ.UInt16)
		v0.AuxInt = ssa.Int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [11] dst src mem)
	// result: (MOVWstore [7] dst (MOVWload [7] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 11 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWload, typ.Int32)
		v0.AuxInt = ssa.Int32ToAuxInt(7)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [12] dst src mem)
	// result: (MOVWstore [8] dst (MOVWUload [8] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 12 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWUload, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [13] dst src mem)
	// result: (MOVVstore [5] dst (MOVVload [5] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 13 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(5)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v0.AuxInt = ssa.Int32ToAuxInt(5)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [14] dst src mem)
	// result: (MOVVstore [6] dst (MOVVload [6] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 14 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v0.AuxInt = ssa.Int32ToAuxInt(6)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [15] dst src mem)
	// result: (MOVVstore [7] dst (MOVVload [7] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 15 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v0.AuxInt = ssa.Int32ToAuxInt(7)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [16] dst src mem)
	// result: (MOVVstore [8] dst (MOVVload [8] src mem) (MOVVstore dst (MOVVload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 16 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v0.AuxInt = ssa.Int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 16 && s < 192 && ssa.LogLargeCopyValue(v, s)
	// result: (LoweredMove [s] dst src mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 16 && s < 192 && ssa.LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.OpLOONG64LoweredMove)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s >= 192 && ssa.LogLargeCopyValue(v, s)
	// result: (LoweredMoveLoop [s] dst src mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s >= 192 && ssa.LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.OpLOONG64LoweredMoveLoop)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	return false
}
func rewriteValue_OpNeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// result: (SGTU (XOR (ZeroExt16to32 x) (ZeroExt16to64 y)) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64XOR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValue_OpNeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq32 x y)
	// result: (SGTU (XOR (ZeroExt32to64 x) (ZeroExt32to64 y)) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64XOR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValue_OpNeq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32F x y)
	// result: (FPFlagFalse (CMPEQF x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64FPFlagFalse)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64CMPEQF, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq64 x y)
	// result: (SGTU (XOR x y) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64XOR, typ.UInt64)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpNeq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64F x y)
	// result: (FPFlagFalse (CMPEQD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64FPFlagFalse)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64CMPEQD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// result: (SGTU (XOR (ZeroExt8to64 x) (ZeroExt8to64 y)) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64XOR, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValue_OpNeqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NeqPtr x y)
	// result: (SGTU (XOR x y) (MOVVconst [0]))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64SGTU)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64XOR, typ.UInt64)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpNot(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Not x)
	// result: (XORconst [1] x)
	for {
		x := v_0
		v.Reset(ssaop.OpLOONG64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValue_OpOffPtr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (OffPtr [off] ptr:(SP))
	// result: (MOVVaddr [int32(off)] ptr)
	for {
		off := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		if ptr.Op != ssaop.OpSP {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVaddr)
		v.AuxInt = ssa.Int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// result: (ADDVconst [off] ptr)
	for {
		off := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		v.Reset(ssaop.OpLOONG64ADDVconst)
		v.AuxInt = ssa.Int64ToAuxInt(off)
		v.AddArg(ptr)
		return true
	}
}
func rewriteValue_OpPopCount16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount16 <t> x)
	// result: (MOVWfpgp <t> (VPCNT16 <typ.Float32> (MOVWgpfp <typ.Float32> (ZeroExt16to32 x))))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpLOONG64MOVWfpgp)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64VPCNT16, typ.Float32)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWgpfp, typ.Float32)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(x)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpPopCount32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount32 <t> x)
	// result: (MOVWfpgp <t> (VPCNT32 <typ.Float32> (MOVWgpfp <typ.Float32> x)))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpLOONG64MOVWfpgp)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64VPCNT32, typ.Float32)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWgpfp, typ.Float32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpPopCount64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount64 <t> x)
	// result: (MOVVfpgp <t> (VPCNT64 <typ.Float64> (MOVVgpfp <typ.Float64> x)))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpLOONG64MOVVfpgp)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64VPCNT64, typ.Float64)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVgpfp, typ.Float64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpPrefetchCache(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PrefetchCache addr mem)
	// result: (PRELD addr mem [0])
	for {
		addr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64PRELD)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(addr, mem)
		return true
	}
}
func rewriteValue_OpPrefetchCacheStreamed(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PrefetchCacheStreamed addr mem)
	// result: (PRELDX addr mem [(((512 << 1) + (1 << 12)) << 5) + 2])
	for {
		addr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64PRELDX)
		v.AuxInt = ssa.Int64ToAuxInt((((512 << 1) + (1 << 12)) << 5) + 2)
		v.AddArg2(addr, mem)
		return true
	}
}
func rewriteValue_OpRotateLeft16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVVconst [c]))
	// result: (Or16 (Lsh16x64 <t> x (MOVVconst [c&15])) (Rsh16Ux64 <t> x (MOVVconst [-c&15])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpOr16)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh16x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(c & 15)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh16Ux64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(-c & 15)
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
		v.Reset(ssaop.OpLOONG64ROTR)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLVconst, t)
		v2.AuxInt = ssa.Int64ToAuxInt(16)
		v2.AddArg(v1)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, typ.Int64)
		v3.AddArg(y)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValue_OpRotateLeft32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RotateLeft32 x y)
	// result: (ROTR x (NEGV <y.Type> y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64ROTR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpRotateLeft64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RotateLeft64 x y)
	// result: (ROTRV x (NEGV <y.Type> y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpLOONG64ROTRV)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpRotateLeft8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVVconst [c]))
	// result: (Or8 (Lsh8x64 <t> x (MOVVconst [c&7])) (Rsh8Ux64 <t> x (MOVVconst [-c&7])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpLOONG64MOVVconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpOr8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh8x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(c & 7)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh8Ux64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(-c & 7)
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
		v.Reset(ssaop.OpLOONG64OR)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SLLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64ANDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(7)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64ANDconst, typ.Int64)
		v4.AuxInt = ssa.Int64ToAuxInt(7)
		v5 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, typ.Int64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v2.AddArg2(v3, v4)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValue_OpRsh16Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLV (ZeroExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLV)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRLV <t> (ZeroExt16to64 x) (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
	return false
}
func rewriteValue_OpRsh16Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLV (ZeroExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLV)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRLV <t> (ZeroExt16to64 x) (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
	return false
}
func rewriteValue_OpRsh16Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLV (ZeroExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLV)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRLV <t> (ZeroExt16to64 x) y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, y)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpRsh16Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLV (ZeroExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLV)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRLV <t> (ZeroExt16to64 x) (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
	return false
}
func rewriteValue_OpRsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt16to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt16to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpRsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt16to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt32to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpRsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt16to64 x) (OR <t> (NEGV <t> (SGTU y (MOVVconst <typ.UInt64> [63]))) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(63)
		v3.AddArg2(y, v4)
		v2.AddArg(v3)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpRsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt16to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt16to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt8to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpRsh32Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRL)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32Ux16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRL <t> x (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [32]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(32)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpRsh32Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRL)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32Ux32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRL <t> x (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [32]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(32)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpRsh32Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRL)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32Ux64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRL <t> x y) (SGTU (MOVVconst <typ.UInt64> [32]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(32)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpRsh32Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRL)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32Ux8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRL <t> x (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [32]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(32)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpRsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRA)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA x (OR <t> (NEGV <t> (SGTU (ZeroExt16to64 y) (MOVVconst <typ.UInt64> [31]))) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(31)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg2(v1, v3)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRA)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA x (OR <t> (NEGV <t> (SGTU (ZeroExt32to64 y) (MOVVconst <typ.UInt64> [31]))) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(31)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg2(v1, v3)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRA)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA x (OR <t> (NEGV <t> (SGTU y (MOVVconst <typ.UInt64> [31]))) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(31)
		v2.AddArg2(y, v3)
		v1.AddArg(v2)
		v0.AddArg2(v1, y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRA)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh32x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA x (OR <t> (NEGV <t> (SGTU (ZeroExt8to64 y) (MOVVconst <typ.UInt64> [31]))) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(31)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg2(v1, v3)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh64Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRLV <t> x (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpRsh64Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRLV <t> x (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpRsh64Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRLV <t> x y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpRsh64Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRLV <t> x (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, v1)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpRsh64x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAV x (OR <t> (NEGV <t> (SGTU (ZeroExt16to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(63)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg2(v1, v3)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh64x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAV x (OR <t> (NEGV <t> (SGTU (ZeroExt32to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(63)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg2(v1, v3)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh64x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAV x (OR <t> (NEGV <t> (SGTU y (MOVVconst <typ.UInt64> [63]))) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(63)
		v2.AddArg2(y, v3)
		v1.AddArg(v2)
		v0.AddArg2(v1, y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh64x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAV x y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAV x (OR <t> (NEGV <t> (SGTU (ZeroExt8to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(63)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg2(v1, v3)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh8Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLV (ZeroExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLV)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRLV <t> (ZeroExt8to64 x) (ZeroExt16to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
	return false
}
func rewriteValue_OpRsh8Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLV (ZeroExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLV)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRLV <t> (ZeroExt8to64 x) (ZeroExt32to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
	return false
}
func rewriteValue_OpRsh8Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLV (ZeroExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLV)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRLV <t> (ZeroExt8to64 x) y) (SGTU (MOVVconst <typ.UInt64> [64]) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg2(v3, y)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpRsh8Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRLV (ZeroExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRLV)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (MASKEQZ (SRLV <t> (ZeroExt8to64 x) (ZeroExt8to64 y)) (SGTU (MOVVconst <typ.UInt64> [64]) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64MASKEQZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SRLV, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(64)
		v3.AddArg2(v4, v2)
		v.AddArg2(v0, v3)
		return true
	}
	return false
}
func rewriteValue_OpRsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x16 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt8to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt16to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpRsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt8to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt32to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpRsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt8to64 x) (OR <t> (NEGV <t> (SGTU y (MOVVconst <typ.UInt64> [63]))) y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(63)
		v3.AddArg2(y, v4)
		v2.AddArg(v3)
		v1.AddArg2(v2, y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpRsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt8to64 x) y)
	for {
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x8 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRAV (SignExt8to64 x) (OR <t> (NEGV <t> (SGTU (ZeroExt8to64 y) (MOVVconst <typ.UInt64> [63]))) (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLOONG64SRAV)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64OR, t)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, typ.Bool)
		v4 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v5 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v5.AuxInt = ssa.Int64ToAuxInt(63)
		v3.AddArg2(v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(v2, v4)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpSelect0(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Select0 (Mul64uhilo x y))
	// result: (MULHVU x y)
	for {
		if v_0.Op != ssaop.OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpLOONG64MULHVU)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select0 (Mul64uover x y))
	// result: (MULV x y)
	for {
		if v_0.Op != ssaop.OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpLOONG64MULV)
		v.AddArg2(x, y)
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
		v.Reset(ssaop.OpLOONG64ADDV)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64ADDV, t)
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
		v.Reset(ssaop.OpLOONG64SUBV)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SUBV, t)
		v0.AddArg2(x, y)
		v.AddArg2(v0, c)
		return true
	}
	return false
}
func rewriteValue_OpSelect1(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Mul64uhilo x y))
	// result: (MULV x y)
	for {
		if v_0.Op != ssaop.OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpLOONG64MULV)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select1 (Mul64uover x y))
	// result: (SGTU <typ.Bool> (MULHVU x y) (MOVVconst <typ.UInt64> [0]))
	for {
		if v_0.Op != ssaop.OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpLOONG64SGTU)
		v.Type = typ.Bool
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MULHVU, typ.UInt64)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, v1)
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
		v.Reset(ssaop.OpLOONG64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, t)
		s := b.NewValue0(v.Pos, ssaop.OpLOONG64ADDV, t)
		s.AddArg2(x, y)
		v0.AddArg2(x, s)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64ADDV, t)
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
		v.Reset(ssaop.OpLOONG64OR)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, t)
		s := b.NewValue0(v.Pos, ssaop.OpLOONG64SUBV, t)
		s.AddArg2(x, y)
		v0.AddArg2(s, x)
		v2 := b.NewValue0(v.Pos, ssaop.OpLOONG64SGTU, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpLOONG64SUBV, t)
		v3.AddArg2(s, c)
		v2.AddArg2(v3, s)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValue_OpSelectN(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (SelectN [0] call:(CALLstatic {sym} dst src (MOVVconst [sz]) mem))
	// cond: sz >= 0 && ssa.IsSameCall(sym, "runtime.memmove") && call.Uses == 1 && ssa.IsInlinableMemmove(dst, src, sz, config) && ssa.Clobber(call)
	// result: (Move [sz] dst src mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		call := v_0
		if call.Op != ssaop.OpLOONG64CALLstatic || len(call.Args) != 4 {
			break
		}
		sym := ssa.AuxToCall(call.Aux)
		mem := call.Args[3]
		dst := call.Args[0]
		src := call.Args[1]
		call_2 := call.Args[2]
		if call_2.Op != ssaop.OpLOONG64MOVVconst {
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
func rewriteValue_OpSlicemask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SRAVconst (NEGV <t> x) [63])
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpLOONG64SRAVconst)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64NEGV, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpStore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
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
		v.Reset(ssaop.OpLOONG64MOVBstore)
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
		v.Reset(ssaop.OpLOONG64MOVHstore)
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
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && !t.IsFloat()
	// result: (MOVVstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && !t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && t.IsFloat()
	// result: (MOVFstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVFstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && t.IsFloat()
	// result: (MOVDstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && t.IsFloat()) {
			break
		}
		v.Reset(ssaop.OpLOONG64MOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpZero(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
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
	// match: (Zero [1] ptr mem)
	// result: (MOVBstore ptr (MOVVconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] ptr mem)
	// result: (MOVHstore ptr (MOVVconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [3] ptr mem)
	// result: (MOVBstore [2] ptr (MOVVconst [0]) (MOVHstore ptr (MOVVconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVHstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// result: (MOVWstore ptr (MOVVconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [5] ptr mem)
	// result: (MOVBstore [4] ptr (MOVVconst [0]) (MOVWstore ptr (MOVVconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 5 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [6] ptr mem)
	// result: (MOVHstore [4] ptr (MOVVconst [0]) (MOVWstore ptr (MOVVconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [7] ptr mem)
	// result: (MOVWstore [3] ptr (MOVVconst [0]) (MOVWstore ptr (MOVVconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 7 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [8] {t} ptr mem)
	// result: (MOVVstore ptr (MOVVconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [9] ptr mem)
	// result: (MOVBstore [8] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 9 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [10] ptr mem)
	// result: (MOVHstore [8] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 10 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [11] ptr mem)
	// result: (MOVWstore [7] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 11 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [12] ptr mem)
	// result: (MOVWstore [8] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 12 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [13] ptr mem)
	// result: (MOVVstore [5] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 13 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(5)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [14] ptr mem)
	// result: (MOVVstore [6] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 14 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [15] ptr mem)
	// result: (MOVVstore [7] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 15 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [16] ptr mem)
	// result: (MOVVstore [8] ptr (MOVVconst [0]) (MOVVstore ptr (MOVVconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 16 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpLOONG64MOVVstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpLOONG64MOVVstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s > 16 && s < 192
	// result: (LoweredZero [s] ptr mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s > 16 && s < 192) {
			break
		}
		v.Reset(ssaop.OpLOONG64LoweredZero)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s >= 192
	// result: (LoweredZeroLoop [s] ptr mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s >= 192) {
			break
		}
		v.Reset(ssaop.OpLOONG64LoweredZeroLoop)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func RewriteBlock(b *ssa.Block) bool {
	typ := &b.Func.Config.Types
	switch b.Kind {
	case block.BlockLOONG64BEQ:
		// match: (BEQ (MOVVconst [0]) cond yes no)
		// result: (EQZ cond yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.ResetWithControl(block.BlockLOONG64EQZ, cond)
			return true
		}
		// match: (BEQ cond (MOVVconst [0]) yes no)
		// result: (EQZ cond yes no)
		for b.Controls[1].Op == ssaop.OpLOONG64MOVVconst {
			cond := b.Controls[0]
			v_1 := b.Controls[1]
			if ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockLOONG64EQZ, cond)
			return true
		}
	case block.BlockLOONG64BGE:
		// match: (BGE (MOVVconst [0]) cond yes no)
		// result: (LEZ cond yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.ResetWithControl(block.BlockLOONG64LEZ, cond)
			return true
		}
		// match: (BGE cond (MOVVconst [0]) yes no)
		// result: (GEZ cond yes no)
		for b.Controls[1].Op == ssaop.OpLOONG64MOVVconst {
			cond := b.Controls[0]
			v_1 := b.Controls[1]
			if ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockLOONG64GEZ, cond)
			return true
		}
	case block.BlockLOONG64BGEU:
		// match: (BGEU (MOVVconst [0]) cond yes no)
		// result: (EQZ cond yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.ResetWithControl(block.BlockLOONG64EQZ, cond)
			return true
		}
	case block.BlockLOONG64BLT:
		// match: (BLT (MOVVconst [0]) cond yes no)
		// result: (GTZ cond yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.ResetWithControl(block.BlockLOONG64GTZ, cond)
			return true
		}
		// match: (BLT cond (MOVVconst [0]) yes no)
		// result: (LTZ cond yes no)
		for b.Controls[1].Op == ssaop.OpLOONG64MOVVconst {
			cond := b.Controls[0]
			v_1 := b.Controls[1]
			if ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockLOONG64LTZ, cond)
			return true
		}
	case block.BlockLOONG64BLTU:
		// match: (BLTU (MOVVconst [0]) cond yes no)
		// result: (NEZ cond yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.ResetWithControl(block.BlockLOONG64NEZ, cond)
			return true
		}
	case block.BlockLOONG64BNE:
		// match: (BNE (MOVVconst [0]) cond yes no)
		// result: (NEZ cond yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			cond := b.Controls[1]
			b.ResetWithControl(block.BlockLOONG64NEZ, cond)
			return true
		}
		// match: (BNE cond (MOVVconst [0]) yes no)
		// result: (NEZ cond yes no)
		for b.Controls[1].Op == ssaop.OpLOONG64MOVVconst {
			cond := b.Controls[0]
			v_1 := b.Controls[1]
			if ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockLOONG64NEZ, cond)
			return true
		}
	case block.BlockLOONG64EQZ:
		// match: (EQZ (FPFlagTrue cmp) yes no)
		// result: (FPF cmp yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64FPFlagTrue {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockLOONG64FPF, cmp)
			return true
		}
		// match: (EQZ (FPFlagFalse cmp) yes no)
		// result: (FPT cmp yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64FPFlagFalse {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockLOONG64FPT, cmp)
			return true
		}
		// match: (EQZ (XORconst [1] cmp:(SGT _ _)) yes no)
		// result: (NEZ cmp yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpLOONG64SGT {
				break
			}
			b.ResetWithControl(block.BlockLOONG64NEZ, cmp)
			return true
		}
		// match: (EQZ (XORconst [1] cmp:(SGTU _ _)) yes no)
		// result: (NEZ cmp yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpLOONG64SGTU {
				break
			}
			b.ResetWithControl(block.BlockLOONG64NEZ, cmp)
			return true
		}
		// match: (EQZ (XORconst [1] cmp:(SGTconst _)) yes no)
		// result: (NEZ cmp yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpLOONG64SGTconst {
				break
			}
			b.ResetWithControl(block.BlockLOONG64NEZ, cmp)
			return true
		}
		// match: (EQZ (XORconst [1] cmp:(SGTUconst _)) yes no)
		// result: (NEZ cmp yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpLOONG64SGTUconst {
				break
			}
			b.ResetWithControl(block.BlockLOONG64NEZ, cmp)
			return true
		}
		// match: (EQZ (SGTUconst [1] x) yes no)
		// result: (NEZ x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTUconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockLOONG64NEZ, x)
			return true
		}
		// match: (EQZ (SGTU x (MOVVconst [0])) yes no)
		// result: (EQZ x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTU {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(v_0_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockLOONG64EQZ, x)
			return true
		}
		// match: (EQZ (SGTconst [0] x) yes no)
		// result: (GEZ x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockLOONG64GEZ, x)
			return true
		}
		// match: (EQZ (SGT x (MOVVconst [0])) yes no)
		// result: (LEZ x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGT {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(v_0_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockLOONG64LEZ, x)
			return true
		}
		// match: (EQZ (SGTU (MOVVconst [c]) y) yes no)
		// cond: c >= -2048 && c <= 2047
		// result: (EQZ (SGTUconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTU {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpLOONG64MOVVconst {
				break
			}
			c := ssa.AuxIntToInt64(v_0_0.AuxInt)
			if !(c >= -2048 && c <= 2047) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpLOONG64SGTUconst, typ.Bool)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockLOONG64EQZ, v0)
			return true
		}
		// match: (EQZ (SUBV x y) yes no)
		// result: (BEQ x y yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SUBV {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.ResetWithControl2(block.BlockLOONG64BEQ, x, y)
			return true
		}
		// match: (EQZ (SGT x y) yes no)
		// result: (BGE y x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGT {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.ResetWithControl2(block.BlockLOONG64BGE, y, x)
			return true
		}
		// match: (EQZ (SGTU x y) yes no)
		// result: (BGEU y x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTU {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.ResetWithControl2(block.BlockLOONG64BGEU, y, x)
			return true
		}
		// match: (EQZ (SGTconst [c] y) yes no)
		// result: (BGE y (MOVVconst [c]) yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			y := v_0.Args[0]
			v0 := b.NewValue0(b.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			b.ResetWithControl2(block.BlockLOONG64BGE, y, v0)
			return true
		}
		// match: (EQZ (SGTUconst [c] y) yes no)
		// result: (BGEU y (MOVVconst [c]) yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTUconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			y := v_0.Args[0]
			v0 := b.NewValue0(b.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			b.ResetWithControl2(block.BlockLOONG64BGEU, y, v0)
			return true
		}
		// match: (EQZ (MOVVconst [0]) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (EQZ (MOVVconst [c]) yes no)
		// cond: c != 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(c != 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (EQZ (NEGV x) yes no)
		// result: (EQZ x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64NEGV {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockLOONG64EQZ, x)
			return true
		}
	case block.BlockLOONG64GEZ:
		// match: (GEZ (MOVVconst [c]) yes no)
		// cond: c >= 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(c >= 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GEZ (MOVVconst [c]) yes no)
		// cond: c < 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(c < 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockLOONG64GTZ:
		// match: (GTZ (MOVVconst [c]) yes no)
		// cond: c > 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(c > 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GTZ (MOVVconst [c]) yes no)
		// cond: c <= 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(c <= 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockIf:
		// match: (If cond yes no)
		// result: (NEZ (MOVBUreg <typ.UInt64> cond) yes no)
		for {
			cond := b.Controls[0]
			v0 := b.NewValue0(cond.Pos, ssaop.OpLOONG64MOVBUreg, typ.UInt64)
			v0.AddArg(cond)
			b.ResetWithControl(block.BlockLOONG64NEZ, v0)
			return true
		}
	case block.BlockJumpTable:
		// match: (JumpTable idx)
		// result: (JUMPTABLE {ssa.MakeJumpTableSym(b)} idx (MOVVaddr <typ.Uintptr> {ssa.MakeJumpTableSym(b)} (SB)))
		for {
			idx := b.Controls[0]
			v0 := b.NewValue0(b.Pos, ssaop.OpLOONG64MOVVaddr, typ.Uintptr)
			v0.Aux = ssa.SymToAux(ssa.MakeJumpTableSym(b))
			v1 := b.NewValue0(b.Pos, ssaop.OpSB, typ.Uintptr)
			v0.AddArg(v1)
			b.ResetWithControl2(block.BlockLOONG64JUMPTABLE, idx, v0)
			b.Aux = ssa.SymToAux(ssa.MakeJumpTableSym(b))
			return true
		}
	case block.BlockLOONG64LEZ:
		// match: (LEZ (MOVVconst [c]) yes no)
		// cond: c <= 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(c <= 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LEZ (MOVVconst [c]) yes no)
		// cond: c > 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(c > 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockLOONG64LTZ:
		// match: (LTZ (MOVVconst [c]) yes no)
		// cond: c < 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(c < 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LTZ (MOVVconst [c]) yes no)
		// cond: c >= 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(c >= 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockLOONG64NEZ:
		// match: (NEZ (FPFlagTrue cmp) yes no)
		// result: (FPT cmp yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64FPFlagTrue {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockLOONG64FPT, cmp)
			return true
		}
		// match: (NEZ (FPFlagFalse cmp) yes no)
		// result: (FPF cmp yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64FPFlagFalse {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockLOONG64FPF, cmp)
			return true
		}
		// match: (NEZ (XORconst [1] cmp:(SGT _ _)) yes no)
		// result: (EQZ cmp yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpLOONG64SGT {
				break
			}
			b.ResetWithControl(block.BlockLOONG64EQZ, cmp)
			return true
		}
		// match: (NEZ (XORconst [1] cmp:(SGTU _ _)) yes no)
		// result: (EQZ cmp yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpLOONG64SGTU {
				break
			}
			b.ResetWithControl(block.BlockLOONG64EQZ, cmp)
			return true
		}
		// match: (NEZ (XORconst [1] cmp:(SGTconst _)) yes no)
		// result: (EQZ cmp yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpLOONG64SGTconst {
				break
			}
			b.ResetWithControl(block.BlockLOONG64EQZ, cmp)
			return true
		}
		// match: (NEZ (XORconst [1] cmp:(SGTUconst _)) yes no)
		// result: (EQZ cmp yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64XORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			cmp := v_0.Args[0]
			if cmp.Op != ssaop.OpLOONG64SGTUconst {
				break
			}
			b.ResetWithControl(block.BlockLOONG64EQZ, cmp)
			return true
		}
		// match: (NEZ (SGTUconst [1] x) yes no)
		// result: (EQZ x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTUconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockLOONG64EQZ, x)
			return true
		}
		// match: (NEZ (SGTU x (MOVVconst [0])) yes no)
		// result: (NEZ x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTU {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(v_0_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockLOONG64NEZ, x)
			return true
		}
		// match: (NEZ (SGTconst [0] x) yes no)
		// result: (LTZ x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockLOONG64LTZ, x)
			return true
		}
		// match: (NEZ (SGT x (MOVVconst [0])) yes no)
		// result: (GTZ x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGT {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.OpLOONG64MOVVconst || ssa.AuxIntToInt64(v_0_1.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockLOONG64GTZ, x)
			return true
		}
		// match: (NEZ (SGTU (MOVVconst [c]) y) yes no)
		// cond: c >= -2048 && c <= 2047
		// result: (NEZ (SGTUconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTU {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpLOONG64MOVVconst {
				break
			}
			c := ssa.AuxIntToInt64(v_0_0.AuxInt)
			if !(c >= -2048 && c <= 2047) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpLOONG64SGTUconst, typ.Bool)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockLOONG64NEZ, v0)
			return true
		}
		// match: (NEZ (SUBV x y) yes no)
		// result: (BNE x y yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SUBV {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.ResetWithControl2(block.BlockLOONG64BNE, x, y)
			return true
		}
		// match: (NEZ (SGT x y) yes no)
		// result: (BLT y x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGT {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.ResetWithControl2(block.BlockLOONG64BLT, y, x)
			return true
		}
		// match: (NEZ (SGTU x y) yes no)
		// result: (BLTU y x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTU {
			v_0 := b.Controls[0]
			y := v_0.Args[1]
			x := v_0.Args[0]
			b.ResetWithControl2(block.BlockLOONG64BLTU, y, x)
			return true
		}
		// match: (NEZ (SGTconst [c] y) yes no)
		// result: (BLT y (MOVVconst [c]) yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			y := v_0.Args[0]
			v0 := b.NewValue0(b.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			b.ResetWithControl2(block.BlockLOONG64BLT, y, v0)
			return true
		}
		// match: (NEZ (SGTUconst [c] y) yes no)
		// result: (BLTU y (MOVVconst [c]) yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64SGTUconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			y := v_0.Args[0]
			v0 := b.NewValue0(b.Pos, ssaop.OpLOONG64MOVVconst, typ.UInt64)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			b.ResetWithControl2(block.BlockLOONG64BLTU, y, v0)
			return true
		}
		// match: (NEZ (MOVVconst [0]) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (NEZ (MOVVconst [c]) yes no)
		// cond: c != 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64MOVVconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(c != 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (NEZ (NEGV x) yes no)
		// result: (NEZ x yes no)
		for b.Controls[0].Op == ssaop.OpLOONG64NEGV {
			v_0 := b.Controls[0]
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockLOONG64NEZ, x)
			return true
		}
	}
	return false
}
