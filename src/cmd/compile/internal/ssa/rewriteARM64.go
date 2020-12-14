// Code generated from gen/ARM64.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

import "cmd/compile/internal/types"

func rewriteValueARM64(v *Value) bool {
	switch v.Op {
	case OpARM64ADCSflags:
		return rewriteValueARM64_OpARM64ADCSflags(v)
	case OpARM64ADD:
		return rewriteValueARM64_OpARM64ADD(v)
	case OpARM64ADDconst:
		return rewriteValueARM64_OpARM64ADDconst(v)
	case OpARM64ADDshiftLL:
		return rewriteValueARM64_OpARM64ADDshiftLL(v)
	case OpARM64ADDshiftRA:
		return rewriteValueARM64_OpARM64ADDshiftRA(v)
	case OpARM64ADDshiftRL:
		return rewriteValueARM64_OpARM64ADDshiftRL(v)
	case OpARM64AND:
		return rewriteValueARM64_OpARM64AND(v)
	case OpARM64ANDconst:
		return rewriteValueARM64_OpARM64ANDconst(v)
	case OpARM64ANDshiftLL:
		return rewriteValueARM64_OpARM64ANDshiftLL(v)
	case OpARM64ANDshiftRA:
		return rewriteValueARM64_OpARM64ANDshiftRA(v)
	case OpARM64ANDshiftRL:
		return rewriteValueARM64_OpARM64ANDshiftRL(v)
	case OpARM64BIC:
		return rewriteValueARM64_OpARM64BIC(v)
	case OpARM64BICshiftLL:
		return rewriteValueARM64_OpARM64BICshiftLL(v)
	case OpARM64BICshiftRA:
		return rewriteValueARM64_OpARM64BICshiftRA(v)
	case OpARM64BICshiftRL:
		return rewriteValueARM64_OpARM64BICshiftRL(v)
	case OpARM64CMN:
		return rewriteValueARM64_OpARM64CMN(v)
	case OpARM64CMNW:
		return rewriteValueARM64_OpARM64CMNW(v)
	case OpARM64CMNWconst:
		return rewriteValueARM64_OpARM64CMNWconst(v)
	case OpARM64CMNconst:
		return rewriteValueARM64_OpARM64CMNconst(v)
	case OpARM64CMNshiftLL:
		return rewriteValueARM64_OpARM64CMNshiftLL(v)
	case OpARM64CMNshiftRA:
		return rewriteValueARM64_OpARM64CMNshiftRA(v)
	case OpARM64CMNshiftRL:
		return rewriteValueARM64_OpARM64CMNshiftRL(v)
	case OpARM64CMP:
		return rewriteValueARM64_OpARM64CMP(v)
	case OpARM64CMPW:
		return rewriteValueARM64_OpARM64CMPW(v)
	case OpARM64CMPWconst:
		return rewriteValueARM64_OpARM64CMPWconst(v)
	case OpARM64CMPconst:
		return rewriteValueARM64_OpARM64CMPconst(v)
	case OpARM64CMPshiftLL:
		return rewriteValueARM64_OpARM64CMPshiftLL(v)
	case OpARM64CMPshiftRA:
		return rewriteValueARM64_OpARM64CMPshiftRA(v)
	case OpARM64CMPshiftRL:
		return rewriteValueARM64_OpARM64CMPshiftRL(v)
	case OpARM64CSEL:
		return rewriteValueARM64_OpARM64CSEL(v)
	case OpARM64CSEL0:
		return rewriteValueARM64_OpARM64CSEL0(v)
	case OpARM64DIV:
		return rewriteValueARM64_OpARM64DIV(v)
	case OpARM64DIVW:
		return rewriteValueARM64_OpARM64DIVW(v)
	case OpARM64EON:
		return rewriteValueARM64_OpARM64EON(v)
	case OpARM64EONshiftLL:
		return rewriteValueARM64_OpARM64EONshiftLL(v)
	case OpARM64EONshiftRA:
		return rewriteValueARM64_OpARM64EONshiftRA(v)
	case OpARM64EONshiftRL:
		return rewriteValueARM64_OpARM64EONshiftRL(v)
	case OpARM64Equal:
		return rewriteValueARM64_OpARM64Equal(v)
	case OpARM64FADDD:
		return rewriteValueARM64_OpARM64FADDD(v)
	case OpARM64FADDS:
		return rewriteValueARM64_OpARM64FADDS(v)
	case OpARM64FCMPD:
		return rewriteValueARM64_OpARM64FCMPD(v)
	case OpARM64FCMPS:
		return rewriteValueARM64_OpARM64FCMPS(v)
	case OpARM64FMOVDfpgp:
		return rewriteValueARM64_OpARM64FMOVDfpgp(v)
	case OpARM64FMOVDgpfp:
		return rewriteValueARM64_OpARM64FMOVDgpfp(v)
	case OpARM64FMOVDload:
		return rewriteValueARM64_OpARM64FMOVDload(v)
	case OpARM64FMOVDloadidx:
		return rewriteValueARM64_OpARM64FMOVDloadidx(v)
	case OpARM64FMOVDstore:
		return rewriteValueARM64_OpARM64FMOVDstore(v)
	case OpARM64FMOVDstoreidx:
		return rewriteValueARM64_OpARM64FMOVDstoreidx(v)
	case OpARM64FMOVSload:
		return rewriteValueARM64_OpARM64FMOVSload(v)
	case OpARM64FMOVSloadidx:
		return rewriteValueARM64_OpARM64FMOVSloadidx(v)
	case OpARM64FMOVSstore:
		return rewriteValueARM64_OpARM64FMOVSstore(v)
	case OpARM64FMOVSstoreidx:
		return rewriteValueARM64_OpARM64FMOVSstoreidx(v)
	case OpARM64FMULD:
		return rewriteValueARM64_OpARM64FMULD(v)
	case OpARM64FMULS:
		return rewriteValueARM64_OpARM64FMULS(v)
	case OpARM64FNEGD:
		return rewriteValueARM64_OpARM64FNEGD(v)
	case OpARM64FNEGS:
		return rewriteValueARM64_OpARM64FNEGS(v)
	case OpARM64FNMULD:
		return rewriteValueARM64_OpARM64FNMULD(v)
	case OpARM64FNMULS:
		return rewriteValueARM64_OpARM64FNMULS(v)
	case OpARM64FSUBD:
		return rewriteValueARM64_OpARM64FSUBD(v)
	case OpARM64FSUBS:
		return rewriteValueARM64_OpARM64FSUBS(v)
	case OpARM64GreaterEqual:
		return rewriteValueARM64_OpARM64GreaterEqual(v)
	case OpARM64GreaterEqualF:
		return rewriteValueARM64_OpARM64GreaterEqualF(v)
	case OpARM64GreaterEqualU:
		return rewriteValueARM64_OpARM64GreaterEqualU(v)
	case OpARM64GreaterThan:
		return rewriteValueARM64_OpARM64GreaterThan(v)
	case OpARM64GreaterThanF:
		return rewriteValueARM64_OpARM64GreaterThanF(v)
	case OpARM64GreaterThanU:
		return rewriteValueARM64_OpARM64GreaterThanU(v)
	case OpARM64LessEqual:
		return rewriteValueARM64_OpARM64LessEqual(v)
	case OpARM64LessEqualF:
		return rewriteValueARM64_OpARM64LessEqualF(v)
	case OpARM64LessEqualU:
		return rewriteValueARM64_OpARM64LessEqualU(v)
	case OpARM64LessThan:
		return rewriteValueARM64_OpARM64LessThan(v)
	case OpARM64LessThanF:
		return rewriteValueARM64_OpARM64LessThanF(v)
	case OpARM64LessThanU:
		return rewriteValueARM64_OpARM64LessThanU(v)
	case OpARM64MADD:
		return rewriteValueARM64_OpARM64MADD(v)
	case OpARM64MADDW:
		return rewriteValueARM64_OpARM64MADDW(v)
	case OpARM64MNEG:
		return rewriteValueARM64_OpARM64MNEG(v)
	case OpARM64MNEGW:
		return rewriteValueARM64_OpARM64MNEGW(v)
	case OpARM64MOD:
		return rewriteValueARM64_OpARM64MOD(v)
	case OpARM64MODW:
		return rewriteValueARM64_OpARM64MODW(v)
	case OpARM64MOVBUload:
		return rewriteValueARM64_OpARM64MOVBUload(v)
	case OpARM64MOVBUloadidx:
		return rewriteValueARM64_OpARM64MOVBUloadidx(v)
	case OpARM64MOVBUreg:
		return rewriteValueARM64_OpARM64MOVBUreg(v)
	case OpARM64MOVBload:
		return rewriteValueARM64_OpARM64MOVBload(v)
	case OpARM64MOVBloadidx:
		return rewriteValueARM64_OpARM64MOVBloadidx(v)
	case OpARM64MOVBreg:
		return rewriteValueARM64_OpARM64MOVBreg(v)
	case OpARM64MOVBstore:
		return rewriteValueARM64_OpARM64MOVBstore(v)
	case OpARM64MOVBstoreidx:
		return rewriteValueARM64_OpARM64MOVBstoreidx(v)
	case OpARM64MOVBstorezero:
		return rewriteValueARM64_OpARM64MOVBstorezero(v)
	case OpARM64MOVBstorezeroidx:
		return rewriteValueARM64_OpARM64MOVBstorezeroidx(v)
	case OpARM64MOVDload:
		return rewriteValueARM64_OpARM64MOVDload(v)
	case OpARM64MOVDloadidx:
		return rewriteValueARM64_OpARM64MOVDloadidx(v)
	case OpARM64MOVDloadidx8:
		return rewriteValueARM64_OpARM64MOVDloadidx8(v)
	case OpARM64MOVDreg:
		return rewriteValueARM64_OpARM64MOVDreg(v)
	case OpARM64MOVDstore:
		return rewriteValueARM64_OpARM64MOVDstore(v)
	case OpARM64MOVDstoreidx:
		return rewriteValueARM64_OpARM64MOVDstoreidx(v)
	case OpARM64MOVDstoreidx8:
		return rewriteValueARM64_OpARM64MOVDstoreidx8(v)
	case OpARM64MOVDstorezero:
		return rewriteValueARM64_OpARM64MOVDstorezero(v)
	case OpARM64MOVDstorezeroidx:
		return rewriteValueARM64_OpARM64MOVDstorezeroidx(v)
	case OpARM64MOVDstorezeroidx8:
		return rewriteValueARM64_OpARM64MOVDstorezeroidx8(v)
	case OpARM64MOVHUload:
		return rewriteValueARM64_OpARM64MOVHUload(v)
	case OpARM64MOVHUloadidx:
		return rewriteValueARM64_OpARM64MOVHUloadidx(v)
	case OpARM64MOVHUloadidx2:
		return rewriteValueARM64_OpARM64MOVHUloadidx2(v)
	case OpARM64MOVHUreg:
		return rewriteValueARM64_OpARM64MOVHUreg(v)
	case OpARM64MOVHload:
		return rewriteValueARM64_OpARM64MOVHload(v)
	case OpARM64MOVHloadidx:
		return rewriteValueARM64_OpARM64MOVHloadidx(v)
	case OpARM64MOVHloadidx2:
		return rewriteValueARM64_OpARM64MOVHloadidx2(v)
	case OpARM64MOVHreg:
		return rewriteValueARM64_OpARM64MOVHreg(v)
	case OpARM64MOVHstore:
		return rewriteValueARM64_OpARM64MOVHstore(v)
	case OpARM64MOVHstoreidx:
		return rewriteValueARM64_OpARM64MOVHstoreidx(v)
	case OpARM64MOVHstoreidx2:
		return rewriteValueARM64_OpARM64MOVHstoreidx2(v)
	case OpARM64MOVHstorezero:
		return rewriteValueARM64_OpARM64MOVHstorezero(v)
	case OpARM64MOVHstorezeroidx:
		return rewriteValueARM64_OpARM64MOVHstorezeroidx(v)
	case OpARM64MOVHstorezeroidx2:
		return rewriteValueARM64_OpARM64MOVHstorezeroidx2(v)
	case OpARM64MOVQstorezero:
		return rewriteValueARM64_OpARM64MOVQstorezero(v)
	case OpARM64MOVWUload:
		return rewriteValueARM64_OpARM64MOVWUload(v)
	case OpARM64MOVWUloadidx:
		return rewriteValueARM64_OpARM64MOVWUloadidx(v)
	case OpARM64MOVWUloadidx4:
		return rewriteValueARM64_OpARM64MOVWUloadidx4(v)
	case OpARM64MOVWUreg:
		return rewriteValueARM64_OpARM64MOVWUreg(v)
	case OpARM64MOVWload:
		return rewriteValueARM64_OpARM64MOVWload(v)
	case OpARM64MOVWloadidx:
		return rewriteValueARM64_OpARM64MOVWloadidx(v)
	case OpARM64MOVWloadidx4:
		return rewriteValueARM64_OpARM64MOVWloadidx4(v)
	case OpARM64MOVWreg:
		return rewriteValueARM64_OpARM64MOVWreg(v)
	case OpARM64MOVWstore:
		return rewriteValueARM64_OpARM64MOVWstore(v)
	case OpARM64MOVWstoreidx:
		return rewriteValueARM64_OpARM64MOVWstoreidx(v)
	case OpARM64MOVWstoreidx4:
		return rewriteValueARM64_OpARM64MOVWstoreidx4(v)
	case OpARM64MOVWstorezero:
		return rewriteValueARM64_OpARM64MOVWstorezero(v)
	case OpARM64MOVWstorezeroidx:
		return rewriteValueARM64_OpARM64MOVWstorezeroidx(v)
	case OpARM64MOVWstorezeroidx4:
		return rewriteValueARM64_OpARM64MOVWstorezeroidx4(v)
	case OpARM64MSUB:
		return rewriteValueARM64_OpARM64MSUB(v)
	case OpARM64MSUBW:
		return rewriteValueARM64_OpARM64MSUBW(v)
	case OpARM64MUL:
		return rewriteValueARM64_OpARM64MUL(v)
	case OpARM64MULW:
		return rewriteValueARM64_OpARM64MULW(v)
	case OpARM64MVN:
		return rewriteValueARM64_OpARM64MVN(v)
	case OpARM64MVNshiftLL:
		return rewriteValueARM64_OpARM64MVNshiftLL(v)
	case OpARM64MVNshiftRA:
		return rewriteValueARM64_OpARM64MVNshiftRA(v)
	case OpARM64MVNshiftRL:
		return rewriteValueARM64_OpARM64MVNshiftRL(v)
	case OpARM64NEG:
		return rewriteValueARM64_OpARM64NEG(v)
	case OpARM64NEGshiftLL:
		return rewriteValueARM64_OpARM64NEGshiftLL(v)
	case OpARM64NEGshiftRA:
		return rewriteValueARM64_OpARM64NEGshiftRA(v)
	case OpARM64NEGshiftRL:
		return rewriteValueARM64_OpARM64NEGshiftRL(v)
	case OpARM64NotEqual:
		return rewriteValueARM64_OpARM64NotEqual(v)
	case OpARM64OR:
		return rewriteValueARM64_OpARM64OR(v)
	case OpARM64ORN:
		return rewriteValueARM64_OpARM64ORN(v)
	case OpARM64ORNshiftLL:
		return rewriteValueARM64_OpARM64ORNshiftLL(v)
	case OpARM64ORNshiftRA:
		return rewriteValueARM64_OpARM64ORNshiftRA(v)
	case OpARM64ORNshiftRL:
		return rewriteValueARM64_OpARM64ORNshiftRL(v)
	case OpARM64ORconst:
		return rewriteValueARM64_OpARM64ORconst(v)
	case OpARM64ORshiftLL:
		return rewriteValueARM64_OpARM64ORshiftLL(v)
	case OpARM64ORshiftRA:
		return rewriteValueARM64_OpARM64ORshiftRA(v)
	case OpARM64ORshiftRL:
		return rewriteValueARM64_OpARM64ORshiftRL(v)
	case OpARM64RORWconst:
		return rewriteValueARM64_OpARM64RORWconst(v)
	case OpARM64RORconst:
		return rewriteValueARM64_OpARM64RORconst(v)
	case OpARM64SBCSflags:
		return rewriteValueARM64_OpARM64SBCSflags(v)
	case OpARM64SLL:
		return rewriteValueARM64_OpARM64SLL(v)
	case OpARM64SLLconst:
		return rewriteValueARM64_OpARM64SLLconst(v)
	case OpARM64SRA:
		return rewriteValueARM64_OpARM64SRA(v)
	case OpARM64SRAconst:
		return rewriteValueARM64_OpARM64SRAconst(v)
	case OpARM64SRL:
		return rewriteValueARM64_OpARM64SRL(v)
	case OpARM64SRLconst:
		return rewriteValueARM64_OpARM64SRLconst(v)
	case OpARM64STP:
		return rewriteValueARM64_OpARM64STP(v)
	case OpARM64SUB:
		return rewriteValueARM64_OpARM64SUB(v)
	case OpARM64SUBconst:
		return rewriteValueARM64_OpARM64SUBconst(v)
	case OpARM64SUBshiftLL:
		return rewriteValueARM64_OpARM64SUBshiftLL(v)
	case OpARM64SUBshiftRA:
		return rewriteValueARM64_OpARM64SUBshiftRA(v)
	case OpARM64SUBshiftRL:
		return rewriteValueARM64_OpARM64SUBshiftRL(v)
	case OpARM64TST:
		return rewriteValueARM64_OpARM64TST(v)
	case OpARM64TSTW:
		return rewriteValueARM64_OpARM64TSTW(v)
	case OpARM64TSTWconst:
		return rewriteValueARM64_OpARM64TSTWconst(v)
	case OpARM64TSTconst:
		return rewriteValueARM64_OpARM64TSTconst(v)
	case OpARM64TSTshiftLL:
		return rewriteValueARM64_OpARM64TSTshiftLL(v)
	case OpARM64TSTshiftRA:
		return rewriteValueARM64_OpARM64TSTshiftRA(v)
	case OpARM64TSTshiftRL:
		return rewriteValueARM64_OpARM64TSTshiftRL(v)
	case OpARM64UBFIZ:
		return rewriteValueARM64_OpARM64UBFIZ(v)
	case OpARM64UBFX:
		return rewriteValueARM64_OpARM64UBFX(v)
	case OpARM64UDIV:
		return rewriteValueARM64_OpARM64UDIV(v)
	case OpARM64UDIVW:
		return rewriteValueARM64_OpARM64UDIVW(v)
	case OpARM64UMOD:
		return rewriteValueARM64_OpARM64UMOD(v)
	case OpARM64UMODW:
		return rewriteValueARM64_OpARM64UMODW(v)
	case OpARM64XOR:
		return rewriteValueARM64_OpARM64XOR(v)
	case OpARM64XORconst:
		return rewriteValueARM64_OpARM64XORconst(v)
	case OpARM64XORshiftLL:
		return rewriteValueARM64_OpARM64XORshiftLL(v)
	case OpARM64XORshiftRA:
		return rewriteValueARM64_OpARM64XORshiftRA(v)
	case OpARM64XORshiftRL:
		return rewriteValueARM64_OpARM64XORshiftRL(v)
	case OpAbs:
		v.Op = OpARM64FABSD
		return true
	case OpAdd16:
		v.Op = OpARM64ADD
		return true
	case OpAdd32:
		v.Op = OpARM64ADD
		return true
	case OpAdd32F:
		v.Op = OpARM64FADDS
		return true
	case OpAdd64:
		v.Op = OpARM64ADD
		return true
	case OpAdd64F:
		v.Op = OpARM64FADDD
		return true
	case OpAdd8:
		v.Op = OpARM64ADD
		return true
	case OpAddPtr:
		v.Op = OpARM64ADD
		return true
	case OpAddr:
		return rewriteValueARM64_OpAddr(v)
	case OpAnd16:
		v.Op = OpARM64AND
		return true
	case OpAnd32:
		v.Op = OpARM64AND
		return true
	case OpAnd64:
		v.Op = OpARM64AND
		return true
	case OpAnd8:
		v.Op = OpARM64AND
		return true
	case OpAndB:
		v.Op = OpARM64AND
		return true
	case OpAtomicAdd32:
		v.Op = OpARM64LoweredAtomicAdd32
		return true
	case OpAtomicAdd32Variant:
		v.Op = OpARM64LoweredAtomicAdd32Variant
		return true
	case OpAtomicAdd64:
		v.Op = OpARM64LoweredAtomicAdd64
		return true
	case OpAtomicAdd64Variant:
		v.Op = OpARM64LoweredAtomicAdd64Variant
		return true
	case OpAtomicAnd32:
		return rewriteValueARM64_OpAtomicAnd32(v)
	case OpAtomicAnd32Variant:
		return rewriteValueARM64_OpAtomicAnd32Variant(v)
	case OpAtomicAnd8:
		return rewriteValueARM64_OpAtomicAnd8(v)
	case OpAtomicAnd8Variant:
		return rewriteValueARM64_OpAtomicAnd8Variant(v)
	case OpAtomicCompareAndSwap32:
		v.Op = OpARM64LoweredAtomicCas32
		return true
	case OpAtomicCompareAndSwap32Variant:
		v.Op = OpARM64LoweredAtomicCas32Variant
		return true
	case OpAtomicCompareAndSwap64:
		v.Op = OpARM64LoweredAtomicCas64
		return true
	case OpAtomicCompareAndSwap64Variant:
		v.Op = OpARM64LoweredAtomicCas64Variant
		return true
	case OpAtomicExchange32:
		v.Op = OpARM64LoweredAtomicExchange32
		return true
	case OpAtomicExchange32Variant:
		v.Op = OpARM64LoweredAtomicExchange32Variant
		return true
	case OpAtomicExchange64:
		v.Op = OpARM64LoweredAtomicExchange64
		return true
	case OpAtomicExchange64Variant:
		v.Op = OpARM64LoweredAtomicExchange64Variant
		return true
	case OpAtomicLoad32:
		v.Op = OpARM64LDARW
		return true
	case OpAtomicLoad64:
		v.Op = OpARM64LDAR
		return true
	case OpAtomicLoad8:
		v.Op = OpARM64LDARB
		return true
	case OpAtomicLoadPtr:
		v.Op = OpARM64LDAR
		return true
	case OpAtomicOr32:
		return rewriteValueARM64_OpAtomicOr32(v)
	case OpAtomicOr32Variant:
		return rewriteValueARM64_OpAtomicOr32Variant(v)
	case OpAtomicOr8:
		return rewriteValueARM64_OpAtomicOr8(v)
	case OpAtomicOr8Variant:
		return rewriteValueARM64_OpAtomicOr8Variant(v)
	case OpAtomicStore32:
		v.Op = OpARM64STLRW
		return true
	case OpAtomicStore64:
		v.Op = OpARM64STLR
		return true
	case OpAtomicStore8:
		v.Op = OpARM64STLRB
		return true
	case OpAtomicStorePtrNoWB:
		v.Op = OpARM64STLR
		return true
	case OpAvg64u:
		return rewriteValueARM64_OpAvg64u(v)
	case OpBitLen32:
		return rewriteValueARM64_OpBitLen32(v)
	case OpBitLen64:
		return rewriteValueARM64_OpBitLen64(v)
	case OpBitRev16:
		return rewriteValueARM64_OpBitRev16(v)
	case OpBitRev32:
		v.Op = OpARM64RBITW
		return true
	case OpBitRev64:
		v.Op = OpARM64RBIT
		return true
	case OpBitRev8:
		return rewriteValueARM64_OpBitRev8(v)
	case OpBswap32:
		v.Op = OpARM64REVW
		return true
	case OpBswap64:
		v.Op = OpARM64REV
		return true
	case OpCeil:
		v.Op = OpARM64FRINTPD
		return true
	case OpClosureCall:
		v.Op = OpARM64CALLclosure
		return true
	case OpCom16:
		v.Op = OpARM64MVN
		return true
	case OpCom32:
		v.Op = OpARM64MVN
		return true
	case OpCom64:
		v.Op = OpARM64MVN
		return true
	case OpCom8:
		v.Op = OpARM64MVN
		return true
	case OpCondSelect:
		return rewriteValueARM64_OpCondSelect(v)
	case OpConst16:
		return rewriteValueARM64_OpConst16(v)
	case OpConst32:
		return rewriteValueARM64_OpConst32(v)
	case OpConst32F:
		return rewriteValueARM64_OpConst32F(v)
	case OpConst64:
		return rewriteValueARM64_OpConst64(v)
	case OpConst64F:
		return rewriteValueARM64_OpConst64F(v)
	case OpConst8:
		return rewriteValueARM64_OpConst8(v)
	case OpConstBool:
		return rewriteValueARM64_OpConstBool(v)
	case OpConstNil:
		return rewriteValueARM64_OpConstNil(v)
	case OpCtz16:
		return rewriteValueARM64_OpCtz16(v)
	case OpCtz16NonZero:
		v.Op = OpCtz32
		return true
	case OpCtz32:
		return rewriteValueARM64_OpCtz32(v)
	case OpCtz32NonZero:
		v.Op = OpCtz32
		return true
	case OpCtz64:
		return rewriteValueARM64_OpCtz64(v)
	case OpCtz64NonZero:
		v.Op = OpCtz64
		return true
	case OpCtz8:
		return rewriteValueARM64_OpCtz8(v)
	case OpCtz8NonZero:
		v.Op = OpCtz32
		return true
	case OpCvt32Fto32:
		v.Op = OpARM64FCVTZSSW
		return true
	case OpCvt32Fto32U:
		v.Op = OpARM64FCVTZUSW
		return true
	case OpCvt32Fto64:
		v.Op = OpARM64FCVTZSS
		return true
	case OpCvt32Fto64F:
		v.Op = OpARM64FCVTSD
		return true
	case OpCvt32Fto64U:
		v.Op = OpARM64FCVTZUS
		return true
	case OpCvt32Uto32F:
		v.Op = OpARM64UCVTFWS
		return true
	case OpCvt32Uto64F:
		v.Op = OpARM64UCVTFWD
		return true
	case OpCvt32to32F:
		v.Op = OpARM64SCVTFWS
		return true
	case OpCvt32to64F:
		v.Op = OpARM64SCVTFWD
		return true
	case OpCvt64Fto32:
		v.Op = OpARM64FCVTZSDW
		return true
	case OpCvt64Fto32F:
		v.Op = OpARM64FCVTDS
		return true
	case OpCvt64Fto32U:
		v.Op = OpARM64FCVTZUDW
		return true
	case OpCvt64Fto64:
		v.Op = OpARM64FCVTZSD
		return true
	case OpCvt64Fto64U:
		v.Op = OpARM64FCVTZUD
		return true
	case OpCvt64Uto32F:
		v.Op = OpARM64UCVTFS
		return true
	case OpCvt64Uto64F:
		v.Op = OpARM64UCVTFD
		return true
	case OpCvt64to32F:
		v.Op = OpARM64SCVTFS
		return true
	case OpCvt64to64F:
		v.Op = OpARM64SCVTFD
		return true
	case OpCvtBoolToUint8:
		v.Op = OpCopy
		return true
	case OpDiv16:
		return rewriteValueARM64_OpDiv16(v)
	case OpDiv16u:
		return rewriteValueARM64_OpDiv16u(v)
	case OpDiv32:
		return rewriteValueARM64_OpDiv32(v)
	case OpDiv32F:
		v.Op = OpARM64FDIVS
		return true
	case OpDiv32u:
		v.Op = OpARM64UDIVW
		return true
	case OpDiv64:
		return rewriteValueARM64_OpDiv64(v)
	case OpDiv64F:
		v.Op = OpARM64FDIVD
		return true
	case OpDiv64u:
		v.Op = OpARM64UDIV
		return true
	case OpDiv8:
		return rewriteValueARM64_OpDiv8(v)
	case OpDiv8u:
		return rewriteValueARM64_OpDiv8u(v)
	case OpEq16:
		return rewriteValueARM64_OpEq16(v)
	case OpEq32:
		return rewriteValueARM64_OpEq32(v)
	case OpEq32F:
		return rewriteValueARM64_OpEq32F(v)
	case OpEq64:
		return rewriteValueARM64_OpEq64(v)
	case OpEq64F:
		return rewriteValueARM64_OpEq64F(v)
	case OpEq8:
		return rewriteValueARM64_OpEq8(v)
	case OpEqB:
		return rewriteValueARM64_OpEqB(v)
	case OpEqPtr:
		return rewriteValueARM64_OpEqPtr(v)
	case OpFMA:
		return rewriteValueARM64_OpFMA(v)
	case OpFloor:
		v.Op = OpARM64FRINTMD
		return true
	case OpGetCallerPC:
		v.Op = OpARM64LoweredGetCallerPC
		return true
	case OpGetCallerSP:
		v.Op = OpARM64LoweredGetCallerSP
		return true
	case OpGetClosurePtr:
		v.Op = OpARM64LoweredGetClosurePtr
		return true
	case OpHmul32:
		return rewriteValueARM64_OpHmul32(v)
	case OpHmul32u:
		return rewriteValueARM64_OpHmul32u(v)
	case OpHmul64:
		v.Op = OpARM64MULH
		return true
	case OpHmul64u:
		v.Op = OpARM64UMULH
		return true
	case OpInterCall:
		v.Op = OpARM64CALLinter
		return true
	case OpIsInBounds:
		return rewriteValueARM64_OpIsInBounds(v)
	case OpIsNonNil:
		return rewriteValueARM64_OpIsNonNil(v)
	case OpIsSliceInBounds:
		return rewriteValueARM64_OpIsSliceInBounds(v)
	case OpLeq16:
		return rewriteValueARM64_OpLeq16(v)
	case OpLeq16U:
		return rewriteValueARM64_OpLeq16U(v)
	case OpLeq32:
		return rewriteValueARM64_OpLeq32(v)
	case OpLeq32F:
		return rewriteValueARM64_OpLeq32F(v)
	case OpLeq32U:
		return rewriteValueARM64_OpLeq32U(v)
	case OpLeq64:
		return rewriteValueARM64_OpLeq64(v)
	case OpLeq64F:
		return rewriteValueARM64_OpLeq64F(v)
	case OpLeq64U:
		return rewriteValueARM64_OpLeq64U(v)
	case OpLeq8:
		return rewriteValueARM64_OpLeq8(v)
	case OpLeq8U:
		return rewriteValueARM64_OpLeq8U(v)
	case OpLess16:
		return rewriteValueARM64_OpLess16(v)
	case OpLess16U:
		return rewriteValueARM64_OpLess16U(v)
	case OpLess32:
		return rewriteValueARM64_OpLess32(v)
	case OpLess32F:
		return rewriteValueARM64_OpLess32F(v)
	case OpLess32U:
		return rewriteValueARM64_OpLess32U(v)
	case OpLess64:
		return rewriteValueARM64_OpLess64(v)
	case OpLess64F:
		return rewriteValueARM64_OpLess64F(v)
	case OpLess64U:
		return rewriteValueARM64_OpLess64U(v)
	case OpLess8:
		return rewriteValueARM64_OpLess8(v)
	case OpLess8U:
		return rewriteValueARM64_OpLess8U(v)
	case OpLoad:
		return rewriteValueARM64_OpLoad(v)
	case OpLocalAddr:
		return rewriteValueARM64_OpLocalAddr(v)
	case OpLsh16x16:
		return rewriteValueARM64_OpLsh16x16(v)
	case OpLsh16x32:
		return rewriteValueARM64_OpLsh16x32(v)
	case OpLsh16x64:
		return rewriteValueARM64_OpLsh16x64(v)
	case OpLsh16x8:
		return rewriteValueARM64_OpLsh16x8(v)
	case OpLsh32x16:
		return rewriteValueARM64_OpLsh32x16(v)
	case OpLsh32x32:
		return rewriteValueARM64_OpLsh32x32(v)
	case OpLsh32x64:
		return rewriteValueARM64_OpLsh32x64(v)
	case OpLsh32x8:
		return rewriteValueARM64_OpLsh32x8(v)
	case OpLsh64x16:
		return rewriteValueARM64_OpLsh64x16(v)
	case OpLsh64x32:
		return rewriteValueARM64_OpLsh64x32(v)
	case OpLsh64x64:
		return rewriteValueARM64_OpLsh64x64(v)
	case OpLsh64x8:
		return rewriteValueARM64_OpLsh64x8(v)
	case OpLsh8x16:
		return rewriteValueARM64_OpLsh8x16(v)
	case OpLsh8x32:
		return rewriteValueARM64_OpLsh8x32(v)
	case OpLsh8x64:
		return rewriteValueARM64_OpLsh8x64(v)
	case OpLsh8x8:
		return rewriteValueARM64_OpLsh8x8(v)
	case OpMod16:
		return rewriteValueARM64_OpMod16(v)
	case OpMod16u:
		return rewriteValueARM64_OpMod16u(v)
	case OpMod32:
		return rewriteValueARM64_OpMod32(v)
	case OpMod32u:
		v.Op = OpARM64UMODW
		return true
	case OpMod64:
		return rewriteValueARM64_OpMod64(v)
	case OpMod64u:
		v.Op = OpARM64UMOD
		return true
	case OpMod8:
		return rewriteValueARM64_OpMod8(v)
	case OpMod8u:
		return rewriteValueARM64_OpMod8u(v)
	case OpMove:
		return rewriteValueARM64_OpMove(v)
	case OpMul16:
		v.Op = OpARM64MULW
		return true
	case OpMul32:
		v.Op = OpARM64MULW
		return true
	case OpMul32F:
		v.Op = OpARM64FMULS
		return true
	case OpMul64:
		v.Op = OpARM64MUL
		return true
	case OpMul64F:
		v.Op = OpARM64FMULD
		return true
	case OpMul64uhilo:
		v.Op = OpARM64LoweredMuluhilo
		return true
	case OpMul8:
		v.Op = OpARM64MULW
		return true
	case OpNeg16:
		v.Op = OpARM64NEG
		return true
	case OpNeg32:
		v.Op = OpARM64NEG
		return true
	case OpNeg32F:
		v.Op = OpARM64FNEGS
		return true
	case OpNeg64:
		v.Op = OpARM64NEG
		return true
	case OpNeg64F:
		v.Op = OpARM64FNEGD
		return true
	case OpNeg8:
		v.Op = OpARM64NEG
		return true
	case OpNeq16:
		return rewriteValueARM64_OpNeq16(v)
	case OpNeq32:
		return rewriteValueARM64_OpNeq32(v)
	case OpNeq32F:
		return rewriteValueARM64_OpNeq32F(v)
	case OpNeq64:
		return rewriteValueARM64_OpNeq64(v)
	case OpNeq64F:
		return rewriteValueARM64_OpNeq64F(v)
	case OpNeq8:
		return rewriteValueARM64_OpNeq8(v)
	case OpNeqB:
		v.Op = OpARM64XOR
		return true
	case OpNeqPtr:
		return rewriteValueARM64_OpNeqPtr(v)
	case OpNilCheck:
		v.Op = OpARM64LoweredNilCheck
		return true
	case OpNot:
		return rewriteValueARM64_OpNot(v)
	case OpOffPtr:
		return rewriteValueARM64_OpOffPtr(v)
	case OpOr16:
		v.Op = OpARM64OR
		return true
	case OpOr32:
		v.Op = OpARM64OR
		return true
	case OpOr64:
		v.Op = OpARM64OR
		return true
	case OpOr8:
		v.Op = OpARM64OR
		return true
	case OpOrB:
		v.Op = OpARM64OR
		return true
	case OpPanicBounds:
		return rewriteValueARM64_OpPanicBounds(v)
	case OpPopCount16:
		return rewriteValueARM64_OpPopCount16(v)
	case OpPopCount32:
		return rewriteValueARM64_OpPopCount32(v)
	case OpPopCount64:
		return rewriteValueARM64_OpPopCount64(v)
	case OpRotateLeft16:
		return rewriteValueARM64_OpRotateLeft16(v)
	case OpRotateLeft32:
		return rewriteValueARM64_OpRotateLeft32(v)
	case OpRotateLeft64:
		return rewriteValueARM64_OpRotateLeft64(v)
	case OpRotateLeft8:
		return rewriteValueARM64_OpRotateLeft8(v)
	case OpRound:
		v.Op = OpARM64FRINTAD
		return true
	case OpRound32F:
		v.Op = OpARM64LoweredRound32F
		return true
	case OpRound64F:
		v.Op = OpARM64LoweredRound64F
		return true
	case OpRoundToEven:
		v.Op = OpARM64FRINTND
		return true
	case OpRsh16Ux16:
		return rewriteValueARM64_OpRsh16Ux16(v)
	case OpRsh16Ux32:
		return rewriteValueARM64_OpRsh16Ux32(v)
	case OpRsh16Ux64:
		return rewriteValueARM64_OpRsh16Ux64(v)
	case OpRsh16Ux8:
		return rewriteValueARM64_OpRsh16Ux8(v)
	case OpRsh16x16:
		return rewriteValueARM64_OpRsh16x16(v)
	case OpRsh16x32:
		return rewriteValueARM64_OpRsh16x32(v)
	case OpRsh16x64:
		return rewriteValueARM64_OpRsh16x64(v)
	case OpRsh16x8:
		return rewriteValueARM64_OpRsh16x8(v)
	case OpRsh32Ux16:
		return rewriteValueARM64_OpRsh32Ux16(v)
	case OpRsh32Ux32:
		return rewriteValueARM64_OpRsh32Ux32(v)
	case OpRsh32Ux64:
		return rewriteValueARM64_OpRsh32Ux64(v)
	case OpRsh32Ux8:
		return rewriteValueARM64_OpRsh32Ux8(v)
	case OpRsh32x16:
		return rewriteValueARM64_OpRsh32x16(v)
	case OpRsh32x32:
		return rewriteValueARM64_OpRsh32x32(v)
	case OpRsh32x64:
		return rewriteValueARM64_OpRsh32x64(v)
	case OpRsh32x8:
		return rewriteValueARM64_OpRsh32x8(v)
	case OpRsh64Ux16:
		return rewriteValueARM64_OpRsh64Ux16(v)
	case OpRsh64Ux32:
		return rewriteValueARM64_OpRsh64Ux32(v)
	case OpRsh64Ux64:
		return rewriteValueARM64_OpRsh64Ux64(v)
	case OpRsh64Ux8:
		return rewriteValueARM64_OpRsh64Ux8(v)
	case OpRsh64x16:
		return rewriteValueARM64_OpRsh64x16(v)
	case OpRsh64x32:
		return rewriteValueARM64_OpRsh64x32(v)
	case OpRsh64x64:
		return rewriteValueARM64_OpRsh64x64(v)
	case OpRsh64x8:
		return rewriteValueARM64_OpRsh64x8(v)
	case OpRsh8Ux16:
		return rewriteValueARM64_OpRsh8Ux16(v)
	case OpRsh8Ux32:
		return rewriteValueARM64_OpRsh8Ux32(v)
	case OpRsh8Ux64:
		return rewriteValueARM64_OpRsh8Ux64(v)
	case OpRsh8Ux8:
		return rewriteValueARM64_OpRsh8Ux8(v)
	case OpRsh8x16:
		return rewriteValueARM64_OpRsh8x16(v)
	case OpRsh8x32:
		return rewriteValueARM64_OpRsh8x32(v)
	case OpRsh8x64:
		return rewriteValueARM64_OpRsh8x64(v)
	case OpRsh8x8:
		return rewriteValueARM64_OpRsh8x8(v)
	case OpSelect0:
		return rewriteValueARM64_OpSelect0(v)
	case OpSelect1:
		return rewriteValueARM64_OpSelect1(v)
	case OpSignExt16to32:
		v.Op = OpARM64MOVHreg
		return true
	case OpSignExt16to64:
		v.Op = OpARM64MOVHreg
		return true
	case OpSignExt32to64:
		v.Op = OpARM64MOVWreg
		return true
	case OpSignExt8to16:
		v.Op = OpARM64MOVBreg
		return true
	case OpSignExt8to32:
		v.Op = OpARM64MOVBreg
		return true
	case OpSignExt8to64:
		v.Op = OpARM64MOVBreg
		return true
	case OpSlicemask:
		return rewriteValueARM64_OpSlicemask(v)
	case OpSqrt:
		v.Op = OpARM64FSQRTD
		return true
	case OpStaticCall:
		v.Op = OpARM64CALLstatic
		return true
	case OpStore:
		return rewriteValueARM64_OpStore(v)
	case OpSub16:
		v.Op = OpARM64SUB
		return true
	case OpSub32:
		v.Op = OpARM64SUB
		return true
	case OpSub32F:
		v.Op = OpARM64FSUBS
		return true
	case OpSub64:
		v.Op = OpARM64SUB
		return true
	case OpSub64F:
		v.Op = OpARM64FSUBD
		return true
	case OpSub8:
		v.Op = OpARM64SUB
		return true
	case OpSubPtr:
		v.Op = OpARM64SUB
		return true
	case OpTrunc:
		v.Op = OpARM64FRINTZD
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
		v.Op = OpARM64LoweredWB
		return true
	case OpXor16:
		v.Op = OpARM64XOR
		return true
	case OpXor32:
		v.Op = OpARM64XOR
		return true
	case OpXor64:
		v.Op = OpARM64XOR
		return true
	case OpXor8:
		v.Op = OpARM64XOR
		return true
	case OpZero:
		return rewriteValueARM64_OpZero(v)
	case OpZeroExt16to32:
		v.Op = OpARM64MOVHUreg
		return true
	case OpZeroExt16to64:
		v.Op = OpARM64MOVHUreg
		return true
	case OpZeroExt32to64:
		v.Op = OpARM64MOVWUreg
		return true
	case OpZeroExt8to16:
		v.Op = OpARM64MOVBUreg
		return true
	case OpZeroExt8to32:
		v.Op = OpARM64MOVBUreg
		return true
	case OpZeroExt8to64:
		v.Op = OpARM64MOVBUreg
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADCSflags(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADCSflags x y (Select1 <types.TypeFlags> (ADDSconstflags [-1] (ADCzerocarry <typ.UInt64> c))))
	// result: (ADCSflags x y c)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpSelect1 || v_2.Type != types.TypeFlags {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpARM64ADDSconstflags || auxIntToInt64(v_2_0.AuxInt) != -1 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpARM64ADCzerocarry || v_2_0_0.Type != typ.UInt64 {
			break
		}
		c := v_2_0_0.Args[0]
		v.reset(OpARM64ADCSflags)
		v.AddArg3(x, y, c)
		return true
	}
	// match: (ADCSflags x y (Select1 <types.TypeFlags> (ADDSconstflags [-1] (MOVDconst [0]))))
	// result: (ADDSflags x y)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpSelect1 || v_2.Type != types.TypeFlags {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpARM64ADDSconstflags || auxIntToInt64(v_2_0.AuxInt) != -1 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_2_0_0.AuxInt) != 0 {
			break
		}
		v.reset(OpARM64ADDSflags)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADD x (MOVDconst [c]))
	// result: (ADDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64ADDconst)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADD a l:(MUL x y))
	// cond: l.Uses==1 && clobber(l)
	// result: (MADD a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			l := v_1
			if l.Op != OpARM64MUL {
				continue
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1 && clobber(l)) {
				continue
			}
			v.reset(OpARM64MADD)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (ADD a l:(MNEG x y))
	// cond: l.Uses==1 && clobber(l)
	// result: (MSUB a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			l := v_1
			if l.Op != OpARM64MNEG {
				continue
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1 && clobber(l)) {
				continue
			}
			v.reset(OpARM64MSUB)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (ADD a l:(MULW x y))
	// cond: a.Type.Size() != 8 && l.Uses==1 && clobber(l)
	// result: (MADDW a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			l := v_1
			if l.Op != OpARM64MULW {
				continue
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(a.Type.Size() != 8 && l.Uses == 1 && clobber(l)) {
				continue
			}
			v.reset(OpARM64MADDW)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (ADD a l:(MNEGW x y))
	// cond: a.Type.Size() != 8 && l.Uses==1 && clobber(l)
	// result: (MSUBW a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			l := v_1
			if l.Op != OpARM64MNEGW {
				continue
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(a.Type.Size() != 8 && l.Uses == 1 && clobber(l)) {
				continue
			}
			v.reset(OpARM64MSUBW)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (ADD x (NEG y))
	// result: (SUB x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64NEG {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpARM64SUB)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADD x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ADDshiftLL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SLLconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64ADDshiftLL)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (ADD x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ADDshiftRL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SRLconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64ADDshiftRL)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (ADD x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ADDshiftRA x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SRAconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64ADDshiftRA)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (ADD (SLL x (ANDconst <t> [63] y)) (CSEL0 <typ.UInt64> [cc] (SRL <typ.UInt64> x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))))
	// cond: cc == OpARM64LessThanU
	// result: (ROR x (NEG <t> y))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64SLL {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpARM64ANDconst {
				continue
			}
			t := v_0_1.Type
			if auxIntToInt64(v_0_1.AuxInt) != 63 {
				continue
			}
			y := v_0_1.Args[0]
			if v_1.Op != OpARM64CSEL0 || v_1.Type != typ.UInt64 {
				continue
			}
			cc := auxIntToOp(v_1.AuxInt)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64SRL || v_1_0.Type != typ.UInt64 {
				continue
			}
			_ = v_1_0.Args[1]
			if x != v_1_0.Args[0] {
				continue
			}
			v_1_0_1 := v_1_0.Args[1]
			if v_1_0_1.Op != OpARM64SUB || v_1_0_1.Type != t {
				continue
			}
			_ = v_1_0_1.Args[1]
			v_1_0_1_0 := v_1_0_1.Args[0]
			if v_1_0_1_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_0_1_0.AuxInt) != 64 {
				continue
			}
			v_1_0_1_1 := v_1_0_1.Args[1]
			if v_1_0_1_1.Op != OpARM64ANDconst || v_1_0_1_1.Type != t || auxIntToInt64(v_1_0_1_1.AuxInt) != 63 || y != v_1_0_1_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpARM64CMPconst || auxIntToInt64(v_1_1.AuxInt) != 64 {
				continue
			}
			v_1_1_0 := v_1_1.Args[0]
			if v_1_1_0.Op != OpARM64SUB || v_1_1_0.Type != t {
				continue
			}
			_ = v_1_1_0.Args[1]
			v_1_1_0_0 := v_1_1_0.Args[0]
			if v_1_1_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_1_0_0.AuxInt) != 64 {
				continue
			}
			v_1_1_0_1 := v_1_1_0.Args[1]
			if v_1_1_0_1.Op != OpARM64ANDconst || v_1_1_0_1.Type != t || auxIntToInt64(v_1_1_0_1.AuxInt) != 63 || y != v_1_1_0_1.Args[0] || !(cc == OpARM64LessThanU) {
				continue
			}
			v.reset(OpARM64ROR)
			v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (ADD (SRL <typ.UInt64> x (ANDconst <t> [63] y)) (CSEL0 <typ.UInt64> [cc] (SLL x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))))
	// cond: cc == OpARM64LessThanU
	// result: (ROR x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64SRL || v_0.Type != typ.UInt64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpARM64ANDconst {
				continue
			}
			t := v_0_1.Type
			if auxIntToInt64(v_0_1.AuxInt) != 63 {
				continue
			}
			y := v_0_1.Args[0]
			if v_1.Op != OpARM64CSEL0 || v_1.Type != typ.UInt64 {
				continue
			}
			cc := auxIntToOp(v_1.AuxInt)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64SLL {
				continue
			}
			_ = v_1_0.Args[1]
			if x != v_1_0.Args[0] {
				continue
			}
			v_1_0_1 := v_1_0.Args[1]
			if v_1_0_1.Op != OpARM64SUB || v_1_0_1.Type != t {
				continue
			}
			_ = v_1_0_1.Args[1]
			v_1_0_1_0 := v_1_0_1.Args[0]
			if v_1_0_1_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_0_1_0.AuxInt) != 64 {
				continue
			}
			v_1_0_1_1 := v_1_0_1.Args[1]
			if v_1_0_1_1.Op != OpARM64ANDconst || v_1_0_1_1.Type != t || auxIntToInt64(v_1_0_1_1.AuxInt) != 63 || y != v_1_0_1_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpARM64CMPconst || auxIntToInt64(v_1_1.AuxInt) != 64 {
				continue
			}
			v_1_1_0 := v_1_1.Args[0]
			if v_1_1_0.Op != OpARM64SUB || v_1_1_0.Type != t {
				continue
			}
			_ = v_1_1_0.Args[1]
			v_1_1_0_0 := v_1_1_0.Args[0]
			if v_1_1_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_1_0_0.AuxInt) != 64 {
				continue
			}
			v_1_1_0_1 := v_1_1_0.Args[1]
			if v_1_1_0_1.Op != OpARM64ANDconst || v_1_1_0_1.Type != t || auxIntToInt64(v_1_1_0_1.AuxInt) != 63 || y != v_1_1_0_1.Args[0] || !(cc == OpARM64LessThanU) {
				continue
			}
			v.reset(OpARM64ROR)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADD (SLL x (ANDconst <t> [31] y)) (CSEL0 <typ.UInt32> [cc] (SRL <typ.UInt32> (MOVWUreg x) (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))))
	// cond: cc == OpARM64LessThanU
	// result: (RORW x (NEG <t> y))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64SLL {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpARM64ANDconst {
				continue
			}
			t := v_0_1.Type
			if auxIntToInt64(v_0_1.AuxInt) != 31 {
				continue
			}
			y := v_0_1.Args[0]
			if v_1.Op != OpARM64CSEL0 || v_1.Type != typ.UInt32 {
				continue
			}
			cc := auxIntToOp(v_1.AuxInt)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64SRL || v_1_0.Type != typ.UInt32 {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpARM64MOVWUreg || x != v_1_0_0.Args[0] {
				continue
			}
			v_1_0_1 := v_1_0.Args[1]
			if v_1_0_1.Op != OpARM64SUB || v_1_0_1.Type != t {
				continue
			}
			_ = v_1_0_1.Args[1]
			v_1_0_1_0 := v_1_0_1.Args[0]
			if v_1_0_1_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_0_1_0.AuxInt) != 32 {
				continue
			}
			v_1_0_1_1 := v_1_0_1.Args[1]
			if v_1_0_1_1.Op != OpARM64ANDconst || v_1_0_1_1.Type != t || auxIntToInt64(v_1_0_1_1.AuxInt) != 31 || y != v_1_0_1_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpARM64CMPconst || auxIntToInt64(v_1_1.AuxInt) != 64 {
				continue
			}
			v_1_1_0 := v_1_1.Args[0]
			if v_1_1_0.Op != OpARM64SUB || v_1_1_0.Type != t {
				continue
			}
			_ = v_1_1_0.Args[1]
			v_1_1_0_0 := v_1_1_0.Args[0]
			if v_1_1_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_1_0_0.AuxInt) != 32 {
				continue
			}
			v_1_1_0_1 := v_1_1_0.Args[1]
			if v_1_1_0_1.Op != OpARM64ANDconst || v_1_1_0_1.Type != t || auxIntToInt64(v_1_1_0_1.AuxInt) != 31 || y != v_1_1_0_1.Args[0] || !(cc == OpARM64LessThanU) {
				continue
			}
			v.reset(OpARM64RORW)
			v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (ADD (SRL <typ.UInt32> (MOVWUreg x) (ANDconst <t> [31] y)) (CSEL0 <typ.UInt32> [cc] (SLL x (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))))
	// cond: cc == OpARM64LessThanU
	// result: (RORW x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64SRL || v_0.Type != typ.UInt32 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpARM64MOVWUreg {
				continue
			}
			x := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpARM64ANDconst {
				continue
			}
			t := v_0_1.Type
			if auxIntToInt64(v_0_1.AuxInt) != 31 {
				continue
			}
			y := v_0_1.Args[0]
			if v_1.Op != OpARM64CSEL0 || v_1.Type != typ.UInt32 {
				continue
			}
			cc := auxIntToOp(v_1.AuxInt)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64SLL {
				continue
			}
			_ = v_1_0.Args[1]
			if x != v_1_0.Args[0] {
				continue
			}
			v_1_0_1 := v_1_0.Args[1]
			if v_1_0_1.Op != OpARM64SUB || v_1_0_1.Type != t {
				continue
			}
			_ = v_1_0_1.Args[1]
			v_1_0_1_0 := v_1_0_1.Args[0]
			if v_1_0_1_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_0_1_0.AuxInt) != 32 {
				continue
			}
			v_1_0_1_1 := v_1_0_1.Args[1]
			if v_1_0_1_1.Op != OpARM64ANDconst || v_1_0_1_1.Type != t || auxIntToInt64(v_1_0_1_1.AuxInt) != 31 || y != v_1_0_1_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpARM64CMPconst || auxIntToInt64(v_1_1.AuxInt) != 64 {
				continue
			}
			v_1_1_0 := v_1_1.Args[0]
			if v_1_1_0.Op != OpARM64SUB || v_1_1_0.Type != t {
				continue
			}
			_ = v_1_1_0.Args[1]
			v_1_1_0_0 := v_1_1_0.Args[0]
			if v_1_1_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_1_0_0.AuxInt) != 32 {
				continue
			}
			v_1_1_0_1 := v_1_1_0.Args[1]
			if v_1_1_0_1.Op != OpARM64ANDconst || v_1_1_0_1.Type != t || auxIntToInt64(v_1_1_0_1.AuxInt) != 31 || y != v_1_1_0_1.Args[0] || !(cc == OpARM64LessThanU) {
				continue
			}
			v.reset(OpARM64RORW)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64ADDconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ADDconst [off1] (MOVDaddr [off2] {sym} ptr))
	// cond: is32Bit(off1+int64(off2))
	// result: (MOVDaddr [int32(off1)+off2] {sym} ptr)
	for {
		off1 := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		if !(is32Bit(off1 + int64(off2))) {
			break
		}
		v.reset(OpARM64MOVDaddr)
		v.AuxInt = int32ToAuxInt(int32(off1) + off2)
		v.Aux = symToAux(sym)
		v.AddArg(ptr)
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
	// match: (ADDconst [c] (MOVDconst [d]))
	// result: (MOVDconst [c+d])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(c + d)
		return true
	}
	// match: (ADDconst [c] (ADDconst [d] x))
	// result: (ADDconst [c+d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (SUBconst [d] x))
	// result: (ADDconst [c-d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SUBconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADDshiftLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDshiftLL (MOVDconst [c]) x [d])
	// result: (ADDconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftLL x (MOVDconst [c]) [d])
	// result: (ADDconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL [c] (SRLconst x [64-c]) x)
	// result: (RORconst [64-c] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != 64-c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = int64ToAuxInt(64 - c)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL <t> [c] (UBFX [bfc] x) x)
	// cond: c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)
	// result: (RORWconst [32-c] x)
	for {
		t := v.Type
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if x != v_1 || !(c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = int64ToAuxInt(32 - c)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL <typ.UInt16> [8] (UBFX <typ.UInt16> [armBFAuxInt(8, 8)] x) x)
	// result: (REV16W x)
	for {
		if v.Type != typ.UInt16 || auxIntToInt64(v.AuxInt) != 8 || v_0.Op != OpARM64UBFX || v_0.Type != typ.UInt16 || auxIntToArm64BitField(v_0.AuxInt) != armBFAuxInt(8, 8) {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL [c] (SRLconst x [64-c]) x2)
	// result: (EXTRconst [64-c] x2 x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != 64-c {
			break
		}
		x := v_0.Args[0]
		x2 := v_1
		v.reset(OpARM64EXTRconst)
		v.AuxInt = int64ToAuxInt(64 - c)
		v.AddArg2(x2, x)
		return true
	}
	// match: (ADDshiftLL <t> [c] (UBFX [bfc] x) x2)
	// cond: c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)
	// result: (EXTRWconst [32-c] x2 x)
	for {
		t := v.Type
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		x2 := v_1
		if !(c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)) {
			break
		}
		v.reset(OpARM64EXTRWconst)
		v.AuxInt = int64ToAuxInt(32 - c)
		v.AddArg2(x2, x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADDshiftRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDshiftRA (MOVDconst [c]) x [d])
	// result: (ADDconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRA x (MOVDconst [c]) [d])
	// result: (ADDconst x [c>>uint64(d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADDshiftRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDshiftRL (MOVDconst [c]) x [d])
	// result: (ADDconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRL x (MOVDconst [c]) [d])
	// result: (ADDconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftRL [c] (SLLconst x [64-c]) x)
	// result: (RORconst [ c] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 64-c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftRL <t> [c] (SLLconst x [32-c]) (MOVWUreg x))
	// cond: c < 32 && t.Size() == 4
	// result: (RORWconst [c] x)
	for {
		t := v.Type
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 32-c {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpARM64MOVWUreg || x != v_1.Args[0] || !(c < 32 && t.Size() == 4) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64AND(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AND x (MOVDconst [c]))
	// result: (ANDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64ANDconst)
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
	// match: (AND x (MVN y))
	// result: (BIC x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MVN {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpARM64BIC)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (AND x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ANDshiftLL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SLLconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64ANDshiftLL)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (AND x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ANDshiftRL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SRLconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64ANDshiftRL)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (AND x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ANDshiftRA x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SRAconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64ANDshiftRA)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64ANDconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (ANDconst [0] _)
	// result: (MOVDconst [0])
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.reset(OpARM64MOVDconst)
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
	// match: (ANDconst [c] (MOVDconst [d]))
	// result: (MOVDconst [c&d])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(c & d)
		return true
	}
	// match: (ANDconst [c] (ANDconst [d] x))
	// result: (ANDconst [c&d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64ANDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVWUreg x))
	// result: (ANDconst [c&(1<<32-1)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c & (1<<32 - 1))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVHUreg x))
	// result: (ANDconst [c&(1<<16-1)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c & (1<<16 - 1))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVBUreg x))
	// result: (ANDconst [c&(1<<8-1)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c & (1<<8 - 1))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [ac] (SLLconst [sc] x))
	// cond: isARM64BFMask(sc, ac, sc)
	// result: (UBFIZ [armBFAuxInt(sc, arm64BFWidth(ac, sc))] x)
	for {
		ac := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, ac, sc)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, arm64BFWidth(ac, sc)))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [ac] (SRLconst [sc] x))
	// cond: isARM64BFMask(sc, ac, 0)
	// result: (UBFX [armBFAuxInt(sc, arm64BFWidth(ac, 0))] x)
	for {
		ac := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, ac, 0)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, arm64BFWidth(ac, 0)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ANDshiftLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftLL (MOVDconst [c]) x [d])
	// result: (ANDconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftLL x (MOVDconst [c]) [d])
	// result: (ANDconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftLL x y:(SLLconst x [c]) [d])
	// cond: c==d
	// result: y
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		y := v_1
		if y.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(y.AuxInt)
		if x != y.Args[0] || !(c == d) {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ANDshiftRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftRA (MOVDconst [c]) x [d])
	// result: (ANDconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRA x (MOVDconst [c]) [d])
	// result: (ANDconst x [c>>uint64(d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftRA x y:(SRAconst x [c]) [d])
	// cond: c==d
	// result: y
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		y := v_1
		if y.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(y.AuxInt)
		if x != y.Args[0] || !(c == d) {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ANDshiftRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftRL (MOVDconst [c]) x [d])
	// result: (ANDconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRL x (MOVDconst [c]) [d])
	// result: (ANDconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftRL x y:(SRLconst x [c]) [d])
	// cond: c==d
	// result: y
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		y := v_1
		if y.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(y.AuxInt)
		if x != y.Args[0] || !(c == d) {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64BIC(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BIC x (MOVDconst [c]))
	// result: (ANDconst [^c] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(^c)
		v.AddArg(x)
		return true
	}
	// match: (BIC x x)
	// result: (MOVDconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (BIC x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (BICshiftLL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64BICshiftLL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (BIC x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (BICshiftRL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64BICshiftRL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (BIC x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (BICshiftRA x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64BICshiftRA)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64BICshiftLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftLL x (MOVDconst [c]) [d])
	// result: (ANDconst x [^int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(^int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64BICshiftRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftRA x (MOVDconst [c]) [d])
	// result: (ANDconst x [^(c>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(^(c >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64BICshiftRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftRL x (MOVDconst [c]) [d])
	// result: (ANDconst x [^int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(^int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMN(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMN x (MOVDconst [c]))
	// result: (CMNconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64CMNconst)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (CMN x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (CMNshiftLL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SLLconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64CMNshiftLL)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (CMN x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (CMNshiftRL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SRLconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64CMNshiftRL)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (CMN x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (CMNshiftRA x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SRAconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64CMNshiftRA)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64CMNW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMNW x (MOVDconst [c]))
	// result: (CMNWconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64CMNWconst)
			v.AuxInt = int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64CMNWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CMNWconst (MOVDconst [x]) [y])
	// result: (FlagConstant [addFlags32(int32(x),y)])
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(addFlags32(int32(x), y))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMNconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CMNconst (MOVDconst [x]) [y])
	// result: (FlagConstant [addFlags64(x,y)])
	for {
		y := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(addFlags64(x, y))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMNshiftLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMNshiftLL (MOVDconst [c]) x [d])
	// result: (CMNconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64CMNconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftLL x (MOVDconst [c]) [d])
	// result: (CMNconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64CMNconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMNshiftRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMNshiftRA (MOVDconst [c]) x [d])
	// result: (CMNconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64CMNconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRA x (MOVDconst [c]) [d])
	// result: (CMNconst x [c>>uint64(d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64CMNconst)
		v.AuxInt = int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMNshiftRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMNshiftRL (MOVDconst [c]) x [d])
	// result: (CMNconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64CMNconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRL x (MOVDconst [c]) [d])
	// result: (CMNconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64CMNconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMP(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMP x (MOVDconst [c]))
	// result: (CMPconst [c] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64CMPconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMP (MOVDconst [c]) x)
	// result: (InvertFlags (CMPconst [c] x))
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x y)
	// cond: x.ID > y.ID
	// result: (InvertFlags (CMP y x))
	for {
		x := v_0
		y := v_1
		if !(x.ID > y.ID) {
			break
		}
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (CMPshiftLL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64CMPshiftLL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (CMP x0:(SLLconst [c] y) x1)
	// cond: clobberIfDead(x0)
	// result: (InvertFlags (CMPshiftLL x1 y [c]))
	for {
		x0 := v_0
		if x0.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(x0.AuxInt)
		y := x0.Args[0]
		x1 := v_1
		if !(clobberIfDead(x0)) {
			break
		}
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPshiftLL, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg2(x1, y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (CMPshiftRL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64CMPshiftRL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (CMP x0:(SRLconst [c] y) x1)
	// cond: clobberIfDead(x0)
	// result: (InvertFlags (CMPshiftRL x1 y [c]))
	for {
		x0 := v_0
		if x0.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(x0.AuxInt)
		y := x0.Args[0]
		x1 := v_1
		if !(clobberIfDead(x0)) {
			break
		}
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPshiftRL, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg2(x1, y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (CMPshiftRA x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64CMPshiftRA)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (CMP x0:(SRAconst [c] y) x1)
	// cond: clobberIfDead(x0)
	// result: (InvertFlags (CMPshiftRA x1 y [c]))
	for {
		x0 := v_0
		if x0.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(x0.AuxInt)
		y := x0.Args[0]
		x1 := v_1
		if !(clobberIfDead(x0)) {
			break
		}
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPshiftRA, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg2(x1, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPW x (MOVDconst [c]))
	// result: (CMPWconst [int32(c)] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64CMPWconst)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPW (MOVDconst [c]) x)
	// result: (InvertFlags (CMPWconst [int32(c)] x))
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMPW x y)
	// cond: x.ID > y.ID
	// result: (InvertFlags (CMPW y x))
	for {
		x := v_0
		y := v_1
		if !(x.ID > y.ID) {
			break
		}
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CMPWconst (MOVDconst [x]) [y])
	// result: (FlagConstant [subFlags32(int32(x),y)])
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(subFlags32(int32(x), y))
		return true
	}
	// match: (CMPWconst (MOVBUreg _) [c])
	// cond: 0xff < c
	// result: (FlagConstant [subFlags64(0,1)])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpARM64MOVBUreg || !(0xff < c) {
			break
		}
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(subFlags64(0, 1))
		return true
	}
	// match: (CMPWconst (MOVHUreg _) [c])
	// cond: 0xffff < c
	// result: (FlagConstant [subFlags64(0,1)])
	for {
		c := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpARM64MOVHUreg || !(0xffff < c) {
			break
		}
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(subFlags64(0, 1))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CMPconst (MOVDconst [x]) [y])
	// result: (FlagConstant [subFlags64(x,y)])
	for {
		y := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(subFlags64(x, y))
		return true
	}
	// match: (CMPconst (MOVBUreg _) [c])
	// cond: 0xff < c
	// result: (FlagConstant [subFlags64(0,1)])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVBUreg || !(0xff < c) {
			break
		}
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(subFlags64(0, 1))
		return true
	}
	// match: (CMPconst (MOVHUreg _) [c])
	// cond: 0xffff < c
	// result: (FlagConstant [subFlags64(0,1)])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVHUreg || !(0xffff < c) {
			break
		}
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(subFlags64(0, 1))
		return true
	}
	// match: (CMPconst (MOVWUreg _) [c])
	// cond: 0xffffffff < c
	// result: (FlagConstant [subFlags64(0,1)])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVWUreg || !(0xffffffff < c) {
			break
		}
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(subFlags64(0, 1))
		return true
	}
	// match: (CMPconst (ANDconst _ [m]) [n])
	// cond: 0 <= m && m < n
	// result: (FlagConstant [subFlags64(0,1)])
	for {
		n := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64ANDconst {
			break
		}
		m := auxIntToInt64(v_0.AuxInt)
		if !(0 <= m && m < n) {
			break
		}
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(subFlags64(0, 1))
		return true
	}
	// match: (CMPconst (SRLconst _ [c]) [n])
	// cond: 0 <= n && 0 < c && c <= 63 && (1<<uint64(64-c)) <= uint64(n)
	// result: (FlagConstant [subFlags64(0,1)])
	for {
		n := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if !(0 <= n && 0 < c && c <= 63 && (1<<uint64(64-c)) <= uint64(n)) {
			break
		}
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(subFlags64(0, 1))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPshiftLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPshiftLL (MOVDconst [c]) x [d])
	// result: (InvertFlags (CMPconst [c] (SLLconst <x.Type> x [d])))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v1.AuxInt = int64ToAuxInt(d)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftLL x (MOVDconst [c]) [d])
	// result: (CMPconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64CMPconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPshiftRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPshiftRA (MOVDconst [c]) x [d])
	// result: (InvertFlags (CMPconst [c] (SRAconst <x.Type> x [d])))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v1.AuxInt = int64ToAuxInt(d)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRA x (MOVDconst [c]) [d])
	// result: (CMPconst x [c>>uint64(d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64CMPconst)
		v.AuxInt = int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPshiftRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPshiftRL (MOVDconst [c]) x [d])
	// result: (InvertFlags (CMPconst [c] (SRLconst <x.Type> x [d])))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v1.AuxInt = int64ToAuxInt(d)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRL x (MOVDconst [c]) [d])
	// result: (CMPconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64CMPconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CSEL(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CSEL [cc] x (MOVDconst [0]) flag)
	// result: (CSEL0 [cc] x flag)
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		flag := v_2
		v.reset(OpARM64CSEL0)
		v.AuxInt = opToAuxInt(cc)
		v.AddArg2(x, flag)
		return true
	}
	// match: (CSEL [cc] (MOVDconst [0]) y flag)
	// result: (CSEL0 [arm64Negate(cc)] y flag)
	for {
		cc := auxIntToOp(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		y := v_1
		flag := v_2
		v.reset(OpARM64CSEL0)
		v.AuxInt = opToAuxInt(arm64Negate(cc))
		v.AddArg2(y, flag)
		return true
	}
	// match: (CSEL [cc] x y (InvertFlags cmp))
	// result: (CSEL [arm64Invert(cc)] x y cmp)
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != OpARM64InvertFlags {
			break
		}
		cmp := v_2.Args[0]
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(arm64Invert(cc))
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (CSEL [cc] x _ flag)
	// cond: ccARM64Eval(cc, flag) > 0
	// result: x
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		flag := v_2
		if !(ccARM64Eval(cc, flag) > 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CSEL [cc] _ y flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: y
	for {
		cc := auxIntToOp(v.AuxInt)
		y := v_1
		flag := v_2
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: (CSEL [cc] x y (CMPWconst [0] boolval))
	// cond: cc == OpARM64NotEqual && flagArg(boolval) != nil
	// result: (CSEL [boolval.Op] x y flagArg(boolval))
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != OpARM64CMPWconst || auxIntToInt32(v_2.AuxInt) != 0 {
			break
		}
		boolval := v_2.Args[0]
		if !(cc == OpARM64NotEqual && flagArg(boolval) != nil) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(boolval.Op)
		v.AddArg3(x, y, flagArg(boolval))
		return true
	}
	// match: (CSEL [cc] x y (CMPWconst [0] boolval))
	// cond: cc == OpARM64Equal && flagArg(boolval) != nil
	// result: (CSEL [arm64Negate(boolval.Op)] x y flagArg(boolval))
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != OpARM64CMPWconst || auxIntToInt32(v_2.AuxInt) != 0 {
			break
		}
		boolval := v_2.Args[0]
		if !(cc == OpARM64Equal && flagArg(boolval) != nil) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(arm64Negate(boolval.Op))
		v.AddArg3(x, y, flagArg(boolval))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CSEL0(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CSEL0 [cc] x (InvertFlags cmp))
	// result: (CSEL0 [arm64Invert(cc)] x cmp)
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64InvertFlags {
			break
		}
		cmp := v_1.Args[0]
		v.reset(OpARM64CSEL0)
		v.AuxInt = opToAuxInt(arm64Invert(cc))
		v.AddArg2(x, cmp)
		return true
	}
	// match: (CSEL0 [cc] x flag)
	// cond: ccARM64Eval(cc, flag) > 0
	// result: x
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		flag := v_1
		if !(ccARM64Eval(cc, flag) > 0) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (CSEL0 [cc] _ flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: (MOVDconst [0])
	for {
		cc := auxIntToOp(v.AuxInt)
		flag := v_1
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (CSEL0 [cc] x (CMPWconst [0] boolval))
	// cond: cc == OpARM64NotEqual && flagArg(boolval) != nil
	// result: (CSEL0 [boolval.Op] x flagArg(boolval))
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64CMPWconst || auxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		boolval := v_1.Args[0]
		if !(cc == OpARM64NotEqual && flagArg(boolval) != nil) {
			break
		}
		v.reset(OpARM64CSEL0)
		v.AuxInt = opToAuxInt(boolval.Op)
		v.AddArg2(x, flagArg(boolval))
		return true
	}
	// match: (CSEL0 [cc] x (CMPWconst [0] boolval))
	// cond: cc == OpARM64Equal && flagArg(boolval) != nil
	// result: (CSEL0 [arm64Negate(boolval.Op)] x flagArg(boolval))
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64CMPWconst || auxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		boolval := v_1.Args[0]
		if !(cc == OpARM64Equal && flagArg(boolval) != nil) {
			break
		}
		v.reset(OpARM64CSEL0)
		v.AuxInt = opToAuxInt(arm64Negate(boolval.Op))
		v.AddArg2(x, flagArg(boolval))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64DIV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIV (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [c/d])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(c / d)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64DIVW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVW (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [int64(int32(c)/int32(d))])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int32(c) / int32(d)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64EON(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EON x (MOVDconst [c]))
	// result: (XORconst [^c] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(^c)
		v.AddArg(x)
		return true
	}
	// match: (EON x x)
	// result: (MOVDconst [-1])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	// match: (EON x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (EONshiftLL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64EONshiftLL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (EON x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (EONshiftRL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64EONshiftRL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (EON x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (EONshiftRA x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64EONshiftRA)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64EONshiftLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EONshiftLL x (MOVDconst [c]) [d])
	// result: (XORconst x [^int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(^int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (EONshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [-1])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64EONshiftRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EONshiftRA x (MOVDconst [c]) [d])
	// result: (XORconst x [^(c>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(^(c >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (EONshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [-1])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64EONshiftRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EONshiftRL x (MOVDconst [c]) [d])
	// result: (XORconst x [^int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(^int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (EONshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [-1])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64Equal(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Equal (FlagConstant [fc]))
	// result: (MOVDconst [b2i(fc.eq())])
	for {
		if v_0.Op != OpARM64FlagConstant {
			break
		}
		fc := auxIntToFlagConstant(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(fc.eq()))
		return true
	}
	// match: (Equal (InvertFlags x))
	// result: (Equal x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64Equal)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FADDD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FADDD a (FMULD x y))
	// result: (FMADDD a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != OpARM64FMULD {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			v.reset(OpARM64FMADDD)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (FADDD a (FNMULD x y))
	// result: (FMSUBD a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != OpARM64FNMULD {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			v.reset(OpARM64FMSUBD)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64FADDS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FADDS a (FMULS x y))
	// result: (FMADDS a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != OpARM64FMULS {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			v.reset(OpARM64FMADDS)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (FADDS a (FNMULS x y))
	// result: (FMSUBS a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != OpARM64FNMULS {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			v.reset(OpARM64FMSUBS)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64FCMPD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (FCMPD x (FMOVDconst [0]))
	// result: (FCMPD0 x)
	for {
		x := v_0
		if v_1.Op != OpARM64FMOVDconst || auxIntToFloat64(v_1.AuxInt) != 0 {
			break
		}
		v.reset(OpARM64FCMPD0)
		v.AddArg(x)
		return true
	}
	// match: (FCMPD (FMOVDconst [0]) x)
	// result: (InvertFlags (FCMPD0 x))
	for {
		if v_0.Op != OpARM64FMOVDconst || auxIntToFloat64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD0, types.TypeFlags)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FCMPS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (FCMPS x (FMOVSconst [0]))
	// result: (FCMPS0 x)
	for {
		x := v_0
		if v_1.Op != OpARM64FMOVSconst || auxIntToFloat64(v_1.AuxInt) != 0 {
			break
		}
		v.reset(OpARM64FCMPS0)
		v.AddArg(x)
		return true
	}
	// match: (FCMPS (FMOVSconst [0]) x)
	// result: (InvertFlags (FCMPS0 x))
	for {
		if v_0.Op != OpARM64FMOVSconst || auxIntToFloat64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS0, types.TypeFlags)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDfpgp(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (FMOVDfpgp <t> (Arg [off] {sym}))
	// result: @b.Func.Entry (Arg <t> [off] {sym})
	for {
		t := v.Type
		if v_0.Op != OpArg {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		sym := auxToSym(v_0.Aux)
		b = b.Func.Entry
		v0 := b.NewValue0(v.Pos, OpArg, t)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDgpfp(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (FMOVDgpfp <t> (Arg [off] {sym}))
	// result: @b.Func.Entry (Arg <t> [off] {sym})
	for {
		t := v.Type
		if v_0.Op != OpArg {
			break
		}
		off := auxIntToInt32(v_0.AuxInt)
		sym := auxToSym(v_0.Aux)
		b = b.Func.Entry
		v0 := b.NewValue0(v.Pos, OpArg, t)
		v.copyOf(v0)
		v0.AuxInt = int32ToAuxInt(off)
		v0.Aux = symToAux(sym)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVDload [off] {sym} ptr (MOVDstore [off] {sym} ptr val _))
	// result: (FMOVDgpfp val)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVDstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpARM64FMOVDgpfp)
		v.AddArg(val)
		return true
	}
	// match: (FMOVDload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVDload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVDload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVDload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVDloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64FMOVDloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (FMOVDload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVDload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVDloadidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (FMOVDload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64FMOVDload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVDloadidx (MOVDconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (FMOVDload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64FMOVDload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVDstore [off] {sym} ptr (FMOVDgpfp val) mem)
	// result: (MOVDstore [off] {sym} ptr val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64FMOVDgpfp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVDstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVDstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVDstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVDstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVDstoreidx ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64FMOVDstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (FMOVDstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVDstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVDstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVDstoreidx ptr (MOVDconst [c]) val mem)
	// cond: is32Bit(c)
	// result: (FMOVDstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64FMOVDstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVDstoreidx (MOVDconst [c]) idx val mem)
	// cond: is32Bit(c)
	// result: (FMOVDstore [int32(c)] idx val mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64FMOVDstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVSload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVSload [off] {sym} ptr (MOVWstore [off] {sym} ptr val _))
	// result: (FMOVSgpfp val)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVWstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpARM64FMOVSgpfp)
		v.AddArg(val)
		return true
	}
	// match: (FMOVSload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVSload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVSload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVSload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVSloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64FMOVSloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (FMOVSload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVSload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVSload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVSloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVSloadidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (FMOVSload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64FMOVSload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVSloadidx (MOVDconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (FMOVSload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64FMOVSload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVSstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVSstore [off] {sym} ptr (FMOVSgpfp val) mem)
	// result: (MOVWstore [off] {sym} ptr val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64FMOVSgpfp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVSstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVSstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVSstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVSstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVSstoreidx ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64FMOVSstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (FMOVSstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVSstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVSstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVSstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVSstoreidx ptr (MOVDconst [c]) val mem)
	// cond: is32Bit(c)
	// result: (FMOVSstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64FMOVSstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVSstoreidx (MOVDconst [c]) idx val mem)
	// cond: is32Bit(c)
	// result: (FMOVSstore [int32(c)] idx val mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64FMOVSstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMULD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMULD (FNEGD x) y)
	// result: (FNMULD x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64FNEGD {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.reset(OpARM64FNMULD)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64FMULS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMULS (FNEGS x) y)
	// result: (FNMULS x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64FNEGS {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.reset(OpARM64FNMULS)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64FNEGD(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FNEGD (FMULD x y))
	// result: (FNMULD x y)
	for {
		if v_0.Op != OpARM64FMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FNMULD)
		v.AddArg2(x, y)
		return true
	}
	// match: (FNEGD (FNMULD x y))
	// result: (FMULD x y)
	for {
		if v_0.Op != OpARM64FNMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FMULD)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FNEGS(v *Value) bool {
	v_0 := v.Args[0]
	// match: (FNEGS (FMULS x y))
	// result: (FNMULS x y)
	for {
		if v_0.Op != OpARM64FMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FNMULS)
		v.AddArg2(x, y)
		return true
	}
	// match: (FNEGS (FNMULS x y))
	// result: (FMULS x y)
	for {
		if v_0.Op != OpARM64FNMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FMULS)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FNMULD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FNMULD (FNEGD x) y)
	// result: (FMULD x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64FNEGD {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.reset(OpARM64FMULD)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64FNMULS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FNMULS (FNEGS x) y)
	// result: (FMULS x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64FNEGS {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.reset(OpARM64FMULS)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64FSUBD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FSUBD a (FMULD x y))
	// result: (FMSUBD a x y)
	for {
		a := v_0
		if v_1.Op != OpARM64FMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARM64FMSUBD)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBD (FMULD x y) a)
	// result: (FNMSUBD a x y)
	for {
		if v_0.Op != OpARM64FMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		v.reset(OpARM64FNMSUBD)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBD a (FNMULD x y))
	// result: (FMADDD a x y)
	for {
		a := v_0
		if v_1.Op != OpARM64FNMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARM64FMADDD)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBD (FNMULD x y) a)
	// result: (FNMADDD a x y)
	for {
		if v_0.Op != OpARM64FNMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		v.reset(OpARM64FNMADDD)
		v.AddArg3(a, x, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FSUBS(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FSUBS a (FMULS x y))
	// result: (FMSUBS a x y)
	for {
		a := v_0
		if v_1.Op != OpARM64FMULS {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARM64FMSUBS)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBS (FMULS x y) a)
	// result: (FNMSUBS a x y)
	for {
		if v_0.Op != OpARM64FMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		v.reset(OpARM64FNMSUBS)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBS a (FNMULS x y))
	// result: (FMADDS a x y)
	for {
		a := v_0
		if v_1.Op != OpARM64FNMULS {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARM64FMADDS)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBS (FNMULS x y) a)
	// result: (FNMADDS a x y)
	for {
		if v_0.Op != OpARM64FNMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		v.reset(OpARM64FNMADDS)
		v.AddArg3(a, x, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64GreaterEqual(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterEqual (FlagConstant [fc]))
	// result: (MOVDconst [b2i(fc.ge())])
	for {
		if v_0.Op != OpARM64FlagConstant {
			break
		}
		fc := auxIntToFlagConstant(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(fc.ge()))
		return true
	}
	// match: (GreaterEqual (InvertFlags x))
	// result: (LessEqual x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64LessEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64GreaterEqualF(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterEqualF (InvertFlags x))
	// result: (LessEqualF x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64LessEqualF)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64GreaterEqualU(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterEqualU (FlagConstant [fc]))
	// result: (MOVDconst [b2i(fc.uge())])
	for {
		if v_0.Op != OpARM64FlagConstant {
			break
		}
		fc := auxIntToFlagConstant(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(fc.uge()))
		return true
	}
	// match: (GreaterEqualU (InvertFlags x))
	// result: (LessEqualU x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64LessEqualU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64GreaterThan(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterThan (FlagConstant [fc]))
	// result: (MOVDconst [b2i(fc.gt())])
	for {
		if v_0.Op != OpARM64FlagConstant {
			break
		}
		fc := auxIntToFlagConstant(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(fc.gt()))
		return true
	}
	// match: (GreaterThan (InvertFlags x))
	// result: (LessThan x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64LessThan)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64GreaterThanF(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterThanF (InvertFlags x))
	// result: (LessThanF x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64LessThanF)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64GreaterThanU(v *Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterThanU (FlagConstant [fc]))
	// result: (MOVDconst [b2i(fc.ugt())])
	for {
		if v_0.Op != OpARM64FlagConstant {
			break
		}
		fc := auxIntToFlagConstant(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(fc.ugt()))
		return true
	}
	// match: (GreaterThanU (InvertFlags x))
	// result: (LessThanU x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64LessThanU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64LessEqual(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LessEqual (FlagConstant [fc]))
	// result: (MOVDconst [b2i(fc.le())])
	for {
		if v_0.Op != OpARM64FlagConstant {
			break
		}
		fc := auxIntToFlagConstant(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(fc.le()))
		return true
	}
	// match: (LessEqual (InvertFlags x))
	// result: (GreaterEqual x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64GreaterEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64LessEqualF(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LessEqualF (InvertFlags x))
	// result: (GreaterEqualF x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64GreaterEqualF)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64LessEqualU(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LessEqualU (FlagConstant [fc]))
	// result: (MOVDconst [b2i(fc.ule())])
	for {
		if v_0.Op != OpARM64FlagConstant {
			break
		}
		fc := auxIntToFlagConstant(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(fc.ule()))
		return true
	}
	// match: (LessEqualU (InvertFlags x))
	// result: (GreaterEqualU x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64GreaterEqualU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64LessThan(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LessThan (FlagConstant [fc]))
	// result: (MOVDconst [b2i(fc.lt())])
	for {
		if v_0.Op != OpARM64FlagConstant {
			break
		}
		fc := auxIntToFlagConstant(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(fc.lt()))
		return true
	}
	// match: (LessThan (InvertFlags x))
	// result: (GreaterThan x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64GreaterThan)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64LessThanF(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LessThanF (InvertFlags x))
	// result: (GreaterThanF x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64GreaterThanF)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64LessThanU(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LessThanU (FlagConstant [fc]))
	// result: (MOVDconst [b2i(fc.ult())])
	for {
		if v_0.Op != OpARM64FlagConstant {
			break
		}
		fc := auxIntToFlagConstant(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(fc.ult()))
		return true
	}
	// match: (LessThanU (InvertFlags x))
	// result: (GreaterThanU x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64GreaterThanU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MADD(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MADD a x (MOVDconst [-1]))
	// result: (SUB a x)
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != -1 {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a _ (MOVDconst [0]))
	// result: a
	for {
		a := v_0
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		v.copyOf(a)
		return true
	}
	// match: (MADD a x (MOVDconst [1]))
	// result: (ADD a x)
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != 1 {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c)
	// result: (ADDshiftLL a x [log64(c)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo64(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c-1) && c>=3
	// result: (ADD a (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo64(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64ADD)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c - 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c+1) && c>=7
	// result: (SUB a (SUBshiftLL <x.Type> x x [log64(c+1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo64(c+1) && c >= 7) {
			break
		}
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c + 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo64(c/3)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%3 == 0 && isPowerOfTwo64(c/3)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 3))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo64(c/5)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%5 == 0 && isPowerOfTwo64(c/5)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 5))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo64(c/7)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%7 == 0 && isPowerOfTwo64(c/7)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 7))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo64(c/9)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%9 == 0 && isPowerOfTwo64(c/9)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 9))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a (MOVDconst [-1]) x)
	// result: (SUB a x)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		x := v_2
		v.reset(OpARM64SUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a (MOVDconst [0]) _)
	// result: a
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(a)
		return true
	}
	// match: (MADD a (MOVDconst [1]) x)
	// result: (ADD a x)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		x := v_2
		v.reset(OpARM64ADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: isPowerOfTwo64(c)
	// result: (ADDshiftLL a x [log64(c)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo64(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: isPowerOfTwo64(c-1) && c>=3
	// result: (ADD a (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo64(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64ADD)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c - 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: isPowerOfTwo64(c+1) && c>=7
	// result: (SUB a (SUBshiftLL <x.Type> x x [log64(c+1)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo64(c+1) && c >= 7) {
			break
		}
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c + 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo64(c/3)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%3 == 0 && isPowerOfTwo64(c/3)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 3))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo64(c/5)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%5 == 0 && isPowerOfTwo64(c/5)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 5))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo64(c/7)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%7 == 0 && isPowerOfTwo64(c/7)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 7))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo64(c/9)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%9 == 0 && isPowerOfTwo64(c/9)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 9))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD (MOVDconst [c]) x y)
	// result: (ADDconst [c] (MUL <x.Type> x y))
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		y := v_2
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) (MOVDconst [d]))
	// result: (ADDconst [c*d] a)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_2.AuxInt)
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c * d)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MADDW(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MADDW a x (MOVDconst [c]))
	// cond: int32(c)==-1
	// result: (SUB a x)
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MADDW a _ (MOVDconst [c]))
	// cond: int32(c)==0
	// result: a
	for {
		a := v_0
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(int32(c) == 0) {
			break
		}
		v.copyOf(a)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: int32(c)==1
	// result: (ADD a x)
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(int32(c) == 1) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c)
	// result: (ADDshiftLL a x [log64(c)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo64(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c-1) && int32(c)>=3
	// result: (ADD a (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo64(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64ADD)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c - 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c+1) && int32(c)>=7
	// result: (SUB a (SUBshiftLL <x.Type> x x [log64(c+1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo64(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c + 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo64(c/3) && is32Bit(c)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%3 == 0 && isPowerOfTwo64(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 3))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo64(c/5) && is32Bit(c)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%5 == 0 && isPowerOfTwo64(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 5))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo64(c/7) && is32Bit(c)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%7 == 0 && isPowerOfTwo64(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 7))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo64(c/9) && is32Bit(c)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%9 == 0 && isPowerOfTwo64(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 9))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: int32(c)==-1
	// result: (SUB a x)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) _)
	// cond: int32(c)==0
	// result: a
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(int32(c) == 0) {
			break
		}
		v.copyOf(a)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: int32(c)==1
	// result: (ADD a x)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(int32(c) == 1) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo64(c)
	// result: (ADDshiftLL a x [log64(c)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo64(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo64(c-1) && int32(c)>=3
	// result: (ADD a (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo64(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64ADD)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c - 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo64(c+1) && int32(c)>=7
	// result: (SUB a (SUBshiftLL <x.Type> x x [log64(c+1)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo64(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c + 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo64(c/3) && is32Bit(c)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%3 == 0 && isPowerOfTwo64(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 3))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo64(c/5) && is32Bit(c)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%5 == 0 && isPowerOfTwo64(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 5))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo64(c/7) && is32Bit(c)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%7 == 0 && isPowerOfTwo64(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 7))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo64(c/9) && is32Bit(c)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%9 == 0 && isPowerOfTwo64(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 9))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADDW (MOVDconst [c]) x y)
	// result: (ADDconst [c] (MULW <x.Type> x y))
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		y := v_2
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) (MOVDconst [d]))
	// result: (ADDconst [int64(int32(c)*int32(d))] a)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_2.AuxInt)
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(int64(int32(c) * int32(d)))
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MNEG(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MNEG x (MOVDconst [-1]))
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != -1 {
				continue
			}
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (MNEG _ (MOVDconst [0]))
	// result: (MOVDconst [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
				continue
			}
			v.reset(OpARM64MOVDconst)
			v.AuxInt = int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [1]))
	// result: (NEG x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 1 {
				continue
			}
			v.reset(OpARM64NEG)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c)
	// result: (NEG (SLLconst <x.Type> [log64(c)] x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo64(c)) {
				continue
			}
			v.reset(OpARM64NEG)
			v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
			v0.AuxInt = int64ToAuxInt(log64(c))
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c-1) && c >= 3
	// result: (NEG (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo64(c-1) && c >= 3) {
				continue
			}
			v.reset(OpARM64NEG)
			v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(log64(c - 1))
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c+1) && c >= 7
	// result: (NEG (ADDshiftLL <x.Type> (NEG <x.Type> x) x [log64(c+1)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo64(c+1) && c >= 7) {
				continue
			}
			v.reset(OpARM64NEG)
			v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(log64(c + 1))
			v1 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
			v1.AddArg(x)
			v0.AddArg2(v1, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo64(c/3)
	// result: (SLLconst <x.Type> [log64(c/3)] (SUBshiftLL <x.Type> x x [2]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%3 == 0 && isPowerOfTwo64(c/3)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.Type = x.Type
			v.AuxInt = int64ToAuxInt(log64(c / 3))
			v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(2)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo64(c/5)
	// result: (NEG (SLLconst <x.Type> [log64(c/5)] (ADDshiftLL <x.Type> x x [2])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%5 == 0 && isPowerOfTwo64(c/5)) {
				continue
			}
			v.reset(OpARM64NEG)
			v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
			v0.AuxInt = int64ToAuxInt(log64(c / 5))
			v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v1.AuxInt = int64ToAuxInt(2)
			v1.AddArg2(x, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo64(c/7)
	// result: (SLLconst <x.Type> [log64(c/7)] (SUBshiftLL <x.Type> x x [3]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%7 == 0 && isPowerOfTwo64(c/7)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.Type = x.Type
			v.AuxInt = int64ToAuxInt(log64(c / 7))
			v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(3)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo64(c/9)
	// result: (NEG (SLLconst <x.Type> [log64(c/9)] (ADDshiftLL <x.Type> x x [3])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%9 == 0 && isPowerOfTwo64(c/9)) {
				continue
			}
			v.reset(OpARM64NEG)
			v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
			v0.AuxInt = int64ToAuxInt(log64(c / 9))
			v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v1.AuxInt = int64ToAuxInt(3)
			v1.AddArg2(x, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [-c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64MOVDconst)
			v.AuxInt = int64ToAuxInt(-c * d)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64MNEGW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MNEGW x (MOVDconst [c]))
	// cond: int32(c)==-1
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(int32(c) == -1) {
				continue
			}
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (MNEGW _ (MOVDconst [c]))
	// cond: int32(c)==0
	// result: (MOVDconst [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(int32(c) == 0) {
				continue
			}
			v.reset(OpARM64MOVDconst)
			v.AuxInt = int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: int32(c)==1
	// result: (NEG x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(int32(c) == 1) {
				continue
			}
			v.reset(OpARM64NEG)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c)
	// result: (NEG (SLLconst <x.Type> [log64(c)] x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo64(c)) {
				continue
			}
			v.reset(OpARM64NEG)
			v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
			v0.AuxInt = int64ToAuxInt(log64(c))
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c-1) && int32(c) >= 3
	// result: (NEG (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo64(c-1) && int32(c) >= 3) {
				continue
			}
			v.reset(OpARM64NEG)
			v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(log64(c - 1))
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c+1) && int32(c) >= 7
	// result: (NEG (ADDshiftLL <x.Type> (NEG <x.Type> x) x [log64(c+1)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo64(c+1) && int32(c) >= 7) {
				continue
			}
			v.reset(OpARM64NEG)
			v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(log64(c + 1))
			v1 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
			v1.AddArg(x)
			v0.AddArg2(v1, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo64(c/3) && is32Bit(c)
	// result: (SLLconst <x.Type> [log64(c/3)] (SUBshiftLL <x.Type> x x [2]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%3 == 0 && isPowerOfTwo64(c/3) && is32Bit(c)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.Type = x.Type
			v.AuxInt = int64ToAuxInt(log64(c / 3))
			v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(2)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo64(c/5) && is32Bit(c)
	// result: (NEG (SLLconst <x.Type> [log64(c/5)] (ADDshiftLL <x.Type> x x [2])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%5 == 0 && isPowerOfTwo64(c/5) && is32Bit(c)) {
				continue
			}
			v.reset(OpARM64NEG)
			v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
			v0.AuxInt = int64ToAuxInt(log64(c / 5))
			v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v1.AuxInt = int64ToAuxInt(2)
			v1.AddArg2(x, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo64(c/7) && is32Bit(c)
	// result: (SLLconst <x.Type> [log64(c/7)] (SUBshiftLL <x.Type> x x [3]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%7 == 0 && isPowerOfTwo64(c/7) && is32Bit(c)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.Type = x.Type
			v.AuxInt = int64ToAuxInt(log64(c / 7))
			v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(3)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo64(c/9) && is32Bit(c)
	// result: (NEG (SLLconst <x.Type> [log64(c/9)] (ADDshiftLL <x.Type> x x [3])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%9 == 0 && isPowerOfTwo64(c/9) && is32Bit(c)) {
				continue
			}
			v.reset(OpARM64NEG)
			v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
			v0.AuxInt = int64ToAuxInt(log64(c / 9))
			v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v1.AuxInt = int64ToAuxInt(3)
			v1.AddArg2(x, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [-int64(int32(c)*int32(d))])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64MOVDconst)
			v.AuxInt = int64ToAuxInt(-int64(int32(c) * int32(d)))
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64MOD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOD (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [c%d])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(c % d)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MODW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MODW (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [int64(int32(c)%int32(d))])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int32(c) % int32(d)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBUload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBUloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVBUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVBUload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBUload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBUload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} ptr (MOVBstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVBstorezero {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MOVBUload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVDconst [int64(read8(sym, int64(off)))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(read8(sym, int64(off))))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBUloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBUloadidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVBUload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVBUload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUloadidx (MOVDconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVBUload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVBUload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUloadidx ptr idx (MOVBstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVBstorezeroidx {
			break
		}
		idx2 := v_2.Args[1]
		ptr2 := v_2.Args[0]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (ANDconst [c] x))
	// result: (ANDconst [c&(1<<8-1)] x)
	for {
		if v_0.Op != OpARM64ANDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c & (1<<8 - 1))
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint8(c))])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint8(c)))
		return true
	}
	// match: (MOVBUreg x)
	// cond: x.Type.IsBoolean()
	// result: (MOVDreg x)
	for {
		x := v_0
		if !(x.Type.IsBoolean()) {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (SLLconst [sc] x))
	// cond: isARM64BFMask(sc, 1<<8-1, sc)
	// result: (UBFIZ [armBFAuxInt(sc, arm64BFWidth(1<<8-1, sc))] x)
	for {
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<8-1, sc)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, arm64BFWidth(1<<8-1, sc)))
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (SRLconst [sc] x))
	// cond: isARM64BFMask(sc, 1<<8-1, 0)
	// result: (UBFX [armBFAuxInt(sc, 8)] x)
	for {
		if v_0.Op != OpARM64SRLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<8-1, 0)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, 8))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVBloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off] {sym} ptr (MOVBstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVBstorezero {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBloadidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVBload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVBload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBloadidx (MOVDconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVBload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVBload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBloadidx ptr idx (MOVBstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVBstorezeroidx {
			break
		}
		idx2 := v_2.Args[1]
		ptr2 := v_2.Args[0]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int8(c))])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int8(c)))
		return true
	}
	// match: (MOVBreg (SLLconst [lc] x))
	// cond: lc < 8
	// result: (SBFIZ [armBFAuxInt(lc, 8-lc)] x)
	for {
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < 8) {
			break
		}
		v.reset(OpARM64SBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc, 8-lc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBstoreidx ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBstore)
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
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpARM64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBreg x) mem)
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVBreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVBstore)
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
		if v_1.Op != OpARM64MOVBUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVBstore)
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
		if v_1.Op != OpARM64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVBstore)
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
		if v_1.Op != OpARM64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVBstore)
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
		if v_1.Op != OpARM64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVBstore)
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
		if v_1.Op != OpARM64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr0 (SRLconst [8] w) x:(MOVBstore [i-1] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 8 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w, mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (SRLconst [8] w) x:(MOVBstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w mem)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 8 {
				continue
			}
			w := v_1.Args[0]
			x := v_2
			if x.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			if w != x.Args[2] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVHstoreidx)
			v.AddArg4(ptr1, idx1, w, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr0 (UBFX [armBFAuxInt(8, 8)] w) x:(MOVBstore [i-1] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64UBFX || auxIntToArm64BitField(v_1.AuxInt) != armBFAuxInt(8, 8) {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w, mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (UBFX [armBFAuxInt(8, 8)] w) x:(MOVBstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w mem)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64UBFX || auxIntToArm64BitField(v_1.AuxInt) != armBFAuxInt(8, 8) {
				continue
			}
			w := v_1.Args[0]
			x := v_2
			if x.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			if w != x.Args[2] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVHstoreidx)
			v.AddArg4(ptr1, idx1, w, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr0 (UBFX [armBFAuxInt(8, 24)] w) x:(MOVBstore [i-1] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64UBFX || auxIntToArm64BitField(v_1.AuxInt) != armBFAuxInt(8, 24) {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w, mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (UBFX [armBFAuxInt(8, 24)] w) x:(MOVBstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w mem)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64UBFX || auxIntToArm64BitField(v_1.AuxInt) != armBFAuxInt(8, 24) {
				continue
			}
			w := v_1.Args[0]
			x := v_2
			if x.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			if w != x.Args[2] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVHstoreidx)
			v.AddArg4(ptr1, idx1, w, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr0 (SRLconst [8] (MOVDreg w)) x:(MOVBstore [i-1] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 8 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVDreg {
			break
		}
		w := v_1_0.Args[0]
		x := v_2
		if x.Op != OpARM64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w, mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (SRLconst [8] (MOVDreg w)) x:(MOVBstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w mem)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 8 {
				continue
			}
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64MOVDreg {
				continue
			}
			w := v_1_0.Args[0]
			x := v_2
			if x.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			if w != x.Args[2] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVHstoreidx)
			v.AddArg4(ptr1, idx1, w, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr0 (SRLconst [j] w) x:(MOVBstore [i-1] {s} ptr1 w0:(SRLconst [j-8] w) mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := auxIntToInt64(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpARM64SRLconst || auxIntToInt64(w0.AuxInt) != j-8 || w != w0.Args[0] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w0, mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (SRLconst [j] w) x:(MOVBstoreidx ptr1 idx1 w0:(SRLconst [j-8] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w0 mem)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64SRLconst {
				continue
			}
			j := auxIntToInt64(v_1.AuxInt)
			w := v_1.Args[0]
			x := v_2
			if x.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			w0 := x.Args[2]
			if w0.Op != OpARM64SRLconst || auxIntToInt64(w0.AuxInt) != j-8 || w != w0.Args[0] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVHstoreidx)
			v.AddArg4(ptr1, idx1, w0, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr0 (UBFX [bfc] w) x:(MOVBstore [i-1] {s} ptr1 w0:(UBFX [bfc2] w) mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && bfc.getARM64BFwidth() == 32 - bfc.getARM64BFlsb() && bfc2.getARM64BFwidth() == 32 - bfc2.getARM64BFlsb() && bfc2.getARM64BFlsb() == bfc.getARM64BFlsb() - 8 && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpARM64UBFX {
			break
		}
		bfc2 := auxIntToArm64BitField(w0.AuxInt)
		if w != w0.Args[0] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && bfc.getARM64BFwidth() == 32-bfc.getARM64BFlsb() && bfc2.getARM64BFwidth() == 32-bfc2.getARM64BFlsb() && bfc2.getARM64BFlsb() == bfc.getARM64BFlsb()-8 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w0, mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (UBFX [bfc] w) x:(MOVBstoreidx ptr1 idx1 w0:(UBFX [bfc2] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && bfc.getARM64BFwidth() == 32 - bfc.getARM64BFlsb() && bfc2.getARM64BFwidth() == 32 - bfc2.getARM64BFlsb() && bfc2.getARM64BFlsb() == bfc.getARM64BFlsb() - 8 && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w0 mem)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64UBFX {
				continue
			}
			bfc := auxIntToArm64BitField(v_1.AuxInt)
			w := v_1.Args[0]
			x := v_2
			if x.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			w0 := x.Args[2]
			if w0.Op != OpARM64UBFX {
				continue
			}
			bfc2 := auxIntToArm64BitField(w0.AuxInt)
			if w != w0.Args[0] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && bfc.getARM64BFwidth() == 32-bfc.getARM64BFlsb() && bfc2.getARM64BFwidth() == 32-bfc2.getARM64BFlsb() && bfc2.getARM64BFlsb() == bfc.getARM64BFlsb()-8 && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVHstoreidx)
			v.AddArg4(ptr1, idx1, w0, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr0 (SRLconst [j] (MOVDreg w)) x:(MOVBstore [i-1] {s} ptr1 w0:(SRLconst [j-8] (MOVDreg w)) mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := auxIntToInt64(v_1.AuxInt)
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVDreg {
			break
		}
		w := v_1_0.Args[0]
		x := v_2
		if x.Op != OpARM64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpARM64SRLconst || auxIntToInt64(w0.AuxInt) != j-8 {
			break
		}
		w0_0 := w0.Args[0]
		if w0_0.Op != OpARM64MOVDreg || w != w0_0.Args[0] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w0, mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (SRLconst [j] (MOVDreg w)) x:(MOVBstoreidx ptr1 idx1 w0:(SRLconst [j-8] (MOVDreg w)) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w0 mem)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64SRLconst {
				continue
			}
			j := auxIntToInt64(v_1.AuxInt)
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64MOVDreg {
				continue
			}
			w := v_1_0.Args[0]
			x := v_2
			if x.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			w0 := x.Args[2]
			if w0.Op != OpARM64SRLconst || auxIntToInt64(w0.AuxInt) != j-8 {
				continue
			}
			w0_0 := w0.Args[0]
			if w0_0.Op != OpARM64MOVDreg || w != w0_0.Args[0] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVHstoreidx)
			v.AddArg4(ptr1, idx1, w0, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr w x0:(MOVBstore [i-1] {s} ptr (SRLconst [8] w) x1:(MOVBstore [i-2] {s} ptr (SRLconst [16] w) x2:(MOVBstore [i-3] {s} ptr (SRLconst [24] w) x3:(MOVBstore [i-4] {s} ptr (SRLconst [32] w) x4:(MOVBstore [i-5] {s} ptr (SRLconst [40] w) x5:(MOVBstore [i-6] {s} ptr (SRLconst [48] w) x6:(MOVBstore [i-7] {s} ptr (SRLconst [56] w) mem))))))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && clobber(x0, x1, x2, x3, x4, x5, x6)
	// result: (MOVDstore [i-7] {s} ptr (REV <w.Type> w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr := v_0
		w := v_1
		x0 := v_2
		if x0.Op != OpARM64MOVBstore || auxIntToInt32(x0.AuxInt) != i-1 || auxToSym(x0.Aux) != s {
			break
		}
		_ = x0.Args[2]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64SRLconst || auxIntToInt64(x0_1.AuxInt) != 8 || w != x0_1.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore || auxIntToInt32(x1.AuxInt) != i-2 || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64SRLconst || auxIntToInt64(x1_1.AuxInt) != 16 || w != x1_1.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstore || auxIntToInt32(x2.AuxInt) != i-3 || auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[2]
		if ptr != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64SRLconst || auxIntToInt64(x2_1.AuxInt) != 24 || w != x2_1.Args[0] {
			break
		}
		x3 := x2.Args[2]
		if x3.Op != OpARM64MOVBstore || auxIntToInt32(x3.AuxInt) != i-4 || auxToSym(x3.Aux) != s {
			break
		}
		_ = x3.Args[2]
		if ptr != x3.Args[0] {
			break
		}
		x3_1 := x3.Args[1]
		if x3_1.Op != OpARM64SRLconst || auxIntToInt64(x3_1.AuxInt) != 32 || w != x3_1.Args[0] {
			break
		}
		x4 := x3.Args[2]
		if x4.Op != OpARM64MOVBstore || auxIntToInt32(x4.AuxInt) != i-5 || auxToSym(x4.Aux) != s {
			break
		}
		_ = x4.Args[2]
		if ptr != x4.Args[0] {
			break
		}
		x4_1 := x4.Args[1]
		if x4_1.Op != OpARM64SRLconst || auxIntToInt64(x4_1.AuxInt) != 40 || w != x4_1.Args[0] {
			break
		}
		x5 := x4.Args[2]
		if x5.Op != OpARM64MOVBstore || auxIntToInt32(x5.AuxInt) != i-6 || auxToSym(x5.Aux) != s {
			break
		}
		_ = x5.Args[2]
		if ptr != x5.Args[0] {
			break
		}
		x5_1 := x5.Args[1]
		if x5_1.Op != OpARM64SRLconst || auxIntToInt64(x5_1.AuxInt) != 48 || w != x5_1.Args[0] {
			break
		}
		x6 := x5.Args[2]
		if x6.Op != OpARM64MOVBstore || auxIntToInt32(x6.AuxInt) != i-7 || auxToSym(x6.Aux) != s {
			break
		}
		mem := x6.Args[2]
		if ptr != x6.Args[0] {
			break
		}
		x6_1 := x6.Args[1]
		if x6_1.Op != OpARM64SRLconst || auxIntToInt64(x6_1.AuxInt) != 56 || w != x6_1.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && clobber(x0, x1, x2, x3, x4, x5, x6)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(i - 7)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x6.Pos, OpARM64REV, w.Type)
		v0.AddArg(w)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [7] {s} p w x0:(MOVBstore [6] {s} p (SRLconst [8] w) x1:(MOVBstore [5] {s} p (SRLconst [16] w) x2:(MOVBstore [4] {s} p (SRLconst [24] w) x3:(MOVBstore [3] {s} p (SRLconst [32] w) x4:(MOVBstore [2] {s} p (SRLconst [40] w) x5:(MOVBstore [1] {s} p1:(ADD ptr1 idx1) (SRLconst [48] w) x6:(MOVBstoreidx ptr0 idx0 (SRLconst [56] w) mem))))))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, x4, x5, x6)
	// result: (MOVDstoreidx ptr0 idx0 (REV <w.Type> w) mem)
	for {
		if auxIntToInt32(v.AuxInt) != 7 {
			break
		}
		s := auxToSym(v.Aux)
		p := v_0
		w := v_1
		x0 := v_2
		if x0.Op != OpARM64MOVBstore || auxIntToInt32(x0.AuxInt) != 6 || auxToSym(x0.Aux) != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64SRLconst || auxIntToInt64(x0_1.AuxInt) != 8 || w != x0_1.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore || auxIntToInt32(x1.AuxInt) != 5 || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64SRLconst || auxIntToInt64(x1_1.AuxInt) != 16 || w != x1_1.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstore || auxIntToInt32(x2.AuxInt) != 4 || auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[2]
		if p != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64SRLconst || auxIntToInt64(x2_1.AuxInt) != 24 || w != x2_1.Args[0] {
			break
		}
		x3 := x2.Args[2]
		if x3.Op != OpARM64MOVBstore || auxIntToInt32(x3.AuxInt) != 3 || auxToSym(x3.Aux) != s {
			break
		}
		_ = x3.Args[2]
		if p != x3.Args[0] {
			break
		}
		x3_1 := x3.Args[1]
		if x3_1.Op != OpARM64SRLconst || auxIntToInt64(x3_1.AuxInt) != 32 || w != x3_1.Args[0] {
			break
		}
		x4 := x3.Args[2]
		if x4.Op != OpARM64MOVBstore || auxIntToInt32(x4.AuxInt) != 2 || auxToSym(x4.Aux) != s {
			break
		}
		_ = x4.Args[2]
		if p != x4.Args[0] {
			break
		}
		x4_1 := x4.Args[1]
		if x4_1.Op != OpARM64SRLconst || auxIntToInt64(x4_1.AuxInt) != 40 || w != x4_1.Args[0] {
			break
		}
		x5 := x4.Args[2]
		if x5.Op != OpARM64MOVBstore || auxIntToInt32(x5.AuxInt) != 1 || auxToSym(x5.Aux) != s {
			break
		}
		_ = x5.Args[2]
		p1 := x5.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		_ = p1.Args[1]
		p1_0 := p1.Args[0]
		p1_1 := p1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, p1_0, p1_1 = _i0+1, p1_1, p1_0 {
			ptr1 := p1_0
			idx1 := p1_1
			x5_1 := x5.Args[1]
			if x5_1.Op != OpARM64SRLconst || auxIntToInt64(x5_1.AuxInt) != 48 || w != x5_1.Args[0] {
				continue
			}
			x6 := x5.Args[2]
			if x6.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x6.Args[3]
			ptr0 := x6.Args[0]
			idx0 := x6.Args[1]
			x6_2 := x6.Args[2]
			if x6_2.Op != OpARM64SRLconst || auxIntToInt64(x6_2.AuxInt) != 56 || w != x6_2.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, x4, x5, x6)) {
				continue
			}
			v.reset(OpARM64MOVDstoreidx)
			v0 := b.NewValue0(x5.Pos, OpARM64REV, w.Type)
			v0.AddArg(w)
			v.AddArg4(ptr0, idx0, v0, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr w x0:(MOVBstore [i-1] {s} ptr (UBFX [armBFAuxInt(8, 24)] w) x1:(MOVBstore [i-2] {s} ptr (UBFX [armBFAuxInt(16, 16)] w) x2:(MOVBstore [i-3] {s} ptr (UBFX [armBFAuxInt(24, 8)] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0, x1, x2)
	// result: (MOVWstore [i-3] {s} ptr (REVW <w.Type> w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr := v_0
		w := v_1
		x0 := v_2
		if x0.Op != OpARM64MOVBstore || auxIntToInt32(x0.AuxInt) != i-1 || auxToSym(x0.Aux) != s {
			break
		}
		_ = x0.Args[2]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64UBFX || auxIntToArm64BitField(x0_1.AuxInt) != armBFAuxInt(8, 24) || w != x0_1.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore || auxIntToInt32(x1.AuxInt) != i-2 || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64UBFX || auxIntToArm64BitField(x1_1.AuxInt) != armBFAuxInt(16, 16) || w != x1_1.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstore || auxIntToInt32(x2.AuxInt) != i-3 || auxToSym(x2.Aux) != s {
			break
		}
		mem := x2.Args[2]
		if ptr != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64UBFX || auxIntToArm64BitField(x2_1.AuxInt) != armBFAuxInt(24, 8) || w != x2_1.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0, x1, x2)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 3)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x2.Pos, OpARM64REVW, w.Type)
		v0.AddArg(w)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [3] {s} p w x0:(MOVBstore [2] {s} p (UBFX [armBFAuxInt(8, 24)] w) x1:(MOVBstore [1] {s} p1:(ADD ptr1 idx1) (UBFX [armBFAuxInt(16, 16)] w) x2:(MOVBstoreidx ptr0 idx0 (UBFX [armBFAuxInt(24, 8)] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2)
	// result: (MOVWstoreidx ptr0 idx0 (REVW <w.Type> w) mem)
	for {
		if auxIntToInt32(v.AuxInt) != 3 {
			break
		}
		s := auxToSym(v.Aux)
		p := v_0
		w := v_1
		x0 := v_2
		if x0.Op != OpARM64MOVBstore || auxIntToInt32(x0.AuxInt) != 2 || auxToSym(x0.Aux) != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64UBFX || auxIntToArm64BitField(x0_1.AuxInt) != armBFAuxInt(8, 24) || w != x0_1.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore || auxIntToInt32(x1.AuxInt) != 1 || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		_ = p1.Args[1]
		p1_0 := p1.Args[0]
		p1_1 := p1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, p1_0, p1_1 = _i0+1, p1_1, p1_0 {
			ptr1 := p1_0
			idx1 := p1_1
			x1_1 := x1.Args[1]
			if x1_1.Op != OpARM64UBFX || auxIntToArm64BitField(x1_1.AuxInt) != armBFAuxInt(16, 16) || w != x1_1.Args[0] {
				continue
			}
			x2 := x1.Args[2]
			if x2.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x2.Args[3]
			ptr0 := x2.Args[0]
			idx0 := x2.Args[1]
			x2_2 := x2.Args[2]
			if x2_2.Op != OpARM64UBFX || auxIntToArm64BitField(x2_2.AuxInt) != armBFAuxInt(24, 8) || w != x2_2.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2)) {
				continue
			}
			v.reset(OpARM64MOVWstoreidx)
			v0 := b.NewValue0(x1.Pos, OpARM64REVW, w.Type)
			v0.AddArg(w)
			v.AddArg4(ptr0, idx0, v0, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr w x0:(MOVBstore [i-1] {s} ptr (SRLconst [8] (MOVDreg w)) x1:(MOVBstore [i-2] {s} ptr (SRLconst [16] (MOVDreg w)) x2:(MOVBstore [i-3] {s} ptr (SRLconst [24] (MOVDreg w)) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0, x1, x2)
	// result: (MOVWstore [i-3] {s} ptr (REVW <w.Type> w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr := v_0
		w := v_1
		x0 := v_2
		if x0.Op != OpARM64MOVBstore || auxIntToInt32(x0.AuxInt) != i-1 || auxToSym(x0.Aux) != s {
			break
		}
		_ = x0.Args[2]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64SRLconst || auxIntToInt64(x0_1.AuxInt) != 8 {
			break
		}
		x0_1_0 := x0_1.Args[0]
		if x0_1_0.Op != OpARM64MOVDreg || w != x0_1_0.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore || auxIntToInt32(x1.AuxInt) != i-2 || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64SRLconst || auxIntToInt64(x1_1.AuxInt) != 16 {
			break
		}
		x1_1_0 := x1_1.Args[0]
		if x1_1_0.Op != OpARM64MOVDreg || w != x1_1_0.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstore || auxIntToInt32(x2.AuxInt) != i-3 || auxToSym(x2.Aux) != s {
			break
		}
		mem := x2.Args[2]
		if ptr != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64SRLconst || auxIntToInt64(x2_1.AuxInt) != 24 {
			break
		}
		x2_1_0 := x2_1.Args[0]
		if x2_1_0.Op != OpARM64MOVDreg || w != x2_1_0.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0, x1, x2)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 3)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x2.Pos, OpARM64REVW, w.Type)
		v0.AddArg(w)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [3] {s} p w x0:(MOVBstore [2] {s} p (SRLconst [8] (MOVDreg w)) x1:(MOVBstore [1] {s} p1:(ADD ptr1 idx1) (SRLconst [16] (MOVDreg w)) x2:(MOVBstoreidx ptr0 idx0 (SRLconst [24] (MOVDreg w)) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2)
	// result: (MOVWstoreidx ptr0 idx0 (REVW <w.Type> w) mem)
	for {
		if auxIntToInt32(v.AuxInt) != 3 {
			break
		}
		s := auxToSym(v.Aux)
		p := v_0
		w := v_1
		x0 := v_2
		if x0.Op != OpARM64MOVBstore || auxIntToInt32(x0.AuxInt) != 2 || auxToSym(x0.Aux) != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64SRLconst || auxIntToInt64(x0_1.AuxInt) != 8 {
			break
		}
		x0_1_0 := x0_1.Args[0]
		if x0_1_0.Op != OpARM64MOVDreg || w != x0_1_0.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore || auxIntToInt32(x1.AuxInt) != 1 || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		_ = p1.Args[1]
		p1_0 := p1.Args[0]
		p1_1 := p1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, p1_0, p1_1 = _i0+1, p1_1, p1_0 {
			ptr1 := p1_0
			idx1 := p1_1
			x1_1 := x1.Args[1]
			if x1_1.Op != OpARM64SRLconst || auxIntToInt64(x1_1.AuxInt) != 16 {
				continue
			}
			x1_1_0 := x1_1.Args[0]
			if x1_1_0.Op != OpARM64MOVDreg || w != x1_1_0.Args[0] {
				continue
			}
			x2 := x1.Args[2]
			if x2.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x2.Args[3]
			ptr0 := x2.Args[0]
			idx0 := x2.Args[1]
			x2_2 := x2.Args[2]
			if x2_2.Op != OpARM64SRLconst || auxIntToInt64(x2_2.AuxInt) != 24 {
				continue
			}
			x2_2_0 := x2_2.Args[0]
			if x2_2_0.Op != OpARM64MOVDreg || w != x2_2_0.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2)) {
				continue
			}
			v.reset(OpARM64MOVWstoreidx)
			v0 := b.NewValue0(x1.Pos, OpARM64REVW, w.Type)
			v0.AddArg(w)
			v.AddArg4(ptr0, idx0, v0, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr w x0:(MOVBstore [i-1] {s} ptr (SRLconst [8] w) x1:(MOVBstore [i-2] {s} ptr (SRLconst [16] w) x2:(MOVBstore [i-3] {s} ptr (SRLconst [24] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0, x1, x2)
	// result: (MOVWstore [i-3] {s} ptr (REVW <w.Type> w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr := v_0
		w := v_1
		x0 := v_2
		if x0.Op != OpARM64MOVBstore || auxIntToInt32(x0.AuxInt) != i-1 || auxToSym(x0.Aux) != s {
			break
		}
		_ = x0.Args[2]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64SRLconst || auxIntToInt64(x0_1.AuxInt) != 8 || w != x0_1.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore || auxIntToInt32(x1.AuxInt) != i-2 || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64SRLconst || auxIntToInt64(x1_1.AuxInt) != 16 || w != x1_1.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstore || auxIntToInt32(x2.AuxInt) != i-3 || auxToSym(x2.Aux) != s {
			break
		}
		mem := x2.Args[2]
		if ptr != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64SRLconst || auxIntToInt64(x2_1.AuxInt) != 24 || w != x2_1.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0, x1, x2)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 3)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x2.Pos, OpARM64REVW, w.Type)
		v0.AddArg(w)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [3] {s} p w x0:(MOVBstore [2] {s} p (SRLconst [8] w) x1:(MOVBstore [1] {s} p1:(ADD ptr1 idx1) (SRLconst [16] w) x2:(MOVBstoreidx ptr0 idx0 (SRLconst [24] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2)
	// result: (MOVWstoreidx ptr0 idx0 (REVW <w.Type> w) mem)
	for {
		if auxIntToInt32(v.AuxInt) != 3 {
			break
		}
		s := auxToSym(v.Aux)
		p := v_0
		w := v_1
		x0 := v_2
		if x0.Op != OpARM64MOVBstore || auxIntToInt32(x0.AuxInt) != 2 || auxToSym(x0.Aux) != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64SRLconst || auxIntToInt64(x0_1.AuxInt) != 8 || w != x0_1.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore || auxIntToInt32(x1.AuxInt) != 1 || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[2]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		_ = p1.Args[1]
		p1_0 := p1.Args[0]
		p1_1 := p1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, p1_0, p1_1 = _i0+1, p1_1, p1_0 {
			ptr1 := p1_0
			idx1 := p1_1
			x1_1 := x1.Args[1]
			if x1_1.Op != OpARM64SRLconst || auxIntToInt64(x1_1.AuxInt) != 16 || w != x1_1.Args[0] {
				continue
			}
			x2 := x1.Args[2]
			if x2.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x2.Args[3]
			ptr0 := x2.Args[0]
			idx0 := x2.Args[1]
			x2_2 := x2.Args[2]
			if x2_2.Op != OpARM64SRLconst || auxIntToInt64(x2_2.AuxInt) != 24 || w != x2_2.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2)) {
				continue
			}
			v.reset(OpARM64MOVWstoreidx)
			v0 := b.NewValue0(x1.Pos, OpARM64REVW, w.Type)
			v0.AddArg(w)
			v.AddArg4(ptr0, idx0, v0, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr w x:(MOVBstore [i-1] {s} ptr (SRLconst [8] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr (REV16W <w.Type> w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr := v_0
		w := v_1
		x := v_2
		if x.Op != OpARM64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpARM64SRLconst || auxIntToInt64(x_1.AuxInt) != 8 || w != x_1.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr1 idx1) w x:(MOVBstoreidx ptr0 idx0 (SRLconst [8] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr0 idx0 (REV16W <w.Type> w) mem)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr1 := v_0_0
			idx1 := v_0_1
			w := v_1
			x := v_2
			if x.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr0 := x.Args[0]
			idx0 := x.Args[1]
			x_2 := x.Args[2]
			if x_2.Op != OpARM64SRLconst || auxIntToInt64(x_2.AuxInt) != 8 || w != x_2.Args[0] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVHstoreidx)
			v0 := b.NewValue0(v.Pos, OpARM64REV16W, w.Type)
			v0.AddArg(w)
			v.AddArg4(ptr0, idx0, v0, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr w x:(MOVBstore [i-1] {s} ptr (UBFX [armBFAuxInt(8, 8)] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr (REV16W <w.Type> w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr := v_0
		w := v_1
		x := v_2
		if x.Op != OpARM64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpARM64UBFX || auxIntToArm64BitField(x_1.AuxInt) != armBFAuxInt(8, 8) || w != x_1.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr1 idx1) w x:(MOVBstoreidx ptr0 idx0 (UBFX [armBFAuxInt(8, 8)] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr0 idx0 (REV16W <w.Type> w) mem)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr1 := v_0_0
			idx1 := v_0_1
			w := v_1
			x := v_2
			if x.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr0 := x.Args[0]
			idx0 := x.Args[1]
			x_2 := x.Args[2]
			if x_2.Op != OpARM64UBFX || auxIntToArm64BitField(x_2.AuxInt) != armBFAuxInt(8, 8) || w != x_2.Args[0] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVHstoreidx)
			v0 := b.NewValue0(v.Pos, OpARM64REV16W, w.Type)
			v0.AddArg(w)
			v.AddArg4(ptr0, idx0, v0, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr w x:(MOVBstore [i-1] {s} ptr (SRLconst [8] (MOVDreg w)) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr (REV16W <w.Type> w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr := v_0
		w := v_1
		x := v_2
		if x.Op != OpARM64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpARM64SRLconst || auxIntToInt64(x_1.AuxInt) != 8 {
			break
		}
		x_1_0 := x_1.Args[0]
		if x_1_0.Op != OpARM64MOVDreg || w != x_1_0.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr1 idx1) w x:(MOVBstoreidx ptr0 idx0 (SRLconst [8] (MOVDreg w)) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr0 idx0 (REV16W <w.Type> w) mem)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr1 := v_0_0
			idx1 := v_0_1
			w := v_1
			x := v_2
			if x.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr0 := x.Args[0]
			idx0 := x.Args[1]
			x_2 := x.Args[2]
			if x_2.Op != OpARM64SRLconst || auxIntToInt64(x_2.AuxInt) != 8 {
				continue
			}
			x_2_0 := x_2.Args[0]
			if x_2_0.Op != OpARM64MOVDreg || w != x_2_0.Args[0] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVHstoreidx)
			v0 := b.NewValue0(v.Pos, OpARM64REV16W, w.Type)
			v0.AddArg(w)
			v.AddArg4(ptr0, idx0, v0, mem)
			return true
		}
		break
	}
	// match: (MOVBstore [i] {s} ptr w x:(MOVBstore [i-1] {s} ptr (UBFX [armBFAuxInt(8, 24)] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr (REV16W <w.Type> w) mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr := v_0
		w := v_1
		x := v_2
		if x.Op != OpARM64MOVBstore || auxIntToInt32(x.AuxInt) != i-1 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpARM64UBFX || auxIntToArm64BitField(x_1.AuxInt) != armBFAuxInt(8, 24) || w != x_1.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(i - 1)
		v.Aux = symToAux(s)
		v0 := b.NewValue0(x.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr1 idx1) w x:(MOVBstoreidx ptr0 idx0 (UBFX [armBFAuxInt(8, 24)] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr0 idx0 (REV16W <w.Type> w) mem)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr1 := v_0_0
			idx1 := v_0_1
			w := v_1
			x := v_2
			if x.Op != OpARM64MOVBstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr0 := x.Args[0]
			idx0 := x.Args[1]
			x_2 := x.Args[2]
			if x_2.Op != OpARM64UBFX || auxIntToArm64BitField(x_2.AuxInt) != armBFAuxInt(8, 24) || w != x_2.Args[0] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVHstoreidx)
			v0 := b.NewValue0(v.Pos, OpARM64REV16W, w.Type)
			v0.AddArg(w)
			v.AddArg4(ptr0, idx0, v0, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MOVBstoreidx ptr (MOVDconst [c]) val mem)
	// cond: is32Bit(c)
	// result: (MOVBstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstoreidx (MOVDconst [c]) idx val mem)
	// cond: is32Bit(c)
	// result: (MOVBstore [int32(c)] idx val mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVDconst [0]) mem)
	// result: (MOVBstorezeroidx ptr idx mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		mem := v_3
		v.reset(OpARM64MOVBstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVBreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVBreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVBUreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVBUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVHreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVHreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVHUreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVHUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVWreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVWUreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr (ADDconst [1] idx) (SRLconst [8] w) x:(MOVBstoreidx ptr idx w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstoreidx ptr idx w mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64ADDconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		idx := v_1.Args[0]
		if v_2.Op != OpARM64SRLconst || auxIntToInt64(v_2.AuxInt) != 8 {
			break
		}
		w := v_2.Args[0]
		x := v_3
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		if ptr != x.Args[0] || idx != x.Args[1] || w != x.Args[2] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg4(ptr, idx, w, mem)
		return true
	}
	// match: (MOVBstoreidx ptr (ADDconst [3] idx) w x0:(MOVBstoreidx ptr (ADDconst [2] idx) (UBFX [armBFAuxInt(8, 24)] w) x1:(MOVBstoreidx ptr (ADDconst [1] idx) (UBFX [armBFAuxInt(16, 16)] w) x2:(MOVBstoreidx ptr idx (UBFX [armBFAuxInt(24, 8)] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0, x1, x2)
	// result: (MOVWstoreidx ptr idx (REVW <w.Type> w) mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64ADDconst || auxIntToInt64(v_1.AuxInt) != 3 {
			break
		}
		idx := v_1.Args[0]
		w := v_2
		x0 := v_3
		if x0.Op != OpARM64MOVBstoreidx {
			break
		}
		_ = x0.Args[3]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64ADDconst || auxIntToInt64(x0_1.AuxInt) != 2 || idx != x0_1.Args[0] {
			break
		}
		x0_2 := x0.Args[2]
		if x0_2.Op != OpARM64UBFX || auxIntToArm64BitField(x0_2.AuxInt) != armBFAuxInt(8, 24) || w != x0_2.Args[0] {
			break
		}
		x1 := x0.Args[3]
		if x1.Op != OpARM64MOVBstoreidx {
			break
		}
		_ = x1.Args[3]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64ADDconst || auxIntToInt64(x1_1.AuxInt) != 1 || idx != x1_1.Args[0] {
			break
		}
		x1_2 := x1.Args[2]
		if x1_2.Op != OpARM64UBFX || auxIntToArm64BitField(x1_2.AuxInt) != armBFAuxInt(16, 16) || w != x1_2.Args[0] {
			break
		}
		x2 := x1.Args[3]
		if x2.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x2.Args[3]
		if ptr != x2.Args[0] || idx != x2.Args[1] {
			break
		}
		x2_2 := x2.Args[2]
		if x2_2.Op != OpARM64UBFX || auxIntToArm64BitField(x2_2.AuxInt) != armBFAuxInt(24, 8) || w != x2_2.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0, x1, x2)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v0 := b.NewValue0(v.Pos, OpARM64REVW, w.Type)
		v0.AddArg(w)
		v.AddArg4(ptr, idx, v0, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx w x0:(MOVBstoreidx ptr (ADDconst [1] idx) (UBFX [armBFAuxInt(8, 24)] w) x1:(MOVBstoreidx ptr (ADDconst [2] idx) (UBFX [armBFAuxInt(16, 16)] w) x2:(MOVBstoreidx ptr (ADDconst [3] idx) (UBFX [armBFAuxInt(24, 8)] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0, x1, x2)
	// result: (MOVWstoreidx ptr idx w mem)
	for {
		ptr := v_0
		idx := v_1
		w := v_2
		x0 := v_3
		if x0.Op != OpARM64MOVBstoreidx {
			break
		}
		_ = x0.Args[3]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64ADDconst || auxIntToInt64(x0_1.AuxInt) != 1 || idx != x0_1.Args[0] {
			break
		}
		x0_2 := x0.Args[2]
		if x0_2.Op != OpARM64UBFX || auxIntToArm64BitField(x0_2.AuxInt) != armBFAuxInt(8, 24) || w != x0_2.Args[0] {
			break
		}
		x1 := x0.Args[3]
		if x1.Op != OpARM64MOVBstoreidx {
			break
		}
		_ = x1.Args[3]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64ADDconst || auxIntToInt64(x1_1.AuxInt) != 2 || idx != x1_1.Args[0] {
			break
		}
		x1_2 := x1.Args[2]
		if x1_2.Op != OpARM64UBFX || auxIntToArm64BitField(x1_2.AuxInt) != armBFAuxInt(16, 16) || w != x1_2.Args[0] {
			break
		}
		x2 := x1.Args[3]
		if x2.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x2.Args[3]
		if ptr != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64ADDconst || auxIntToInt64(x2_1.AuxInt) != 3 || idx != x2_1.Args[0] {
			break
		}
		x2_2 := x2.Args[2]
		if x2_2.Op != OpARM64UBFX || auxIntToArm64BitField(x2_2.AuxInt) != armBFAuxInt(24, 8) || w != x2_2.Args[0] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0, x1, x2)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg4(ptr, idx, w, mem)
		return true
	}
	// match: (MOVBstoreidx ptr (ADDconst [1] idx) w x:(MOVBstoreidx ptr idx (UBFX [armBFAuxInt(8, 8)] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstoreidx ptr idx (REV16W <w.Type> w) mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64ADDconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		idx := v_1.Args[0]
		w := v_2
		x := v_3
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		if ptr != x.Args[0] || idx != x.Args[1] {
			break
		}
		x_2 := x.Args[2]
		if x_2.Op != OpARM64UBFX || auxIntToArm64BitField(x_2.AuxInt) != armBFAuxInt(8, 8) || w != x_2.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v0 := b.NewValue0(v.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg4(ptr, idx, v0, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx w x:(MOVBstoreidx ptr (ADDconst [1] idx) (UBFX [armBFAuxInt(8, 8)] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstoreidx ptr idx w mem)
	for {
		ptr := v_0
		idx := v_1
		w := v_2
		x := v_3
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		if ptr != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpARM64ADDconst || auxIntToInt64(x_1.AuxInt) != 1 || idx != x_1.Args[0] {
			break
		}
		x_2 := x.Args[2]
		if x_2.Op != OpARM64UBFX || auxIntToArm64BitField(x_2.AuxInt) != armBFAuxInt(8, 8) || w != x_2.Args[0] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg4(ptr, idx, w, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstorezero [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstorezero [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBstorezeroidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVBstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVBstorezero [i] {s} ptr0 x:(MOVBstorezero [j] {s} ptr1 mem))
	// cond: x.Uses == 1 && areAdjacentOffsets(int64(i),int64(j),1) && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstorezero [int32(min(int64(i),int64(j)))] {s} ptr0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		x := v_1
		if x.Op != OpARM64MOVBstorezero {
			break
		}
		j := auxIntToInt32(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		ptr1 := x.Args[0]
		if !(x.Uses == 1 && areAdjacentOffsets(int64(i), int64(j), 1) && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(int32(min(int64(i), int64(j))))
		v.Aux = symToAux(s)
		v.AddArg2(ptr0, mem)
		return true
	}
	// match: (MOVBstorezero [1] {s} (ADD ptr0 idx0) x:(MOVBstorezeroidx ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstorezeroidx ptr1 idx1 mem)
	for {
		if auxIntToInt32(v.AuxInt) != 1 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			x := v_1
			if x.Op != OpARM64MOVBstorezeroidx {
				continue
			}
			mem := x.Args[2]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVHstorezeroidx)
			v.AddArg3(ptr1, idx1, mem)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstorezeroidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstorezeroidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVBstorezero [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBstorezeroidx (MOVDconst [c]) idx mem)
	// cond: is32Bit(c)
	// result: (MOVBstorezero [int32(c)] idx mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVBstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(idx, mem)
		return true
	}
	// match: (MOVBstorezeroidx ptr (ADDconst [1] idx) x:(MOVBstorezeroidx ptr idx mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstorezeroidx ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64ADDconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		idx := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVBstorezeroidx {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] || idx != x.Args[1] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVDload [off] {sym} ptr (FMOVDstore [off] {sym} ptr val _))
	// result: (FMOVDfpgp val)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64FMOVDstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpARM64FMOVDfpgp)
		v.AddArg(val)
		return true
	}
	// match: (MOVDload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVDload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVDloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVDload [off] {sym} (ADDshiftLL [3] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDloadidx8 ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVDloadidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVDload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off] {sym} ptr (MOVDstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVDstorezero {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MOVDload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVDconst [int64(read64(sym, int64(off), config.ctxt.Arch.ByteOrder))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(read64(sym, int64(off), config.ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDloadidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVDload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVDload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDloadidx (MOVDconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVDload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVDload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDloadidx ptr (SLLconst [3] idx) mem)
	// result: (MOVDloadidx8 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 3 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVDloadidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVDloadidx (SLLconst [3] idx) ptr mem)
	// result: (MOVDloadidx8 ptr idx mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.reset(OpARM64MOVDloadidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVDloadidx ptr idx (MOVDstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVDstorezeroidx {
			break
		}
		idx2 := v_2.Args[1]
		ptr2 := v_2.Args[0]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDloadidx8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDloadidx8 ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c<<3)
	// result: (MOVDload [int32(c)<<3] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c << 3)) {
			break
		}
		v.reset(OpARM64MOVDload)
		v.AuxInt = int32ToAuxInt(int32(c) << 3)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDloadidx8 ptr idx (MOVDstorezeroidx8 ptr2 idx2 _))
	// cond: isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)
	// result: (MOVDconst [0])
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVDstorezeroidx8 {
			break
		}
		idx2 := v_2.Args[1]
		ptr2 := v_2.Args[0]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDreg x)
	// cond: x.Uses == 1
	// result: (MOVDnop x)
	for {
		x := v_0
		if !(x.Uses == 1) {
			break
		}
		v.reset(OpARM64MOVDnop)
		v.AddArg(x)
		return true
	}
	// match: (MOVDreg (MOVDconst [c]))
	// result: (MOVDconst [c])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVDstore [off] {sym} ptr (FMOVDfpgp val) mem)
	// result: (FMOVDstore [off] {sym} ptr val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64FMOVDfpgp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64FMOVDstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstoreidx ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVDstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} (ADDshiftLL [3] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstoreidx8 ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVDstoreidx8)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVDstore)
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
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstoreidx ptr (MOVDconst [c]) val mem)
	// cond: is32Bit(c)
	// result: (MOVDstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstoreidx (MOVDconst [c]) idx val mem)
	// cond: is32Bit(c)
	// result: (MOVDstore [int32(c)] idx val mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (MOVDstoreidx ptr (SLLconst [3] idx) val mem)
	// result: (MOVDstoreidx8 ptr idx val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 3 {
			break
		}
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.reset(OpARM64MOVDstoreidx8)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVDstoreidx (SLLconst [3] idx) ptr val mem)
	// result: (MOVDstoreidx8 ptr idx val mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.reset(OpARM64MOVDstoreidx8)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVDstoreidx ptr idx (MOVDconst [0]) mem)
	// result: (MOVDstorezeroidx ptr idx mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		mem := v_3
		v.reset(OpARM64MOVDstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstoreidx8(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstoreidx8 ptr (MOVDconst [c]) val mem)
	// cond: is32Bit(c<<3)
	// result: (MOVDstore [int32(c)<<3] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c << 3)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(int32(c) << 3)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstoreidx8 ptr idx (MOVDconst [0]) mem)
	// result: (MOVDstorezeroidx8 ptr idx mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		mem := v_3
		v.reset(OpARM64MOVDstorezeroidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVDstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDstorezero [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDstorezero [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstorezeroidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVDstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVDstorezero [off] {sym} (ADDshiftLL [3] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstorezeroidx8 ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVDstorezeroidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVDstorezero [i] {s} ptr0 x:(MOVDstorezero [j] {s} ptr1 mem))
	// cond: x.Uses == 1 && areAdjacentOffsets(int64(i),int64(j),8) && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVQstorezero [int32(min(int64(i),int64(j)))] {s} ptr0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		x := v_1
		if x.Op != OpARM64MOVDstorezero {
			break
		}
		j := auxIntToInt32(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		ptr1 := x.Args[0]
		if !(x.Uses == 1 && areAdjacentOffsets(int64(i), int64(j), 8) && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVQstorezero)
		v.AuxInt = int32ToAuxInt(int32(min(int64(i), int64(j))))
		v.Aux = symToAux(s)
		v.AddArg2(ptr0, mem)
		return true
	}
	// match: (MOVDstorezero [8] {s} p0:(ADD ptr0 idx0) x:(MOVDstorezeroidx ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVQstorezero [0] {s} p0 mem)
	for {
		if auxIntToInt32(v.AuxInt) != 8 {
			break
		}
		s := auxToSym(v.Aux)
		p0 := v_0
		if p0.Op != OpARM64ADD {
			break
		}
		_ = p0.Args[1]
		p0_0 := p0.Args[0]
		p0_1 := p0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, p0_0, p0_1 = _i0+1, p0_1, p0_0 {
			ptr0 := p0_0
			idx0 := p0_1
			x := v_1
			if x.Op != OpARM64MOVDstorezeroidx {
				continue
			}
			mem := x.Args[2]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVQstorezero)
			v.AuxInt = int32ToAuxInt(0)
			v.Aux = symToAux(s)
			v.AddArg2(p0, mem)
			return true
		}
		break
	}
	// match: (MOVDstorezero [8] {s} p0:(ADDshiftLL [3] ptr0 idx0) x:(MOVDstorezeroidx8 ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVQstorezero [0] {s} p0 mem)
	for {
		if auxIntToInt32(v.AuxInt) != 8 {
			break
		}
		s := auxToSym(v.Aux)
		p0 := v_0
		if p0.Op != OpARM64ADDshiftLL || auxIntToInt64(p0.AuxInt) != 3 {
			break
		}
		idx0 := p0.Args[1]
		ptr0 := p0.Args[0]
		x := v_1
		if x.Op != OpARM64MOVDstorezeroidx8 {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVQstorezero)
		v.AuxInt = int32ToAuxInt(0)
		v.Aux = symToAux(s)
		v.AddArg2(p0, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstorezeroidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstorezeroidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVDstorezero [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDstorezeroidx (MOVDconst [c]) idx mem)
	// cond: is32Bit(c)
	// result: (MOVDstorezero [int32(c)] idx mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(idx, mem)
		return true
	}
	// match: (MOVDstorezeroidx ptr (SLLconst [3] idx) mem)
	// result: (MOVDstorezeroidx8 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 3 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVDstorezeroidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVDstorezeroidx (SLLconst [3] idx) ptr mem)
	// result: (MOVDstorezeroidx8 ptr idx mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.reset(OpARM64MOVDstorezeroidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstorezeroidx8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstorezeroidx8 ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c<<3)
	// result: (MOVDstorezero [int32(c<<3)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c << 3)) {
			break
		}
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(int32(c << 3))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHUload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHUloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (ADDshiftLL [1] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHUloadidx2 ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHUloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHUload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHUload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHUload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} ptr (MOVHstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVHstorezero {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MOVHUload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVDconst [int64(read16(sym, int64(off), config.ctxt.Arch.ByteOrder))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(read16(sym, int64(off), config.ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHUloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHUloadidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVHUload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVHUload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUloadidx (MOVDconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVHUload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVHUload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUloadidx ptr (SLLconst [1] idx) mem)
	// result: (MOVHUloadidx2 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVHUloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHUloadidx ptr (ADD idx idx) mem)
	// result: (MOVHUloadidx2 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64ADD {
			break
		}
		idx := v_1.Args[1]
		if idx != v_1.Args[0] {
			break
		}
		mem := v_2
		v.reset(OpARM64MOVHUloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHUloadidx (ADD idx idx) ptr mem)
	// result: (MOVHUloadidx2 ptr idx mem)
	for {
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		if idx != v_0.Args[0] {
			break
		}
		ptr := v_1
		mem := v_2
		v.reset(OpARM64MOVHUloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHUloadidx ptr idx (MOVHstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVHstorezeroidx {
			break
		}
		idx2 := v_2.Args[1]
		ptr2 := v_2.Args[0]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHUloadidx2(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHUloadidx2 ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c<<1)
	// result: (MOVHUload [int32(c)<<1] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c << 1)) {
			break
		}
		v.reset(OpARM64MOVHUload)
		v.AuxInt = int32ToAuxInt(int32(c) << 1)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUloadidx2 ptr idx (MOVHstorezeroidx2 ptr2 idx2 _))
	// cond: isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)
	// result: (MOVDconst [0])
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVHstorezeroidx2 {
			break
		}
		idx2 := v_2.Args[1]
		ptr2 := v_2.Args[0]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUloadidx2 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (ANDconst [c] x))
	// result: (ANDconst [c&(1<<16-1)] x)
	for {
		if v_0.Op != OpARM64ANDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c & (1<<16 - 1))
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint16(c))])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint16(c)))
		return true
	}
	// match: (MOVHUreg (SLLconst [sc] x))
	// cond: isARM64BFMask(sc, 1<<16-1, sc)
	// result: (UBFIZ [armBFAuxInt(sc, arm64BFWidth(1<<16-1, sc))] x)
	for {
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<16-1, sc)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, arm64BFWidth(1<<16-1, sc)))
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (SRLconst [sc] x))
	// cond: isARM64BFMask(sc, 1<<16-1, 0)
	// result: (UBFX [armBFAuxInt(sc, 16)] x)
	for {
		if v_0.Op != OpARM64SRLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<16-1, 0)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, 16))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHload [off] {sym} (ADDshiftLL [1] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHloadidx2 ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off] {sym} ptr (MOVHstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVHstorezero {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHloadidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVHload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVHload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHloadidx (MOVDconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVHload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVHload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHloadidx ptr (SLLconst [1] idx) mem)
	// result: (MOVHloadidx2 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVHloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHloadidx ptr (ADD idx idx) mem)
	// result: (MOVHloadidx2 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64ADD {
			break
		}
		idx := v_1.Args[1]
		if idx != v_1.Args[0] {
			break
		}
		mem := v_2
		v.reset(OpARM64MOVHloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHloadidx (ADD idx idx) ptr mem)
	// result: (MOVHloadidx2 ptr idx mem)
	for {
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		if idx != v_0.Args[0] {
			break
		}
		ptr := v_1
		mem := v_2
		v.reset(OpARM64MOVHloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHloadidx ptr idx (MOVHstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVHstorezeroidx {
			break
		}
		idx2 := v_2.Args[1]
		ptr2 := v_2.Args[0]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHloadidx2(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHloadidx2 ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c<<1)
	// result: (MOVHload [int32(c)<<1] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c << 1)) {
			break
		}
		v.reset(OpARM64MOVHload)
		v.AuxInt = int32ToAuxInt(int32(c) << 1)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHloadidx2 ptr idx (MOVHstorezeroidx2 ptr2 idx2 _))
	// cond: isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)
	// result: (MOVDconst [0])
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVHstorezeroidx2 {
			break
		}
		idx2 := v_2.Args[1]
		ptr2 := v_2.Args[0]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHloadidx2 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int16(c))])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int16(c)))
		return true
	}
	// match: (MOVHreg (SLLconst [lc] x))
	// cond: lc < 16
	// result: (SBFIZ [armBFAuxInt(lc, 16-lc)] x)
	for {
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < 16) {
			break
		}
		v.reset(OpARM64SBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc, 16-lc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstoreidx ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} (ADDshiftLL [1] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHstore)
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
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHreg x) mem)
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVHstore)
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
		if v_1.Op != OpARM64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVHstore)
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
		if v_1.Op != OpARM64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVHstore)
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
		if v_1.Op != OpARM64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [i] {s} ptr0 (SRLconst [16] w) x:(MOVHstore [i-2] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVWstore [i-2] {s} ptr0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 16 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVHstore || auxIntToInt32(x.AuxInt) != i-2 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 2)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w, mem)
		return true
	}
	// match: (MOVHstore [2] {s} (ADD ptr0 idx0) (SRLconst [16] w) x:(MOVHstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVWstoreidx ptr1 idx1 w mem)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 16 {
				continue
			}
			w := v_1.Args[0]
			x := v_2
			if x.Op != OpARM64MOVHstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			if w != x.Args[2] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVWstoreidx)
			v.AddArg4(ptr1, idx1, w, mem)
			return true
		}
		break
	}
	// match: (MOVHstore [2] {s} (ADDshiftLL [1] ptr0 idx0) (SRLconst [16] w) x:(MOVHstoreidx2 ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVWstoreidx ptr1 (SLLconst <idx1.Type> [1] idx1) w mem)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 16 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVHstoreidx2 {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] || !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = int64ToAuxInt(1)
		v0.AddArg(idx1)
		v.AddArg4(ptr1, v0, w, mem)
		return true
	}
	// match: (MOVHstore [i] {s} ptr0 (UBFX [armBFAuxInt(16, 16)] w) x:(MOVHstore [i-2] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVWstore [i-2] {s} ptr0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64UBFX || auxIntToArm64BitField(v_1.AuxInt) != armBFAuxInt(16, 16) {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVHstore || auxIntToInt32(x.AuxInt) != i-2 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 2)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w, mem)
		return true
	}
	// match: (MOVHstore [2] {s} (ADD ptr0 idx0) (UBFX [armBFAuxInt(16, 16)] w) x:(MOVHstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVWstoreidx ptr1 idx1 w mem)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64UBFX || auxIntToArm64BitField(v_1.AuxInt) != armBFAuxInt(16, 16) {
				continue
			}
			w := v_1.Args[0]
			x := v_2
			if x.Op != OpARM64MOVHstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			if w != x.Args[2] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVWstoreidx)
			v.AddArg4(ptr1, idx1, w, mem)
			return true
		}
		break
	}
	// match: (MOVHstore [2] {s} (ADDshiftLL [1] ptr0 idx0) (UBFX [armBFAuxInt(16, 16)] w) x:(MOVHstoreidx2 ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVWstoreidx ptr1 (SLLconst <idx1.Type> [1] idx1) w mem)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		if v_1.Op != OpARM64UBFX || auxIntToArm64BitField(v_1.AuxInt) != armBFAuxInt(16, 16) {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVHstoreidx2 {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] || !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = int64ToAuxInt(1)
		v0.AddArg(idx1)
		v.AddArg4(ptr1, v0, w, mem)
		return true
	}
	// match: (MOVHstore [i] {s} ptr0 (SRLconst [16] (MOVDreg w)) x:(MOVHstore [i-2] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVWstore [i-2] {s} ptr0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 16 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVDreg {
			break
		}
		w := v_1_0.Args[0]
		x := v_2
		if x.Op != OpARM64MOVHstore || auxIntToInt32(x.AuxInt) != i-2 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 2)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w, mem)
		return true
	}
	// match: (MOVHstore [2] {s} (ADD ptr0 idx0) (SRLconst [16] (MOVDreg w)) x:(MOVHstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVWstoreidx ptr1 idx1 w mem)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 16 {
				continue
			}
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64MOVDreg {
				continue
			}
			w := v_1_0.Args[0]
			x := v_2
			if x.Op != OpARM64MOVHstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			if w != x.Args[2] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVWstoreidx)
			v.AddArg4(ptr1, idx1, w, mem)
			return true
		}
		break
	}
	// match: (MOVHstore [2] {s} (ADDshiftLL [1] ptr0 idx0) (SRLconst [16] (MOVDreg w)) x:(MOVHstoreidx2 ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVWstoreidx ptr1 (SLLconst <idx1.Type> [1] idx1) w mem)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 16 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVDreg {
			break
		}
		w := v_1_0.Args[0]
		x := v_2
		if x.Op != OpARM64MOVHstoreidx2 {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] || !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = int64ToAuxInt(1)
		v0.AddArg(idx1)
		v.AddArg4(ptr1, v0, w, mem)
		return true
	}
	// match: (MOVHstore [i] {s} ptr0 (SRLconst [j] w) x:(MOVHstore [i-2] {s} ptr1 w0:(SRLconst [j-16] w) mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVWstore [i-2] {s} ptr0 w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := auxIntToInt64(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVHstore || auxIntToInt32(x.AuxInt) != i-2 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpARM64SRLconst || auxIntToInt64(w0.AuxInt) != j-16 || w != w0.Args[0] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(i - 2)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w0, mem)
		return true
	}
	// match: (MOVHstore [2] {s} (ADD ptr0 idx0) (SRLconst [j] w) x:(MOVHstoreidx ptr1 idx1 w0:(SRLconst [j-16] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVWstoreidx ptr1 idx1 w0 mem)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64SRLconst {
				continue
			}
			j := auxIntToInt64(v_1.AuxInt)
			w := v_1.Args[0]
			x := v_2
			if x.Op != OpARM64MOVHstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			w0 := x.Args[2]
			if w0.Op != OpARM64SRLconst || auxIntToInt64(w0.AuxInt) != j-16 || w != w0.Args[0] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVWstoreidx)
			v.AddArg4(ptr1, idx1, w0, mem)
			return true
		}
		break
	}
	// match: (MOVHstore [2] {s} (ADDshiftLL [1] ptr0 idx0) (SRLconst [j] w) x:(MOVHstoreidx2 ptr1 idx1 w0:(SRLconst [j-16] w) mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVWstoreidx ptr1 (SLLconst <idx1.Type> [1] idx1) w0 mem)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := auxIntToInt64(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVHstoreidx2 {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		w0 := x.Args[2]
		if w0.Op != OpARM64SRLconst || auxIntToInt64(w0.AuxInt) != j-16 || w != w0.Args[0] || !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = int64ToAuxInt(1)
		v0.AddArg(idx1)
		v.AddArg4(ptr1, v0, w0, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstoreidx ptr (MOVDconst [c]) val mem)
	// cond: is32Bit(c)
	// result: (MOVHstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstoreidx (MOVDconst [c]) idx val mem)
	// cond: is32Bit(c)
	// result: (MOVHstore [int32(c)] idx val mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx ptr (SLLconst [1] idx) val mem)
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx ptr (ADD idx idx) val mem)
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64ADD {
			break
		}
		idx := v_1.Args[1]
		if idx != v_1.Args[0] {
			break
		}
		val := v_2
		mem := v_3
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx (SLLconst [1] idx) ptr val mem)
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx (ADD idx idx) ptr val mem)
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		if idx != v_0.Args[0] {
			break
		}
		ptr := v_1
		val := v_2
		mem := v_3
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVDconst [0]) mem)
	// result: (MOVHstorezeroidx ptr idx mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		mem := v_3
		v.reset(OpARM64MOVHstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVHreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVHreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVHUreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVHUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVWreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVWUreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr (ADDconst [2] idx) (SRLconst [16] w) x:(MOVHstoreidx ptr idx w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstoreidx ptr idx w mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64ADDconst || auxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		if v_2.Op != OpARM64SRLconst || auxIntToInt64(v_2.AuxInt) != 16 {
			break
		}
		w := v_2.Args[0]
		x := v_3
		if x.Op != OpARM64MOVHstoreidx {
			break
		}
		mem := x.Args[3]
		if ptr != x.Args[0] || idx != x.Args[1] || w != x.Args[2] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg4(ptr, idx, w, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstoreidx2(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstoreidx2 ptr (MOVDconst [c]) val mem)
	// cond: is32Bit(c<<1)
	// result: (MOVHstore [int32(c)<<1] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c << 1)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(int32(c) << 1)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVDconst [0]) mem)
	// result: (MOVHstorezeroidx2 ptr idx mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		mem := v_3
		v.reset(OpARM64MOVHstorezeroidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVHreg x) mem)
	// result: (MOVHstoreidx2 ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVHreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVHUreg x) mem)
	// result: (MOVHstoreidx2 ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVHUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVWreg x) mem)
	// result: (MOVHstoreidx2 ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVWUreg x) mem)
	// result: (MOVHstoreidx2 ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstorezero [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstorezero [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstorezeroidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHstorezero [off] {sym} (ADDshiftLL [1] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstorezeroidx2 ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHstorezeroidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHstorezero [i] {s} ptr0 x:(MOVHstorezero [j] {s} ptr1 mem))
	// cond: x.Uses == 1 && areAdjacentOffsets(int64(i),int64(j),2) && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVWstorezero [int32(min(int64(i),int64(j)))] {s} ptr0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		x := v_1
		if x.Op != OpARM64MOVHstorezero {
			break
		}
		j := auxIntToInt32(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		ptr1 := x.Args[0]
		if !(x.Uses == 1 && areAdjacentOffsets(int64(i), int64(j), 2) && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(int32(min(int64(i), int64(j))))
		v.Aux = symToAux(s)
		v.AddArg2(ptr0, mem)
		return true
	}
	// match: (MOVHstorezero [2] {s} (ADD ptr0 idx0) x:(MOVHstorezeroidx ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVWstorezeroidx ptr1 idx1 mem)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			x := v_1
			if x.Op != OpARM64MOVHstorezeroidx {
				continue
			}
			mem := x.Args[2]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVWstorezeroidx)
			v.AddArg3(ptr1, idx1, mem)
			return true
		}
		break
	}
	// match: (MOVHstorezero [2] {s} (ADDshiftLL [1] ptr0 idx0) x:(MOVHstorezeroidx2 ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVWstorezeroidx ptr1 (SLLconst <idx1.Type> [1] idx1) mem)
	for {
		if auxIntToInt32(v.AuxInt) != 2 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		x := v_1
		if x.Op != OpARM64MOVHstorezeroidx2 {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstorezeroidx)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = int64ToAuxInt(1)
		v0.AddArg(idx1)
		v.AddArg3(ptr1, v0, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstorezeroidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstorezeroidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVHstorezero [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHstorezeroidx (MOVDconst [c]) idx mem)
	// cond: is32Bit(c)
	// result: (MOVHstorezero [int32(c)] idx mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(idx, mem)
		return true
	}
	// match: (MOVHstorezeroidx ptr (SLLconst [1] idx) mem)
	// result: (MOVHstorezeroidx2 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVHstorezeroidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHstorezeroidx ptr (ADD idx idx) mem)
	// result: (MOVHstorezeroidx2 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64ADD {
			break
		}
		idx := v_1.Args[1]
		if idx != v_1.Args[0] {
			break
		}
		mem := v_2
		v.reset(OpARM64MOVHstorezeroidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHstorezeroidx (SLLconst [1] idx) ptr mem)
	// result: (MOVHstorezeroidx2 ptr idx mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.reset(OpARM64MOVHstorezeroidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHstorezeroidx (ADD idx idx) ptr mem)
	// result: (MOVHstorezeroidx2 ptr idx mem)
	for {
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		if idx != v_0.Args[0] {
			break
		}
		ptr := v_1
		mem := v_2
		v.reset(OpARM64MOVHstorezeroidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHstorezeroidx ptr (ADDconst [2] idx) x:(MOVHstorezeroidx ptr idx mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstorezeroidx ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64ADDconst || auxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVHstorezeroidx {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] || idx != x.Args[1] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstorezeroidx2(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstorezeroidx2 ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c<<1)
	// result: (MOVHstorezero [int32(c<<1)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c << 1)) {
			break
		}
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = int32ToAuxInt(int32(c << 1))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVQstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVQstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVQstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVQstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVQstorezero [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVQstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVQstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWUload [off] {sym} ptr (FMOVSstore [off] {sym} ptr val _))
	// result: (FMOVSfpgp val)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64FMOVSstore || auxIntToInt32(v_1.AuxInt) != off || auxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.reset(OpARM64FMOVSfpgp)
		v.AddArg(val)
		return true
	}
	// match: (MOVWUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWUload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWUloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWUload [off] {sym} (ADDshiftLL [2] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWUloadidx4 ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWUloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWUload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWUload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWUload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUload [off] {sym} ptr (MOVWstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVWstorezero {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MOVWUload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVDconst [int64(read32(sym, int64(off), config.ctxt.Arch.ByteOrder))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(read32(sym, int64(off), config.ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWUloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWUloadidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVWUload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUloadidx (MOVDconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVWUload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUloadidx ptr (SLLconst [2] idx) mem)
	// result: (MOVWUloadidx4 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVWUloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWUloadidx (SLLconst [2] idx) ptr mem)
	// result: (MOVWUloadidx4 ptr idx mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.reset(OpARM64MOVWUloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWUloadidx ptr idx (MOVWstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWstorezeroidx {
			break
		}
		idx2 := v_2.Args[1]
		ptr2 := v_2.Args[0]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWUloadidx4(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWUloadidx4 ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c<<2)
	// result: (MOVWUload [int32(c)<<2] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c << 2)) {
			break
		}
		v.reset(OpARM64MOVWUload)
		v.AuxInt = int32ToAuxInt(int32(c) << 2)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUloadidx4 ptr idx (MOVWstorezeroidx4 ptr2 idx2 _))
	// cond: isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)
	// result: (MOVDconst [0])
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWstorezeroidx4 {
			break
		}
		idx2 := v_2.Args[1]
		ptr2 := v_2.Args[0]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWUreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWUreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUloadidx2 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUloadidx4 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWUloadidx4 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (ANDconst [c] x))
	// result: (ANDconst [c&(1<<32-1)] x)
	for {
		if v_0.Op != OpARM64ANDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c & (1<<32 - 1))
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint32(c))])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint32(c)))
		return true
	}
	// match: (MOVWUreg (SLLconst [sc] x))
	// cond: isARM64BFMask(sc, 1<<32-1, sc)
	// result: (UBFIZ [armBFAuxInt(sc, arm64BFWidth(1<<32-1, sc))] x)
	for {
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<32-1, sc)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, arm64BFWidth(1<<32-1, sc)))
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (SRLconst [sc] x))
	// cond: isARM64BFMask(sc, 1<<32-1, 0)
	// result: (UBFX [armBFAuxInt(sc, 32)] x)
	for {
		if v_0.Op != OpARM64SRLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<32-1, 0)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, 32))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWload)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWloadidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (ADDshiftLL [2] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWloadidx4 ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off] {sym} ptr (MOVWstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVWstorezero {
			break
		}
		off2 := auxIntToInt32(v_1.AuxInt)
		sym2 := auxToSym(v_1.Aux)
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWloadidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWloadidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVWload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWloadidx (MOVDconst [c]) ptr mem)
	// cond: is32Bit(c)
	// result: (MOVWload [int32(c)] ptr mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWload)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWloadidx ptr (SLLconst [2] idx) mem)
	// result: (MOVWloadidx4 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVWloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWloadidx (SLLconst [2] idx) ptr mem)
	// result: (MOVWloadidx4 ptr idx mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.reset(OpARM64MOVWloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWloadidx ptr idx (MOVWstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWstorezeroidx {
			break
		}
		idx2 := v_2.Args[1]
		ptr2 := v_2.Args[0]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWloadidx4(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWloadidx4 ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c<<2)
	// result: (MOVWload [int32(c)<<2] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c << 2)) {
			break
		}
		v.reset(OpARM64MOVWload)
		v.AuxInt = int32ToAuxInt(int32(c) << 2)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWloadidx4 ptr idx (MOVWstorezeroidx4 ptr2 idx2 _))
	// cond: isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)
	// result: (MOVDconst [0])
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWstorezeroidx4 {
			break
		}
		idx2 := v_2.Args[1]
		ptr2 := v_2.Args[0]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWreg(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWreg x:(MOVBload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWload _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWload {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWloadidx _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWloadidx {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHloadidx2 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUloadidx2 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHUloadidx2 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWloadidx4 _ _ _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWloadidx4 {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVHreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWreg _))
	// result: (MOVDreg x)
	for {
		x := v_0
		if x.Op != OpARM64MOVWreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int32(c))])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int32(c)))
		return true
	}
	// match: (MOVWreg (SLLconst [lc] x))
	// cond: lc < 32
	// result: (SBFIZ [armBFAuxInt(lc, 32-lc)] x)
	for {
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < 32) {
			break
		}
		v.reset(OpARM64SBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc, 32-lc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstore(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstore [off] {sym} ptr (FMOVSfpgp val) mem)
	// result: (FMOVSstore [off] {sym} ptr val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64FMOVSfpgp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64FMOVSstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstoreidx ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} (ADDshiftLL [2] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstoreidx4 ptr idx val mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWstoreidx4)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWstore)
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
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		mem := v_2
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWreg x) mem)
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVWstore)
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
		if v_1.Op != OpARM64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVWstore [i] {s} ptr0 (SRLconst [32] w) x:(MOVWstore [i-4] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVDstore [i-4] {s} ptr0 w mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 32 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVWstore || auxIntToInt32(x.AuxInt) != i-4 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(i - 4)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w, mem)
		return true
	}
	// match: (MOVWstore [4] {s} (ADD ptr0 idx0) (SRLconst [32] w) x:(MOVWstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVDstoreidx ptr1 idx1 w mem)
	for {
		if auxIntToInt32(v.AuxInt) != 4 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 32 {
				continue
			}
			w := v_1.Args[0]
			x := v_2
			if x.Op != OpARM64MOVWstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			if w != x.Args[2] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVDstoreidx)
			v.AddArg4(ptr1, idx1, w, mem)
			return true
		}
		break
	}
	// match: (MOVWstore [4] {s} (ADDshiftLL [2] ptr0 idx0) (SRLconst [32] w) x:(MOVWstoreidx4 ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVDstoreidx ptr1 (SLLconst <idx1.Type> [2] idx1) w mem)
	for {
		if auxIntToInt32(v.AuxInt) != 4 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != 32 {
			break
		}
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVWstoreidx4 {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] || !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstoreidx)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg(idx1)
		v.AddArg4(ptr1, v0, w, mem)
		return true
	}
	// match: (MOVWstore [i] {s} ptr0 (SRLconst [j] w) x:(MOVWstore [i-4] {s} ptr1 w0:(SRLconst [j-32] w) mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVDstore [i-4] {s} ptr0 w0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := auxIntToInt64(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVWstore || auxIntToInt32(x.AuxInt) != i-4 || auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpARM64SRLconst || auxIntToInt64(w0.AuxInt) != j-32 || w != w0.Args[0] || !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(i - 4)
		v.Aux = symToAux(s)
		v.AddArg3(ptr0, w0, mem)
		return true
	}
	// match: (MOVWstore [4] {s} (ADD ptr0 idx0) (SRLconst [j] w) x:(MOVWstoreidx ptr1 idx1 w0:(SRLconst [j-32] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVDstoreidx ptr1 idx1 w0 mem)
	for {
		if auxIntToInt32(v.AuxInt) != 4 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			if v_1.Op != OpARM64SRLconst {
				continue
			}
			j := auxIntToInt64(v_1.AuxInt)
			w := v_1.Args[0]
			x := v_2
			if x.Op != OpARM64MOVWstoreidx {
				continue
			}
			mem := x.Args[3]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			w0 := x.Args[2]
			if w0.Op != OpARM64SRLconst || auxIntToInt64(w0.AuxInt) != j-32 || w != w0.Args[0] || !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVDstoreidx)
			v.AddArg4(ptr1, idx1, w0, mem)
			return true
		}
		break
	}
	// match: (MOVWstore [4] {s} (ADDshiftLL [2] ptr0 idx0) (SRLconst [j] w) x:(MOVWstoreidx4 ptr1 idx1 w0:(SRLconst [j-32] w) mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVDstoreidx ptr1 (SLLconst <idx1.Type> [2] idx1) w0 mem)
	for {
		if auxIntToInt32(v.AuxInt) != 4 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := auxIntToInt64(v_1.AuxInt)
		w := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVWstoreidx4 {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		w0 := x.Args[2]
		if w0.Op != OpARM64SRLconst || auxIntToInt64(w0.AuxInt) != j-32 || w != w0.Args[0] || !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstoreidx)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg(idx1)
		v.AddArg4(ptr1, v0, w0, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreidx ptr (MOVDconst [c]) val mem)
	// cond: is32Bit(c)
	// result: (MOVWstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstoreidx (MOVDconst [c]) idx val mem)
	// cond: is32Bit(c)
	// result: (MOVWstore [int32(c)] idx val mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx ptr (SLLconst [2] idx) val mem)
	// result: (MOVWstoreidx4 ptr idx val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.reset(OpARM64MOVWstoreidx4)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx (SLLconst [2] idx) ptr val mem)
	// result: (MOVWstoreidx4 ptr idx val mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.reset(OpARM64MOVWstoreidx4)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVDconst [0]) mem)
	// result: (MOVWstorezeroidx ptr idx mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		mem := v_3
		v.reset(OpARM64MOVWstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVWreg x) mem)
	// result: (MOVWstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVWUreg x) mem)
	// result: (MOVWstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVWstoreidx ptr (ADDconst [4] idx) (SRLconst [32] w) x:(MOVWstoreidx ptr idx w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVDstoreidx ptr idx w mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64ADDconst || auxIntToInt64(v_1.AuxInt) != 4 {
			break
		}
		idx := v_1.Args[0]
		if v_2.Op != OpARM64SRLconst || auxIntToInt64(v_2.AuxInt) != 32 {
			break
		}
		w := v_2.Args[0]
		x := v_3
		if x.Op != OpARM64MOVWstoreidx {
			break
		}
		mem := x.Args[3]
		if ptr != x.Args[0] || idx != x.Args[1] || w != x.Args[2] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstoreidx)
		v.AddArg4(ptr, idx, w, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstoreidx4(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreidx4 ptr (MOVDconst [c]) val mem)
	// cond: is32Bit(c<<2)
	// result: (MOVWstore [int32(c)<<2] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(is32Bit(c << 2)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(int32(c) << 2)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstoreidx4 ptr idx (MOVDconst [0]) mem)
	// result: (MOVWstorezeroidx4 ptr idx mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		mem := v_3
		v.reset(OpARM64MOVWstorezeroidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWstoreidx4 ptr idx (MOVWreg x) mem)
	// result: (MOVWstoreidx4 ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVWstoreidx4)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVWstoreidx4 ptr idx (MOVWUreg x) mem)
	// result: (MOVWstoreidx4 ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.reset(OpARM64MOVWstoreidx4)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstorezero(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstorezero [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstorezero [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstorezero [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstorezeroidx ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWstorezero [off] {sym} (ADDshiftLL [2] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstorezeroidx4 ptr idx mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWstorezeroidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWstorezero [i] {s} ptr0 x:(MOVWstorezero [j] {s} ptr1 mem))
	// cond: x.Uses == 1 && areAdjacentOffsets(int64(i),int64(j),4) && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVDstorezero [int32(min(int64(i),int64(j)))] {s} ptr0 mem)
	for {
		i := auxIntToInt32(v.AuxInt)
		s := auxToSym(v.Aux)
		ptr0 := v_0
		x := v_1
		if x.Op != OpARM64MOVWstorezero {
			break
		}
		j := auxIntToInt32(x.AuxInt)
		if auxToSym(x.Aux) != s {
			break
		}
		mem := x.Args[1]
		ptr1 := x.Args[0]
		if !(x.Uses == 1 && areAdjacentOffsets(int64(i), int64(j), 4) && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = int32ToAuxInt(int32(min(int64(i), int64(j))))
		v.Aux = symToAux(s)
		v.AddArg2(ptr0, mem)
		return true
	}
	// match: (MOVWstorezero [4] {s} (ADD ptr0 idx0) x:(MOVWstorezeroidx ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVDstorezeroidx ptr1 idx1 mem)
	for {
		if auxIntToInt32(v.AuxInt) != 4 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			ptr0 := v_0_0
			idx0 := v_0_1
			x := v_1
			if x.Op != OpARM64MOVWstorezeroidx {
				continue
			}
			mem := x.Args[2]
			ptr1 := x.Args[0]
			idx1 := x.Args[1]
			if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
				continue
			}
			v.reset(OpARM64MOVDstorezeroidx)
			v.AddArg3(ptr1, idx1, mem)
			return true
		}
		break
	}
	// match: (MOVWstorezero [4] {s} (ADDshiftLL [2] ptr0 idx0) x:(MOVWstorezeroidx4 ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVDstorezeroidx ptr1 (SLLconst <idx1.Type> [2] idx1) mem)
	for {
		if auxIntToInt32(v.AuxInt) != 4 {
			break
		}
		s := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDshiftLL || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		x := v_1
		if x.Op != OpARM64MOVWstorezeroidx4 {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstorezeroidx)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg(idx1)
		v.AddArg3(ptr1, v0, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstorezeroidx(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstorezeroidx ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c)
	// result: (MOVWstorezero [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWstorezeroidx (MOVDconst [c]) idx mem)
	// cond: is32Bit(c)
	// result: (MOVWstorezero [int32(c)] idx mem)
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		idx := v_1
		mem := v_2
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(int32(c))
		v.AddArg2(idx, mem)
		return true
	}
	// match: (MOVWstorezeroidx ptr (SLLconst [2] idx) mem)
	// result: (MOVWstorezeroidx4 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64MOVWstorezeroidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWstorezeroidx (SLLconst [2] idx) ptr mem)
	// result: (MOVWstorezeroidx4 ptr idx mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.reset(OpARM64MOVWstorezeroidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWstorezeroidx ptr (ADDconst [4] idx) x:(MOVWstorezeroidx ptr idx mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVDstorezeroidx ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64ADDconst || auxIntToInt64(v_1.AuxInt) != 4 {
			break
		}
		idx := v_1.Args[0]
		x := v_2
		if x.Op != OpARM64MOVWstorezeroidx {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] || idx != x.Args[1] || !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstorezeroidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstorezeroidx4(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstorezeroidx4 ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c<<2)
	// result: (MOVWstorezero [int32(c<<2)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(is32Bit(c << 2)) {
			break
		}
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = int32ToAuxInt(int32(c << 2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MSUB(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MSUB a x (MOVDconst [-1]))
	// result: (ADD a x)
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != -1 {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a _ (MOVDconst [0]))
	// result: a
	for {
		a := v_0
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		v.copyOf(a)
		return true
	}
	// match: (MSUB a x (MOVDconst [1]))
	// result: (SUB a x)
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != 1 {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c)
	// result: (SUBshiftLL a x [log64(c)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo64(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c-1) && c>=3
	// result: (SUB a (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo64(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c - 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c+1) && c>=7
	// result: (ADD a (SUBshiftLL <x.Type> x x [log64(c+1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo64(c+1) && c >= 7) {
			break
		}
		v.reset(OpARM64ADD)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c + 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo64(c/3)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%3 == 0 && isPowerOfTwo64(c/3)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 3))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo64(c/5)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%5 == 0 && isPowerOfTwo64(c/5)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 5))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo64(c/7)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%7 == 0 && isPowerOfTwo64(c/7)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 7))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo64(c/9)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%9 == 0 && isPowerOfTwo64(c/9)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 9))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a (MOVDconst [-1]) x)
	// result: (ADD a x)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		x := v_2
		v.reset(OpARM64ADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a (MOVDconst [0]) _)
	// result: a
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.copyOf(a)
		return true
	}
	// match: (MSUB a (MOVDconst [1]) x)
	// result: (SUB a x)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		x := v_2
		v.reset(OpARM64SUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: isPowerOfTwo64(c)
	// result: (SUBshiftLL a x [log64(c)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo64(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: isPowerOfTwo64(c-1) && c>=3
	// result: (SUB a (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo64(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c - 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: isPowerOfTwo64(c+1) && c>=7
	// result: (ADD a (SUBshiftLL <x.Type> x x [log64(c+1)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo64(c+1) && c >= 7) {
			break
		}
		v.reset(OpARM64ADD)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c + 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo64(c/3)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%3 == 0 && isPowerOfTwo64(c/3)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 3))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo64(c/5)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%5 == 0 && isPowerOfTwo64(c/5)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 5))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo64(c/7)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%7 == 0 && isPowerOfTwo64(c/7)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 7))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo64(c/9)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%9 == 0 && isPowerOfTwo64(c/9)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 9))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB (MOVDconst [c]) x y)
	// result: (ADDconst [c] (MNEG <x.Type> x y))
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		y := v_2
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64MNEG, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) (MOVDconst [d]))
	// result: (SUBconst [c*d] a)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_2.AuxInt)
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64ToAuxInt(c * d)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MSUBW(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: int32(c)==-1
	// result: (ADD a x)
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUBW a _ (MOVDconst [c]))
	// cond: int32(c)==0
	// result: a
	for {
		a := v_0
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(int32(c) == 0) {
			break
		}
		v.copyOf(a)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: int32(c)==1
	// result: (SUB a x)
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(int32(c) == 1) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c)
	// result: (SUBshiftLL a x [log64(c)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo64(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c-1) && int32(c)>=3
	// result: (SUB a (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo64(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c - 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c+1) && int32(c)>=7
	// result: (ADD a (SUBshiftLL <x.Type> x x [log64(c+1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo64(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64ADD)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c + 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo64(c/3) && is32Bit(c)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%3 == 0 && isPowerOfTwo64(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 3))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo64(c/5) && is32Bit(c)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%5 == 0 && isPowerOfTwo64(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 5))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo64(c/7) && is32Bit(c)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%7 == 0 && isPowerOfTwo64(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 7))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo64(c/9) && is32Bit(c)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%9 == 0 && isPowerOfTwo64(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 9))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: int32(c)==-1
	// result: (ADD a x)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) _)
	// cond: int32(c)==0
	// result: a
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(int32(c) == 0) {
			break
		}
		v.copyOf(a)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: int32(c)==1
	// result: (SUB a x)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(int32(c) == 1) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo64(c)
	// result: (SUBshiftLL a x [log64(c)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo64(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo64(c-1) && int32(c)>=3
	// result: (SUB a (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo64(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c - 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo64(c+1) && int32(c)>=7
	// result: (ADD a (SUBshiftLL <x.Type> x x [log64(c+1)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo64(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64ADD)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(log64(c + 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo64(c/3) && is32Bit(c)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%3 == 0 && isPowerOfTwo64(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 3))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo64(c/5) && is32Bit(c)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%5 == 0 && isPowerOfTwo64(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 5))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo64(c/7) && is32Bit(c)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%7 == 0 && isPowerOfTwo64(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 7))
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo64(c/9) && is32Bit(c)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%9 == 0 && isPowerOfTwo64(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c / 9))
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUBW (MOVDconst [c]) x y)
	// result: (ADDconst [c] (MNEGW <x.Type> x y))
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		y := v_2
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64MNEGW, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) (MOVDconst [d]))
	// result: (SUBconst [int64(int32(c)*int32(d))] a)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_2.AuxInt)
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64ToAuxInt(int64(int32(c) * int32(d)))
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MUL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MUL (NEG x) y)
	// result: (MNEG x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64NEG {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.reset(OpARM64MNEG)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [-1]))
	// result: (NEG x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != -1 {
				continue
			}
			v.reset(OpARM64NEG)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MUL _ (MOVDconst [0]))
	// result: (MOVDconst [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
				continue
			}
			v.reset(OpARM64MOVDconst)
			v.AuxInt = int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [1]))
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 1 {
				continue
			}
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c)
	// result: (SLLconst [log64(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo64(c)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.AuxInt = int64ToAuxInt(log64(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c-1) && c >= 3
	// result: (ADDshiftLL x x [log64(c-1)])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo64(c-1) && c >= 3) {
				continue
			}
			v.reset(OpARM64ADDshiftLL)
			v.AuxInt = int64ToAuxInt(log64(c - 1))
			v.AddArg2(x, x)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c+1) && c >= 7
	// result: (ADDshiftLL (NEG <x.Type> x) x [log64(c+1)])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo64(c+1) && c >= 7) {
				continue
			}
			v.reset(OpARM64ADDshiftLL)
			v.AuxInt = int64ToAuxInt(log64(c + 1))
			v0 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
			v0.AddArg(x)
			v.AddArg2(v0, x)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo64(c/3)
	// result: (SLLconst [log64(c/3)] (ADDshiftLL <x.Type> x x [1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%3 == 0 && isPowerOfTwo64(c/3)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.AuxInt = int64ToAuxInt(log64(c / 3))
			v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(1)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo64(c/5)
	// result: (SLLconst [log64(c/5)] (ADDshiftLL <x.Type> x x [2]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%5 == 0 && isPowerOfTwo64(c/5)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.AuxInt = int64ToAuxInt(log64(c / 5))
			v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(2)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo64(c/7)
	// result: (SLLconst [log64(c/7)] (ADDshiftLL <x.Type> (NEG <x.Type> x) x [3]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%7 == 0 && isPowerOfTwo64(c/7)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.AuxInt = int64ToAuxInt(log64(c / 7))
			v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(3)
			v1 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
			v1.AddArg(x)
			v0.AddArg2(v1, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo64(c/9)
	// result: (SLLconst [log64(c/9)] (ADDshiftLL <x.Type> x x [3]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%9 == 0 && isPowerOfTwo64(c/9)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.AuxInt = int64ToAuxInt(log64(c / 9))
			v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(3)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MUL (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64MOVDconst)
			v.AuxInt = int64ToAuxInt(c * d)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64MULW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MULW (NEG x) y)
	// result: (MNEGW x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64NEG {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.reset(OpARM64MNEGW)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: int32(c)==-1
	// result: (NEG x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(int32(c) == -1) {
				continue
			}
			v.reset(OpARM64NEG)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MULW _ (MOVDconst [c]))
	// cond: int32(c)==0
	// result: (MOVDconst [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(int32(c) == 0) {
				continue
			}
			v.reset(OpARM64MOVDconst)
			v.AuxInt = int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: int32(c)==1
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(int32(c) == 1) {
				continue
			}
			v.copyOf(x)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c)
	// result: (SLLconst [log64(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo64(c)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.AuxInt = int64ToAuxInt(log64(c))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c-1) && int32(c) >= 3
	// result: (ADDshiftLL x x [log64(c-1)])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo64(c-1) && int32(c) >= 3) {
				continue
			}
			v.reset(OpARM64ADDshiftLL)
			v.AuxInt = int64ToAuxInt(log64(c - 1))
			v.AddArg2(x, x)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c+1) && int32(c) >= 7
	// result: (ADDshiftLL (NEG <x.Type> x) x [log64(c+1)])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo64(c+1) && int32(c) >= 7) {
				continue
			}
			v.reset(OpARM64ADDshiftLL)
			v.AuxInt = int64ToAuxInt(log64(c + 1))
			v0 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
			v0.AddArg(x)
			v.AddArg2(v0, x)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo64(c/3) && is32Bit(c)
	// result: (SLLconst [log64(c/3)] (ADDshiftLL <x.Type> x x [1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%3 == 0 && isPowerOfTwo64(c/3) && is32Bit(c)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.AuxInt = int64ToAuxInt(log64(c / 3))
			v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(1)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo64(c/5) && is32Bit(c)
	// result: (SLLconst [log64(c/5)] (ADDshiftLL <x.Type> x x [2]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%5 == 0 && isPowerOfTwo64(c/5) && is32Bit(c)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.AuxInt = int64ToAuxInt(log64(c / 5))
			v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(2)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo64(c/7) && is32Bit(c)
	// result: (SLLconst [log64(c/7)] (ADDshiftLL <x.Type> (NEG <x.Type> x) x [3]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%7 == 0 && isPowerOfTwo64(c/7) && is32Bit(c)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.AuxInt = int64ToAuxInt(log64(c / 7))
			v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(3)
			v1 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
			v1.AddArg(x)
			v0.AddArg2(v1, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo64(c/9) && is32Bit(c)
	// result: (SLLconst [log64(c/9)] (ADDshiftLL <x.Type> x x [3]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%9 == 0 && isPowerOfTwo64(c/9) && is32Bit(c)) {
				continue
			}
			v.reset(OpARM64SLLconst)
			v.AuxInt = int64ToAuxInt(log64(c / 9))
			v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = int64ToAuxInt(3)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MULW (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [int64(int32(c)*int32(d))])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_0.AuxInt)
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			d := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64MOVDconst)
			v.AuxInt = int64ToAuxInt(int64(int32(c) * int32(d)))
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64MVN(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MVN (XOR x y))
	// result: (EON x y)
	for {
		if v_0.Op != OpARM64XOR {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64EON)
		v.AddArg2(x, y)
		return true
	}
	// match: (MVN (MOVDconst [c]))
	// result: (MOVDconst [^c])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(^c)
		return true
	}
	// match: (MVN x:(SLLconst [c] y))
	// cond: clobberIfDead(x)
	// result: (MVNshiftLL [c] y)
	for {
		x := v_0
		if x.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64MVNshiftLL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	// match: (MVN x:(SRLconst [c] y))
	// cond: clobberIfDead(x)
	// result: (MVNshiftRL [c] y)
	for {
		x := v_0
		if x.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64MVNshiftRL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	// match: (MVN x:(SRAconst [c] y))
	// cond: clobberIfDead(x)
	// result: (MVNshiftRA [c] y)
	for {
		x := v_0
		if x.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64MVNshiftRA)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MVNshiftLL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MVNshiftLL (MOVDconst [c]) [d])
	// result: (MOVDconst [^int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(^int64(uint64(c) << uint64(d)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MVNshiftRA(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MVNshiftRA (MOVDconst [c]) [d])
	// result: (MOVDconst [^(c>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(^(c >> uint64(d)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MVNshiftRL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MVNshiftRL (MOVDconst [c]) [d])
	// result: (MOVDconst [^int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(^int64(uint64(c) >> uint64(d)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64NEG(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NEG (MUL x y))
	// result: (MNEG x y)
	for {
		if v_0.Op != OpARM64MUL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64MNEG)
		v.AddArg2(x, y)
		return true
	}
	// match: (NEG (MULW x y))
	// result: (MNEGW x y)
	for {
		if v_0.Op != OpARM64MULW {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64MNEGW)
		v.AddArg2(x, y)
		return true
	}
	// match: (NEG (MOVDconst [c]))
	// result: (MOVDconst [-c])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-c)
		return true
	}
	// match: (NEG x:(SLLconst [c] y))
	// cond: clobberIfDead(x)
	// result: (NEGshiftLL [c] y)
	for {
		x := v_0
		if x.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64NEGshiftLL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	// match: (NEG x:(SRLconst [c] y))
	// cond: clobberIfDead(x)
	// result: (NEGshiftRL [c] y)
	for {
		x := v_0
		if x.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64NEGshiftRL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	// match: (NEG x:(SRAconst [c] y))
	// cond: clobberIfDead(x)
	// result: (NEGshiftRA [c] y)
	for {
		x := v_0
		if x.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64NEGshiftRA)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64NEGshiftLL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NEGshiftLL (MOVDconst [c]) [d])
	// result: (MOVDconst [-int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-int64(uint64(c) << uint64(d)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64NEGshiftRA(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NEGshiftRA (MOVDconst [c]) [d])
	// result: (MOVDconst [-(c>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-(c >> uint64(d)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64NEGshiftRL(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NEGshiftRL (MOVDconst [c]) [d])
	// result: (MOVDconst [-int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-int64(uint64(c) >> uint64(d)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64NotEqual(v *Value) bool {
	v_0 := v.Args[0]
	// match: (NotEqual (FlagConstant [fc]))
	// result: (MOVDconst [b2i(fc.ne())])
	for {
		if v_0.Op != OpARM64FlagConstant {
			break
		}
		fc := auxIntToFlagConstant(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(fc.ne()))
		return true
	}
	// match: (NotEqual (InvertFlags x))
	// result: (NotEqual x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64NotEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64OR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OR x (MOVDconst [c]))
	// result: (ORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64ORconst)
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
	// match: (OR x (MVN y))
	// result: (ORN x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MVN {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpARM64ORN)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORshiftLL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SLLconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64ORshiftLL)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (OR x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORshiftRL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SRLconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64ORshiftRL)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (OR x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORshiftRA x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SRAconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64ORshiftRA)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (OR (SLL x (ANDconst <t> [63] y)) (CSEL0 <typ.UInt64> [cc] (SRL <typ.UInt64> x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))))
	// cond: cc == OpARM64LessThanU
	// result: (ROR x (NEG <t> y))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64SLL {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpARM64ANDconst {
				continue
			}
			t := v_0_1.Type
			if auxIntToInt64(v_0_1.AuxInt) != 63 {
				continue
			}
			y := v_0_1.Args[0]
			if v_1.Op != OpARM64CSEL0 || v_1.Type != typ.UInt64 {
				continue
			}
			cc := auxIntToOp(v_1.AuxInt)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64SRL || v_1_0.Type != typ.UInt64 {
				continue
			}
			_ = v_1_0.Args[1]
			if x != v_1_0.Args[0] {
				continue
			}
			v_1_0_1 := v_1_0.Args[1]
			if v_1_0_1.Op != OpARM64SUB || v_1_0_1.Type != t {
				continue
			}
			_ = v_1_0_1.Args[1]
			v_1_0_1_0 := v_1_0_1.Args[0]
			if v_1_0_1_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_0_1_0.AuxInt) != 64 {
				continue
			}
			v_1_0_1_1 := v_1_0_1.Args[1]
			if v_1_0_1_1.Op != OpARM64ANDconst || v_1_0_1_1.Type != t || auxIntToInt64(v_1_0_1_1.AuxInt) != 63 || y != v_1_0_1_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpARM64CMPconst || auxIntToInt64(v_1_1.AuxInt) != 64 {
				continue
			}
			v_1_1_0 := v_1_1.Args[0]
			if v_1_1_0.Op != OpARM64SUB || v_1_1_0.Type != t {
				continue
			}
			_ = v_1_1_0.Args[1]
			v_1_1_0_0 := v_1_1_0.Args[0]
			if v_1_1_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_1_0_0.AuxInt) != 64 {
				continue
			}
			v_1_1_0_1 := v_1_1_0.Args[1]
			if v_1_1_0_1.Op != OpARM64ANDconst || v_1_1_0_1.Type != t || auxIntToInt64(v_1_1_0_1.AuxInt) != 63 || y != v_1_1_0_1.Args[0] || !(cc == OpARM64LessThanU) {
				continue
			}
			v.reset(OpARM64ROR)
			v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (SRL <typ.UInt64> x (ANDconst <t> [63] y)) (CSEL0 <typ.UInt64> [cc] (SLL x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))))
	// cond: cc == OpARM64LessThanU
	// result: (ROR x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64SRL || v_0.Type != typ.UInt64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpARM64ANDconst {
				continue
			}
			t := v_0_1.Type
			if auxIntToInt64(v_0_1.AuxInt) != 63 {
				continue
			}
			y := v_0_1.Args[0]
			if v_1.Op != OpARM64CSEL0 || v_1.Type != typ.UInt64 {
				continue
			}
			cc := auxIntToOp(v_1.AuxInt)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64SLL {
				continue
			}
			_ = v_1_0.Args[1]
			if x != v_1_0.Args[0] {
				continue
			}
			v_1_0_1 := v_1_0.Args[1]
			if v_1_0_1.Op != OpARM64SUB || v_1_0_1.Type != t {
				continue
			}
			_ = v_1_0_1.Args[1]
			v_1_0_1_0 := v_1_0_1.Args[0]
			if v_1_0_1_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_0_1_0.AuxInt) != 64 {
				continue
			}
			v_1_0_1_1 := v_1_0_1.Args[1]
			if v_1_0_1_1.Op != OpARM64ANDconst || v_1_0_1_1.Type != t || auxIntToInt64(v_1_0_1_1.AuxInt) != 63 || y != v_1_0_1_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpARM64CMPconst || auxIntToInt64(v_1_1.AuxInt) != 64 {
				continue
			}
			v_1_1_0 := v_1_1.Args[0]
			if v_1_1_0.Op != OpARM64SUB || v_1_1_0.Type != t {
				continue
			}
			_ = v_1_1_0.Args[1]
			v_1_1_0_0 := v_1_1_0.Args[0]
			if v_1_1_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_1_0_0.AuxInt) != 64 {
				continue
			}
			v_1_1_0_1 := v_1_1_0.Args[1]
			if v_1_1_0_1.Op != OpARM64ANDconst || v_1_1_0_1.Type != t || auxIntToInt64(v_1_1_0_1.AuxInt) != 63 || y != v_1_1_0_1.Args[0] || !(cc == OpARM64LessThanU) {
				continue
			}
			v.reset(OpARM64ROR)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR (SLL x (ANDconst <t> [31] y)) (CSEL0 <typ.UInt32> [cc] (SRL <typ.UInt32> (MOVWUreg x) (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))))
	// cond: cc == OpARM64LessThanU
	// result: (RORW x (NEG <t> y))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64SLL {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpARM64ANDconst {
				continue
			}
			t := v_0_1.Type
			if auxIntToInt64(v_0_1.AuxInt) != 31 {
				continue
			}
			y := v_0_1.Args[0]
			if v_1.Op != OpARM64CSEL0 || v_1.Type != typ.UInt32 {
				continue
			}
			cc := auxIntToOp(v_1.AuxInt)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64SRL || v_1_0.Type != typ.UInt32 {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpARM64MOVWUreg || x != v_1_0_0.Args[0] {
				continue
			}
			v_1_0_1 := v_1_0.Args[1]
			if v_1_0_1.Op != OpARM64SUB || v_1_0_1.Type != t {
				continue
			}
			_ = v_1_0_1.Args[1]
			v_1_0_1_0 := v_1_0_1.Args[0]
			if v_1_0_1_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_0_1_0.AuxInt) != 32 {
				continue
			}
			v_1_0_1_1 := v_1_0_1.Args[1]
			if v_1_0_1_1.Op != OpARM64ANDconst || v_1_0_1_1.Type != t || auxIntToInt64(v_1_0_1_1.AuxInt) != 31 || y != v_1_0_1_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpARM64CMPconst || auxIntToInt64(v_1_1.AuxInt) != 64 {
				continue
			}
			v_1_1_0 := v_1_1.Args[0]
			if v_1_1_0.Op != OpARM64SUB || v_1_1_0.Type != t {
				continue
			}
			_ = v_1_1_0.Args[1]
			v_1_1_0_0 := v_1_1_0.Args[0]
			if v_1_1_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_1_0_0.AuxInt) != 32 {
				continue
			}
			v_1_1_0_1 := v_1_1_0.Args[1]
			if v_1_1_0_1.Op != OpARM64ANDconst || v_1_1_0_1.Type != t || auxIntToInt64(v_1_1_0_1.AuxInt) != 31 || y != v_1_1_0_1.Args[0] || !(cc == OpARM64LessThanU) {
				continue
			}
			v.reset(OpARM64RORW)
			v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (OR (SRL <typ.UInt32> (MOVWUreg x) (ANDconst <t> [31] y)) (CSEL0 <typ.UInt32> [cc] (SLL x (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))))
	// cond: cc == OpARM64LessThanU
	// result: (RORW x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64SRL || v_0.Type != typ.UInt32 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpARM64MOVWUreg {
				continue
			}
			x := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpARM64ANDconst {
				continue
			}
			t := v_0_1.Type
			if auxIntToInt64(v_0_1.AuxInt) != 31 {
				continue
			}
			y := v_0_1.Args[0]
			if v_1.Op != OpARM64CSEL0 || v_1.Type != typ.UInt32 {
				continue
			}
			cc := auxIntToOp(v_1.AuxInt)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64SLL {
				continue
			}
			_ = v_1_0.Args[1]
			if x != v_1_0.Args[0] {
				continue
			}
			v_1_0_1 := v_1_0.Args[1]
			if v_1_0_1.Op != OpARM64SUB || v_1_0_1.Type != t {
				continue
			}
			_ = v_1_0_1.Args[1]
			v_1_0_1_0 := v_1_0_1.Args[0]
			if v_1_0_1_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_0_1_0.AuxInt) != 32 {
				continue
			}
			v_1_0_1_1 := v_1_0_1.Args[1]
			if v_1_0_1_1.Op != OpARM64ANDconst || v_1_0_1_1.Type != t || auxIntToInt64(v_1_0_1_1.AuxInt) != 31 || y != v_1_0_1_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpARM64CMPconst || auxIntToInt64(v_1_1.AuxInt) != 64 {
				continue
			}
			v_1_1_0 := v_1_1.Args[0]
			if v_1_1_0.Op != OpARM64SUB || v_1_1_0.Type != t {
				continue
			}
			_ = v_1_1_0.Args[1]
			v_1_1_0_0 := v_1_1_0.Args[0]
			if v_1_1_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_1_0_0.AuxInt) != 32 {
				continue
			}
			v_1_1_0_1 := v_1_1_0.Args[1]
			if v_1_1_0_1.Op != OpARM64ANDconst || v_1_1_0_1.Type != t || auxIntToInt64(v_1_1_0_1.AuxInt) != 31 || y != v_1_1_0_1.Args[0] || !(cc == OpARM64LessThanU) {
				continue
			}
			v.reset(OpARM64RORW)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR (UBFIZ [bfc] x) (ANDconst [ac] y))
	// cond: ac == ^((1<<uint(bfc.getARM64BFwidth())-1) << uint(bfc.getARM64BFlsb()))
	// result: (BFI [bfc] y x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64UBFIZ {
				continue
			}
			bfc := auxIntToArm64BitField(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != OpARM64ANDconst {
				continue
			}
			ac := auxIntToInt64(v_1.AuxInt)
			y := v_1.Args[0]
			if !(ac == ^((1<<uint(bfc.getARM64BFwidth()) - 1) << uint(bfc.getARM64BFlsb()))) {
				continue
			}
			v.reset(OpARM64BFI)
			v.AuxInt = arm64BitFieldToAuxInt(bfc)
			v.AddArg2(y, x)
			return true
		}
		break
	}
	// match: (OR (UBFX [bfc] x) (ANDconst [ac] y))
	// cond: ac == ^(1<<uint(bfc.getARM64BFwidth())-1)
	// result: (BFXIL [bfc] y x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64UBFX {
				continue
			}
			bfc := auxIntToArm64BitField(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != OpARM64ANDconst {
				continue
			}
			ac := auxIntToInt64(v_1.AuxInt)
			y := v_1.Args[0]
			if !(ac == ^(1<<uint(bfc.getARM64BFwidth()) - 1)) {
				continue
			}
			v.reset(OpARM64BFXIL)
			v.AuxInt = arm64BitFieldToAuxInt(bfc)
			v.AddArg2(y, x)
			return true
		}
		break
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [i3] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i2] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i1] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && clobber(x0, x1, x2, x3, y0, y1, y2, y3, o0, o1, s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (MOVWUload <t> {s} (OffPtr <p.Type> [int64(i0)] p) mem)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 8 {
				continue
			}
			_ = o0.Args[1]
			o1 := o0.Args[0]
			if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 16 {
				continue
			}
			_ = o1.Args[1]
			s0 := o1.Args[0]
			if s0.Op != OpARM64SLLconst || auxIntToInt64(s0.AuxInt) != 24 {
				continue
			}
			y0 := s0.Args[0]
			if y0.Op != OpARM64MOVDnop {
				continue
			}
			x0 := y0.Args[0]
			if x0.Op != OpARM64MOVBUload {
				continue
			}
			i3 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			y1 := o1.Args[1]
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUload {
				continue
			}
			i2 := auxIntToInt32(x1.AuxInt)
			if auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] {
				continue
			}
			y2 := o0.Args[1]
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUload {
				continue
			}
			i1 := auxIntToInt32(x2.AuxInt)
			if auxToSym(x2.Aux) != s {
				continue
			}
			_ = x2.Args[1]
			if p != x2.Args[0] || mem != x2.Args[1] {
				continue
			}
			y3 := v_1
			if y3.Op != OpARM64MOVDnop {
				continue
			}
			x3 := y3.Args[0]
			if x3.Op != OpARM64MOVBUload {
				continue
			}
			i0 := auxIntToInt32(x3.AuxInt)
			if auxToSym(x3.Aux) != s {
				continue
			}
			_ = x3.Args[1]
			if p != x3.Args[0] || mem != x3.Args[1] || !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && clobber(x0, x1, x2, x3, y0, y1, y2, y3, o0, o1, s0)) {
				continue
			}
			b = mergePoint(b, x0, x1, x2, x3)
			v0 := b.NewValue0(x3.Pos, OpARM64MOVWUload, t)
			v.copyOf(v0)
			v0.Aux = symToAux(s)
			v1 := b.NewValue0(x3.Pos, OpOffPtr, p.Type)
			v1.AuxInt = int64ToAuxInt(int64(i0))
			v1.AddArg(p)
			v0.AddArg2(v1, mem)
			return true
		}
		break
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [3] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [2] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr0 idx0 mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, y0, y1, y2, y3, o0, o1, s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (MOVWUloadidx <t> ptr0 idx0 mem)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 8 {
				continue
			}
			_ = o0.Args[1]
			o1 := o0.Args[0]
			if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 16 {
				continue
			}
			_ = o1.Args[1]
			s0 := o1.Args[0]
			if s0.Op != OpARM64SLLconst || auxIntToInt64(s0.AuxInt) != 24 {
				continue
			}
			y0 := s0.Args[0]
			if y0.Op != OpARM64MOVDnop {
				continue
			}
			x0 := y0.Args[0]
			if x0.Op != OpARM64MOVBUload || auxIntToInt32(x0.AuxInt) != 3 {
				continue
			}
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			y1 := o1.Args[1]
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUload || auxIntToInt32(x1.AuxInt) != 2 || auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] {
				continue
			}
			y2 := o0.Args[1]
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUload || auxIntToInt32(x2.AuxInt) != 1 || auxToSym(x2.Aux) != s {
				continue
			}
			_ = x2.Args[1]
			p1 := x2.Args[0]
			if p1.Op != OpARM64ADD {
				continue
			}
			_ = p1.Args[1]
			p1_0 := p1.Args[0]
			p1_1 := p1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, p1_0, p1_1 = _i1+1, p1_1, p1_0 {
				ptr1 := p1_0
				idx1 := p1_1
				if mem != x2.Args[1] {
					continue
				}
				y3 := v_1
				if y3.Op != OpARM64MOVDnop {
					continue
				}
				x3 := y3.Args[0]
				if x3.Op != OpARM64MOVBUloadidx {
					continue
				}
				_ = x3.Args[2]
				ptr0 := x3.Args[0]
				idx0 := x3.Args[1]
				if mem != x3.Args[2] || !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, y0, y1, y2, y3, o0, o1, s0)) {
					continue
				}
				b = mergePoint(b, x0, x1, x2, x3)
				v0 := b.NewValue0(x2.Pos, OpARM64MOVWUloadidx, t)
				v.copyOf(v0)
				v0.AddArg3(ptr0, idx0, mem)
				return true
			}
		}
		break
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUloadidx ptr (ADDconst [3] idx) mem))) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr idx mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && clobber(x0, x1, x2, x3, y0, y1, y2, y3, o0, o1, s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (MOVWUloadidx <t> ptr idx mem)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 8 {
				continue
			}
			_ = o0.Args[1]
			o1 := o0.Args[0]
			if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 16 {
				continue
			}
			_ = o1.Args[1]
			s0 := o1.Args[0]
			if s0.Op != OpARM64SLLconst || auxIntToInt64(s0.AuxInt) != 24 {
				continue
			}
			y0 := s0.Args[0]
			if y0.Op != OpARM64MOVDnop {
				continue
			}
			x0 := y0.Args[0]
			if x0.Op != OpARM64MOVBUloadidx {
				continue
			}
			mem := x0.Args[2]
			ptr := x0.Args[0]
			x0_1 := x0.Args[1]
			if x0_1.Op != OpARM64ADDconst || auxIntToInt64(x0_1.AuxInt) != 3 {
				continue
			}
			idx := x0_1.Args[0]
			y1 := o1.Args[1]
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x1.Args[2]
			if ptr != x1.Args[0] {
				continue
			}
			x1_1 := x1.Args[1]
			if x1_1.Op != OpARM64ADDconst || auxIntToInt64(x1_1.AuxInt) != 2 || idx != x1_1.Args[0] || mem != x1.Args[2] {
				continue
			}
			y2 := o0.Args[1]
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x2.Args[2]
			if ptr != x2.Args[0] {
				continue
			}
			x2_1 := x2.Args[1]
			if x2_1.Op != OpARM64ADDconst || auxIntToInt64(x2_1.AuxInt) != 1 || idx != x2_1.Args[0] || mem != x2.Args[2] {
				continue
			}
			y3 := v_1
			if y3.Op != OpARM64MOVDnop {
				continue
			}
			x3 := y3.Args[0]
			if x3.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x3.Args[2]
			if ptr != x3.Args[0] || idx != x3.Args[1] || mem != x3.Args[2] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && clobber(x0, x1, x2, x3, y0, y1, y2, y3, o0, o1, s0)) {
				continue
			}
			b = mergePoint(b, x0, x1, x2, x3)
			v0 := b.NewValue0(v.Pos, OpARM64MOVWUloadidx, t)
			v.copyOf(v0)
			v0.AddArg3(ptr, idx, mem)
			return true
		}
		break
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [i7] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i6] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i4] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i3] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [i2] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [i1] {s} p mem))) y7:(MOVDnop x7:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && clobber(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, o0, o1, o2, o3, o4, o5, s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (MOVDload <t> {s} (OffPtr <p.Type> [int64(i0)] p) mem)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 8 {
				continue
			}
			_ = o0.Args[1]
			o1 := o0.Args[0]
			if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 16 {
				continue
			}
			_ = o1.Args[1]
			o2 := o1.Args[0]
			if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 24 {
				continue
			}
			_ = o2.Args[1]
			o3 := o2.Args[0]
			if o3.Op != OpARM64ORshiftLL || auxIntToInt64(o3.AuxInt) != 32 {
				continue
			}
			_ = o3.Args[1]
			o4 := o3.Args[0]
			if o4.Op != OpARM64ORshiftLL || auxIntToInt64(o4.AuxInt) != 40 {
				continue
			}
			_ = o4.Args[1]
			o5 := o4.Args[0]
			if o5.Op != OpARM64ORshiftLL || auxIntToInt64(o5.AuxInt) != 48 {
				continue
			}
			_ = o5.Args[1]
			s0 := o5.Args[0]
			if s0.Op != OpARM64SLLconst || auxIntToInt64(s0.AuxInt) != 56 {
				continue
			}
			y0 := s0.Args[0]
			if y0.Op != OpARM64MOVDnop {
				continue
			}
			x0 := y0.Args[0]
			if x0.Op != OpARM64MOVBUload {
				continue
			}
			i7 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			y1 := o5.Args[1]
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUload {
				continue
			}
			i6 := auxIntToInt32(x1.AuxInt)
			if auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] {
				continue
			}
			y2 := o4.Args[1]
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUload {
				continue
			}
			i5 := auxIntToInt32(x2.AuxInt)
			if auxToSym(x2.Aux) != s {
				continue
			}
			_ = x2.Args[1]
			if p != x2.Args[0] || mem != x2.Args[1] {
				continue
			}
			y3 := o3.Args[1]
			if y3.Op != OpARM64MOVDnop {
				continue
			}
			x3 := y3.Args[0]
			if x3.Op != OpARM64MOVBUload {
				continue
			}
			i4 := auxIntToInt32(x3.AuxInt)
			if auxToSym(x3.Aux) != s {
				continue
			}
			_ = x3.Args[1]
			if p != x3.Args[0] || mem != x3.Args[1] {
				continue
			}
			y4 := o2.Args[1]
			if y4.Op != OpARM64MOVDnop {
				continue
			}
			x4 := y4.Args[0]
			if x4.Op != OpARM64MOVBUload {
				continue
			}
			i3 := auxIntToInt32(x4.AuxInt)
			if auxToSym(x4.Aux) != s {
				continue
			}
			_ = x4.Args[1]
			if p != x4.Args[0] || mem != x4.Args[1] {
				continue
			}
			y5 := o1.Args[1]
			if y5.Op != OpARM64MOVDnop {
				continue
			}
			x5 := y5.Args[0]
			if x5.Op != OpARM64MOVBUload {
				continue
			}
			i2 := auxIntToInt32(x5.AuxInt)
			if auxToSym(x5.Aux) != s {
				continue
			}
			_ = x5.Args[1]
			if p != x5.Args[0] || mem != x5.Args[1] {
				continue
			}
			y6 := o0.Args[1]
			if y6.Op != OpARM64MOVDnop {
				continue
			}
			x6 := y6.Args[0]
			if x6.Op != OpARM64MOVBUload {
				continue
			}
			i1 := auxIntToInt32(x6.AuxInt)
			if auxToSym(x6.Aux) != s {
				continue
			}
			_ = x6.Args[1]
			if p != x6.Args[0] || mem != x6.Args[1] {
				continue
			}
			y7 := v_1
			if y7.Op != OpARM64MOVDnop {
				continue
			}
			x7 := y7.Args[0]
			if x7.Op != OpARM64MOVBUload {
				continue
			}
			i0 := auxIntToInt32(x7.AuxInt)
			if auxToSym(x7.Aux) != s {
				continue
			}
			_ = x7.Args[1]
			if p != x7.Args[0] || mem != x7.Args[1] || !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && clobber(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, o0, o1, o2, o3, o4, o5, s0)) {
				continue
			}
			b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
			v0 := b.NewValue0(x7.Pos, OpARM64MOVDload, t)
			v.copyOf(v0)
			v0.Aux = symToAux(s)
			v1 := b.NewValue0(x7.Pos, OpOffPtr, p.Type)
			v1.AuxInt = int64ToAuxInt(int64(i0))
			v1.AddArg(p)
			v0.AddArg2(v1, mem)
			return true
		}
		break
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [7] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [6] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [4] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [3] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [2] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y7:(MOVDnop x7:(MOVBUloadidx ptr0 idx0 mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, o0, o1, o2, o3, o4, o5, s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (MOVDloadidx <t> ptr0 idx0 mem)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 8 {
				continue
			}
			_ = o0.Args[1]
			o1 := o0.Args[0]
			if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 16 {
				continue
			}
			_ = o1.Args[1]
			o2 := o1.Args[0]
			if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 24 {
				continue
			}
			_ = o2.Args[1]
			o3 := o2.Args[0]
			if o3.Op != OpARM64ORshiftLL || auxIntToInt64(o3.AuxInt) != 32 {
				continue
			}
			_ = o3.Args[1]
			o4 := o3.Args[0]
			if o4.Op != OpARM64ORshiftLL || auxIntToInt64(o4.AuxInt) != 40 {
				continue
			}
			_ = o4.Args[1]
			o5 := o4.Args[0]
			if o5.Op != OpARM64ORshiftLL || auxIntToInt64(o5.AuxInt) != 48 {
				continue
			}
			_ = o5.Args[1]
			s0 := o5.Args[0]
			if s0.Op != OpARM64SLLconst || auxIntToInt64(s0.AuxInt) != 56 {
				continue
			}
			y0 := s0.Args[0]
			if y0.Op != OpARM64MOVDnop {
				continue
			}
			x0 := y0.Args[0]
			if x0.Op != OpARM64MOVBUload || auxIntToInt32(x0.AuxInt) != 7 {
				continue
			}
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			y1 := o5.Args[1]
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUload || auxIntToInt32(x1.AuxInt) != 6 || auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] {
				continue
			}
			y2 := o4.Args[1]
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUload || auxIntToInt32(x2.AuxInt) != 5 || auxToSym(x2.Aux) != s {
				continue
			}
			_ = x2.Args[1]
			if p != x2.Args[0] || mem != x2.Args[1] {
				continue
			}
			y3 := o3.Args[1]
			if y3.Op != OpARM64MOVDnop {
				continue
			}
			x3 := y3.Args[0]
			if x3.Op != OpARM64MOVBUload || auxIntToInt32(x3.AuxInt) != 4 || auxToSym(x3.Aux) != s {
				continue
			}
			_ = x3.Args[1]
			if p != x3.Args[0] || mem != x3.Args[1] {
				continue
			}
			y4 := o2.Args[1]
			if y4.Op != OpARM64MOVDnop {
				continue
			}
			x4 := y4.Args[0]
			if x4.Op != OpARM64MOVBUload || auxIntToInt32(x4.AuxInt) != 3 || auxToSym(x4.Aux) != s {
				continue
			}
			_ = x4.Args[1]
			if p != x4.Args[0] || mem != x4.Args[1] {
				continue
			}
			y5 := o1.Args[1]
			if y5.Op != OpARM64MOVDnop {
				continue
			}
			x5 := y5.Args[0]
			if x5.Op != OpARM64MOVBUload || auxIntToInt32(x5.AuxInt) != 2 || auxToSym(x5.Aux) != s {
				continue
			}
			_ = x5.Args[1]
			if p != x5.Args[0] || mem != x5.Args[1] {
				continue
			}
			y6 := o0.Args[1]
			if y6.Op != OpARM64MOVDnop {
				continue
			}
			x6 := y6.Args[0]
			if x6.Op != OpARM64MOVBUload || auxIntToInt32(x6.AuxInt) != 1 || auxToSym(x6.Aux) != s {
				continue
			}
			_ = x6.Args[1]
			p1 := x6.Args[0]
			if p1.Op != OpARM64ADD {
				continue
			}
			_ = p1.Args[1]
			p1_0 := p1.Args[0]
			p1_1 := p1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, p1_0, p1_1 = _i1+1, p1_1, p1_0 {
				ptr1 := p1_0
				idx1 := p1_1
				if mem != x6.Args[1] {
					continue
				}
				y7 := v_1
				if y7.Op != OpARM64MOVDnop {
					continue
				}
				x7 := y7.Args[0]
				if x7.Op != OpARM64MOVBUloadidx {
					continue
				}
				_ = x7.Args[2]
				ptr0 := x7.Args[0]
				idx0 := x7.Args[1]
				if mem != x7.Args[2] || !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, o0, o1, o2, o3, o4, o5, s0)) {
					continue
				}
				b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
				v0 := b.NewValue0(x6.Pos, OpARM64MOVDloadidx, t)
				v.copyOf(v0)
				v0.AddArg3(ptr0, idx0, mem)
				return true
			}
		}
		break
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUloadidx ptr (ADDconst [7] idx) mem))) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [6] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [5] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [4] idx) mem))) y4:(MOVDnop x4:(MOVBUloadidx ptr (ADDconst [3] idx) mem))) y5:(MOVDnop x5:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y6:(MOVDnop x6:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y7:(MOVDnop x7:(MOVBUloadidx ptr idx mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && clobber(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, o0, o1, o2, o3, o4, o5, s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (MOVDloadidx <t> ptr idx mem)
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 8 {
				continue
			}
			_ = o0.Args[1]
			o1 := o0.Args[0]
			if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 16 {
				continue
			}
			_ = o1.Args[1]
			o2 := o1.Args[0]
			if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 24 {
				continue
			}
			_ = o2.Args[1]
			o3 := o2.Args[0]
			if o3.Op != OpARM64ORshiftLL || auxIntToInt64(o3.AuxInt) != 32 {
				continue
			}
			_ = o3.Args[1]
			o4 := o3.Args[0]
			if o4.Op != OpARM64ORshiftLL || auxIntToInt64(o4.AuxInt) != 40 {
				continue
			}
			_ = o4.Args[1]
			o5 := o4.Args[0]
			if o5.Op != OpARM64ORshiftLL || auxIntToInt64(o5.AuxInt) != 48 {
				continue
			}
			_ = o5.Args[1]
			s0 := o5.Args[0]
			if s0.Op != OpARM64SLLconst || auxIntToInt64(s0.AuxInt) != 56 {
				continue
			}
			y0 := s0.Args[0]
			if y0.Op != OpARM64MOVDnop {
				continue
			}
			x0 := y0.Args[0]
			if x0.Op != OpARM64MOVBUloadidx {
				continue
			}
			mem := x0.Args[2]
			ptr := x0.Args[0]
			x0_1 := x0.Args[1]
			if x0_1.Op != OpARM64ADDconst || auxIntToInt64(x0_1.AuxInt) != 7 {
				continue
			}
			idx := x0_1.Args[0]
			y1 := o5.Args[1]
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x1.Args[2]
			if ptr != x1.Args[0] {
				continue
			}
			x1_1 := x1.Args[1]
			if x1_1.Op != OpARM64ADDconst || auxIntToInt64(x1_1.AuxInt) != 6 || idx != x1_1.Args[0] || mem != x1.Args[2] {
				continue
			}
			y2 := o4.Args[1]
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x2.Args[2]
			if ptr != x2.Args[0] {
				continue
			}
			x2_1 := x2.Args[1]
			if x2_1.Op != OpARM64ADDconst || auxIntToInt64(x2_1.AuxInt) != 5 || idx != x2_1.Args[0] || mem != x2.Args[2] {
				continue
			}
			y3 := o3.Args[1]
			if y3.Op != OpARM64MOVDnop {
				continue
			}
			x3 := y3.Args[0]
			if x3.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x3.Args[2]
			if ptr != x3.Args[0] {
				continue
			}
			x3_1 := x3.Args[1]
			if x3_1.Op != OpARM64ADDconst || auxIntToInt64(x3_1.AuxInt) != 4 || idx != x3_1.Args[0] || mem != x3.Args[2] {
				continue
			}
			y4 := o2.Args[1]
			if y4.Op != OpARM64MOVDnop {
				continue
			}
			x4 := y4.Args[0]
			if x4.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x4.Args[2]
			if ptr != x4.Args[0] {
				continue
			}
			x4_1 := x4.Args[1]
			if x4_1.Op != OpARM64ADDconst || auxIntToInt64(x4_1.AuxInt) != 3 || idx != x4_1.Args[0] || mem != x4.Args[2] {
				continue
			}
			y5 := o1.Args[1]
			if y5.Op != OpARM64MOVDnop {
				continue
			}
			x5 := y5.Args[0]
			if x5.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x5.Args[2]
			if ptr != x5.Args[0] {
				continue
			}
			x5_1 := x5.Args[1]
			if x5_1.Op != OpARM64ADDconst || auxIntToInt64(x5_1.AuxInt) != 2 || idx != x5_1.Args[0] || mem != x5.Args[2] {
				continue
			}
			y6 := o0.Args[1]
			if y6.Op != OpARM64MOVDnop {
				continue
			}
			x6 := y6.Args[0]
			if x6.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x6.Args[2]
			if ptr != x6.Args[0] {
				continue
			}
			x6_1 := x6.Args[1]
			if x6_1.Op != OpARM64ADDconst || auxIntToInt64(x6_1.AuxInt) != 1 || idx != x6_1.Args[0] || mem != x6.Args[2] {
				continue
			}
			y7 := v_1
			if y7.Op != OpARM64MOVDnop {
				continue
			}
			x7 := y7.Args[0]
			if x7.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x7.Args[2]
			if ptr != x7.Args[0] || idx != x7.Args[1] || mem != x7.Args[2] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && clobber(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, o0, o1, o2, o3, o4, o5, s0)) {
				continue
			}
			b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
			v0 := b.NewValue0(v.Pos, OpARM64MOVDloadidx, t)
			v.copyOf(v0)
			v0.AddArg3(ptr, idx, mem)
			return true
		}
		break
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i3] {s} p mem)))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && clobber(x0, x1, x2, x3, y0, y1, y2, y3, o0, o1, s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (REVW <t> (MOVWUload <t> {s} (OffPtr <p.Type> [int64(i0)] p) mem))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 8 {
				continue
			}
			_ = o0.Args[1]
			o1 := o0.Args[0]
			if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 16 {
				continue
			}
			_ = o1.Args[1]
			s0 := o1.Args[0]
			if s0.Op != OpARM64SLLconst || auxIntToInt64(s0.AuxInt) != 24 {
				continue
			}
			y0 := s0.Args[0]
			if y0.Op != OpARM64MOVDnop {
				continue
			}
			x0 := y0.Args[0]
			if x0.Op != OpARM64MOVBUload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			y1 := o1.Args[1]
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			if auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] {
				continue
			}
			y2 := o0.Args[1]
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUload {
				continue
			}
			i2 := auxIntToInt32(x2.AuxInt)
			if auxToSym(x2.Aux) != s {
				continue
			}
			_ = x2.Args[1]
			if p != x2.Args[0] || mem != x2.Args[1] {
				continue
			}
			y3 := v_1
			if y3.Op != OpARM64MOVDnop {
				continue
			}
			x3 := y3.Args[0]
			if x3.Op != OpARM64MOVBUload {
				continue
			}
			i3 := auxIntToInt32(x3.AuxInt)
			if auxToSym(x3.Aux) != s {
				continue
			}
			_ = x3.Args[1]
			if p != x3.Args[0] || mem != x3.Args[1] || !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && clobber(x0, x1, x2, x3, y0, y1, y2, y3, o0, o1, s0)) {
				continue
			}
			b = mergePoint(b, x0, x1, x2, x3)
			v0 := b.NewValue0(x3.Pos, OpARM64REVW, t)
			v.copyOf(v0)
			v1 := b.NewValue0(x3.Pos, OpARM64MOVWUload, t)
			v1.Aux = symToAux(s)
			v2 := b.NewValue0(x3.Pos, OpOffPtr, p.Type)
			v2.AuxInt = int64ToAuxInt(int64(i0))
			v2.AddArg(p)
			v1.AddArg2(v2, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUloadidx ptr0 idx0 mem))) y1:(MOVDnop x1:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [3] {s} p mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, y0, y1, y2, y3, o0, o1, s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (REVW <t> (MOVWUloadidx <t> ptr0 idx0 mem))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 8 {
				continue
			}
			_ = o0.Args[1]
			o1 := o0.Args[0]
			if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 16 {
				continue
			}
			_ = o1.Args[1]
			s0 := o1.Args[0]
			if s0.Op != OpARM64SLLconst || auxIntToInt64(s0.AuxInt) != 24 {
				continue
			}
			y0 := s0.Args[0]
			if y0.Op != OpARM64MOVDnop {
				continue
			}
			x0 := y0.Args[0]
			if x0.Op != OpARM64MOVBUloadidx {
				continue
			}
			mem := x0.Args[2]
			ptr0 := x0.Args[0]
			idx0 := x0.Args[1]
			y1 := o1.Args[1]
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUload || auxIntToInt32(x1.AuxInt) != 1 {
				continue
			}
			s := auxToSym(x1.Aux)
			_ = x1.Args[1]
			p1 := x1.Args[0]
			if p1.Op != OpARM64ADD {
				continue
			}
			_ = p1.Args[1]
			p1_0 := p1.Args[0]
			p1_1 := p1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, p1_0, p1_1 = _i1+1, p1_1, p1_0 {
				ptr1 := p1_0
				idx1 := p1_1
				if mem != x1.Args[1] {
					continue
				}
				y2 := o0.Args[1]
				if y2.Op != OpARM64MOVDnop {
					continue
				}
				x2 := y2.Args[0]
				if x2.Op != OpARM64MOVBUload || auxIntToInt32(x2.AuxInt) != 2 || auxToSym(x2.Aux) != s {
					continue
				}
				_ = x2.Args[1]
				p := x2.Args[0]
				if mem != x2.Args[1] {
					continue
				}
				y3 := v_1
				if y3.Op != OpARM64MOVDnop {
					continue
				}
				x3 := y3.Args[0]
				if x3.Op != OpARM64MOVBUload || auxIntToInt32(x3.AuxInt) != 3 || auxToSym(x3.Aux) != s {
					continue
				}
				_ = x3.Args[1]
				if p != x3.Args[0] || mem != x3.Args[1] || !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, y0, y1, y2, y3, o0, o1, s0)) {
					continue
				}
				b = mergePoint(b, x0, x1, x2, x3)
				v0 := b.NewValue0(x3.Pos, OpARM64REVW, t)
				v.copyOf(v0)
				v1 := b.NewValue0(x3.Pos, OpARM64MOVWUloadidx, t)
				v1.AddArg3(ptr0, idx0, mem)
				v0.AddArg(v1)
				return true
			}
		}
		break
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUloadidx ptr idx mem))) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [3] idx) mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && clobber(x0, x1, x2, x3, y0, y1, y2, y3, o0, o1, s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (REVW <t> (MOVWUloadidx <t> ptr idx mem))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 8 {
				continue
			}
			_ = o0.Args[1]
			o1 := o0.Args[0]
			if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 16 {
				continue
			}
			_ = o1.Args[1]
			s0 := o1.Args[0]
			if s0.Op != OpARM64SLLconst || auxIntToInt64(s0.AuxInt) != 24 {
				continue
			}
			y0 := s0.Args[0]
			if y0.Op != OpARM64MOVDnop {
				continue
			}
			x0 := y0.Args[0]
			if x0.Op != OpARM64MOVBUloadidx {
				continue
			}
			mem := x0.Args[2]
			ptr := x0.Args[0]
			idx := x0.Args[1]
			y1 := o1.Args[1]
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x1.Args[2]
			if ptr != x1.Args[0] {
				continue
			}
			x1_1 := x1.Args[1]
			if x1_1.Op != OpARM64ADDconst || auxIntToInt64(x1_1.AuxInt) != 1 || idx != x1_1.Args[0] || mem != x1.Args[2] {
				continue
			}
			y2 := o0.Args[1]
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x2.Args[2]
			if ptr != x2.Args[0] {
				continue
			}
			x2_1 := x2.Args[1]
			if x2_1.Op != OpARM64ADDconst || auxIntToInt64(x2_1.AuxInt) != 2 || idx != x2_1.Args[0] || mem != x2.Args[2] {
				continue
			}
			y3 := v_1
			if y3.Op != OpARM64MOVDnop {
				continue
			}
			x3 := y3.Args[0]
			if x3.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x3.Args[2]
			if ptr != x3.Args[0] {
				continue
			}
			x3_1 := x3.Args[1]
			if x3_1.Op != OpARM64ADDconst || auxIntToInt64(x3_1.AuxInt) != 3 || idx != x3_1.Args[0] || mem != x3.Args[2] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && clobber(x0, x1, x2, x3, y0, y1, y2, y3, o0, o1, s0)) {
				continue
			}
			b = mergePoint(b, x0, x1, x2, x3)
			v0 := b.NewValue0(v.Pos, OpARM64REVW, t)
			v.copyOf(v0)
			v1 := b.NewValue0(v.Pos, OpARM64MOVWUloadidx, t)
			v1.AddArg3(ptr, idx, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i3] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i4] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [i5] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [i6] {s} p mem))) y7:(MOVDnop x7:(MOVBUload [i7] {s} p mem)))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && clobber(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, o0, o1, o2, o3, o4, o5, s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (REV <t> (MOVDload <t> {s} (OffPtr <p.Type> [int64(i0)] p) mem))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 8 {
				continue
			}
			_ = o0.Args[1]
			o1 := o0.Args[0]
			if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 16 {
				continue
			}
			_ = o1.Args[1]
			o2 := o1.Args[0]
			if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 24 {
				continue
			}
			_ = o2.Args[1]
			o3 := o2.Args[0]
			if o3.Op != OpARM64ORshiftLL || auxIntToInt64(o3.AuxInt) != 32 {
				continue
			}
			_ = o3.Args[1]
			o4 := o3.Args[0]
			if o4.Op != OpARM64ORshiftLL || auxIntToInt64(o4.AuxInt) != 40 {
				continue
			}
			_ = o4.Args[1]
			o5 := o4.Args[0]
			if o5.Op != OpARM64ORshiftLL || auxIntToInt64(o5.AuxInt) != 48 {
				continue
			}
			_ = o5.Args[1]
			s0 := o5.Args[0]
			if s0.Op != OpARM64SLLconst || auxIntToInt64(s0.AuxInt) != 56 {
				continue
			}
			y0 := s0.Args[0]
			if y0.Op != OpARM64MOVDnop {
				continue
			}
			x0 := y0.Args[0]
			if x0.Op != OpARM64MOVBUload {
				continue
			}
			i0 := auxIntToInt32(x0.AuxInt)
			s := auxToSym(x0.Aux)
			mem := x0.Args[1]
			p := x0.Args[0]
			y1 := o5.Args[1]
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUload {
				continue
			}
			i1 := auxIntToInt32(x1.AuxInt)
			if auxToSym(x1.Aux) != s {
				continue
			}
			_ = x1.Args[1]
			if p != x1.Args[0] || mem != x1.Args[1] {
				continue
			}
			y2 := o4.Args[1]
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUload {
				continue
			}
			i2 := auxIntToInt32(x2.AuxInt)
			if auxToSym(x2.Aux) != s {
				continue
			}
			_ = x2.Args[1]
			if p != x2.Args[0] || mem != x2.Args[1] {
				continue
			}
			y3 := o3.Args[1]
			if y3.Op != OpARM64MOVDnop {
				continue
			}
			x3 := y3.Args[0]
			if x3.Op != OpARM64MOVBUload {
				continue
			}
			i3 := auxIntToInt32(x3.AuxInt)
			if auxToSym(x3.Aux) != s {
				continue
			}
			_ = x3.Args[1]
			if p != x3.Args[0] || mem != x3.Args[1] {
				continue
			}
			y4 := o2.Args[1]
			if y4.Op != OpARM64MOVDnop {
				continue
			}
			x4 := y4.Args[0]
			if x4.Op != OpARM64MOVBUload {
				continue
			}
			i4 := auxIntToInt32(x4.AuxInt)
			if auxToSym(x4.Aux) != s {
				continue
			}
			_ = x4.Args[1]
			if p != x4.Args[0] || mem != x4.Args[1] {
				continue
			}
			y5 := o1.Args[1]
			if y5.Op != OpARM64MOVDnop {
				continue
			}
			x5 := y5.Args[0]
			if x5.Op != OpARM64MOVBUload {
				continue
			}
			i5 := auxIntToInt32(x5.AuxInt)
			if auxToSym(x5.Aux) != s {
				continue
			}
			_ = x5.Args[1]
			if p != x5.Args[0] || mem != x5.Args[1] {
				continue
			}
			y6 := o0.Args[1]
			if y6.Op != OpARM64MOVDnop {
				continue
			}
			x6 := y6.Args[0]
			if x6.Op != OpARM64MOVBUload {
				continue
			}
			i6 := auxIntToInt32(x6.AuxInt)
			if auxToSym(x6.Aux) != s {
				continue
			}
			_ = x6.Args[1]
			if p != x6.Args[0] || mem != x6.Args[1] {
				continue
			}
			y7 := v_1
			if y7.Op != OpARM64MOVDnop {
				continue
			}
			x7 := y7.Args[0]
			if x7.Op != OpARM64MOVBUload {
				continue
			}
			i7 := auxIntToInt32(x7.AuxInt)
			if auxToSym(x7.Aux) != s {
				continue
			}
			_ = x7.Args[1]
			if p != x7.Args[0] || mem != x7.Args[1] || !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && clobber(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, o0, o1, o2, o3, o4, o5, s0)) {
				continue
			}
			b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
			v0 := b.NewValue0(x7.Pos, OpARM64REV, t)
			v.copyOf(v0)
			v1 := b.NewValue0(x7.Pos, OpARM64MOVDload, t)
			v1.Aux = symToAux(s)
			v2 := b.NewValue0(x7.Pos, OpOffPtr, p.Type)
			v2.AuxInt = int64ToAuxInt(int64(i0))
			v2.AddArg(p)
			v1.AddArg2(v2, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUloadidx ptr0 idx0 mem))) y1:(MOVDnop x1:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [3] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [4] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [5] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [6] {s} p mem))) y7:(MOVDnop x7:(MOVBUload [7] {s} p mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, o0, o1, o2, o3, o4, o5, s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (REV <t> (MOVDloadidx <t> ptr0 idx0 mem))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 8 {
				continue
			}
			_ = o0.Args[1]
			o1 := o0.Args[0]
			if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 16 {
				continue
			}
			_ = o1.Args[1]
			o2 := o1.Args[0]
			if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 24 {
				continue
			}
			_ = o2.Args[1]
			o3 := o2.Args[0]
			if o3.Op != OpARM64ORshiftLL || auxIntToInt64(o3.AuxInt) != 32 {
				continue
			}
			_ = o3.Args[1]
			o4 := o3.Args[0]
			if o4.Op != OpARM64ORshiftLL || auxIntToInt64(o4.AuxInt) != 40 {
				continue
			}
			_ = o4.Args[1]
			o5 := o4.Args[0]
			if o5.Op != OpARM64ORshiftLL || auxIntToInt64(o5.AuxInt) != 48 {
				continue
			}
			_ = o5.Args[1]
			s0 := o5.Args[0]
			if s0.Op != OpARM64SLLconst || auxIntToInt64(s0.AuxInt) != 56 {
				continue
			}
			y0 := s0.Args[0]
			if y0.Op != OpARM64MOVDnop {
				continue
			}
			x0 := y0.Args[0]
			if x0.Op != OpARM64MOVBUloadidx {
				continue
			}
			mem := x0.Args[2]
			ptr0 := x0.Args[0]
			idx0 := x0.Args[1]
			y1 := o5.Args[1]
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUload || auxIntToInt32(x1.AuxInt) != 1 {
				continue
			}
			s := auxToSym(x1.Aux)
			_ = x1.Args[1]
			p1 := x1.Args[0]
			if p1.Op != OpARM64ADD {
				continue
			}
			_ = p1.Args[1]
			p1_0 := p1.Args[0]
			p1_1 := p1.Args[1]
			for _i1 := 0; _i1 <= 1; _i1, p1_0, p1_1 = _i1+1, p1_1, p1_0 {
				ptr1 := p1_0
				idx1 := p1_1
				if mem != x1.Args[1] {
					continue
				}
				y2 := o4.Args[1]
				if y2.Op != OpARM64MOVDnop {
					continue
				}
				x2 := y2.Args[0]
				if x2.Op != OpARM64MOVBUload || auxIntToInt32(x2.AuxInt) != 2 || auxToSym(x2.Aux) != s {
					continue
				}
				_ = x2.Args[1]
				p := x2.Args[0]
				if mem != x2.Args[1] {
					continue
				}
				y3 := o3.Args[1]
				if y3.Op != OpARM64MOVDnop {
					continue
				}
				x3 := y3.Args[0]
				if x3.Op != OpARM64MOVBUload || auxIntToInt32(x3.AuxInt) != 3 || auxToSym(x3.Aux) != s {
					continue
				}
				_ = x3.Args[1]
				if p != x3.Args[0] || mem != x3.Args[1] {
					continue
				}
				y4 := o2.Args[1]
				if y4.Op != OpARM64MOVDnop {
					continue
				}
				x4 := y4.Args[0]
				if x4.Op != OpARM64MOVBUload || auxIntToInt32(x4.AuxInt) != 4 || auxToSym(x4.Aux) != s {
					continue
				}
				_ = x4.Args[1]
				if p != x4.Args[0] || mem != x4.Args[1] {
					continue
				}
				y5 := o1.Args[1]
				if y5.Op != OpARM64MOVDnop {
					continue
				}
				x5 := y5.Args[0]
				if x5.Op != OpARM64MOVBUload || auxIntToInt32(x5.AuxInt) != 5 || auxToSym(x5.Aux) != s {
					continue
				}
				_ = x5.Args[1]
				if p != x5.Args[0] || mem != x5.Args[1] {
					continue
				}
				y6 := o0.Args[1]
				if y6.Op != OpARM64MOVDnop {
					continue
				}
				x6 := y6.Args[0]
				if x6.Op != OpARM64MOVBUload || auxIntToInt32(x6.AuxInt) != 6 || auxToSym(x6.Aux) != s {
					continue
				}
				_ = x6.Args[1]
				if p != x6.Args[0] || mem != x6.Args[1] {
					continue
				}
				y7 := v_1
				if y7.Op != OpARM64MOVDnop {
					continue
				}
				x7 := y7.Args[0]
				if x7.Op != OpARM64MOVBUload || auxIntToInt32(x7.AuxInt) != 7 || auxToSym(x7.Aux) != s {
					continue
				}
				_ = x7.Args[1]
				if p != x7.Args[0] || mem != x7.Args[1] || !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, o0, o1, o2, o3, o4, o5, s0)) {
					continue
				}
				b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
				v0 := b.NewValue0(x7.Pos, OpARM64REV, t)
				v.copyOf(v0)
				v1 := b.NewValue0(x7.Pos, OpARM64MOVDloadidx, t)
				v1.AddArg3(ptr0, idx0, mem)
				v0.AddArg(v1)
				return true
			}
		}
		break
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUloadidx ptr idx mem))) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [3] idx) mem))) y4:(MOVDnop x4:(MOVBUloadidx ptr (ADDconst [4] idx) mem))) y5:(MOVDnop x5:(MOVBUloadidx ptr (ADDconst [5] idx) mem))) y6:(MOVDnop x6:(MOVBUloadidx ptr (ADDconst [6] idx) mem))) y7:(MOVDnop x7:(MOVBUloadidx ptr (ADDconst [7] idx) mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && clobber(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, o0, o1, o2, o3, o4, o5, s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (REV <t> (MOVDloadidx <t> ptr idx mem))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			o0 := v_0
			if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 8 {
				continue
			}
			_ = o0.Args[1]
			o1 := o0.Args[0]
			if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 16 {
				continue
			}
			_ = o1.Args[1]
			o2 := o1.Args[0]
			if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 24 {
				continue
			}
			_ = o2.Args[1]
			o3 := o2.Args[0]
			if o3.Op != OpARM64ORshiftLL || auxIntToInt64(o3.AuxInt) != 32 {
				continue
			}
			_ = o3.Args[1]
			o4 := o3.Args[0]
			if o4.Op != OpARM64ORshiftLL || auxIntToInt64(o4.AuxInt) != 40 {
				continue
			}
			_ = o4.Args[1]
			o5 := o4.Args[0]
			if o5.Op != OpARM64ORshiftLL || auxIntToInt64(o5.AuxInt) != 48 {
				continue
			}
			_ = o5.Args[1]
			s0 := o5.Args[0]
			if s0.Op != OpARM64SLLconst || auxIntToInt64(s0.AuxInt) != 56 {
				continue
			}
			y0 := s0.Args[0]
			if y0.Op != OpARM64MOVDnop {
				continue
			}
			x0 := y0.Args[0]
			if x0.Op != OpARM64MOVBUloadidx {
				continue
			}
			mem := x0.Args[2]
			ptr := x0.Args[0]
			idx := x0.Args[1]
			y1 := o5.Args[1]
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x1.Args[2]
			if ptr != x1.Args[0] {
				continue
			}
			x1_1 := x1.Args[1]
			if x1_1.Op != OpARM64ADDconst || auxIntToInt64(x1_1.AuxInt) != 1 || idx != x1_1.Args[0] || mem != x1.Args[2] {
				continue
			}
			y2 := o4.Args[1]
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x2.Args[2]
			if ptr != x2.Args[0] {
				continue
			}
			x2_1 := x2.Args[1]
			if x2_1.Op != OpARM64ADDconst || auxIntToInt64(x2_1.AuxInt) != 2 || idx != x2_1.Args[0] || mem != x2.Args[2] {
				continue
			}
			y3 := o3.Args[1]
			if y3.Op != OpARM64MOVDnop {
				continue
			}
			x3 := y3.Args[0]
			if x3.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x3.Args[2]
			if ptr != x3.Args[0] {
				continue
			}
			x3_1 := x3.Args[1]
			if x3_1.Op != OpARM64ADDconst || auxIntToInt64(x3_1.AuxInt) != 3 || idx != x3_1.Args[0] || mem != x3.Args[2] {
				continue
			}
			y4 := o2.Args[1]
			if y4.Op != OpARM64MOVDnop {
				continue
			}
			x4 := y4.Args[0]
			if x4.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x4.Args[2]
			if ptr != x4.Args[0] {
				continue
			}
			x4_1 := x4.Args[1]
			if x4_1.Op != OpARM64ADDconst || auxIntToInt64(x4_1.AuxInt) != 4 || idx != x4_1.Args[0] || mem != x4.Args[2] {
				continue
			}
			y5 := o1.Args[1]
			if y5.Op != OpARM64MOVDnop {
				continue
			}
			x5 := y5.Args[0]
			if x5.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x5.Args[2]
			if ptr != x5.Args[0] {
				continue
			}
			x5_1 := x5.Args[1]
			if x5_1.Op != OpARM64ADDconst || auxIntToInt64(x5_1.AuxInt) != 5 || idx != x5_1.Args[0] || mem != x5.Args[2] {
				continue
			}
			y6 := o0.Args[1]
			if y6.Op != OpARM64MOVDnop {
				continue
			}
			x6 := y6.Args[0]
			if x6.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x6.Args[2]
			if ptr != x6.Args[0] {
				continue
			}
			x6_1 := x6.Args[1]
			if x6_1.Op != OpARM64ADDconst || auxIntToInt64(x6_1.AuxInt) != 6 || idx != x6_1.Args[0] || mem != x6.Args[2] {
				continue
			}
			y7 := v_1
			if y7.Op != OpARM64MOVDnop {
				continue
			}
			x7 := y7.Args[0]
			if x7.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x7.Args[2]
			if ptr != x7.Args[0] {
				continue
			}
			x7_1 := x7.Args[1]
			if x7_1.Op != OpARM64ADDconst || auxIntToInt64(x7_1.AuxInt) != 7 || idx != x7_1.Args[0] || mem != x7.Args[2] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && clobber(x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7, o0, o1, o2, o3, o4, o5, s0)) {
				continue
			}
			b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
			v0 := b.NewValue0(v.Pos, OpARM64REV, t)
			v.copyOf(v0)
			v1 := b.NewValue0(v.Pos, OpARM64MOVDloadidx, t)
			v1.AddArg3(ptr, idx, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64ORN(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORN x (MOVDconst [c]))
	// result: (ORconst [^c] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(^c)
		v.AddArg(x)
		return true
	}
	// match: (ORN x x)
	// result: (MOVDconst [-1])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	// match: (ORN x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORNshiftLL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORNshiftLL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (ORN x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORNshiftRL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORNshiftRL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (ORN x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORNshiftRA x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORNshiftRA)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORNshiftLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORNshiftLL x (MOVDconst [c]) [d])
	// result: (ORconst x [^int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(^int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORNshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [-1])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORNshiftRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORNshiftRA x (MOVDconst [c]) [d])
	// result: (ORconst x [^(c>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(^(c >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORNshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [-1])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORNshiftRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORNshiftRL x (MOVDconst [c]) [d])
	// result: (ORconst x [^int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(^int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORNshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [-1])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORconst(v *Value) bool {
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
	// result: (MOVDconst [-1])
	for {
		if auxIntToInt64(v.AuxInt) != -1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	// match: (ORconst [c] (MOVDconst [d]))
	// result: (MOVDconst [c|d])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(c | d)
		return true
	}
	// match: (ORconst [c] (ORconst [d] x))
	// result: (ORconst [c|d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64ORconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(c | d)
		v.AddArg(x)
		return true
	}
	// match: (ORconst [c1] (ANDconst [c2] x))
	// cond: c2|c1 == ^0
	// result: (ORconst [c1] x)
	for {
		c1 := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64ANDconst {
			break
		}
		c2 := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c2|c1 == ^0) {
			break
		}
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(c1)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORshiftLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ORshiftLL (MOVDconst [c]) x [d])
	// result: (ORconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftLL x (MOVDconst [c]) [d])
	// result: (ORconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL x y:(SLLconst x [c]) [d])
	// cond: c==d
	// result: y
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		y := v_1
		if y.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(y.AuxInt)
		if x != y.Args[0] || !(c == d) {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: ( ORshiftLL [c] (SRLconst x [64-c]) x)
	// result: (RORconst [64-c] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != 64-c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = int64ToAuxInt(64 - c)
		v.AddArg(x)
		return true
	}
	// match: ( ORshiftLL <t> [c] (UBFX [bfc] x) x)
	// cond: c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)
	// result: (RORWconst [32-c] x)
	for {
		t := v.Type
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if x != v_1 || !(c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = int64ToAuxInt(32 - c)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL <typ.UInt16> [8] (UBFX <typ.UInt16> [armBFAuxInt(8, 8)] x) x)
	// result: (REV16W x)
	for {
		if v.Type != typ.UInt16 || auxIntToInt64(v.AuxInt) != 8 || v_0.Op != OpARM64UBFX || v_0.Type != typ.UInt16 || auxIntToArm64BitField(v_0.AuxInt) != armBFAuxInt(8, 8) {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: ( ORshiftLL [c] (SRLconst x [64-c]) x2)
	// result: (EXTRconst [64-c] x2 x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != 64-c {
			break
		}
		x := v_0.Args[0]
		x2 := v_1
		v.reset(OpARM64EXTRconst)
		v.AuxInt = int64ToAuxInt(64 - c)
		v.AddArg2(x2, x)
		return true
	}
	// match: ( ORshiftLL <t> [c] (UBFX [bfc] x) x2)
	// cond: c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)
	// result: (EXTRWconst [32-c] x2 x)
	for {
		t := v.Type
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		x2 := v_1
		if !(c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)) {
			break
		}
		v.reset(OpARM64EXTRWconst)
		v.AuxInt = int64ToAuxInt(32 - c)
		v.AddArg2(x2, x)
		return true
	}
	// match: (ORshiftLL [sc] (UBFX [bfc] x) (SRLconst [sc] y))
	// cond: sc == bfc.getARM64BFwidth()
	// result: (BFXIL [bfc] y x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if v_1.Op != OpARM64SRLconst || auxIntToInt64(v_1.AuxInt) != sc {
			break
		}
		y := v_1.Args[0]
		if !(sc == bfc.getARM64BFwidth()) {
			break
		}
		v.reset(OpARM64BFXIL)
		v.AuxInt = arm64BitFieldToAuxInt(bfc)
		v.AddArg2(y, x)
		return true
	}
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem)))
	// cond: i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, y0, y1)
	// result: @mergePoint(b,x0,x1) (MOVHUload <t> {s} (OffPtr <p.Type> [int64(i0)] p) mem)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		y0 := v_0
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		i0 := auxIntToInt32(x0.AuxInt)
		s := auxToSym(x0.Aux)
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := v_1
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i1 := auxIntToInt32(x1.AuxInt)
		if auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] || mem != x1.Args[1] || !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, y0, y1)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(x1.Pos, OpARM64MOVHUload, t)
		v.copyOf(v0)
		v0.Aux = symToAux(s)
		v1 := b.NewValue0(x1.Pos, OpOffPtr, p.Type)
		v1.AuxInt = int64ToAuxInt(int64(i0))
		v1.AddArg(p)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUloadidx ptr0 idx0 mem)) y1:(MOVDnop x1:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b,x0,x1) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x0, x1, y0, y1)
	// result: @mergePoint(b,x0,x1) (MOVHUloadidx <t> ptr0 idx0 mem)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		y0 := v_0
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUloadidx {
			break
		}
		mem := x0.Args[2]
		ptr0 := x0.Args[0]
		idx0 := x0.Args[1]
		y1 := v_1
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload || auxIntToInt32(x1.AuxInt) != 1 {
			break
		}
		s := auxToSym(x1.Aux)
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		_ = p1.Args[1]
		p1_0 := p1.Args[0]
		p1_1 := p1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, p1_0, p1_1 = _i0+1, p1_1, p1_0 {
			ptr1 := p1_0
			idx1 := p1_1
			if mem != x1.Args[1] || !(s == nil && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b, x0, x1) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x0, x1, y0, y1)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x1.Pos, OpARM64MOVHUloadidx, t)
			v.copyOf(v0)
			v0.AddArg3(ptr0, idx0, mem)
			return true
		}
		break
	}
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUloadidx ptr idx mem)) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [1] idx) mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, y0, y1)
	// result: @mergePoint(b,x0,x1) (MOVHUloadidx <t> ptr idx mem)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		y0 := v_0
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUloadidx {
			break
		}
		mem := x0.Args[2]
		ptr := x0.Args[0]
		idx := x0.Args[1]
		y1 := v_1
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64ADDconst || auxIntToInt64(x1_1.AuxInt) != 1 || idx != x1_1.Args[0] || mem != x1.Args[2] || !(x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, y0, y1)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpARM64MOVHUloadidx, t)
		v.copyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] x0:(MOVHUload [i0] {s} p mem) y1:(MOVDnop x1:(MOVBUload [i2] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i3] {s} p mem)))
	// cond: i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && clobber(x0, x1, x2, y1, y2, o0)
	// result: @mergePoint(b,x0,x1,x2) (MOVWUload <t> {s} (OffPtr <p.Type> [int64(i0)] p) mem)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 24 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 16 {
			break
		}
		_ = o0.Args[1]
		x0 := o0.Args[0]
		if x0.Op != OpARM64MOVHUload {
			break
		}
		i0 := auxIntToInt32(x0.AuxInt)
		s := auxToSym(x0.Aux)
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := o0.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i2 := auxIntToInt32(x1.AuxInt)
		if auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] || mem != x1.Args[1] {
			break
		}
		y2 := v_1
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i3 := auxIntToInt32(x2.AuxInt)
		if auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] || mem != x2.Args[1] || !(i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && clobber(x0, x1, x2, y1, y2, o0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2)
		v0 := b.NewValue0(x2.Pos, OpARM64MOVWUload, t)
		v.copyOf(v0)
		v0.Aux = symToAux(s)
		v1 := b.NewValue0(x2.Pos, OpOffPtr, p.Type)
		v1.AuxInt = int64ToAuxInt(int64(i0))
		v1.AddArg(p)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] x0:(MOVHUloadidx ptr0 idx0 mem) y1:(MOVDnop x1:(MOVBUload [2] {s} p1:(ADD ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [3] {s} p mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, y1, y2, o0)
	// result: @mergePoint(b,x0,x1,x2) (MOVWUloadidx <t> ptr0 idx0 mem)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 24 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 16 {
			break
		}
		_ = o0.Args[1]
		x0 := o0.Args[0]
		if x0.Op != OpARM64MOVHUloadidx {
			break
		}
		mem := x0.Args[2]
		ptr0 := x0.Args[0]
		idx0 := x0.Args[1]
		y1 := o0.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload || auxIntToInt32(x1.AuxInt) != 2 {
			break
		}
		s := auxToSym(x1.Aux)
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		_ = p1.Args[1]
		p1_0 := p1.Args[0]
		p1_1 := p1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, p1_0, p1_1 = _i0+1, p1_1, p1_0 {
			ptr1 := p1_0
			idx1 := p1_1
			if mem != x1.Args[1] {
				continue
			}
			y2 := v_1
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUload || auxIntToInt32(x2.AuxInt) != 3 || auxToSym(x2.Aux) != s {
				continue
			}
			_ = x2.Args[1]
			p := x2.Args[0]
			if mem != x2.Args[1] || !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, y1, y2, o0)) {
				continue
			}
			b = mergePoint(b, x0, x1, x2)
			v0 := b.NewValue0(x2.Pos, OpARM64MOVWUloadidx, t)
			v.copyOf(v0)
			v0.AddArg3(ptr0, idx0, mem)
			return true
		}
		break
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] x0:(MOVHUloadidx ptr idx mem) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [3] idx) mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && clobber(x0, x1, x2, y1, y2, o0)
	// result: @mergePoint(b,x0,x1,x2) (MOVWUloadidx <t> ptr idx mem)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 24 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 16 {
			break
		}
		_ = o0.Args[1]
		x0 := o0.Args[0]
		if x0.Op != OpARM64MOVHUloadidx {
			break
		}
		mem := x0.Args[2]
		ptr := x0.Args[0]
		idx := x0.Args[1]
		y1 := o0.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64ADDconst || auxIntToInt64(x1_1.AuxInt) != 2 || idx != x1_1.Args[0] || mem != x1.Args[2] {
			break
		}
		y2 := v_1
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x2.Args[2]
		if ptr != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64ADDconst || auxIntToInt64(x2_1.AuxInt) != 3 || idx != x2_1.Args[0] || mem != x2.Args[2] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && clobber(x0, x1, x2, y1, y2, o0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2)
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUloadidx, t)
		v.copyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] x0:(MOVHUloadidx2 ptr0 idx0 mem) y1:(MOVDnop x1:(MOVBUload [2] {s} p1:(ADDshiftLL [1] ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [3] {s} p mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && isSamePtr(p1, p) && clobber(x0, x1, x2, y1, y2, o0)
	// result: @mergePoint(b,x0,x1,x2) (MOVWUloadidx <t> ptr0 (SLLconst <idx0.Type> [1] idx0) mem)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 24 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 16 {
			break
		}
		_ = o0.Args[1]
		x0 := o0.Args[0]
		if x0.Op != OpARM64MOVHUloadidx2 {
			break
		}
		mem := x0.Args[2]
		ptr0 := x0.Args[0]
		idx0 := x0.Args[1]
		y1 := o0.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload || auxIntToInt32(x1.AuxInt) != 2 {
			break
		}
		s := auxToSym(x1.Aux)
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADDshiftLL || auxIntToInt64(p1.AuxInt) != 1 {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x1.Args[1] {
			break
		}
		y2 := v_1
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload || auxIntToInt32(x2.AuxInt) != 3 || auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[1]
		p := x2.Args[0]
		if mem != x2.Args[1] || !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && isSamePtr(p1, p) && clobber(x0, x1, x2, y1, y2, o0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2)
		v0 := b.NewValue0(x2.Pos, OpARM64MOVWUloadidx, t)
		v.copyOf(v0)
		v1 := b.NewValue0(x2.Pos, OpARM64SLLconst, idx0.Type)
		v1.AuxInt = int64ToAuxInt(1)
		v1.AddArg(idx0)
		v0.AddArg3(ptr0, v1, mem)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] x0:(MOVWUload [i0] {s} p mem) y1:(MOVDnop x1:(MOVBUload [i4] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i6] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i7] {s} p mem)))
	// cond: i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && clobber(x0, x1, x2, x3, x4, y1, y2, y3, y4, o0, o1, o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (MOVDload <t> {s} (OffPtr <p.Type> [int64(i0)] p) mem)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 56 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 32 {
			break
		}
		_ = o2.Args[1]
		x0 := o2.Args[0]
		if x0.Op != OpARM64MOVWUload {
			break
		}
		i0 := auxIntToInt32(x0.AuxInt)
		s := auxToSym(x0.Aux)
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := o2.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i4 := auxIntToInt32(x1.AuxInt)
		if auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] || mem != x1.Args[1] {
			break
		}
		y2 := o1.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i5 := auxIntToInt32(x2.AuxInt)
		if auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] || mem != x2.Args[1] {
			break
		}
		y3 := o0.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		i6 := auxIntToInt32(x3.AuxInt)
		if auxToSym(x3.Aux) != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] || mem != x3.Args[1] {
			break
		}
		y4 := v_1
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		i7 := auxIntToInt32(x4.AuxInt)
		if auxToSym(x4.Aux) != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] || mem != x4.Args[1] || !(i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && clobber(x0, x1, x2, x3, x4, y1, y2, y3, y4, o0, o1, o2)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4)
		v0 := b.NewValue0(x4.Pos, OpARM64MOVDload, t)
		v.copyOf(v0)
		v0.Aux = symToAux(s)
		v1 := b.NewValue0(x4.Pos, OpOffPtr, p.Type)
		v1.AuxInt = int64ToAuxInt(int64(i0))
		v1.AddArg(p)
		v0.AddArg2(v1, mem)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] x0:(MOVWUloadidx ptr0 idx0 mem) y1:(MOVDnop x1:(MOVBUload [4] {s} p1:(ADD ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [6] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [7] {s} p mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, x4, y1, y2, y3, y4, o0, o1, o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (MOVDloadidx <t> ptr0 idx0 mem)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 56 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 32 {
			break
		}
		_ = o2.Args[1]
		x0 := o2.Args[0]
		if x0.Op != OpARM64MOVWUloadidx {
			break
		}
		mem := x0.Args[2]
		ptr0 := x0.Args[0]
		idx0 := x0.Args[1]
		y1 := o2.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload || auxIntToInt32(x1.AuxInt) != 4 {
			break
		}
		s := auxToSym(x1.Aux)
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		_ = p1.Args[1]
		p1_0 := p1.Args[0]
		p1_1 := p1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, p1_0, p1_1 = _i0+1, p1_1, p1_0 {
			ptr1 := p1_0
			idx1 := p1_1
			if mem != x1.Args[1] {
				continue
			}
			y2 := o1.Args[1]
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUload || auxIntToInt32(x2.AuxInt) != 5 || auxToSym(x2.Aux) != s {
				continue
			}
			_ = x2.Args[1]
			p := x2.Args[0]
			if mem != x2.Args[1] {
				continue
			}
			y3 := o0.Args[1]
			if y3.Op != OpARM64MOVDnop {
				continue
			}
			x3 := y3.Args[0]
			if x3.Op != OpARM64MOVBUload || auxIntToInt32(x3.AuxInt) != 6 || auxToSym(x3.Aux) != s {
				continue
			}
			_ = x3.Args[1]
			if p != x3.Args[0] || mem != x3.Args[1] {
				continue
			}
			y4 := v_1
			if y4.Op != OpARM64MOVDnop {
				continue
			}
			x4 := y4.Args[0]
			if x4.Op != OpARM64MOVBUload || auxIntToInt32(x4.AuxInt) != 7 || auxToSym(x4.Aux) != s {
				continue
			}
			_ = x4.Args[1]
			if p != x4.Args[0] || mem != x4.Args[1] || !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, x4, y1, y2, y3, y4, o0, o1, o2)) {
				continue
			}
			b = mergePoint(b, x0, x1, x2, x3, x4)
			v0 := b.NewValue0(x4.Pos, OpARM64MOVDloadidx, t)
			v.copyOf(v0)
			v0.AddArg3(ptr0, idx0, mem)
			return true
		}
		break
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] x0:(MOVWUloadidx4 ptr0 idx0 mem) y1:(MOVDnop x1:(MOVBUload [4] {s} p1:(ADDshiftLL [2] ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [6] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [7] {s} p mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, x4, y1, y2, y3, y4, o0, o1, o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (MOVDloadidx <t> ptr0 (SLLconst <idx0.Type> [2] idx0) mem)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 56 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 32 {
			break
		}
		_ = o2.Args[1]
		x0 := o2.Args[0]
		if x0.Op != OpARM64MOVWUloadidx4 {
			break
		}
		mem := x0.Args[2]
		ptr0 := x0.Args[0]
		idx0 := x0.Args[1]
		y1 := o2.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload || auxIntToInt32(x1.AuxInt) != 4 {
			break
		}
		s := auxToSym(x1.Aux)
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADDshiftLL || auxIntToInt64(p1.AuxInt) != 2 {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x1.Args[1] {
			break
		}
		y2 := o1.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload || auxIntToInt32(x2.AuxInt) != 5 || auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[1]
		p := x2.Args[0]
		if mem != x2.Args[1] {
			break
		}
		y3 := o0.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload || auxIntToInt32(x3.AuxInt) != 6 || auxToSym(x3.Aux) != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] || mem != x3.Args[1] {
			break
		}
		y4 := v_1
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload || auxIntToInt32(x4.AuxInt) != 7 || auxToSym(x4.Aux) != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] || mem != x4.Args[1] || !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, x4, y1, y2, y3, y4, o0, o1, o2)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4)
		v0 := b.NewValue0(x4.Pos, OpARM64MOVDloadidx, t)
		v.copyOf(v0)
		v1 := b.NewValue0(x4.Pos, OpARM64SLLconst, idx0.Type)
		v1.AuxInt = int64ToAuxInt(2)
		v1.AddArg(idx0)
		v0.AddArg3(ptr0, v1, mem)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] x0:(MOVWUloadidx ptr idx mem) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [4] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [5] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [6] idx) mem))) y4:(MOVDnop x4:(MOVBUloadidx ptr (ADDconst [7] idx) mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && clobber(x0, x1, x2, x3, x4, y1, y2, y3, y4, o0, o1, o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (MOVDloadidx <t> ptr idx mem)
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 56 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 32 {
			break
		}
		_ = o2.Args[1]
		x0 := o2.Args[0]
		if x0.Op != OpARM64MOVWUloadidx {
			break
		}
		mem := x0.Args[2]
		ptr := x0.Args[0]
		idx := x0.Args[1]
		y1 := o2.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64ADDconst || auxIntToInt64(x1_1.AuxInt) != 4 || idx != x1_1.Args[0] || mem != x1.Args[2] {
			break
		}
		y2 := o1.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x2.Args[2]
		if ptr != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64ADDconst || auxIntToInt64(x2_1.AuxInt) != 5 || idx != x2_1.Args[0] || mem != x2.Args[2] {
			break
		}
		y3 := o0.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x3.Args[2]
		if ptr != x3.Args[0] {
			break
		}
		x3_1 := x3.Args[1]
		if x3_1.Op != OpARM64ADDconst || auxIntToInt64(x3_1.AuxInt) != 6 || idx != x3_1.Args[0] || mem != x3.Args[2] {
			break
		}
		y4 := v_1
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x4.Args[2]
		if ptr != x4.Args[0] {
			break
		}
		x4_1 := x4.Args[1]
		if x4_1.Op != OpARM64ADDconst || auxIntToInt64(x4_1.AuxInt) != 7 || idx != x4_1.Args[0] || mem != x4.Args[2] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && clobber(x0, x1, x2, x3, x4, y1, y2, y3, y4, o0, o1, o2)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDloadidx, t)
		v.copyOf(v0)
		v0.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUload [i1] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, y0, y1)
	// result: @mergePoint(b,x0,x1) (REV16W <t> (MOVHUload <t> [i0] {s} p mem))
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		y0 := v_0
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		i1 := auxIntToInt32(x0.AuxInt)
		s := auxToSym(x0.Aux)
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := v_1
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i0 := auxIntToInt32(x1.AuxInt)
		if auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] || mem != x1.Args[1] || !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, y0, y1)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(x1.Pos, OpARM64REV16W, t)
		v.copyOf(v0)
		v1 := b.NewValue0(x1.Pos, OpARM64MOVHUload, t)
		v1.AuxInt = int32ToAuxInt(i0)
		v1.Aux = symToAux(s)
		v1.AddArg2(p, mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem)) y1:(MOVDnop x1:(MOVBUloadidx ptr0 idx0 mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b,x0,x1) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x0, x1, y0, y1)
	// result: @mergePoint(b,x0,x1) (REV16W <t> (MOVHUloadidx <t> ptr0 idx0 mem))
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		y0 := v_0
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload || auxIntToInt32(x0.AuxInt) != 1 {
			break
		}
		s := auxToSym(x0.Aux)
		mem := x0.Args[1]
		p1 := x0.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		_ = p1.Args[1]
		p1_0 := p1.Args[0]
		p1_1 := p1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, p1_0, p1_1 = _i0+1, p1_1, p1_0 {
			ptr1 := p1_0
			idx1 := p1_1
			y1 := v_1
			if y1.Op != OpARM64MOVDnop {
				continue
			}
			x1 := y1.Args[0]
			if x1.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x1.Args[2]
			ptr0 := x1.Args[0]
			idx0 := x1.Args[1]
			if mem != x1.Args[2] || !(s == nil && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b, x0, x1) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x0, x1, y0, y1)) {
				continue
			}
			b = mergePoint(b, x0, x1)
			v0 := b.NewValue0(x0.Pos, OpARM64REV16W, t)
			v.copyOf(v0)
			v1 := b.NewValue0(x0.Pos, OpARM64MOVHUloadidx, t)
			v1.AddArg3(ptr0, idx0, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUloadidx ptr (ADDconst [1] idx) mem)) y1:(MOVDnop x1:(MOVBUloadidx ptr idx mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0, x1, y0, y1)
	// result: @mergePoint(b,x0,x1) (REV16W <t> (MOVHUloadidx <t> ptr idx mem))
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		y0 := v_0
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUloadidx {
			break
		}
		mem := x0.Args[2]
		ptr := x0.Args[0]
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64ADDconst || auxIntToInt64(x0_1.AuxInt) != 1 {
			break
		}
		idx := x0_1.Args[0]
		y1 := v_1
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] || idx != x1.Args[1] || mem != x1.Args[2] || !(x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0, x1, y0, y1)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpARM64REV16W, t)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHUloadidx, t)
		v1.AddArg3(ptr, idx, mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] y0:(REV16W x0:(MOVHUload [i2] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 && i2 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && clobber(x0, x1, x2, y0, y1, y2, o0)
	// result: @mergePoint(b,x0,x1,x2) (REVW <t> (MOVWUload <t> {s} (OffPtr <p.Type> [int64(i0)] p) mem))
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 24 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 16 {
			break
		}
		_ = o0.Args[1]
		y0 := o0.Args[0]
		if y0.Op != OpARM64REV16W {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVHUload {
			break
		}
		i2 := auxIntToInt32(x0.AuxInt)
		s := auxToSym(x0.Aux)
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := o0.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i1 := auxIntToInt32(x1.AuxInt)
		if auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] || mem != x1.Args[1] {
			break
		}
		y2 := v_1
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i0 := auxIntToInt32(x2.AuxInt)
		if auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] || mem != x2.Args[1] || !(i1 == i0+1 && i2 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && clobber(x0, x1, x2, y0, y1, y2, o0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2)
		v0 := b.NewValue0(x2.Pos, OpARM64REVW, t)
		v.copyOf(v0)
		v1 := b.NewValue0(x2.Pos, OpARM64MOVWUload, t)
		v1.Aux = symToAux(s)
		v2 := b.NewValue0(x2.Pos, OpOffPtr, p.Type)
		v2.AuxInt = int64ToAuxInt(int64(i0))
		v2.AddArg(p)
		v1.AddArg2(v2, mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] y0:(REV16W x0:(MOVHUload [2] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr0 idx0 mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, y0, y1, y2, o0)
	// result: @mergePoint(b,x0,x1,x2) (REVW <t> (MOVWUloadidx <t> ptr0 idx0 mem))
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 24 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 16 {
			break
		}
		_ = o0.Args[1]
		y0 := o0.Args[0]
		if y0.Op != OpARM64REV16W {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVHUload || auxIntToInt32(x0.AuxInt) != 2 {
			break
		}
		s := auxToSym(x0.Aux)
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := o0.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload || auxIntToInt32(x1.AuxInt) != 1 || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		_ = p1.Args[1]
		p1_0 := p1.Args[0]
		p1_1 := p1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, p1_0, p1_1 = _i0+1, p1_1, p1_0 {
			ptr1 := p1_0
			idx1 := p1_1
			if mem != x1.Args[1] {
				continue
			}
			y2 := v_1
			if y2.Op != OpARM64MOVDnop {
				continue
			}
			x2 := y2.Args[0]
			if x2.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x2.Args[2]
			ptr0 := x2.Args[0]
			idx0 := x2.Args[1]
			if mem != x2.Args[2] || !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, y0, y1, y2, o0)) {
				continue
			}
			b = mergePoint(b, x0, x1, x2)
			v0 := b.NewValue0(x1.Pos, OpARM64REVW, t)
			v.copyOf(v0)
			v1 := b.NewValue0(x1.Pos, OpARM64MOVWUloadidx, t)
			v1.AddArg3(ptr0, idx0, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] y0:(REV16W x0:(MOVHUloadidx ptr (ADDconst [2] idx) mem)) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr idx mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && clobber(x0, x1, x2, y0, y1, y2, o0)
	// result: @mergePoint(b,x0,x1,x2) (REVW <t> (MOVWUloadidx <t> ptr idx mem))
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 24 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 16 {
			break
		}
		_ = o0.Args[1]
		y0 := o0.Args[0]
		if y0.Op != OpARM64REV16W {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVHUloadidx {
			break
		}
		mem := x0.Args[2]
		ptr := x0.Args[0]
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64ADDconst || auxIntToInt64(x0_1.AuxInt) != 2 {
			break
		}
		idx := x0_1.Args[0]
		y1 := o0.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64ADDconst || auxIntToInt64(x1_1.AuxInt) != 1 || idx != x1_1.Args[0] || mem != x1.Args[2] {
			break
		}
		y2 := v_1
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x2.Args[2]
		if ptr != x2.Args[0] || idx != x2.Args[1] || mem != x2.Args[2] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && clobber(x0, x1, x2, y0, y1, y2, o0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2)
		v0 := b.NewValue0(v.Pos, OpARM64REVW, t)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWUloadidx, t)
		v1.AddArg3(ptr, idx, mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] y0:(REVW x0:(MOVWUload [i4] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [i3] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i1] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && clobber(x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, o0, o1, o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (REV <t> (MOVDload <t> {s} (OffPtr <p.Type> [int64(i0)] p) mem))
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 56 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 32 {
			break
		}
		_ = o2.Args[1]
		y0 := o2.Args[0]
		if y0.Op != OpARM64REVW {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVWUload {
			break
		}
		i4 := auxIntToInt32(x0.AuxInt)
		s := auxToSym(x0.Aux)
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := o2.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i3 := auxIntToInt32(x1.AuxInt)
		if auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] || mem != x1.Args[1] {
			break
		}
		y2 := o1.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i2 := auxIntToInt32(x2.AuxInt)
		if auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] || mem != x2.Args[1] {
			break
		}
		y3 := o0.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		i1 := auxIntToInt32(x3.AuxInt)
		if auxToSym(x3.Aux) != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] || mem != x3.Args[1] {
			break
		}
		y4 := v_1
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		i0 := auxIntToInt32(x4.AuxInt)
		if auxToSym(x4.Aux) != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] || mem != x4.Args[1] || !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && clobber(x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, o0, o1, o2)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4)
		v0 := b.NewValue0(x4.Pos, OpARM64REV, t)
		v.copyOf(v0)
		v1 := b.NewValue0(x4.Pos, OpARM64MOVDload, t)
		v1.Aux = symToAux(s)
		v2 := b.NewValue0(x4.Pos, OpOffPtr, p.Type)
		v2.AuxInt = int64ToAuxInt(int64(i0))
		v2.AddArg(p)
		v1.AddArg2(v2, mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] y0:(REVW x0:(MOVWUload [4] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [3] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y4:(MOVDnop x4:(MOVBUloadidx ptr0 idx0 mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, o0, o1, o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (REV <t> (MOVDloadidx <t> ptr0 idx0 mem))
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 56 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 32 {
			break
		}
		_ = o2.Args[1]
		y0 := o2.Args[0]
		if y0.Op != OpARM64REVW {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVWUload || auxIntToInt32(x0.AuxInt) != 4 {
			break
		}
		s := auxToSym(x0.Aux)
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := o2.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload || auxIntToInt32(x1.AuxInt) != 3 || auxToSym(x1.Aux) != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] || mem != x1.Args[1] {
			break
		}
		y2 := o1.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload || auxIntToInt32(x2.AuxInt) != 2 || auxToSym(x2.Aux) != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] || mem != x2.Args[1] {
			break
		}
		y3 := o0.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload || auxIntToInt32(x3.AuxInt) != 1 || auxToSym(x3.Aux) != s {
			break
		}
		_ = x3.Args[1]
		p1 := x3.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		_ = p1.Args[1]
		p1_0 := p1.Args[0]
		p1_1 := p1.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, p1_0, p1_1 = _i0+1, p1_1, p1_0 {
			ptr1 := p1_0
			idx1 := p1_1
			if mem != x3.Args[1] {
				continue
			}
			y4 := v_1
			if y4.Op != OpARM64MOVDnop {
				continue
			}
			x4 := y4.Args[0]
			if x4.Op != OpARM64MOVBUloadidx {
				continue
			}
			_ = x4.Args[2]
			ptr0 := x4.Args[0]
			idx0 := x4.Args[1]
			if mem != x4.Args[2] || !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, o0, o1, o2)) {
				continue
			}
			b = mergePoint(b, x0, x1, x2, x3, x4)
			v0 := b.NewValue0(x3.Pos, OpARM64REV, t)
			v.copyOf(v0)
			v1 := b.NewValue0(x3.Pos, OpARM64MOVDloadidx, t)
			v1.AddArg3(ptr0, idx0, mem)
			v0.AddArg(v1)
			return true
		}
		break
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] y0:(REVW x0:(MOVWUloadidx ptr (ADDconst [4] idx) mem)) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [3] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y4:(MOVDnop x4:(MOVBUloadidx ptr idx mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && clobber(x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, o0, o1, o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (REV <t> (MOVDloadidx <t> ptr idx mem))
	for {
		t := v.Type
		if auxIntToInt64(v.AuxInt) != 56 {
			break
		}
		o0 := v_0
		if o0.Op != OpARM64ORshiftLL || auxIntToInt64(o0.AuxInt) != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL || auxIntToInt64(o1.AuxInt) != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL || auxIntToInt64(o2.AuxInt) != 32 {
			break
		}
		_ = o2.Args[1]
		y0 := o2.Args[0]
		if y0.Op != OpARM64REVW {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVWUloadidx {
			break
		}
		mem := x0.Args[2]
		ptr := x0.Args[0]
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64ADDconst || auxIntToInt64(x0_1.AuxInt) != 4 {
			break
		}
		idx := x0_1.Args[0]
		y1 := o2.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64ADDconst || auxIntToInt64(x1_1.AuxInt) != 3 || idx != x1_1.Args[0] || mem != x1.Args[2] {
			break
		}
		y2 := o1.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x2.Args[2]
		if ptr != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64ADDconst || auxIntToInt64(x2_1.AuxInt) != 2 || idx != x2_1.Args[0] || mem != x2.Args[2] {
			break
		}
		y3 := o0.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x3.Args[2]
		if ptr != x3.Args[0] {
			break
		}
		x3_1 := x3.Args[1]
		if x3_1.Op != OpARM64ADDconst || auxIntToInt64(x3_1.AuxInt) != 1 || idx != x3_1.Args[0] || mem != x3.Args[2] {
			break
		}
		y4 := v_1
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x4.Args[2]
		if ptr != x4.Args[0] || idx != x4.Args[1] || mem != x4.Args[2] || !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && clobber(x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, o0, o1, o2)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4)
		v0 := b.NewValue0(v.Pos, OpARM64REV, t)
		v.copyOf(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDloadidx, t)
		v1.AddArg3(ptr, idx, mem)
		v0.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORshiftRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ORshiftRA (MOVDconst [c]) x [d])
	// result: (ORconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRA x (MOVDconst [c]) [d])
	// result: (ORconst x [c>>uint64(d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRA x y:(SRAconst x [c]) [d])
	// cond: c==d
	// result: y
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		y := v_1
		if y.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(y.AuxInt)
		if x != y.Args[0] || !(c == d) {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORshiftRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ORshiftRL (MOVDconst [c]) x [d])
	// result: (ORconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRL x (MOVDconst [c]) [d])
	// result: (ORconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRL x y:(SRLconst x [c]) [d])
	// cond: c==d
	// result: y
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		y := v_1
		if y.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(y.AuxInt)
		if x != y.Args[0] || !(c == d) {
			break
		}
		v.copyOf(y)
		return true
	}
	// match: ( ORshiftRL [c] (SLLconst x [64-c]) x)
	// result: (RORconst [ c] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 64-c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: ( ORshiftRL <t> [c] (SLLconst x [32-c]) (MOVWUreg x))
	// cond: c < 32 && t.Size() == 4
	// result: (RORWconst [c] x)
	for {
		t := v.Type
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 32-c {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpARM64MOVWUreg || x != v_1.Args[0] || !(c < 32 && t.Size() == 4) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRL [rc] (ANDconst [ac] x) (SLLconst [lc] y))
	// cond: lc > rc && ac == ^((1<<uint(64-lc)-1) << uint64(lc-rc))
	// result: (BFI [armBFAuxInt(lc-rc, 64-lc)] x y)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64ANDconst {
			break
		}
		ac := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_1.AuxInt)
		y := v_1.Args[0]
		if !(lc > rc && ac == ^((1<<uint(64-lc)-1)<<uint64(lc-rc))) {
			break
		}
		v.reset(OpARM64BFI)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc-rc, 64-lc))
		v.AddArg2(x, y)
		return true
	}
	// match: (ORshiftRL [rc] (ANDconst [ac] y) (SLLconst [lc] x))
	// cond: lc < rc && ac == ^((1<<uint(64-rc)-1))
	// result: (BFXIL [armBFAuxInt(rc-lc, 64-rc)] y x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64ANDconst {
			break
		}
		ac := auxIntToInt64(v_0.AuxInt)
		y := v_0.Args[0]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_1.AuxInt)
		x := v_1.Args[0]
		if !(lc < rc && ac == ^(1<<uint(64-rc)-1)) {
			break
		}
		v.reset(OpARM64BFXIL)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(rc-lc, 64-rc))
		v.AddArg2(y, x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64RORWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RORWconst [c] (RORWconst [d] x))
	// result: (RORWconst [(c+d)&31] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64RORWconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpARM64RORWconst)
		v.AuxInt = int64ToAuxInt((c + d) & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64RORconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (RORconst [c] (RORconst [d] x))
	// result: (RORconst [(c+d)&63] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64RORconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpARM64RORconst)
		v.AuxInt = int64ToAuxInt((c + d) & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SBCSflags(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SBCSflags x y (Select1 <types.TypeFlags> (NEGSflags (NEG <typ.UInt64> (NGCzerocarry <typ.UInt64> bo)))))
	// result: (SBCSflags x y bo)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpSelect1 || v_2.Type != types.TypeFlags {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpARM64NEGSflags {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpARM64NEG || v_2_0_0.Type != typ.UInt64 {
			break
		}
		v_2_0_0_0 := v_2_0_0.Args[0]
		if v_2_0_0_0.Op != OpARM64NGCzerocarry || v_2_0_0_0.Type != typ.UInt64 {
			break
		}
		bo := v_2_0_0_0.Args[0]
		v.reset(OpARM64SBCSflags)
		v.AddArg3(x, y, bo)
		return true
	}
	// match: (SBCSflags x y (Select1 <types.TypeFlags> (NEGSflags (MOVDconst [0]))))
	// result: (SUBSflags x y)
	for {
		x := v_0
		y := v_1
		if v_2.Op != OpSelect1 || v_2.Type != types.TypeFlags {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpARM64NEGSflags {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_2_0_0.AuxInt) != 0 {
			break
		}
		v.reset(OpARM64SUBSflags)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLL x (MOVDconst [c]))
	// result: (SLLconst x [c&63])
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64SLLconst)
		v.AuxInt = int64ToAuxInt(c & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SLLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SLLconst [c] (MOVDconst [d]))
	// result: (MOVDconst [d<<uint64(c)])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(d << uint64(c))
		return true
	}
	// match: (SLLconst [c] (SRLconst [c] x))
	// cond: 0 < c && c < 64
	// result: (ANDconst [^(1<<uint(c)-1)] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if !(0 < c && c < 64) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(^(1<<uint(c) - 1))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [sc] (ANDconst [ac] x))
	// cond: isARM64BFMask(sc, ac, 0)
	// result: (UBFIZ [armBFAuxInt(sc, arm64BFWidth(ac, 0))] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64ANDconst {
			break
		}
		ac := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, ac, 0)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, arm64BFWidth(ac, 0)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [sc] (MOVWUreg x))
	// cond: isARM64BFMask(sc, 1<<32-1, 0)
	// result: (UBFIZ [armBFAuxInt(sc, 32)] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<32-1, 0)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, 32))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [sc] (MOVHUreg x))
	// cond: isARM64BFMask(sc, 1<<16-1, 0)
	// result: (UBFIZ [armBFAuxInt(sc, 16)] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<16-1, 0)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, 16))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [sc] (MOVBUreg x))
	// cond: isARM64BFMask(sc, 1<<8-1, 0)
	// result: (UBFIZ [armBFAuxInt(sc, 8)] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<8-1, 0)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, 8))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [sc] (UBFIZ [bfc] x))
	// cond: sc+bfc.getARM64BFwidth()+bfc.getARM64BFlsb() < 64
	// result: (UBFIZ [armBFAuxInt(bfc.getARM64BFlsb()+sc, bfc.getARM64BFwidth())] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc+bfc.getARM64BFwidth()+bfc.getARM64BFlsb() < 64) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.getARM64BFlsb()+sc, bfc.getARM64BFwidth()))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRA x (MOVDconst [c]))
	// result: (SRAconst x [c&63])
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64SRAconst)
		v.AuxInt = int64ToAuxInt(c & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SRAconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SRAconst [c] (MOVDconst [d]))
	// result: (MOVDconst [d>>uint64(c)])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(d >> uint64(c))
		return true
	}
	// match: (SRAconst [rc] (SLLconst [lc] x))
	// cond: lc > rc
	// result: (SBFIZ [armBFAuxInt(lc-rc, 64-lc)] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc > rc) {
			break
		}
		v.reset(OpARM64SBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc-rc, 64-lc))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [rc] (SLLconst [lc] x))
	// cond: lc <= rc
	// result: (SBFX [armBFAuxInt(rc-lc, 64-rc)] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc <= rc) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(rc-lc, 64-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [rc] (MOVWreg x))
	// cond: rc < 32
	// result: (SBFX [armBFAuxInt(rc, 32-rc)] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVWreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 32) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(rc, 32-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [rc] (MOVHreg x))
	// cond: rc < 16
	// result: (SBFX [armBFAuxInt(rc, 16-rc)] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVHreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 16) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(rc, 16-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [rc] (MOVBreg x))
	// cond: rc < 8
	// result: (SBFX [armBFAuxInt(rc, 8-rc)] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVBreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 8) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(rc, 8-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [sc] (SBFIZ [bfc] x))
	// cond: sc < bfc.getARM64BFlsb()
	// result: (SBFIZ [armBFAuxInt(bfc.getARM64BFlsb()-sc, bfc.getARM64BFwidth())] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SBFIZ {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.getARM64BFlsb()) {
			break
		}
		v.reset(OpARM64SBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.getARM64BFlsb()-sc, bfc.getARM64BFwidth()))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [sc] (SBFIZ [bfc] x))
	// cond: sc >= bfc.getARM64BFlsb() && sc < bfc.getARM64BFlsb()+bfc.getARM64BFwidth()
	// result: (SBFX [armBFAuxInt(sc-bfc.getARM64BFlsb(), bfc.getARM64BFlsb()+bfc.getARM64BFwidth()-sc)] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SBFIZ {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc >= bfc.getARM64BFlsb() && sc < bfc.getARM64BFlsb()+bfc.getARM64BFwidth()) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc-bfc.getARM64BFlsb(), bfc.getARM64BFlsb()+bfc.getARM64BFwidth()-sc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRL x (MOVDconst [c]))
	// result: (SRLconst x [c&63])
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64SRLconst)
		v.AuxInt = int64ToAuxInt(c & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SRLconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SRLconst [c] (MOVDconst [d]))
	// result: (MOVDconst [int64(uint64(d)>>uint64(c))])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(d) >> uint64(c)))
		return true
	}
	// match: (SRLconst [c] (SLLconst [c] x))
	// cond: 0 < c && c < 64
	// result: (ANDconst [1<<uint(64-c)-1] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if !(0 < c && c < 64) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(1<<uint(64-c) - 1)
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [rc] (SLLconst [lc] x))
	// cond: lc > rc
	// result: (UBFIZ [armBFAuxInt(lc-rc, 64-lc)] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc > rc) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc-rc, 64-lc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (ANDconst [ac] x))
	// cond: isARM64BFMask(sc, ac, sc)
	// result: (UBFX [armBFAuxInt(sc, arm64BFWidth(ac, sc))] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64ANDconst {
			break
		}
		ac := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, ac, sc)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, arm64BFWidth(ac, sc)))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (MOVWUreg x))
	// cond: isARM64BFMask(sc, 1<<32-1, sc)
	// result: (UBFX [armBFAuxInt(sc, arm64BFWidth(1<<32-1, sc))] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<32-1, sc)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, arm64BFWidth(1<<32-1, sc)))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (MOVHUreg x))
	// cond: isARM64BFMask(sc, 1<<16-1, sc)
	// result: (UBFX [armBFAuxInt(sc, arm64BFWidth(1<<16-1, sc))] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<16-1, sc)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, arm64BFWidth(1<<16-1, sc)))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (MOVBUreg x))
	// cond: isARM64BFMask(sc, 1<<8-1, sc)
	// result: (UBFX [armBFAuxInt(sc, arm64BFWidth(1<<8-1, sc))] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<8-1, sc)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc, arm64BFWidth(1<<8-1, sc)))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [rc] (SLLconst [lc] x))
	// cond: lc < rc
	// result: (UBFX [armBFAuxInt(rc-lc, 64-rc)] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < rc) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(rc-lc, 64-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFX [bfc] x))
	// cond: sc < bfc.getARM64BFwidth()
	// result: (UBFX [armBFAuxInt(bfc.getARM64BFlsb()+sc, bfc.getARM64BFwidth()-sc)] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.getARM64BFwidth()) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.getARM64BFlsb()+sc, bfc.getARM64BFwidth()-sc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFIZ [bfc] x))
	// cond: sc == bfc.getARM64BFlsb()
	// result: (ANDconst [1<<uint(bfc.getARM64BFwidth())-1] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc == bfc.getARM64BFlsb()) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(1<<uint(bfc.getARM64BFwidth()) - 1)
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFIZ [bfc] x))
	// cond: sc < bfc.getARM64BFlsb()
	// result: (UBFIZ [armBFAuxInt(bfc.getARM64BFlsb()-sc, bfc.getARM64BFwidth())] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.getARM64BFlsb()) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.getARM64BFlsb()-sc, bfc.getARM64BFwidth()))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFIZ [bfc] x))
	// cond: sc > bfc.getARM64BFlsb() && sc < bfc.getARM64BFlsb()+bfc.getARM64BFwidth()
	// result: (UBFX [armBFAuxInt(sc-bfc.getARM64BFlsb(), bfc.getARM64BFlsb()+bfc.getARM64BFwidth()-sc)] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc > bfc.getARM64BFlsb() && sc < bfc.getARM64BFlsb()+bfc.getARM64BFwidth()) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc-bfc.getARM64BFlsb(), bfc.getARM64BFlsb()+bfc.getARM64BFwidth()-sc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64STP(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (STP [off1] {sym} (ADDconst [off2] ptr) val1 val2 mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (STP [off1+int32(off2)] {sym} ptr val1 val2 mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val1 := v_1
		val2 := v_2
		mem := v_3
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64STP)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg4(ptr, val1, val2, mem)
		return true
	}
	// match: (STP [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val1 val2 mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (STP [off1+off2] {mergeSym(sym1,sym2)} ptr val1 val2 mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym1 := auxToSym(v.Aux)
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := auxIntToInt32(v_0.AuxInt)
		sym2 := auxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val1 := v_1
		val2 := v_2
		mem := v_3
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64STP)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg4(ptr, val1, val2, mem)
		return true
	}
	// match: (STP [off] {sym} ptr (MOVDconst [0]) (MOVDconst [0]) mem)
	// result: (MOVQstorezero [off] {sym} ptr mem)
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 || v_2.Op != OpARM64MOVDconst || auxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		mem := v_3
		v.reset(OpARM64MOVQstorezero)
		v.AuxInt = int32ToAuxInt(off)
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SUB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUB x (MOVDconst [c]))
	// result: (SUBconst [c] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUB a l:(MUL x y))
	// cond: l.Uses==1 && clobber(l)
	// result: (MSUB a x y)
	for {
		a := v_0
		l := v_1
		if l.Op != OpARM64MUL {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MSUB)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (SUB a l:(MNEG x y))
	// cond: l.Uses==1 && clobber(l)
	// result: (MADD a x y)
	for {
		a := v_0
		l := v_1
		if l.Op != OpARM64MNEG {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MADD)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (SUB a l:(MULW x y))
	// cond: a.Type.Size() != 8 && l.Uses==1 && clobber(l)
	// result: (MSUBW a x y)
	for {
		a := v_0
		l := v_1
		if l.Op != OpARM64MULW {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(a.Type.Size() != 8 && l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MSUBW)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (SUB a l:(MNEGW x y))
	// cond: a.Type.Size() != 8 && l.Uses==1 && clobber(l)
	// result: (MADDW a x y)
	for {
		a := v_0
		l := v_1
		if l.Op != OpARM64MNEGW {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(a.Type.Size() != 8 && l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MADDW)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (SUB x x)
	// result: (MOVDconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SUB x (SUB y z))
	// result: (SUB (ADD <v.Type> x z) y)
	for {
		x := v_0
		if v_1.Op != OpARM64SUB {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, v.Type)
		v0.AddArg2(x, z)
		v.AddArg2(v0, y)
		return true
	}
	// match: (SUB (SUB x y) z)
	// result: (SUB x (ADD <y.Type> y z))
	for {
		if v_0.Op != OpARM64SUB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, y.Type)
		v0.AddArg2(y, z)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SUB x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (SUBshiftLL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (SUB x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (SUBshiftRL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64SUBshiftRL)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (SUB x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (SUBshiftRA x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64SUBshiftRA)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SUBconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SUBconst [0] x)
	// result: x
	for {
		if auxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.copyOf(x)
		return true
	}
	// match: (SUBconst [c] (MOVDconst [d]))
	// result: (MOVDconst [d-c])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(d - c)
		return true
	}
	// match: (SUBconst [c] (SUBconst [d] x))
	// result: (ADDconst [-c-d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SUBconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(-c - d)
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] (ADDconst [d] x))
	// result: (ADDconst [-c+d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(-c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SUBshiftLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBshiftLL x (MOVDconst [c]) [d])
	// result: (SUBconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SUBshiftRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBshiftRA x (MOVDconst [c]) [d])
	// result: (SUBconst x [c>>uint64(d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SUBshiftRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBshiftRL x (MOVDconst [c]) [d])
	// result: (SUBconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64TST(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (TST x (MOVDconst [c]))
	// result: (TSTconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64TSTconst)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (TST x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (TSTshiftLL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SLLconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64TSTshiftLL)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (TST x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (TSTshiftRL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SRLconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64TSTshiftRL)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (TST x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (TSTshiftRA x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SRAconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64TSTshiftRA)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64TSTW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (TSTW x (MOVDconst [c]))
	// result: (TSTWconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64TSTWconst)
			v.AuxInt = int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64TSTWconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TSTWconst (MOVDconst [x]) [y])
	// result: (FlagConstant [logicFlags32(int32(x)&y)])
	for {
		y := auxIntToInt32(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(logicFlags32(int32(x) & y))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64TSTconst(v *Value) bool {
	v_0 := v.Args[0]
	// match: (TSTconst (MOVDconst [x]) [y])
	// result: (FlagConstant [logicFlags64(x&y)])
	for {
		y := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64FlagConstant)
		v.AuxInt = flagConstantToAuxInt(logicFlags64(x & y))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64TSTshiftLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftLL (MOVDconst [c]) x [d])
	// result: (TSTconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64TSTconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftLL x (MOVDconst [c]) [d])
	// result: (TSTconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64TSTconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64TSTshiftRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftRA (MOVDconst [c]) x [d])
	// result: (TSTconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64TSTconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRA x (MOVDconst [c]) [d])
	// result: (TSTconst x [c>>uint64(d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64TSTconst)
		v.AuxInt = int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64TSTshiftRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftRL (MOVDconst [c]) x [d])
	// result: (TSTconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64TSTconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRL x (MOVDconst [c]) [d])
	// result: (TSTconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64TSTconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UBFIZ(v *Value) bool {
	v_0 := v.Args[0]
	// match: (UBFIZ [bfc] (SLLconst [sc] x))
	// cond: sc < bfc.getARM64BFwidth()
	// result: (UBFIZ [armBFAuxInt(bfc.getARM64BFlsb()+sc, bfc.getARM64BFwidth()-sc)] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.getARM64BFwidth()) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.getARM64BFlsb()+sc, bfc.getARM64BFwidth()-sc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UBFX(v *Value) bool {
	v_0 := v.Args[0]
	// match: (UBFX [bfc] (SRLconst [sc] x))
	// cond: sc+bfc.getARM64BFwidth()+bfc.getARM64BFlsb() < 64
	// result: (UBFX [armBFAuxInt(bfc.getARM64BFlsb()+sc, bfc.getARM64BFwidth())] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		if v_0.Op != OpARM64SRLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc+bfc.getARM64BFwidth()+bfc.getARM64BFlsb() < 64) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.getARM64BFlsb()+sc, bfc.getARM64BFwidth()))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SLLconst [sc] x))
	// cond: sc == bfc.getARM64BFlsb()
	// result: (ANDconst [1<<uint(bfc.getARM64BFwidth())-1] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc == bfc.getARM64BFlsb()) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(1<<uint(bfc.getARM64BFwidth()) - 1)
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SLLconst [sc] x))
	// cond: sc < bfc.getARM64BFlsb()
	// result: (UBFX [armBFAuxInt(bfc.getARM64BFlsb()-sc, bfc.getARM64BFwidth())] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.getARM64BFlsb()) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.getARM64BFlsb()-sc, bfc.getARM64BFwidth()))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SLLconst [sc] x))
	// cond: sc > bfc.getARM64BFlsb() && sc < bfc.getARM64BFlsb()+bfc.getARM64BFwidth()
	// result: (UBFIZ [armBFAuxInt(sc-bfc.getARM64BFlsb(), bfc.getARM64BFlsb()+bfc.getARM64BFwidth()-sc)] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc > bfc.getARM64BFlsb() && sc < bfc.getARM64BFlsb()+bfc.getARM64BFwidth()) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc-bfc.getARM64BFlsb(), bfc.getARM64BFlsb()+bfc.getARM64BFwidth()-sc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UDIV(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (UDIV x (MOVDconst [1]))
	// result: x
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (UDIV x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c)
	// result: (SRLconst [log64(c)] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo64(c)) {
			break
		}
		v.reset(OpARM64SRLconst)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg(x)
		return true
	}
	// match: (UDIV (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [int64(uint64(c)/uint64(d))])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) / uint64(d)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UDIVW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (UDIVW x (MOVDconst [c]))
	// cond: uint32(c)==1
	// result: x
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint32(c) == 1) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (UDIVW x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c) && is32Bit(c)
	// result: (SRLconst [log64(c)] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo64(c) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SRLconst)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg(x)
		return true
	}
	// match: (UDIVW (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [int64(uint32(c)/uint32(d))])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint32(c) / uint32(d)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UMOD(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (UMOD <typ.UInt64> x y)
	// result: (MSUB <typ.UInt64> x y (UDIV <typ.UInt64> x y))
	for {
		if v.Type != typ.UInt64 {
			break
		}
		x := v_0
		y := v_1
		v.reset(OpARM64MSUB)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpARM64UDIV, typ.UInt64)
		v0.AddArg2(x, y)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (UMOD _ (MOVDconst [1]))
	// result: (MOVDconst [0])
	for {
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (UMOD x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c)
	// result: (ANDconst [c-1] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo64(c)) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c - 1)
		v.AddArg(x)
		return true
	}
	// match: (UMOD (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [int64(uint64(c)%uint64(d))])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) % uint64(d)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UMODW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (UMODW <typ.UInt32> x y)
	// result: (MSUBW <typ.UInt32> x y (UDIVW <typ.UInt32> x y))
	for {
		if v.Type != typ.UInt32 {
			break
		}
		x := v_0
		y := v_1
		v.reset(OpARM64MSUBW)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, OpARM64UDIVW, typ.UInt32)
		v0.AddArg2(x, y)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (UMODW _ (MOVDconst [c]))
	// cond: uint32(c)==1
	// result: (MOVDconst [0])
	for {
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint32(c) == 1) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (UMODW x (MOVDconst [c]))
	// cond: isPowerOfTwo64(c) && is32Bit(c)
	// result: (ANDconst [c-1] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo64(c) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c - 1)
		v.AddArg(x)
		return true
	}
	// match: (UMODW (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [int64(uint32(c)%uint32(d))])
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(uint32(c) % uint32(d)))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64XOR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (XOR x (MOVDconst [c]))
	// result: (XORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64XORconst)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XOR x x)
	// result: (MOVDconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (XOR x (MVN y))
	// result: (EON x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MVN {
				continue
			}
			y := v_1.Args[0]
			v.reset(OpARM64EON)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (XOR x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (XORshiftLL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SLLconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64XORshiftLL)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (XOR x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (XORshiftRL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SRLconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64XORshiftRL)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (XOR x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (XORshiftRA x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64SRAconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64XORshiftRA)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (XOR (SLL x (ANDconst <t> [63] y)) (CSEL0 <typ.UInt64> [cc] (SRL <typ.UInt64> x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))))
	// cond: cc == OpARM64LessThanU
	// result: (ROR x (NEG <t> y))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64SLL {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpARM64ANDconst {
				continue
			}
			t := v_0_1.Type
			if auxIntToInt64(v_0_1.AuxInt) != 63 {
				continue
			}
			y := v_0_1.Args[0]
			if v_1.Op != OpARM64CSEL0 || v_1.Type != typ.UInt64 {
				continue
			}
			cc := auxIntToOp(v_1.AuxInt)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64SRL || v_1_0.Type != typ.UInt64 {
				continue
			}
			_ = v_1_0.Args[1]
			if x != v_1_0.Args[0] {
				continue
			}
			v_1_0_1 := v_1_0.Args[1]
			if v_1_0_1.Op != OpARM64SUB || v_1_0_1.Type != t {
				continue
			}
			_ = v_1_0_1.Args[1]
			v_1_0_1_0 := v_1_0_1.Args[0]
			if v_1_0_1_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_0_1_0.AuxInt) != 64 {
				continue
			}
			v_1_0_1_1 := v_1_0_1.Args[1]
			if v_1_0_1_1.Op != OpARM64ANDconst || v_1_0_1_1.Type != t || auxIntToInt64(v_1_0_1_1.AuxInt) != 63 || y != v_1_0_1_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpARM64CMPconst || auxIntToInt64(v_1_1.AuxInt) != 64 {
				continue
			}
			v_1_1_0 := v_1_1.Args[0]
			if v_1_1_0.Op != OpARM64SUB || v_1_1_0.Type != t {
				continue
			}
			_ = v_1_1_0.Args[1]
			v_1_1_0_0 := v_1_1_0.Args[0]
			if v_1_1_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_1_0_0.AuxInt) != 64 {
				continue
			}
			v_1_1_0_1 := v_1_1_0.Args[1]
			if v_1_1_0_1.Op != OpARM64ANDconst || v_1_1_0_1.Type != t || auxIntToInt64(v_1_1_0_1.AuxInt) != 63 || y != v_1_1_0_1.Args[0] || !(cc == OpARM64LessThanU) {
				continue
			}
			v.reset(OpARM64ROR)
			v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (XOR (SRL <typ.UInt64> x (ANDconst <t> [63] y)) (CSEL0 <typ.UInt64> [cc] (SLL x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))))
	// cond: cc == OpARM64LessThanU
	// result: (ROR x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64SRL || v_0.Type != typ.UInt64 {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpARM64ANDconst {
				continue
			}
			t := v_0_1.Type
			if auxIntToInt64(v_0_1.AuxInt) != 63 {
				continue
			}
			y := v_0_1.Args[0]
			if v_1.Op != OpARM64CSEL0 || v_1.Type != typ.UInt64 {
				continue
			}
			cc := auxIntToOp(v_1.AuxInt)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64SLL {
				continue
			}
			_ = v_1_0.Args[1]
			if x != v_1_0.Args[0] {
				continue
			}
			v_1_0_1 := v_1_0.Args[1]
			if v_1_0_1.Op != OpARM64SUB || v_1_0_1.Type != t {
				continue
			}
			_ = v_1_0_1.Args[1]
			v_1_0_1_0 := v_1_0_1.Args[0]
			if v_1_0_1_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_0_1_0.AuxInt) != 64 {
				continue
			}
			v_1_0_1_1 := v_1_0_1.Args[1]
			if v_1_0_1_1.Op != OpARM64ANDconst || v_1_0_1_1.Type != t || auxIntToInt64(v_1_0_1_1.AuxInt) != 63 || y != v_1_0_1_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpARM64CMPconst || auxIntToInt64(v_1_1.AuxInt) != 64 {
				continue
			}
			v_1_1_0 := v_1_1.Args[0]
			if v_1_1_0.Op != OpARM64SUB || v_1_1_0.Type != t {
				continue
			}
			_ = v_1_1_0.Args[1]
			v_1_1_0_0 := v_1_1_0.Args[0]
			if v_1_1_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_1_0_0.AuxInt) != 64 {
				continue
			}
			v_1_1_0_1 := v_1_1_0.Args[1]
			if v_1_1_0_1.Op != OpARM64ANDconst || v_1_1_0_1.Type != t || auxIntToInt64(v_1_1_0_1.AuxInt) != 63 || y != v_1_1_0_1.Args[0] || !(cc == OpARM64LessThanU) {
				continue
			}
			v.reset(OpARM64ROR)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (XOR (SLL x (ANDconst <t> [31] y)) (CSEL0 <typ.UInt32> [cc] (SRL <typ.UInt32> (MOVWUreg x) (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))))
	// cond: cc == OpARM64LessThanU
	// result: (RORW x (NEG <t> y))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64SLL {
				continue
			}
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpARM64ANDconst {
				continue
			}
			t := v_0_1.Type
			if auxIntToInt64(v_0_1.AuxInt) != 31 {
				continue
			}
			y := v_0_1.Args[0]
			if v_1.Op != OpARM64CSEL0 || v_1.Type != typ.UInt32 {
				continue
			}
			cc := auxIntToOp(v_1.AuxInt)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64SRL || v_1_0.Type != typ.UInt32 {
				continue
			}
			_ = v_1_0.Args[1]
			v_1_0_0 := v_1_0.Args[0]
			if v_1_0_0.Op != OpARM64MOVWUreg || x != v_1_0_0.Args[0] {
				continue
			}
			v_1_0_1 := v_1_0.Args[1]
			if v_1_0_1.Op != OpARM64SUB || v_1_0_1.Type != t {
				continue
			}
			_ = v_1_0_1.Args[1]
			v_1_0_1_0 := v_1_0_1.Args[0]
			if v_1_0_1_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_0_1_0.AuxInt) != 32 {
				continue
			}
			v_1_0_1_1 := v_1_0_1.Args[1]
			if v_1_0_1_1.Op != OpARM64ANDconst || v_1_0_1_1.Type != t || auxIntToInt64(v_1_0_1_1.AuxInt) != 31 || y != v_1_0_1_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpARM64CMPconst || auxIntToInt64(v_1_1.AuxInt) != 64 {
				continue
			}
			v_1_1_0 := v_1_1.Args[0]
			if v_1_1_0.Op != OpARM64SUB || v_1_1_0.Type != t {
				continue
			}
			_ = v_1_1_0.Args[1]
			v_1_1_0_0 := v_1_1_0.Args[0]
			if v_1_1_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_1_0_0.AuxInt) != 32 {
				continue
			}
			v_1_1_0_1 := v_1_1_0.Args[1]
			if v_1_1_0_1.Op != OpARM64ANDconst || v_1_1_0_1.Type != t || auxIntToInt64(v_1_1_0_1.AuxInt) != 31 || y != v_1_1_0_1.Args[0] || !(cc == OpARM64LessThanU) {
				continue
			}
			v.reset(OpARM64RORW)
			v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
			v0.AddArg(y)
			v.AddArg2(x, v0)
			return true
		}
		break
	}
	// match: (XOR (SRL <typ.UInt32> (MOVWUreg x) (ANDconst <t> [31] y)) (CSEL0 <typ.UInt32> [cc] (SLL x (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))))
	// cond: cc == OpARM64LessThanU
	// result: (RORW x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != OpARM64SRL || v_0.Type != typ.UInt32 {
				continue
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpARM64MOVWUreg {
				continue
			}
			x := v_0_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != OpARM64ANDconst {
				continue
			}
			t := v_0_1.Type
			if auxIntToInt64(v_0_1.AuxInt) != 31 {
				continue
			}
			y := v_0_1.Args[0]
			if v_1.Op != OpARM64CSEL0 || v_1.Type != typ.UInt32 {
				continue
			}
			cc := auxIntToOp(v_1.AuxInt)
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpARM64SLL {
				continue
			}
			_ = v_1_0.Args[1]
			if x != v_1_0.Args[0] {
				continue
			}
			v_1_0_1 := v_1_0.Args[1]
			if v_1_0_1.Op != OpARM64SUB || v_1_0_1.Type != t {
				continue
			}
			_ = v_1_0_1.Args[1]
			v_1_0_1_0 := v_1_0_1.Args[0]
			if v_1_0_1_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_0_1_0.AuxInt) != 32 {
				continue
			}
			v_1_0_1_1 := v_1_0_1.Args[1]
			if v_1_0_1_1.Op != OpARM64ANDconst || v_1_0_1_1.Type != t || auxIntToInt64(v_1_0_1_1.AuxInt) != 31 || y != v_1_0_1_1.Args[0] {
				continue
			}
			v_1_1 := v_1.Args[1]
			if v_1_1.Op != OpARM64CMPconst || auxIntToInt64(v_1_1.AuxInt) != 64 {
				continue
			}
			v_1_1_0 := v_1_1.Args[0]
			if v_1_1_0.Op != OpARM64SUB || v_1_1_0.Type != t {
				continue
			}
			_ = v_1_1_0.Args[1]
			v_1_1_0_0 := v_1_1_0.Args[0]
			if v_1_1_0_0.Op != OpARM64MOVDconst || auxIntToInt64(v_1_1_0_0.AuxInt) != 32 {
				continue
			}
			v_1_1_0_1 := v_1_1_0.Args[1]
			if v_1_1_0_1.Op != OpARM64ANDconst || v_1_1_0_1.Type != t || auxIntToInt64(v_1_1_0_1.AuxInt) != 31 || y != v_1_1_0_1.Args[0] || !(cc == OpARM64LessThanU) {
				continue
			}
			v.reset(OpARM64RORW)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64XORconst(v *Value) bool {
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
	// result: (MVN x)
	for {
		if auxIntToInt64(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.reset(OpARM64MVN)
		v.AddArg(x)
		return true
	}
	// match: (XORconst [c] (MOVDconst [d]))
	// result: (MOVDconst [c^d])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(c ^ d)
		return true
	}
	// match: (XORconst [c] (XORconst [d] x))
	// result: (XORconst [c^d] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64XORconst {
			break
		}
		d := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(c ^ d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64XORshiftLL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (XORshiftLL (MOVDconst [c]) x [d])
	// result: (XORconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftLL x (MOVDconst [c]) [d])
	// result: (XORconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (XORshiftLL [c] (SRLconst x [64-c]) x)
	// result: (RORconst [64-c] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != 64-c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = int64ToAuxInt(64 - c)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL <t> [c] (UBFX [bfc] x) x)
	// cond: c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)
	// result: (RORWconst [32-c] x)
	for {
		t := v.Type
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if x != v_1 || !(c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = int64ToAuxInt(32 - c)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL <typ.UInt16> [8] (UBFX <typ.UInt16> [armBFAuxInt(8, 8)] x) x)
	// result: (REV16W x)
	for {
		if v.Type != typ.UInt16 || auxIntToInt64(v.AuxInt) != 8 || v_0.Op != OpARM64UBFX || v_0.Type != typ.UInt16 || auxIntToArm64BitField(v_0.AuxInt) != armBFAuxInt(8, 8) {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL [c] (SRLconst x [64-c]) x2)
	// result: (EXTRconst [64-c] x2 x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != 64-c {
			break
		}
		x := v_0.Args[0]
		x2 := v_1
		v.reset(OpARM64EXTRconst)
		v.AuxInt = int64ToAuxInt(64 - c)
		v.AddArg2(x2, x)
		return true
	}
	// match: (XORshiftLL <t> [c] (UBFX [bfc] x) x2)
	// cond: c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)
	// result: (EXTRWconst [32-c] x2 x)
	for {
		t := v.Type
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		x2 := v_1
		if !(c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)) {
			break
		}
		v.reset(OpARM64EXTRWconst)
		v.AuxInt = int64ToAuxInt(32 - c)
		v.AddArg2(x2, x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64XORshiftRA(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XORshiftRA (MOVDconst [c]) x [d])
	// result: (XORconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRA x (MOVDconst [c]) [d])
	// result: (XORconst x [c>>uint64(d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64XORshiftRL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XORshiftRL (MOVDconst [c]) x [d])
	// result: (XORconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRL x (MOVDconst [c]) [d])
	// result: (XORconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (XORshiftRL [c] (SLLconst x [64-c]) x)
	// result: (RORconst [ c] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 64-c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRL <t> [c] (SLLconst x [32-c]) (MOVWUreg x))
	// cond: c < 32 && t.Size() == 4
	// result: (RORWconst [c] x)
	for {
		t := v.Type
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 32-c {
			break
		}
		x := v_0.Args[0]
		if v_1.Op != OpARM64MOVWUreg || x != v_1.Args[0] || !(c < 32 && t.Size() == 4) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (MOVDaddr {sym} base)
	for {
		sym := auxToSym(v.Aux)
		base := v_0
		v.reset(OpARM64MOVDaddr)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValueARM64_OpAtomicAnd32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicAnd32 ptr val mem)
	// result: (Select1 (LoweredAtomicAnd32 ptr val mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpARM64LoweredAtomicAnd32, types.NewTuple(typ.UInt32, types.TypeMem))
		v0.AddArg3(ptr, val, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpAtomicAnd32Variant(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicAnd32Variant ptr val mem)
	// result: (Select1 (LoweredAtomicAnd32Variant ptr val mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpARM64LoweredAtomicAnd32Variant, types.NewTuple(typ.UInt32, types.TypeMem))
		v0.AddArg3(ptr, val, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpAtomicAnd8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicAnd8 ptr val mem)
	// result: (Select1 (LoweredAtomicAnd8 ptr val mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpARM64LoweredAtomicAnd8, types.NewTuple(typ.UInt8, types.TypeMem))
		v0.AddArg3(ptr, val, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpAtomicAnd8Variant(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicAnd8Variant ptr val mem)
	// result: (Select1 (LoweredAtomicAnd8Variant ptr val mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpARM64LoweredAtomicAnd8Variant, types.NewTuple(typ.UInt8, types.TypeMem))
		v0.AddArg3(ptr, val, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpAtomicOr32(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicOr32 ptr val mem)
	// result: (Select1 (LoweredAtomicOr32 ptr val mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpARM64LoweredAtomicOr32, types.NewTuple(typ.UInt32, types.TypeMem))
		v0.AddArg3(ptr, val, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpAtomicOr32Variant(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicOr32Variant ptr val mem)
	// result: (Select1 (LoweredAtomicOr32Variant ptr val mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpARM64LoweredAtomicOr32Variant, types.NewTuple(typ.UInt32, types.TypeMem))
		v0.AddArg3(ptr, val, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpAtomicOr8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicOr8 ptr val mem)
	// result: (Select1 (LoweredAtomicOr8 ptr val mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpARM64LoweredAtomicOr8, types.NewTuple(typ.UInt8, types.TypeMem))
		v0.AddArg3(ptr, val, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpAtomicOr8Variant(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicOr8Variant ptr val mem)
	// result: (Select1 (LoweredAtomicOr8Variant ptr val mem))
	for {
		ptr := v_0
		val := v_1
		mem := v_2
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpARM64LoweredAtomicOr8Variant, types.NewTuple(typ.UInt8, types.TypeMem))
		v0.AddArg3(ptr, val, mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpAvg64u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Avg64u <t> x y)
	// result: (ADD (SRLconst <t> (SUB <t> x y) [1]) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64ADD)
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, t)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpARM64SUB, t)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValueARM64_OpBitLen32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen32 x)
	// result: (SUB (MOVDconst [32]) (CLZW <typ.Int> x))
	for {
		x := v_0
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(32)
		v1 := b.NewValue0(v.Pos, OpARM64CLZW, typ.Int)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpBitLen64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen64 x)
	// result: (SUB (MOVDconst [64]) (CLZ <typ.Int> x))
	for {
		x := v_0
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(64)
		v1 := b.NewValue0(v.Pos, OpARM64CLZ, typ.Int)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpBitRev16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitRev16 x)
	// result: (SRLconst [48] (RBIT <typ.UInt64> x))
	for {
		x := v_0
		v.reset(OpARM64SRLconst)
		v.AuxInt = int64ToAuxInt(48)
		v0 := b.NewValue0(v.Pos, OpARM64RBIT, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpBitRev8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitRev8 x)
	// result: (SRLconst [56] (RBIT <typ.UInt64> x))
	for {
		x := v_0
		v.reset(OpARM64SRLconst)
		v.AuxInt = int64ToAuxInt(56)
		v0 := b.NewValue0(v.Pos, OpARM64RBIT, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpCondSelect(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CondSelect x y boolval)
	// cond: flagArg(boolval) != nil
	// result: (CSEL [boolval.Op] x y flagArg(boolval))
	for {
		x := v_0
		y := v_1
		boolval := v_2
		if !(flagArg(boolval) != nil) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(boolval.Op)
		v.AddArg3(x, y, flagArg(boolval))
		return true
	}
	// match: (CondSelect x y boolval)
	// cond: flagArg(boolval) == nil
	// result: (CSEL [OpARM64NotEqual] x y (CMPWconst [0] boolval))
	for {
		x := v_0
		y := v_1
		boolval := v_2
		if !(flagArg(boolval) == nil) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(0)
		v0.AddArg(boolval)
		v.AddArg3(x, y, v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpConst16(v *Value) bool {
	// match: (Const16 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := auxIntToInt16(v.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueARM64_OpConst32(v *Value) bool {
	// match: (Const32 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := auxIntToInt32(v.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueARM64_OpConst32F(v *Value) bool {
	// match: (Const32F [val])
	// result: (FMOVSconst [float64(val)])
	for {
		val := auxIntToFloat32(v.AuxInt)
		v.reset(OpARM64FMOVSconst)
		v.AuxInt = float64ToAuxInt(float64(val))
		return true
	}
}
func rewriteValueARM64_OpConst64(v *Value) bool {
	// match: (Const64 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := auxIntToInt64(v.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueARM64_OpConst64F(v *Value) bool {
	// match: (Const64F [val])
	// result: (FMOVDconst [float64(val)])
	for {
		val := auxIntToFloat64(v.AuxInt)
		v.reset(OpARM64FMOVDconst)
		v.AuxInt = float64ToAuxInt(float64(val))
		return true
	}
}
func rewriteValueARM64_OpConst8(v *Value) bool {
	// match: (Const8 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := auxIntToInt8(v.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValueARM64_OpConstBool(v *Value) bool {
	// match: (ConstBool [b])
	// result: (MOVDconst [b2i(b)])
	for {
		b := auxIntToBool(v.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(b))
		return true
	}
}
func rewriteValueARM64_OpConstNil(v *Value) bool {
	// match: (ConstNil)
	// result: (MOVDconst [0])
	for {
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
}
func rewriteValueARM64_OpCtz16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 <t> x)
	// result: (CLZW <t> (RBITW <typ.UInt32> (ORconst <typ.UInt32> [0x10000] x)))
	for {
		t := v.Type
		x := v_0
		v.reset(OpARM64CLZW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARM64RBITW, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpARM64ORconst, typ.UInt32)
		v1.AuxInt = int64ToAuxInt(0x10000)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpCtz32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Ctz32 <t> x)
	// result: (CLZW (RBITW <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpARM64CLZW)
		v0 := b.NewValue0(v.Pos, OpARM64RBITW, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpCtz64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Ctz64 <t> x)
	// result: (CLZ (RBIT <t> x))
	for {
		t := v.Type
		x := v_0
		v.reset(OpARM64CLZ)
		v0 := b.NewValue0(v.Pos, OpARM64RBIT, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpCtz8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 <t> x)
	// result: (CLZW <t> (RBITW <typ.UInt32> (ORconst <typ.UInt32> [0x100] x)))
	for {
		t := v.Type
		x := v_0
		v.reset(OpARM64CLZW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARM64RBITW, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpARM64ORconst, typ.UInt32)
		v1.AuxInt = int64ToAuxInt(0x100)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpDiv16(v *Value) bool {
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
		v.reset(OpARM64DIVW)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpDiv16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (UDIVW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64UDIVW)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpDiv32(v *Value) bool {
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
		v.reset(OpARM64DIVW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpDiv64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64 [false] x y)
	// result: (DIV x y)
	for {
		if auxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.reset(OpARM64DIV)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpDiv8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (DIVW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64DIVW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpDiv8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (UDIVW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64UDIVW)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpEq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// result: (Equal (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpEq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32 x y)
	// result: (Equal (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpEq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32F x y)
	// result: (Equal (FCMPS x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpEq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64 x y)
	// result: (Equal (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpEq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64F x y)
	// result: (Equal (FCMPD x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpEq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// result: (Equal (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpEqB(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// result: (XOR (MOVDconst [1]) (XOR <typ.Bool> x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64XOR)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, OpARM64XOR, typ.Bool)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpEqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (EqPtr x y)
	// result: (Equal (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpFMA(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMA x y z)
	// result: (FMADDD z x y)
	for {
		x := v_0
		y := v_1
		z := v_2
		v.reset(OpARM64FMADDD)
		v.AddArg3(z, x, y)
		return true
	}
}
func rewriteValueARM64_OpHmul32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32 x y)
	// result: (SRAconst (MULL <typ.Int64> x y) [32])
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRAconst)
		v.AuxInt = int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, OpARM64MULL, typ.Int64)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpHmul32u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32u x y)
	// result: (SRAconst (UMULL <typ.UInt64> x y) [32])
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRAconst)
		v.AuxInt = int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, OpARM64UMULL, typ.UInt64)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpIsInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsInBounds idx len)
	// result: (LessThanU (CMP idx len))
	for {
		idx := v_0
		len := v_1
		v.reset(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpIsNonNil(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsNonNil ptr)
	// result: (NotEqual (CMPconst [0] ptr))
	for {
		ptr := v_0
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(0)
		v0.AddArg(ptr)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpIsSliceInBounds(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsSliceInBounds idx len)
	// result: (LessEqualU (CMP idx len))
	for {
		idx := v_0
		len := v_1
		v.reset(OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (LessEqual (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x zero:(MOVDconst [0]))
	// result: (Eq16 x zero)
	for {
		x := v_0
		zero := v_1
		if zero.Op != OpARM64MOVDconst || auxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		v.reset(OpEq16)
		v.AddArg2(x, zero)
		return true
	}
	// match: (Leq16U (MOVDconst [1]) x)
	// result: (Neq16 (MOVDconst [0]) x)
	for {
		if v_0.Op != OpARM64MOVDconst || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.reset(OpNeq16)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq16U x y)
	// result: (LessEqualU (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32 x y)
	// result: (LessEqual (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32F x y)
	// result: (LessEqualF (FCMPS x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessEqualF)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32U x zero:(MOVDconst [0]))
	// result: (Eq32 x zero)
	for {
		x := v_0
		zero := v_1
		if zero.Op != OpARM64MOVDconst || auxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		v.reset(OpEq32)
		v.AddArg2(x, zero)
		return true
	}
	// match: (Leq32U (MOVDconst [1]) x)
	// result: (Neq32 (MOVDconst [0]) x)
	for {
		if v_0.Op != OpARM64MOVDconst || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.reset(OpNeq32)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq32U x y)
	// result: (LessEqualU (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64 x y)
	// result: (LessEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64F x y)
	// result: (LessEqualF (FCMPD x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessEqualF)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64U x zero:(MOVDconst [0]))
	// result: (Eq64 x zero)
	for {
		x := v_0
		zero := v_1
		if zero.Op != OpARM64MOVDconst || auxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		v.reset(OpEq64)
		v.AddArg2(x, zero)
		return true
	}
	// match: (Leq64U (MOVDconst [1]) x)
	// result: (Neq64 (MOVDconst [0]) x)
	for {
		if v_0.Op != OpARM64MOVDconst || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.reset(OpNeq64)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq64U x y)
	// result: (LessEqualU (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (LessEqual (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x zero:(MOVDconst [0]))
	// result: (Eq8 x zero)
	for {
		x := v_0
		zero := v_1
		if zero.Op != OpARM64MOVDconst || auxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		v.reset(OpEq8)
		v.AddArg2(x, zero)
		return true
	}
	// match: (Leq8U (MOVDconst [1]) x)
	// result: (Neq8 (MOVDconst [0]) x)
	for {
		if v_0.Op != OpARM64MOVDconst || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.reset(OpNeq8)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq8U x y)
	// result: (LessEqualU (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (LessThan (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess16U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U zero:(MOVDconst [0]) x)
	// result: (Neq16 zero x)
	for {
		zero := v_0
		if zero.Op != OpARM64MOVDconst || auxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		x := v_1
		v.reset(OpNeq16)
		v.AddArg2(zero, x)
		return true
	}
	// match: (Less16U x (MOVDconst [1]))
	// result: (Eq16 x (MOVDconst [0]))
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpEq16)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less16U x y)
	// result: (LessThanU (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32 x y)
	// result: (LessThan (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32F x y)
	// result: (LessThanF (FCMPS x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessThanF)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess32U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32U zero:(MOVDconst [0]) x)
	// result: (Neq32 zero x)
	for {
		zero := v_0
		if zero.Op != OpARM64MOVDconst || auxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		x := v_1
		v.reset(OpNeq32)
		v.AddArg2(zero, x)
		return true
	}
	// match: (Less32U x (MOVDconst [1]))
	// result: (Eq32 x (MOVDconst [0]))
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpEq32)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less32U x y)
	// result: (LessThanU (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64 x y)
	// result: (LessThan (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64F x y)
	// result: (LessThanF (FCMPD x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessThanF)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess64U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less64U zero:(MOVDconst [0]) x)
	// result: (Neq64 zero x)
	for {
		zero := v_0
		if zero.Op != OpARM64MOVDconst || auxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		x := v_1
		v.reset(OpNeq64)
		v.AddArg2(zero, x)
		return true
	}
	// match: (Less64U x (MOVDconst [1]))
	// result: (Eq64 x (MOVDconst [0]))
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpEq64)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less64U x y)
	// result: (LessThanU (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (LessThan (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess8U(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U zero:(MOVDconst [0]) x)
	// result: (Neq8 zero x)
	for {
		zero := v_0
		if zero.Op != OpARM64MOVDconst || auxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		x := v_1
		v.reset(OpNeq8)
		v.AddArg2(zero, x)
		return true
	}
	// match: (Less8U x (MOVDconst [1]))
	// result: (Eq8 x (MOVDconst [0]))
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.reset(OpEq8)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less8U x y)
	// result: (LessThanU (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLoad(v *Value) bool {
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
		v.reset(OpARM64MOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is8BitInt(t) && isSigned(t))
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is8BitInt(t) && isSigned(t)) {
			break
		}
		v.reset(OpARM64MOVBload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is8BitInt(t) && !isSigned(t))
	// result: (MOVBUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is8BitInt(t) && !isSigned(t)) {
			break
		}
		v.reset(OpARM64MOVBUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is16BitInt(t) && isSigned(t))
	// result: (MOVHload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is16BitInt(t) && isSigned(t)) {
			break
		}
		v.reset(OpARM64MOVHload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is16BitInt(t) && !isSigned(t))
	// result: (MOVHUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is16BitInt(t) && !isSigned(t)) {
			break
		}
		v.reset(OpARM64MOVHUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is32BitInt(t) && isSigned(t))
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitInt(t) && isSigned(t)) {
			break
		}
		v.reset(OpARM64MOVWload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is32BitInt(t) && !isSigned(t))
	// result: (MOVWUload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is32BitInt(t) && !isSigned(t)) {
			break
		}
		v.reset(OpARM64MOVWUload)
		v.AddArg2(ptr, mem)
		return true
	}
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
		v.reset(OpARM64MOVDload)
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
		v.reset(OpARM64FMOVSload)
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
		v.reset(OpARM64FMOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpLocalAddr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (LocalAddr {sym} base _)
	// result: (MOVDaddr {sym} base)
	for {
		sym := auxToSym(v.Aux)
		base := v_0
		v.reset(OpARM64MOVDaddr)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValueARM64_OpLsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpLsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpLsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x64 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueARM64_OpLsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpLsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpLsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpLsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x64 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueARM64_OpLsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpLsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpLsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpLsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x64 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueARM64_OpLsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpLsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpLsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpLsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x64 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueARM64_OpLsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpMod16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// result: (MODW (SignExt16to32 x) (SignExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64MODW)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpMod16u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (UMODW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64UMODW)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpMod32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod32 x y)
	// result: (MODW x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64MODW)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueARM64_OpMod64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod64 x y)
	// result: (MOD x y)
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64MOD)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValueARM64_OpMod8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (MODW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64MODW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpMod8u(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (UMODW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64UMODW)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpMove(v *Value) bool {
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
		v.reset(OpARM64MOVBstore)
		v0 := b.NewValue0(v.Pos, OpARM64MOVBUload, typ.UInt8)
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
		v.reset(OpARM64MOVHstore)
		v0 := b.NewValue0(v.Pos, OpARM64MOVHUload, typ.UInt16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
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
		v.reset(OpARM64MOVWstore)
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUload, typ.UInt32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [8] dst src mem)
	// result: (MOVDstore dst (MOVDload src mem) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64MOVDstore)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
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
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpARM64MOVBUload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpARM64MOVHUload, typ.UInt16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
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
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpARM64MOVBUload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpARM64MOVWUload, typ.UInt32)
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
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpARM64MOVHUload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpARM64MOVWUload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [7] dst src mem)
	// result: (MOVBstore [6] dst (MOVBUload [6] src mem) (MOVHstore [4] dst (MOVHUload [4] src mem) (MOVWstore dst (MOVWUload src mem) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 7 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpARM64MOVBUload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(6)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, OpARM64MOVHUload, typ.UInt16)
		v2.AuxInt = int32ToAuxInt(4)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, OpARM64MOVWUload, typ.UInt32)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [12] dst src mem)
	// result: (MOVWstore [8] dst (MOVWUload [8] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 12 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [16] dst src mem)
	// result: (MOVDstore [8] dst (MOVDload [8] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [24] dst src mem)
	// result: (MOVDstore [16] dst (MOVDload [16] src mem) (MOVDstore [8] dst (MOVDload [8] src mem) (MOVDstore dst (MOVDload src mem) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 24 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(16)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(16)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v2.AuxInt = int32ToAuxInt(8)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s%8 != 0 && s > 8
	// result: (Move [s%8] (OffPtr <dst.Type> dst [s-s%8]) (OffPtr <src.Type> src [s-s%8]) (Move [s-s%8] dst src mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s%8 != 0 && s > 8) {
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
	// cond: s > 32 && s <= 16*64 && s%16 == 8 && !config.noDuffDevice && logLargeCopy(v, s)
	// result: (MOVDstore [int32(s-8)] dst (MOVDload [int32(s-8)] src mem) (DUFFCOPY <types.TypeMem> [8*(64-(s-8)/16)] dst src mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 32 && s <= 16*64 && s%16 == 8 && !config.noDuffDevice && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(int32(s - 8))
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(int32(s - 8))
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64DUFFCOPY, types.TypeMem)
		v1.AuxInt = int64ToAuxInt(8 * (64 - (s-8)/16))
		v1.AddArg3(dst, src, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 32 && s <= 16*64 && s%16 == 0 && !config.noDuffDevice && logLargeCopy(v, s)
	// result: (DUFFCOPY [8 * (64 - s/16)] dst src mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 32 && s <= 16*64 && s%16 == 0 && !config.noDuffDevice && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpARM64DUFFCOPY)
		v.AuxInt = int64ToAuxInt(8 * (64 - s/16))
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 24 && s%8 == 0 && logLargeCopy(v, s)
	// result: (LoweredMove dst src (ADDconst <src.Type> src [s-8]) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 24 && s%8 == 0 && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpARM64LoweredMove)
		v0 := b.NewValue0(v.Pos, OpARM64ADDconst, src.Type)
		v0.AuxInt = int64ToAuxInt(s - 8)
		v0.AddArg(src)
		v.AddArg4(dst, src, v0, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpNeq16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// result: (NotEqual (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNeq32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32 x y)
	// result: (NotEqual (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNeq32F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32F x y)
	// result: (NotEqual (FCMPS x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNeq64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64 x y)
	// result: (NotEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNeq64F(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64F x y)
	// result: (NotEqual (FCMPD x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNeq8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// result: (NotEqual (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNeqPtr(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (NeqPtr x y)
	// result: (NotEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNot(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Not x)
	// result: (XOR (MOVDconst [1]) x)
	for {
		x := v_0
		v.reset(OpARM64XOR)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(1)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValueARM64_OpOffPtr(v *Value) bool {
	v_0 := v.Args[0]
	// match: (OffPtr [off] ptr:(SP))
	// cond: is32Bit(off)
	// result: (MOVDaddr [int32(off)] ptr)
	for {
		off := auxIntToInt64(v.AuxInt)
		ptr := v_0
		if ptr.Op != OpSP || !(is32Bit(off)) {
			break
		}
		v.reset(OpARM64MOVDaddr)
		v.AuxInt = int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// result: (ADDconst [off] ptr)
	for {
		off := auxIntToInt64(v.AuxInt)
		ptr := v_0
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(off)
		v.AddArg(ptr)
		return true
	}
}
func rewriteValueARM64_OpPanicBounds(v *Value) bool {
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
		v.reset(OpARM64LoweredPanicBoundsA)
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
		v.reset(OpARM64LoweredPanicBoundsB)
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
		v.reset(OpARM64LoweredPanicBoundsC)
		v.AuxInt = int64ToAuxInt(kind)
		v.AddArg3(x, y, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpPopCount16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount16 <t> x)
	// result: (FMOVDfpgp <t> (VUADDLV <typ.Float64> (VCNT <typ.Float64> (FMOVDgpfp <typ.Float64> (ZeroExt16to64 x)))))
	for {
		t := v.Type
		x := v_0
		v.reset(OpARM64FMOVDfpgp)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARM64VUADDLV, typ.Float64)
		v1 := b.NewValue0(v.Pos, OpARM64VCNT, typ.Float64)
		v2 := b.NewValue0(v.Pos, OpARM64FMOVDgpfp, typ.Float64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(x)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpPopCount32(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount32 <t> x)
	// result: (FMOVDfpgp <t> (VUADDLV <typ.Float64> (VCNT <typ.Float64> (FMOVDgpfp <typ.Float64> (ZeroExt32to64 x)))))
	for {
		t := v.Type
		x := v_0
		v.reset(OpARM64FMOVDfpgp)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARM64VUADDLV, typ.Float64)
		v1 := b.NewValue0(v.Pos, OpARM64VCNT, typ.Float64)
		v2 := b.NewValue0(v.Pos, OpARM64FMOVDgpfp, typ.Float64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(x)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpPopCount64(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount64 <t> x)
	// result: (FMOVDfpgp <t> (VUADDLV <typ.Float64> (VCNT <typ.Float64> (FMOVDgpfp <typ.Float64> x))))
	for {
		t := v.Type
		x := v_0
		v.reset(OpARM64FMOVDfpgp)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARM64VUADDLV, typ.Float64)
		v1 := b.NewValue0(v.Pos, OpARM64VCNT, typ.Float64)
		v2 := b.NewValue0(v.Pos, OpARM64FMOVDgpfp, typ.Float64)
		v2.AddArg(x)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpRotateLeft16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVDconst [c]))
	// result: (Or16 (Lsh16x64 <t> x (MOVDconst [c&15])) (Rsh16Ux64 <t> x (MOVDconst [-c&15])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpLsh16x64, t)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(c & 15)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpRsh16Ux64, t)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(-c & 15)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpRotateLeft32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RotateLeft32 x y)
	// result: (RORW x (NEG <y.Type> y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64RORW)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueARM64_OpRotateLeft64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RotateLeft64 x y)
	// result: (ROR x (NEG <y.Type> y))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64ROR)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, y.Type)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueARM64_OpRotateLeft8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVDconst [c]))
	// result: (Or8 (Lsh8x64 <t> x (MOVDconst [c&7])) (Rsh8Ux64 <t> x (MOVDconst [-c&7])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpLsh8x64, t)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(c & 7)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpRsh8Ux64, t)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = int64ToAuxInt(-c & 7)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh16Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt16to64 x) (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValueARM64_OpRsh16Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt16to64 x) (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValueARM64_OpRsh16Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt16to64 x) y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpRsh16Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt16to64 x) (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValueARM64_OpRsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 x y)
	// result: (SRA (SignExt16to64 x) (CSEL [OpARM64LessThanU] <y.Type> (ZeroExt16to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = int64ToAuxInt(63)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpRsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 x y)
	// result: (SRA (SignExt16to64 x) (CSEL [OpARM64LessThanU] <y.Type> (ZeroExt32to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = int64ToAuxInt(63)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpRsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 x y)
	// result: (SRA (SignExt16to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(y)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpRsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 x y)
	// result: (SRA (SignExt16to64 x) (CSEL [OpARM64LessThanU] <y.Type> (ZeroExt8to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64 y))))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = int64ToAuxInt(63)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpRsh32Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt32to64 x) (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValueARM64_OpRsh32Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt32to64 x) (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValueARM64_OpRsh32Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt32to64 x) y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpRsh32Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt32to64 x) (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValueARM64_OpRsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 x y)
	// result: (SRA (SignExt32to64 x) (CSEL [OpARM64LessThanU] <y.Type> (ZeroExt16to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = int64ToAuxInt(63)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpRsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 x y)
	// result: (SRA (SignExt32to64 x) (CSEL [OpARM64LessThanU] <y.Type> (ZeroExt32to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = int64ToAuxInt(63)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpRsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x64 x y)
	// result: (SRA (SignExt32to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(y)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpRsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 x y)
	// result: (SRA (SignExt32to64 x) (CSEL [OpARM64LessThanU] <y.Type> (ZeroExt8to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64 y))))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = int64ToAuxInt(63)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpRsh64Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpRsh64Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpRsh64Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux64 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueARM64_OpRsh64Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> x (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpRsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 x y)
	// result: (SRA x (CSEL [OpARM64LessThanU] <y.Type> (ZeroExt16to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v0.AuxInt = opToAuxInt(OpARM64LessThanU)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v0.AddArg3(v1, v2, v3)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueARM64_OpRsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x32 x y)
	// result: (SRA x (CSEL [OpARM64LessThanU] <y.Type> (ZeroExt32to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v0.AuxInt = opToAuxInt(OpARM64LessThanU)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v0.AddArg3(v1, v2, v3)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueARM64_OpRsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x64 x y)
	// result: (SRA x (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v0.AuxInt = opToAuxInt(OpARM64LessThanU)
		v1 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v1.AuxInt = int64ToAuxInt(63)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v2.AddArg(y)
		v0.AddArg3(y, v1, v2)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueARM64_OpRsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 x y)
	// result: (SRA x (CSEL [OpARM64LessThanU] <y.Type> (ZeroExt8to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64 y))))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v0.AuxInt = opToAuxInt(OpARM64LessThanU)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(v1)
		v0.AddArg3(v1, v2, v3)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueARM64_OpRsh8Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt8to64 x) (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValueARM64_OpRsh8Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt8to64 x) (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValueARM64_OpRsh8Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt8to64 x) y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(y)
		v.AddArg3(v0, v2, v3)
		return true
	}
}
func rewriteValueARM64_OpRsh8Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 <t> x y)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt8to64 x) (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = int64ToAuxInt(0)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v.AddArg3(v0, v3, v4)
		return true
	}
}
func rewriteValueARM64_OpRsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 x y)
	// result: (SRA (SignExt8to64 x) (CSEL [OpARM64LessThanU] <y.Type> (ZeroExt16to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = int64ToAuxInt(63)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpRsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 x y)
	// result: (SRA (SignExt8to64 x) (CSEL [OpARM64LessThanU] <y.Type> (ZeroExt32to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = int64ToAuxInt(63)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpRsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 x y)
	// result: (SRA (SignExt8to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v3.AddArg(y)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpRsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 x y)
	// result: (SRA (SignExt8to64 x) (CSEL [OpARM64LessThanU] <y.Type> (ZeroExt8to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64 y))))
	for {
		x := v_0
		y := v_1
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = int64ToAuxInt(63)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = int64ToAuxInt(64)
		v4.AddArg(v2)
		v1.AddArg3(v2, v3, v4)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM64_OpSelect0(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Add64carry x y c))
	// result: (Select0 <typ.UInt64> (ADCSflags x y (Select1 <types.TypeFlags> (ADDSconstflags [-1] c))))
	for {
		if v_0.Op != OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpARM64ADCSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v1 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v2 := b.NewValue0(v.Pos, OpARM64ADDSconstflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v2.AuxInt = int64ToAuxInt(-1)
		v2.AddArg(c)
		v1.AddArg(v2)
		v0.AddArg3(x, y, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 (Sub64borrow x y bo))
	// result: (Select0 <typ.UInt64> (SBCSflags x y (Select1 <types.TypeFlags> (NEGSflags bo))))
	for {
		if v_0.Op != OpSub64borrow {
			break
		}
		bo := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpARM64SBCSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v1 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v2 := b.NewValue0(v.Pos, OpARM64NEGSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v2.AddArg(bo)
		v1.AddArg(v2)
		v0.AddArg3(x, y, v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpSelect1(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Add64carry x y c))
	// result: (ADCzerocarry <typ.UInt64> (Select1 <types.TypeFlags> (ADCSflags x y (Select1 <types.TypeFlags> (ADDSconstflags [-1] c)))))
	for {
		if v_0.Op != OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpARM64ADCzerocarry)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64ADCSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpARM64ADDSconstflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v3.AuxInt = int64ToAuxInt(-1)
		v3.AddArg(c)
		v2.AddArg(v3)
		v1.AddArg3(x, y, v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select1 (Sub64borrow x y bo))
	// result: (NEG <typ.UInt64> (NGCzerocarry <typ.UInt64> (Select1 <types.TypeFlags> (SBCSflags x y (Select1 <types.TypeFlags> (NEGSflags bo))))))
	for {
		if v_0.Op != OpSub64borrow {
			break
		}
		bo := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpARM64NEG)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpARM64NGCzerocarry, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v2 := b.NewValue0(v.Pos, OpARM64SBCSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpARM64NEGSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v4.AddArg(bo)
		v3.AddArg(v4)
		v2.AddArg3(x, y, v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpSlicemask(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SRAconst (NEG <t> x) [63])
	for {
		t := v.Type
		x := v_0
		v.reset(OpARM64SRAconst)
		v.AuxInt = int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpStore(v *Value) bool {
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
		v.reset(OpARM64MOVBstore)
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
		v.reset(OpARM64MOVHstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && !is32BitFloat(val.Type)
	// result: (MOVWstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && !is32BitFloat(val.Type)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && !is64BitFloat(val.Type)
	// result: (MOVDstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && !is64BitFloat(val.Type)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 4 && is32BitFloat(val.Type)
	// result: (FMOVSstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 4 && is32BitFloat(val.Type)) {
			break
		}
		v.reset(OpARM64FMOVSstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && is64BitFloat(val.Type)
	// result: (FMOVDstore ptr val mem)
	for {
		t := auxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && is64BitFloat(val.Type)) {
			break
		}
		v.reset(OpARM64FMOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpZero(v *Value) bool {
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
	// result: (MOVBstore ptr (MOVDconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 1 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVBstore)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] ptr mem)
	// result: (MOVHstore ptr (MOVDconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 2 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVHstore)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [4] ptr mem)
	// result: (MOVWstore ptr (MOVDconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 4 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVWstore)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [8] ptr mem)
	// result: (MOVDstore ptr (MOVDconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 8 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVDstore)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [3] ptr mem)
	// result: (MOVBstore [2] ptr (MOVDconst [0]) (MOVHstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 3 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [5] ptr mem)
	// result: (MOVBstore [4] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 5 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [6] ptr mem)
	// result: (MOVHstore [4] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 6 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [7] ptr mem)
	// result: (MOVBstore [6] ptr (MOVDconst [0]) (MOVHstore [4] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 7 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(4)
		v2 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [9] ptr mem)
	// result: (MOVBstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 9 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [10] ptr mem)
	// result: (MOVHstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 10 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [11] ptr mem)
	// result: (MOVBstore [10] ptr (MOVDconst [0]) (MOVHstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 11 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(10)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [12] ptr mem)
	// result: (MOVWstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 12 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [13] ptr mem)
	// result: (MOVBstore [12] ptr (MOVDconst [0]) (MOVWstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 13 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(12)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [14] ptr mem)
	// result: (MOVHstore [12] ptr (MOVDconst [0]) (MOVWstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 14 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(12)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(8)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [15] ptr mem)
	// result: (MOVBstore [14] ptr (MOVDconst [0]) (MOVHstore [12] ptr (MOVDconst [0]) (MOVWstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))))
	for {
		if auxIntToInt64(v.AuxInt) != 15 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(14)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHstore, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(12)
		v2 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v2.AuxInt = int32ToAuxInt(8)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v3.AddArg3(ptr, v0, mem)
		v2.AddArg3(ptr, v0, v3)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [16] ptr mem)
	// result: (STP [0] ptr (MOVDconst [0]) (MOVDconst [0]) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64STP)
		v.AuxInt = int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v.AddArg4(ptr, v0, v0, mem)
		return true
	}
	// match: (Zero [32] ptr mem)
	// result: (STP [16] ptr (MOVDconst [0]) (MOVDconst [0]) (STP [0] ptr (MOVDconst [0]) (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 32 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64STP)
		v.AuxInt = int32ToAuxInt(16)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(0)
		v1.AddArg4(ptr, v0, v0, mem)
		v.AddArg4(ptr, v0, v0, v1)
		return true
	}
	// match: (Zero [48] ptr mem)
	// result: (STP [32] ptr (MOVDconst [0]) (MOVDconst [0]) (STP [16] ptr (MOVDconst [0]) (MOVDconst [0]) (STP [0] ptr (MOVDconst [0]) (MOVDconst [0]) mem)))
	for {
		if auxIntToInt64(v.AuxInt) != 48 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64STP)
		v.AuxInt = int32ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(16)
		v2 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v2.AuxInt = int32ToAuxInt(0)
		v2.AddArg4(ptr, v0, v0, mem)
		v1.AddArg4(ptr, v0, v0, v2)
		v.AddArg4(ptr, v0, v0, v1)
		return true
	}
	// match: (Zero [64] ptr mem)
	// result: (STP [48] ptr (MOVDconst [0]) (MOVDconst [0]) (STP [32] ptr (MOVDconst [0]) (MOVDconst [0]) (STP [16] ptr (MOVDconst [0]) (MOVDconst [0]) (STP [0] ptr (MOVDconst [0]) (MOVDconst [0]) mem))))
	for {
		if auxIntToInt64(v.AuxInt) != 64 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64STP)
		v.AuxInt = int32ToAuxInt(48)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(32)
		v2 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v2.AuxInt = int32ToAuxInt(16)
		v3 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v3.AuxInt = int32ToAuxInt(0)
		v3.AddArg4(ptr, v0, v0, mem)
		v2.AddArg4(ptr, v0, v0, v3)
		v1.AddArg4(ptr, v0, v0, v2)
		v.AddArg4(ptr, v0, v0, v1)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s%16 != 0 && s%16 <= 8 && s > 16
	// result: (Zero [8] (OffPtr <ptr.Type> ptr [s-8]) (Zero [s-s%16] ptr mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s%16 != 0 && s%16 <= 8 && s > 16) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpOffPtr, ptr.Type)
		v0.AuxInt = int64ToAuxInt(s - 8)
		v0.AddArg(ptr)
		v1 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v1.AuxInt = int64ToAuxInt(s - s%16)
		v1.AddArg2(ptr, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s%16 != 0 && s%16 > 8 && s > 16
	// result: (Zero [16] (OffPtr <ptr.Type> ptr [s-16]) (Zero [s-s%16] ptr mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s%16 != 0 && s%16 > 8 && s > 16) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = int64ToAuxInt(16)
		v0 := b.NewValue0(v.Pos, OpOffPtr, ptr.Type)
		v0.AuxInt = int64ToAuxInt(s - 16)
		v0.AddArg(ptr)
		v1 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v1.AuxInt = int64ToAuxInt(s - s%16)
		v1.AddArg2(ptr, mem)
		v.AddArg2(v0, v1)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s%16 == 0 && s > 64 && s <= 16*64 && !config.noDuffDevice
	// result: (DUFFZERO [4 * (64 - s/16)] ptr mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s%16 == 0 && s > 64 && s <= 16*64 && !config.noDuffDevice) {
			break
		}
		v.reset(OpARM64DUFFZERO)
		v.AuxInt = int64ToAuxInt(4 * (64 - s/16))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s%16 == 0 && (s > 16*64 || config.noDuffDevice)
	// result: (LoweredZero ptr (ADDconst <ptr.Type> [s-16] ptr) mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s%16 == 0 && (s > 16*64 || config.noDuffDevice)) {
			break
		}
		v.reset(OpARM64LoweredZero)
		v0 := b.NewValue0(v.Pos, OpARM64ADDconst, ptr.Type)
		v0.AuxInt = int64ToAuxInt(s - 16)
		v0.AddArg(ptr)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	return false
}
func rewriteBlockARM64(b *Block) bool {
	switch b.Kind {
	case BlockARM64EQ:
		// match: (EQ (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (EQ (TSTWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (TST x y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64TST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (TSTW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64TSTW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (EQ (TSTconst [c] y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (EQ (CMNconst [c] y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (EQ (CMNWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (CMN x y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (CMNW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (CMN x y) yes no)
		for b.Controls[0].Op == OpARM64CMP {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (CMNW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPW {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] x) yes no)
		// result: (Z x yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.resetWithControl(BlockARM64Z, x)
			return true
		}
		// match: (EQ (CMPWconst [0] x) yes no)
		// result: (ZW x yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.resetWithControl(BlockARM64ZW, x)
			return true
		}
		// match: (EQ (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (EQ (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (EQ (CMP a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMP, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (EQ (CMNW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (EQ (CMPW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (TSTconst [c] x) yes no)
		// cond: oneBit(c)
		// result: (TBZ [int64(ntz64(c))] x yes no)
		for b.Controls[0].Op == OpARM64TSTconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			x := v_0.Args[0]
			if !(oneBit(c)) {
				break
			}
			b.resetWithControl(BlockARM64TBZ, x)
			b.AuxInt = int64ToAuxInt(int64(ntz64(c)))
			return true
		}
		// match: (EQ (TSTWconst [c] x) yes no)
		// cond: oneBit(int64(uint32(c)))
		// result: (TBZ [int64(ntz64(int64(uint32(c))))] x yes no)
		for b.Controls[0].Op == OpARM64TSTWconst {
			v_0 := b.Controls[0]
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if !(oneBit(int64(uint32(c)))) {
				break
			}
			b.resetWithControl(BlockARM64TBZ, x)
			b.AuxInt = int64ToAuxInt(int64(ntz64(int64(uint32(c)))))
			return true
		}
		// match: (EQ (FlagConstant [fc]) yes no)
		// cond: fc.eq()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.eq()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (EQ (FlagConstant [fc]) yes no)
		// cond: !fc.eq()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.eq()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (InvertFlags cmp) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64EQ, cmp)
			return true
		}
	case BlockARM64FGE:
		// match: (FGE (InvertFlags cmp) yes no)
		// result: (FLE cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64FLE, cmp)
			return true
		}
	case BlockARM64FGT:
		// match: (FGT (InvertFlags cmp) yes no)
		// result: (FLT cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64FLT, cmp)
			return true
		}
	case BlockARM64FLE:
		// match: (FLE (InvertFlags cmp) yes no)
		// result: (FGE cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64FGE, cmp)
			return true
		}
	case BlockARM64FLT:
		// match: (FLT (InvertFlags cmp) yes no)
		// result: (FGT cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64FGT, cmp)
			return true
		}
	case BlockARM64GE:
		// match: (GE (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GE (TSTWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64GE, v0)
			return true
		}
		// match: (GE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (TST x y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64TST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64GE, v0)
				return true
			}
			break
		}
		// match: (GE (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (TSTW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64TSTW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64GE, v0)
				return true
			}
			break
		}
		// match: (GE (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GE (TSTconst [c] y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64GE, v0)
			return true
		}
		// match: (GE (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GEnoov (CMNconst [c] y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64GEnoov, v0)
			return true
		}
		// match: (GE (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GEnoov (CMNWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64GEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (GEnoov (CMN x y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64GEnoov, v0)
				return true
			}
			break
		}
		// match: (GE (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (GEnoov (CMNW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64GEnoov, v0)
				return true
			}
			break
		}
		// match: (GE (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (CMN x y) yes no)
		for b.Controls[0].Op == OpARM64CMP {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64GE, v0)
			return true
		}
		// match: (GE (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (CMNW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPW {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64GE, v0)
			return true
		}
		// match: (GE (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (GEnoov (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64GEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (GEnoov (CMP a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMP, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64GEnoov, v0)
			return true
		}
		// match: (GE (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (GEnoov (CMNW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64GEnoov, v0)
			return true
		}
		// match: (GE (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (GEnoov (CMPW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64GEnoov, v0)
			return true
		}
		// match: (GE (CMPWconst [0] x) yes no)
		// result: (TBZ [31] x yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.resetWithControl(BlockARM64TBZ, x)
			b.AuxInt = int64ToAuxInt(31)
			return true
		}
		// match: (GE (CMPconst [0] x) yes no)
		// result: (TBZ [63] x yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.resetWithControl(BlockARM64TBZ, x)
			b.AuxInt = int64ToAuxInt(63)
			return true
		}
		// match: (GE (FlagConstant [fc]) yes no)
		// cond: fc.ge()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.ge()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (GE (FlagConstant [fc]) yes no)
		// cond: !fc.ge()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.ge()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GE (InvertFlags cmp) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64LE, cmp)
			return true
		}
	case BlockARM64GEnoov:
		// match: (GEnoov (FlagConstant [fc]) yes no)
		// cond: fc.geNoov()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.geNoov()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (GEnoov (FlagConstant [fc]) yes no)
		// cond: !fc.geNoov()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.geNoov()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GEnoov (InvertFlags cmp) yes no)
		// result: (LEnoov cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64LEnoov, cmp)
			return true
		}
	case BlockARM64GT:
		// match: (GT (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GT (TSTWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64GT, v0)
			return true
		}
		// match: (GT (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (TST x y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64TST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64GT, v0)
				return true
			}
			break
		}
		// match: (GT (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (TSTW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64TSTW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64GT, v0)
				return true
			}
			break
		}
		// match: (GT (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GT (TSTconst [c] y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64GT, v0)
			return true
		}
		// match: (GT (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GTnoov (CMNconst [c] y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64GTnoov, v0)
			return true
		}
		// match: (GT (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GTnoov (CMNWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64GTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (GTnoov (CMN x y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64GTnoov, v0)
				return true
			}
			break
		}
		// match: (GT (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (GTnoov (CMNW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64GTnoov, v0)
				return true
			}
			break
		}
		// match: (GT (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (CMN x y) yes no)
		for b.Controls[0].Op == OpARM64CMP {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64GT, v0)
			return true
		}
		// match: (GT (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (CMNW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPW {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64GT, v0)
			return true
		}
		// match: (GT (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (GTnoov (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64GTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (GTnoov (CMP a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMP, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64GTnoov, v0)
			return true
		}
		// match: (GT (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (GTnoov (CMNW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64GTnoov, v0)
			return true
		}
		// match: (GT (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (GTnoov (CMPW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64GTnoov, v0)
			return true
		}
		// match: (GT (FlagConstant [fc]) yes no)
		// cond: fc.gt()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.gt()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (GT (FlagConstant [fc]) yes no)
		// cond: !fc.gt()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.gt()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GT (InvertFlags cmp) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64LT, cmp)
			return true
		}
	case BlockARM64GTnoov:
		// match: (GTnoov (FlagConstant [fc]) yes no)
		// cond: fc.gtNoov()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.gtNoov()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (GTnoov (FlagConstant [fc]) yes no)
		// cond: !fc.gtNoov()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.gtNoov()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (GTnoov (InvertFlags cmp) yes no)
		// result: (LTnoov cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64LTnoov, cmp)
			return true
		}
	case BlockIf:
		// match: (If (Equal cc) yes no)
		// result: (EQ cc yes no)
		for b.Controls[0].Op == OpARM64Equal {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64EQ, cc)
			return true
		}
		// match: (If (NotEqual cc) yes no)
		// result: (NE cc yes no)
		for b.Controls[0].Op == OpARM64NotEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64NE, cc)
			return true
		}
		// match: (If (LessThan cc) yes no)
		// result: (LT cc yes no)
		for b.Controls[0].Op == OpARM64LessThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64LT, cc)
			return true
		}
		// match: (If (LessThanU cc) yes no)
		// result: (ULT cc yes no)
		for b.Controls[0].Op == OpARM64LessThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64ULT, cc)
			return true
		}
		// match: (If (LessEqual cc) yes no)
		// result: (LE cc yes no)
		for b.Controls[0].Op == OpARM64LessEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64LE, cc)
			return true
		}
		// match: (If (LessEqualU cc) yes no)
		// result: (ULE cc yes no)
		for b.Controls[0].Op == OpARM64LessEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64ULE, cc)
			return true
		}
		// match: (If (GreaterThan cc) yes no)
		// result: (GT cc yes no)
		for b.Controls[0].Op == OpARM64GreaterThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64GT, cc)
			return true
		}
		// match: (If (GreaterThanU cc) yes no)
		// result: (UGT cc yes no)
		for b.Controls[0].Op == OpARM64GreaterThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64UGT, cc)
			return true
		}
		// match: (If (GreaterEqual cc) yes no)
		// result: (GE cc yes no)
		for b.Controls[0].Op == OpARM64GreaterEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64GE, cc)
			return true
		}
		// match: (If (GreaterEqualU cc) yes no)
		// result: (UGE cc yes no)
		for b.Controls[0].Op == OpARM64GreaterEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64UGE, cc)
			return true
		}
		// match: (If (LessThanF cc) yes no)
		// result: (FLT cc yes no)
		for b.Controls[0].Op == OpARM64LessThanF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64FLT, cc)
			return true
		}
		// match: (If (LessEqualF cc) yes no)
		// result: (FLE cc yes no)
		for b.Controls[0].Op == OpARM64LessEqualF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64FLE, cc)
			return true
		}
		// match: (If (GreaterThanF cc) yes no)
		// result: (FGT cc yes no)
		for b.Controls[0].Op == OpARM64GreaterThanF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64FGT, cc)
			return true
		}
		// match: (If (GreaterEqualF cc) yes no)
		// result: (FGE cc yes no)
		for b.Controls[0].Op == OpARM64GreaterEqualF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64FGE, cc)
			return true
		}
		// match: (If cond yes no)
		// result: (NZ cond yes no)
		for {
			cond := b.Controls[0]
			b.resetWithControl(BlockARM64NZ, cond)
			return true
		}
	case BlockARM64LE:
		// match: (LE (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LE (TSTWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64LE, v0)
			return true
		}
		// match: (LE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (TST x y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64TST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64LE, v0)
				return true
			}
			break
		}
		// match: (LE (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (TSTW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64TSTW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64LE, v0)
				return true
			}
			break
		}
		// match: (LE (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LE (TSTconst [c] y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64LE, v0)
			return true
		}
		// match: (LE (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LEnoov (CMNconst [c] y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64LEnoov, v0)
			return true
		}
		// match: (LE (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LEnoov (CMNWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64LEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (LEnoov (CMN x y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64LEnoov, v0)
				return true
			}
			break
		}
		// match: (LE (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (LEnoov (CMNW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64LEnoov, v0)
				return true
			}
			break
		}
		// match: (LE (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (CMN x y) yes no)
		for b.Controls[0].Op == OpARM64CMP {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64LE, v0)
			return true
		}
		// match: (LE (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (CMNW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPW {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64LE, v0)
			return true
		}
		// match: (LE (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (LEnoov (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64LEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (LEnoov (CMP a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMP, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64LEnoov, v0)
			return true
		}
		// match: (LE (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (LEnoov (CMNW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64LEnoov, v0)
			return true
		}
		// match: (LE (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (LEnoov (CMPW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64LEnoov, v0)
			return true
		}
		// match: (LE (FlagConstant [fc]) yes no)
		// cond: fc.le()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.le()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (LE (FlagConstant [fc]) yes no)
		// cond: !fc.le()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.le()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LE (InvertFlags cmp) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64GE, cmp)
			return true
		}
	case BlockARM64LEnoov:
		// match: (LEnoov (FlagConstant [fc]) yes no)
		// cond: fc.leNoov()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.leNoov()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (LEnoov (FlagConstant [fc]) yes no)
		// cond: !fc.leNoov()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.leNoov()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LEnoov (InvertFlags cmp) yes no)
		// result: (GEnoov cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64GEnoov, cmp)
			return true
		}
	case BlockARM64LT:
		// match: (LT (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LT (TSTWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64LT, v0)
			return true
		}
		// match: (LT (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (TST x y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64TST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64LT, v0)
				return true
			}
			break
		}
		// match: (LT (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (TSTW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64TSTW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64LT, v0)
				return true
			}
			break
		}
		// match: (LT (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LT (TSTconst [c] y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64LT, v0)
			return true
		}
		// match: (LT (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LTnoov (CMNconst [c] y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64LTnoov, v0)
			return true
		}
		// match: (LT (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LTnoov (CMNWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64LTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (LTnoov (CMN x y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64LTnoov, v0)
				return true
			}
			break
		}
		// match: (LT (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (LTnoov (CMNW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64LTnoov, v0)
				return true
			}
			break
		}
		// match: (LT (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (CMN x y) yes no)
		for b.Controls[0].Op == OpARM64CMP {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64LT, v0)
			return true
		}
		// match: (LT (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (CMNW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPW {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64LT, v0)
			return true
		}
		// match: (LT (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (LTnoov (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64LTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (LTnoov (CMP a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMP, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64LTnoov, v0)
			return true
		}
		// match: (LT (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (LTnoov (CMNW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64LTnoov, v0)
			return true
		}
		// match: (LT (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (LTnoov (CMPW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64LTnoov, v0)
			return true
		}
		// match: (LT (CMPWconst [0] x) yes no)
		// result: (TBNZ [31] x yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.resetWithControl(BlockARM64TBNZ, x)
			b.AuxInt = int64ToAuxInt(31)
			return true
		}
		// match: (LT (CMPconst [0] x) yes no)
		// result: (TBNZ [63] x yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.resetWithControl(BlockARM64TBNZ, x)
			b.AuxInt = int64ToAuxInt(63)
			return true
		}
		// match: (LT (FlagConstant [fc]) yes no)
		// cond: fc.lt()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.lt()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (LT (FlagConstant [fc]) yes no)
		// cond: !fc.lt()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.lt()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LT (InvertFlags cmp) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64GT, cmp)
			return true
		}
	case BlockARM64LTnoov:
		// match: (LTnoov (FlagConstant [fc]) yes no)
		// cond: fc.ltNoov()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.ltNoov()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (LTnoov (FlagConstant [fc]) yes no)
		// cond: !fc.ltNoov()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.ltNoov()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (LTnoov (InvertFlags cmp) yes no)
		// result: (GTnoov cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64GTnoov, cmp)
			return true
		}
	case BlockARM64NE:
		// match: (NE (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (NE (TSTWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (TST x y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64TST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64NE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (TSTW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64TSTW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64NE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (NE (TSTconst [c] y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (NE (CMNconst [c] y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (NE (CMNWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := auxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (CMN x y) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64NE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (CMNW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.resetWithControl(BlockARM64NE, v0)
				return true
			}
			break
		}
		// match: (NE (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (CMN x y) yes no)
		for b.Controls[0].Op == OpARM64CMP {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (CMNW x y) yes no)
		for b.Controls[0].Op == OpARM64CMPW {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPconst [0] x) yes no)
		// result: (NZ x yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.resetWithControl(BlockARM64NZ, x)
			return true
		}
		// match: (NE (CMPWconst [0] x) yes no)
		// result: (NZW x yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.resetWithControl(BlockARM64NZW, x)
			return true
		}
		// match: (NE (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (NE (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (NE (CMP a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMP, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (NE (CMNW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMNW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (NE (CMPW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, OpARM64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (NE (TSTconst [c] x) yes no)
		// cond: oneBit(c)
		// result: (TBNZ [int64(ntz64(c))] x yes no)
		for b.Controls[0].Op == OpARM64TSTconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			x := v_0.Args[0]
			if !(oneBit(c)) {
				break
			}
			b.resetWithControl(BlockARM64TBNZ, x)
			b.AuxInt = int64ToAuxInt(int64(ntz64(c)))
			return true
		}
		// match: (NE (TSTWconst [c] x) yes no)
		// cond: oneBit(int64(uint32(c)))
		// result: (TBNZ [int64(ntz64(int64(uint32(c))))] x yes no)
		for b.Controls[0].Op == OpARM64TSTWconst {
			v_0 := b.Controls[0]
			c := auxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if !(oneBit(int64(uint32(c)))) {
				break
			}
			b.resetWithControl(BlockARM64TBNZ, x)
			b.AuxInt = int64ToAuxInt(int64(ntz64(int64(uint32(c)))))
			return true
		}
		// match: (NE (FlagConstant [fc]) yes no)
		// cond: fc.ne()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.ne()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (NE (FlagConstant [fc]) yes no)
		// cond: !fc.ne()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.ne()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (NE (InvertFlags cmp) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64NE, cmp)
			return true
		}
	case BlockARM64NZ:
		// match: (NZ (Equal cc) yes no)
		// result: (EQ cc yes no)
		for b.Controls[0].Op == OpARM64Equal {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64EQ, cc)
			return true
		}
		// match: (NZ (NotEqual cc) yes no)
		// result: (NE cc yes no)
		for b.Controls[0].Op == OpARM64NotEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64NE, cc)
			return true
		}
		// match: (NZ (LessThan cc) yes no)
		// result: (LT cc yes no)
		for b.Controls[0].Op == OpARM64LessThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64LT, cc)
			return true
		}
		// match: (NZ (LessThanU cc) yes no)
		// result: (ULT cc yes no)
		for b.Controls[0].Op == OpARM64LessThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64ULT, cc)
			return true
		}
		// match: (NZ (LessEqual cc) yes no)
		// result: (LE cc yes no)
		for b.Controls[0].Op == OpARM64LessEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64LE, cc)
			return true
		}
		// match: (NZ (LessEqualU cc) yes no)
		// result: (ULE cc yes no)
		for b.Controls[0].Op == OpARM64LessEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64ULE, cc)
			return true
		}
		// match: (NZ (GreaterThan cc) yes no)
		// result: (GT cc yes no)
		for b.Controls[0].Op == OpARM64GreaterThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64GT, cc)
			return true
		}
		// match: (NZ (GreaterThanU cc) yes no)
		// result: (UGT cc yes no)
		for b.Controls[0].Op == OpARM64GreaterThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64UGT, cc)
			return true
		}
		// match: (NZ (GreaterEqual cc) yes no)
		// result: (GE cc yes no)
		for b.Controls[0].Op == OpARM64GreaterEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64GE, cc)
			return true
		}
		// match: (NZ (GreaterEqualU cc) yes no)
		// result: (UGE cc yes no)
		for b.Controls[0].Op == OpARM64GreaterEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64UGE, cc)
			return true
		}
		// match: (NZ (LessThanF cc) yes no)
		// result: (FLT cc yes no)
		for b.Controls[0].Op == OpARM64LessThanF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64FLT, cc)
			return true
		}
		// match: (NZ (LessEqualF cc) yes no)
		// result: (FLE cc yes no)
		for b.Controls[0].Op == OpARM64LessEqualF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64FLE, cc)
			return true
		}
		// match: (NZ (GreaterThanF cc) yes no)
		// result: (FGT cc yes no)
		for b.Controls[0].Op == OpARM64GreaterThanF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64FGT, cc)
			return true
		}
		// match: (NZ (GreaterEqualF cc) yes no)
		// result: (FGE cc yes no)
		for b.Controls[0].Op == OpARM64GreaterEqualF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.resetWithControl(BlockARM64FGE, cc)
			return true
		}
		// match: (NZ (ANDconst [c] x) yes no)
		// cond: oneBit(c)
		// result: (TBNZ [int64(ntz64(c))] x yes no)
		for b.Controls[0].Op == OpARM64ANDconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			x := v_0.Args[0]
			if !(oneBit(c)) {
				break
			}
			b.resetWithControl(BlockARM64TBNZ, x)
			b.AuxInt = int64ToAuxInt(int64(ntz64(c)))
			return true
		}
		// match: (NZ (MOVDconst [0]) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64MOVDconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (NZ (MOVDconst [c]) yes no)
		// cond: c != 0
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64MOVDconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(c != 0) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
	case BlockARM64NZW:
		// match: (NZW (ANDconst [c] x) yes no)
		// cond: oneBit(int64(uint32(c)))
		// result: (TBNZ [int64(ntz64(int64(uint32(c))))] x yes no)
		for b.Controls[0].Op == OpARM64ANDconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			x := v_0.Args[0]
			if !(oneBit(int64(uint32(c)))) {
				break
			}
			b.resetWithControl(BlockARM64TBNZ, x)
			b.AuxInt = int64ToAuxInt(int64(ntz64(int64(uint32(c)))))
			return true
		}
		// match: (NZW (MOVDconst [c]) yes no)
		// cond: int32(c) == 0
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64MOVDconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(int32(c) == 0) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (NZW (MOVDconst [c]) yes no)
		// cond: int32(c) != 0
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64MOVDconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(int32(c) != 0) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
	case BlockARM64UGE:
		// match: (UGE (FlagConstant [fc]) yes no)
		// cond: fc.uge()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.uge()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (UGE (FlagConstant [fc]) yes no)
		// cond: !fc.uge()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.uge()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (UGE (InvertFlags cmp) yes no)
		// result: (ULE cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64ULE, cmp)
			return true
		}
	case BlockARM64UGT:
		// match: (UGT (FlagConstant [fc]) yes no)
		// cond: fc.ugt()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.ugt()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (UGT (FlagConstant [fc]) yes no)
		// cond: !fc.ugt()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.ugt()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (UGT (InvertFlags cmp) yes no)
		// result: (ULT cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64ULT, cmp)
			return true
		}
	case BlockARM64ULE:
		// match: (ULE (FlagConstant [fc]) yes no)
		// cond: fc.ule()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.ule()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (ULE (FlagConstant [fc]) yes no)
		// cond: !fc.ule()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.ule()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (ULE (InvertFlags cmp) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64UGE, cmp)
			return true
		}
	case BlockARM64ULT:
		// match: (ULT (FlagConstant [fc]) yes no)
		// cond: fc.ult()
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(fc.ult()) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (ULT (FlagConstant [fc]) yes no)
		// cond: !fc.ult()
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := auxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.ult()) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
		// match: (ULT (InvertFlags cmp) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.resetWithControl(BlockARM64UGT, cmp)
			return true
		}
	case BlockARM64Z:
		// match: (Z (ANDconst [c] x) yes no)
		// cond: oneBit(c)
		// result: (TBZ [int64(ntz64(c))] x yes no)
		for b.Controls[0].Op == OpARM64ANDconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			x := v_0.Args[0]
			if !(oneBit(c)) {
				break
			}
			b.resetWithControl(BlockARM64TBZ, x)
			b.AuxInt = int64ToAuxInt(int64(ntz64(c)))
			return true
		}
		// match: (Z (MOVDconst [0]) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64MOVDconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (Z (MOVDconst [c]) yes no)
		// cond: c != 0
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64MOVDconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(c != 0) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	case BlockARM64ZW:
		// match: (ZW (ANDconst [c] x) yes no)
		// cond: oneBit(int64(uint32(c)))
		// result: (TBZ [int64(ntz64(int64(uint32(c))))] x yes no)
		for b.Controls[0].Op == OpARM64ANDconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			x := v_0.Args[0]
			if !(oneBit(int64(uint32(c)))) {
				break
			}
			b.resetWithControl(BlockARM64TBZ, x)
			b.AuxInt = int64ToAuxInt(int64(ntz64(int64(uint32(c)))))
			return true
		}
		// match: (ZW (MOVDconst [c]) yes no)
		// cond: int32(c) == 0
		// result: (First yes no)
		for b.Controls[0].Op == OpARM64MOVDconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(int32(c) == 0) {
				break
			}
			b.Reset(BlockFirst)
			return true
		}
		// match: (ZW (MOVDconst [c]) yes no)
		// cond: int32(c) != 0
		// result: (First no yes)
		for b.Controls[0].Op == OpARM64MOVDconst {
			v_0 := b.Controls[0]
			c := auxIntToInt64(v_0.AuxInt)
			if !(int32(c) != 0) {
				break
			}
			b.Reset(BlockFirst)
			b.swapSuccessors()
			return true
		}
	}
	return false
}
