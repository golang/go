// Code generated from _gen/ARM64.rules using 'go generate'; DO NOT EDIT.

package ssa

import "cmd/compile/internal/types"

func rewriteValueARM64(v *Value) bool {
	switch v.Op {
	case OpARM64ADCSflags:
		return rewriteValueARM64_OpARM64ADCSflags(v)
	case OpARM64ADD:
		return rewriteValueARM64_OpARM64ADD(v)
	case OpARM64ADDSflags:
		return rewriteValueARM64_OpARM64ADDSflags(v)
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
	case OpARM64ANDshiftRO:
		return rewriteValueARM64_OpARM64ANDshiftRO(v)
	case OpARM64BIC:
		return rewriteValueARM64_OpARM64BIC(v)
	case OpARM64BICshiftLL:
		return rewriteValueARM64_OpARM64BICshiftLL(v)
	case OpARM64BICshiftRA:
		return rewriteValueARM64_OpARM64BICshiftRA(v)
	case OpARM64BICshiftRL:
		return rewriteValueARM64_OpARM64BICshiftRL(v)
	case OpARM64BICshiftRO:
		return rewriteValueARM64_OpARM64BICshiftRO(v)
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
	case OpARM64CSETM:
		return rewriteValueARM64_OpARM64CSETM(v)
	case OpARM64CSINC:
		return rewriteValueARM64_OpARM64CSINC(v)
	case OpARM64CSINV:
		return rewriteValueARM64_OpARM64CSINV(v)
	case OpARM64CSNEG:
		return rewriteValueARM64_OpARM64CSNEG(v)
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
	case OpARM64EONshiftRO:
		return rewriteValueARM64_OpARM64EONshiftRO(v)
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
	case OpARM64FMOVDloadidx8:
		return rewriteValueARM64_OpARM64FMOVDloadidx8(v)
	case OpARM64FMOVDstore:
		return rewriteValueARM64_OpARM64FMOVDstore(v)
	case OpARM64FMOVDstoreidx:
		return rewriteValueARM64_OpARM64FMOVDstoreidx(v)
	case OpARM64FMOVDstoreidx8:
		return rewriteValueARM64_OpARM64FMOVDstoreidx8(v)
	case OpARM64FMOVSload:
		return rewriteValueARM64_OpARM64FMOVSload(v)
	case OpARM64FMOVSloadidx:
		return rewriteValueARM64_OpARM64FMOVSloadidx(v)
	case OpARM64FMOVSloadidx4:
		return rewriteValueARM64_OpARM64FMOVSloadidx4(v)
	case OpARM64FMOVSstore:
		return rewriteValueARM64_OpARM64FMOVSstore(v)
	case OpARM64FMOVSstoreidx:
		return rewriteValueARM64_OpARM64FMOVSstoreidx(v)
	case OpARM64FMOVSstoreidx4:
		return rewriteValueARM64_OpARM64FMOVSstoreidx4(v)
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
	case OpARM64GreaterEqualNoov:
		return rewriteValueARM64_OpARM64GreaterEqualNoov(v)
	case OpARM64GreaterEqualU:
		return rewriteValueARM64_OpARM64GreaterEqualU(v)
	case OpARM64GreaterThan:
		return rewriteValueARM64_OpARM64GreaterThan(v)
	case OpARM64GreaterThanF:
		return rewriteValueARM64_OpARM64GreaterThanF(v)
	case OpARM64GreaterThanU:
		return rewriteValueARM64_OpARM64GreaterThanU(v)
	case OpARM64LDP:
		return rewriteValueARM64_OpARM64LDP(v)
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
	case OpARM64LessThanNoov:
		return rewriteValueARM64_OpARM64LessThanNoov(v)
	case OpARM64LessThanU:
		return rewriteValueARM64_OpARM64LessThanU(v)
	case OpARM64LoweredPanicBoundsCR:
		return rewriteValueARM64_OpARM64LoweredPanicBoundsCR(v)
	case OpARM64LoweredPanicBoundsRC:
		return rewriteValueARM64_OpARM64LoweredPanicBoundsRC(v)
	case OpARM64LoweredPanicBoundsRR:
		return rewriteValueARM64_OpARM64LoweredPanicBoundsRR(v)
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
	case OpARM64MOVDload:
		return rewriteValueARM64_OpARM64MOVDload(v)
	case OpARM64MOVDloadidx:
		return rewriteValueARM64_OpARM64MOVDloadidx(v)
	case OpARM64MOVDloadidx8:
		return rewriteValueARM64_OpARM64MOVDloadidx8(v)
	case OpARM64MOVDnop:
		return rewriteValueARM64_OpARM64MOVDnop(v)
	case OpARM64MOVDreg:
		return rewriteValueARM64_OpARM64MOVDreg(v)
	case OpARM64MOVDstore:
		return rewriteValueARM64_OpARM64MOVDstore(v)
	case OpARM64MOVDstoreidx:
		return rewriteValueARM64_OpARM64MOVDstoreidx(v)
	case OpARM64MOVDstoreidx8:
		return rewriteValueARM64_OpARM64MOVDstoreidx8(v)
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
	case OpARM64MVNshiftRO:
		return rewriteValueARM64_OpARM64MVNshiftRO(v)
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
	case OpARM64ORNshiftRO:
		return rewriteValueARM64_OpARM64ORNshiftRO(v)
	case OpARM64ORconst:
		return rewriteValueARM64_OpARM64ORconst(v)
	case OpARM64ORshiftLL:
		return rewriteValueARM64_OpARM64ORshiftLL(v)
	case OpARM64ORshiftRA:
		return rewriteValueARM64_OpARM64ORshiftRA(v)
	case OpARM64ORshiftRL:
		return rewriteValueARM64_OpARM64ORshiftRL(v)
	case OpARM64ORshiftRO:
		return rewriteValueARM64_OpARM64ORshiftRO(v)
	case OpARM64REV:
		return rewriteValueARM64_OpARM64REV(v)
	case OpARM64REVW:
		return rewriteValueARM64_OpARM64REVW(v)
	case OpARM64ROR:
		return rewriteValueARM64_OpARM64ROR(v)
	case OpARM64RORW:
		return rewriteValueARM64_OpARM64RORW(v)
	case OpARM64SBCSflags:
		return rewriteValueARM64_OpARM64SBCSflags(v)
	case OpARM64SBFX:
		return rewriteValueARM64_OpARM64SBFX(v)
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
	case OpARM64TSTshiftRO:
		return rewriteValueARM64_OpARM64TSTshiftRO(v)
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
	case OpARM64XORshiftRO:
		return rewriteValueARM64_OpARM64XORshiftRO(v)
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
	case OpAtomicAnd32value:
		v.Op = OpARM64LoweredAtomicAnd32
		return true
	case OpAtomicAnd32valueVariant:
		v.Op = OpARM64LoweredAtomicAnd32Variant
		return true
	case OpAtomicAnd64value:
		v.Op = OpARM64LoweredAtomicAnd64
		return true
	case OpAtomicAnd64valueVariant:
		v.Op = OpARM64LoweredAtomicAnd64Variant
		return true
	case OpAtomicAnd8value:
		v.Op = OpARM64LoweredAtomicAnd8
		return true
	case OpAtomicAnd8valueVariant:
		v.Op = OpARM64LoweredAtomicAnd8Variant
		return true
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
	case OpAtomicExchange8:
		v.Op = OpARM64LoweredAtomicExchange8
		return true
	case OpAtomicExchange8Variant:
		v.Op = OpARM64LoweredAtomicExchange8Variant
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
	case OpAtomicOr32value:
		v.Op = OpARM64LoweredAtomicOr32
		return true
	case OpAtomicOr32valueVariant:
		v.Op = OpARM64LoweredAtomicOr32Variant
		return true
	case OpAtomicOr64value:
		v.Op = OpARM64LoweredAtomicOr64
		return true
	case OpAtomicOr64valueVariant:
		v.Op = OpARM64LoweredAtomicOr64Variant
		return true
	case OpAtomicOr8value:
		v.Op = OpARM64LoweredAtomicOr8
		return true
	case OpAtomicOr8valueVariant:
		v.Op = OpARM64LoweredAtomicOr8Variant
		return true
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
	case OpBitLen16:
		return rewriteValueARM64_OpBitLen16(v)
	case OpBitLen32:
		return rewriteValueARM64_OpBitLen32(v)
	case OpBitLen64:
		return rewriteValueARM64_OpBitLen64(v)
	case OpBitLen8:
		return rewriteValueARM64_OpBitLen8(v)
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
	case OpBswap16:
		v.Op = OpARM64REV16W
		return true
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
	case OpMax32F:
		v.Op = OpARM64FMAXS
		return true
	case OpMax64F:
		v.Op = OpARM64FMAXD
		return true
	case OpMemEq:
		v.Op = OpARM64LoweredMemEq
		return true
	case OpMin32F:
		v.Op = OpARM64FMINS
		return true
	case OpMin64F:
		v.Op = OpARM64FMIND
		return true
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
		v.Op = OpARM64LoweredPanicBoundsRR
		return true
	case OpPopCount16:
		return rewriteValueARM64_OpPopCount16(v)
	case OpPopCount32:
		return rewriteValueARM64_OpPopCount32(v)
	case OpPopCount64:
		return rewriteValueARM64_OpPopCount64(v)
	case OpPrefetchCache:
		return rewriteValueARM64_OpPrefetchCache(v)
	case OpPrefetchCacheStreamed:
		return rewriteValueARM64_OpPrefetchCacheStreamed(v)
	case OpPubBarrier:
		return rewriteValueARM64_OpPubBarrier(v)
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
	case OpSelectN:
		return rewriteValueARM64_OpSelectN(v)
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
	case OpSqrt32:
		v.Op = OpARM64FSQRTS
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
	case OpTailCall:
		v.Op = OpARM64CALLtail
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
	// match: (ADD x (MOVDconst <t> [c]))
	// cond: !t.IsPtr()
	// result: (ADDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			t := v_1.Type
			c := auxIntToInt64(v_1.AuxInt)
			if !(!t.IsPtr()) {
				continue
			}
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
	// cond: v.Type.Size() <= 4 && l.Uses==1 && clobber(l)
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
			if !(v.Type.Size() <= 4 && l.Uses == 1 && clobber(l)) {
				continue
			}
			v.reset(OpARM64MADDW)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (ADD a l:(MNEGW x y))
	// cond: v.Type.Size() <= 4 && l.Uses==1 && clobber(l)
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
			if !(v.Type.Size() <= 4 && l.Uses == 1 && clobber(l)) {
				continue
			}
			v.reset(OpARM64MSUBW)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (ADD <t> a p:(ADDconst [c] m:(MUL _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (ADDconst [c] (ADD <v.Type> a m))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			p := v_1
			if p.Op != OpARM64ADDconst {
				continue
			}
			c := auxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != OpARM64MUL || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.reset(OpARM64ADDconst)
			v.AuxInt = int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, OpARM64ADD, v.Type)
			v0.AddArg2(a, m)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (ADD <t> a p:(ADDconst [c] m:(MULW _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (ADDconst [c] (ADD <v.Type> a m))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			p := v_1
			if p.Op != OpARM64ADDconst {
				continue
			}
			c := auxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != OpARM64MULW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.reset(OpARM64ADDconst)
			v.AuxInt = int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, OpARM64ADD, v.Type)
			v0.AddArg2(a, m)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (ADD <t> a p:(ADDconst [c] m:(MNEG _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (ADDconst [c] (ADD <v.Type> a m))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			p := v_1
			if p.Op != OpARM64ADDconst {
				continue
			}
			c := auxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != OpARM64MNEG || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.reset(OpARM64ADDconst)
			v.AuxInt = int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, OpARM64ADD, v.Type)
			v0.AddArg2(a, m)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (ADD <t> a p:(ADDconst [c] m:(MNEGW _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (ADDconst [c] (ADD <v.Type> a m))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			p := v_1
			if p.Op != OpARM64ADDconst {
				continue
			}
			c := auxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != OpARM64MNEGW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.reset(OpARM64ADDconst)
			v.AuxInt = int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, OpARM64ADD, v.Type)
			v0.AddArg2(a, m)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (ADD <t> a p:(SUBconst [c] m:(MUL _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (SUBconst [c] (ADD <v.Type> a m))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			p := v_1
			if p.Op != OpARM64SUBconst {
				continue
			}
			c := auxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != OpARM64MUL || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.reset(OpARM64SUBconst)
			v.AuxInt = int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, OpARM64ADD, v.Type)
			v0.AddArg2(a, m)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (ADD <t> a p:(SUBconst [c] m:(MULW _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (SUBconst [c] (ADD <v.Type> a m))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			p := v_1
			if p.Op != OpARM64SUBconst {
				continue
			}
			c := auxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != OpARM64MULW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.reset(OpARM64SUBconst)
			v.AuxInt = int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, OpARM64ADD, v.Type)
			v0.AddArg2(a, m)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (ADD <t> a p:(SUBconst [c] m:(MNEG _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (SUBconst [c] (ADD <v.Type> a m))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			p := v_1
			if p.Op != OpARM64SUBconst {
				continue
			}
			c := auxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != OpARM64MNEG || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.reset(OpARM64SUBconst)
			v.AuxInt = int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, OpARM64ADD, v.Type)
			v0.AddArg2(a, m)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (ADD <t> a p:(SUBconst [c] m:(MNEGW _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (SUBconst [c] (ADD <v.Type> a m))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			p := v_1
			if p.Op != OpARM64SUBconst {
				continue
			}
			c := auxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != OpARM64MNEGW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.reset(OpARM64SUBconst)
			v.AuxInt = int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, OpARM64ADD, v.Type)
			v0.AddArg2(a, m)
			v.AddArg(v0)
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
	// match: (ADD x0 x1:(ANDshiftRA x2:(SLLconst [sl] y) z [63]))
	// cond: x1.Uses == 1 && x2.Uses == 1
	// result: (ADDshiftLL x0 (ANDshiftRA <y.Type> y z [63]) [sl])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64ANDshiftRA || auxIntToInt64(x1.AuxInt) != 63 {
				continue
			}
			z := x1.Args[1]
			x2 := x1.Args[0]
			if x2.Op != OpARM64SLLconst {
				continue
			}
			sl := auxIntToInt64(x2.AuxInt)
			y := x2.Args[0]
			if !(x1.Uses == 1 && x2.Uses == 1) {
				continue
			}
			v.reset(OpARM64ADDshiftLL)
			v.AuxInt = int64ToAuxInt(sl)
			v0 := b.NewValue0(v.Pos, OpARM64ANDshiftRA, y.Type)
			v0.AuxInt = int64ToAuxInt(63)
			v0.AddArg2(y, z)
			v.AddArg2(x0, v0)
			return true
		}
		break
	}
	// match: (ADD x0 x1:(ANDshiftLL x2:(SRAconst [63] z) y [sl]))
	// cond: x1.Uses == 1 && x2.Uses == 1
	// result: (ADDshiftLL x0 (ANDshiftRA <y.Type> y z [63]) [sl])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64ANDshiftLL {
				continue
			}
			sl := auxIntToInt64(x1.AuxInt)
			y := x1.Args[1]
			x2 := x1.Args[0]
			if x2.Op != OpARM64SRAconst || auxIntToInt64(x2.AuxInt) != 63 {
				continue
			}
			z := x2.Args[0]
			if !(x1.Uses == 1 && x2.Uses == 1) {
				continue
			}
			v.reset(OpARM64ADDshiftLL)
			v.AuxInt = int64ToAuxInt(sl)
			v0 := b.NewValue0(v.Pos, OpARM64ANDshiftRA, y.Type)
			v0.AuxInt = int64ToAuxInt(63)
			v0.AddArg2(y, z)
			v.AddArg2(x0, v0)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM64_OpARM64ADDSflags(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDSflags x (MOVDconst [c]))
	// result: (ADDSconstflags [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			v.reset(OpARM64ADDSconstflags)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg(x)
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
	// match: (ADDconst [c] y)
	// cond: c < 0
	// result: (SUBconst [-c] y)
	for {
		c := auxIntToInt64(v.AuxInt)
		y := v_0
		if !(c < 0) {
			break
		}
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64ToAuxInt(-c)
		v.AddArg(y)
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
	// match: (ADDshiftLL [8] (UBFX [armBFAuxInt(8, 24)] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff
	// result: (REV16W x)
	for {
		if auxIntToInt64(v.AuxInt) != 8 || v_0.Op != OpARM64UBFX || auxIntToArm64BitField(v_0.AuxInt) != armBFAuxInt(8, 24) {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64ANDconst {
			break
		}
		c1 := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != OpARM64ANDconst {
			break
		}
		c2 := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff) {
			break
		}
		v.reset(OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL [8] (SRLconst [8] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: (uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff)
	// result: (REV16 x)
	for {
		if auxIntToInt64(v.AuxInt) != 8 || v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != 8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64ANDconst {
			break
		}
		c1 := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != OpARM64ANDconst {
			break
		}
		c2 := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff) {
			break
		}
		v.reset(OpARM64REV16)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL [8] (SRLconst [8] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: (uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff)
	// result: (REV16 (ANDconst <x.Type> [0xffffffff] x))
	for {
		if auxIntToInt64(v.AuxInt) != 8 || v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != 8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64ANDconst {
			break
		}
		c1 := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != OpARM64ANDconst {
			break
		}
		c2 := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff) {
			break
		}
		v.reset(OpARM64REV16)
		v0 := b.NewValue0(v.Pos, OpARM64ANDconst, x.Type)
		v0.AuxInt = int64ToAuxInt(0xffffffff)
		v0.AddArg(x)
		v.AddArg(v0)
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
	// match: (AND x0 x1:(RORconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ANDshiftRO x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64RORconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64ANDshiftRO)
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
	// match: (ANDconst [c] (UBFX [bfc] x))
	// cond: isARM64BFMask(0, c, 0)
	// result: (UBFX [armBFAuxInt(bfc.lsb(), min(bfc.width(), arm64BFWidth(c, 0)))] x)
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(0, c, 0)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.lsb(), min(bfc.width(), arm64BFWidth(c, 0))))
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
	// match: (ANDshiftLL y:(SLLconst x [c]) x [c])
	// result: y
	for {
		c := auxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != OpARM64SLLconst || auxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
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
	// match: (ANDshiftRA y:(SRAconst x [c]) x [c])
	// result: y
	for {
		c := auxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != OpARM64SRAconst || auxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
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
	// match: (ANDshiftRL y:(SRLconst x [c]) x [c])
	// result: y
	for {
		c := auxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != OpARM64SRLconst || auxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ANDshiftRO(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftRO (MOVDconst [c]) x [d])
	// result: (ANDconst [c] (RORconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64RORconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRO x (MOVDconst [c]) [d])
	// result: (ANDconst x [rotateRight64(c, d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftRO y:(RORconst x [c]) x [c])
	// result: y
	for {
		c := auxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != OpARM64RORconst || auxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
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
	// match: (BIC x0 x1:(RORconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (BICshiftRO x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64RORconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64BICshiftRO)
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
	// match: (BICshiftLL (SLLconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	// match: (BICshiftRA (SRAconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRAconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	// match: (BICshiftRL (SRLconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64BICshiftRO(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftRO x (MOVDconst [c]) [d])
	// result: (ANDconst x [^rotateRight64(c, d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(^rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftRO (RORconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64RORconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	// match: (CMNWconst [c] y)
	// cond: c < 0 && c != -1<<31
	// result: (CMPWconst [-c] y)
	for {
		c := auxIntToInt32(v.AuxInt)
		y := v_0
		if !(c < 0 && c != -1<<31) {
			break
		}
		v.reset(OpARM64CMPWconst)
		v.AuxInt = int32ToAuxInt(-c)
		v.AddArg(y)
		return true
	}
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
	// match: (CMNconst [c] y)
	// cond: c < 0 && c != -1<<63
	// result: (CMPconst [-c] y)
	for {
		c := auxIntToInt64(v.AuxInt)
		y := v_0
		if !(c < 0 && c != -1<<63) {
			break
		}
		v.reset(OpARM64CMPconst)
		v.AuxInt = int64ToAuxInt(-c)
		v.AddArg(y)
		return true
	}
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
	// cond: canonLessThan(x,y)
	// result: (InvertFlags (CMP y x))
	for {
		x := v_0
		y := v_1
		if !(canonLessThan(x, y)) {
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
	// cond: canonLessThan(x,y)
	// result: (InvertFlags (CMPW y x))
	for {
		x := v_0
		y := v_1
		if !(canonLessThan(x, y)) {
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
	// match: (CMPWconst [c] y)
	// cond: c < 0 && c != -1<<31
	// result: (CMNWconst [-c] y)
	for {
		c := auxIntToInt32(v.AuxInt)
		y := v_0
		if !(c < 0 && c != -1<<31) {
			break
		}
		v.reset(OpARM64CMNWconst)
		v.AuxInt = int32ToAuxInt(-c)
		v.AddArg(y)
		return true
	}
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
	// match: (CMPconst [c] y)
	// cond: c < 0 && c != -1<<63
	// result: (CMNconst [-c] y)
	for {
		c := auxIntToInt64(v.AuxInt)
		y := v_0
		if !(c < 0 && c != -1<<63) {
			break
		}
		v.reset(OpARM64CMNconst)
		v.AuxInt = int64ToAuxInt(-c)
		v.AddArg(y)
		return true
	}
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
	// match: (CSEL [cc] (MOVDconst [-1]) (MOVDconst [0]) flag)
	// result: (CSETM [cc] flag)
	for {
		cc := auxIntToOp(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst || auxIntToInt64(v_0.AuxInt) != -1 || v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		flag := v_2
		v.reset(OpARM64CSETM)
		v.AuxInt = opToAuxInt(cc)
		v.AddArg(flag)
		return true
	}
	// match: (CSEL [cc] (MOVDconst [0]) (MOVDconst [-1]) flag)
	// result: (CSETM [arm64Negate(cc)] flag)
	for {
		cc := auxIntToOp(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst || auxIntToInt64(v_0.AuxInt) != 0 || v_1.Op != OpARM64MOVDconst || auxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		flag := v_2
		v.reset(OpARM64CSETM)
		v.AuxInt = opToAuxInt(arm64Negate(cc))
		v.AddArg(flag)
		return true
	}
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
	// match: (CSEL [cc] x (ADDconst [1] a) flag)
	// result: (CSINC [cc] x a flag)
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64ADDconst || auxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		a := v_1.Args[0]
		flag := v_2
		v.reset(OpARM64CSINC)
		v.AuxInt = opToAuxInt(cc)
		v.AddArg3(x, a, flag)
		return true
	}
	// match: (CSEL [cc] (ADDconst [1] a) x flag)
	// result: (CSINC [arm64Negate(cc)] x a flag)
	for {
		cc := auxIntToOp(v.AuxInt)
		if v_0.Op != OpARM64ADDconst || auxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		a := v_0.Args[0]
		x := v_1
		flag := v_2
		v.reset(OpARM64CSINC)
		v.AuxInt = opToAuxInt(arm64Negate(cc))
		v.AddArg3(x, a, flag)
		return true
	}
	// match: (CSEL [cc] x (MVN a) flag)
	// result: (CSINV [cc] x a flag)
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MVN {
			break
		}
		a := v_1.Args[0]
		flag := v_2
		v.reset(OpARM64CSINV)
		v.AuxInt = opToAuxInt(cc)
		v.AddArg3(x, a, flag)
		return true
	}
	// match: (CSEL [cc] (MVN a) x flag)
	// result: (CSINV [arm64Negate(cc)] x a flag)
	for {
		cc := auxIntToOp(v.AuxInt)
		if v_0.Op != OpARM64MVN {
			break
		}
		a := v_0.Args[0]
		x := v_1
		flag := v_2
		v.reset(OpARM64CSINV)
		v.AuxInt = opToAuxInt(arm64Negate(cc))
		v.AddArg3(x, a, flag)
		return true
	}
	// match: (CSEL [cc] x (NEG a) flag)
	// result: (CSNEG [cc] x a flag)
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64NEG {
			break
		}
		a := v_1.Args[0]
		flag := v_2
		v.reset(OpARM64CSNEG)
		v.AuxInt = opToAuxInt(cc)
		v.AddArg3(x, a, flag)
		return true
	}
	// match: (CSEL [cc] (NEG a) x flag)
	// result: (CSNEG [arm64Negate(cc)] x a flag)
	for {
		cc := auxIntToOp(v.AuxInt)
		if v_0.Op != OpARM64NEG {
			break
		}
		a := v_0.Args[0]
		x := v_1
		flag := v_2
		v.reset(OpARM64CSNEG)
		v.AuxInt = opToAuxInt(arm64Negate(cc))
		v.AddArg3(x, a, flag)
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
func rewriteValueARM64_OpARM64CSETM(v *Value) bool {
	v_0 := v.Args[0]
	// match: (CSETM [cc] (InvertFlags cmp))
	// result: (CSETM [arm64Invert(cc)] cmp)
	for {
		cc := auxIntToOp(v.AuxInt)
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		cmp := v_0.Args[0]
		v.reset(OpARM64CSETM)
		v.AuxInt = opToAuxInt(arm64Invert(cc))
		v.AddArg(cmp)
		return true
	}
	// match: (CSETM [cc] flag)
	// cond: ccARM64Eval(cc, flag) > 0
	// result: (MOVDconst [-1])
	for {
		cc := auxIntToOp(v.AuxInt)
		flag := v_0
		if !(ccARM64Eval(cc, flag) > 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	// match: (CSETM [cc] flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: (MOVDconst [0])
	for {
		cc := auxIntToOp(v.AuxInt)
		flag := v_0
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CSINC(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CSINC [cc] x y (InvertFlags cmp))
	// result: (CSINC [arm64Invert(cc)] x y cmp)
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != OpARM64InvertFlags {
			break
		}
		cmp := v_2.Args[0]
		v.reset(OpARM64CSINC)
		v.AuxInt = opToAuxInt(arm64Invert(cc))
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (CSINC [cc] x _ flag)
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
	// match: (CSINC [cc] _ y flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: (ADDconst [1] y)
	for {
		cc := auxIntToOp(v.AuxInt)
		y := v_1
		flag := v_2
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CSINV(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CSINV [cc] x y (InvertFlags cmp))
	// result: (CSINV [arm64Invert(cc)] x y cmp)
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != OpARM64InvertFlags {
			break
		}
		cmp := v_2.Args[0]
		v.reset(OpARM64CSINV)
		v.AuxInt = opToAuxInt(arm64Invert(cc))
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (CSINV [cc] x _ flag)
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
	// match: (CSINV [cc] _ y flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: (Not y)
	for {
		cc := auxIntToOp(v.AuxInt)
		y := v_1
		flag := v_2
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.reset(OpNot)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CSNEG(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CSNEG [cc] x y (InvertFlags cmp))
	// result: (CSNEG [arm64Invert(cc)] x y cmp)
	for {
		cc := auxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != OpARM64InvertFlags {
			break
		}
		cmp := v_2.Args[0]
		v.reset(OpARM64CSNEG)
		v.AuxInt = opToAuxInt(arm64Invert(cc))
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (CSNEG [cc] x _ flag)
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
	// match: (CSNEG [cc] _ y flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: (NEG y)
	for {
		cc := auxIntToOp(v.AuxInt)
		y := v_1
		flag := v_2
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.reset(OpARM64NEG)
		v.AddArg(y)
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
	// result: (MOVDconst [int64(uint32(int32(c)/int32(d)))])
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
		v.AuxInt = int64ToAuxInt(int64(uint32(int32(c) / int32(d))))
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
	// match: (EON x0 x1:(RORconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (EONshiftRO x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64RORconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64EONshiftRO)
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
	// match: (EONshiftLL (SLLconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	// match: (EONshiftRA (SRAconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRAconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	// match: (EONshiftRL (SRLconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64EONshiftRO(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EONshiftRO x (MOVDconst [c]) [d])
	// result: (XORconst x [^rotateRight64(c, d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(^rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	// match: (EONshiftRO (RORconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64RORconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	b := v.Block
	// match: (Equal (CMPconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (Equal (TST x y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64TST, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPWconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (Equal (TSTWconst [int32(c)] y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPWconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (Equal (TSTW x y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64TSTW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (Equal (TSTconst [c] y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64TSTconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMP x z:(NEG y)))
	// cond: z.Uses == 1
	// result: (Equal (CMN x y))
	for {
		if v_0.Op != OpARM64CMP {
			break
		}
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
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPW x z:(NEG y)))
	// cond: z.Uses == 1
	// result: (Equal (CMNW x y))
	for {
		if v_0.Op != OpARM64CMPW {
			break
		}
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
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (Equal (CMNconst [c] y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMNconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPWconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (Equal (CMNWconst [int32(c)] y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMNWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (Equal (CMN x y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPWconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (Equal (CMNW x y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPconst [0] z:(MADD a x y)))
	// cond: z.Uses == 1
	// result: (Equal (CMN a (MUL <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPconst [0] z:(MSUB a x y)))
	// cond: z.Uses == 1
	// result: (Equal (CMP a (MUL <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPWconst [0] z:(MADDW a x y)))
	// cond: z.Uses == 1
	// result: (Equal (CMNW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPWconst [0] z:(MSUBW a x y)))
	// cond: z.Uses == 1
	// result: (Equal (CMPW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
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
	// cond: a.Block.Func.useFMA(v)
	// result: (FMADDD a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != OpARM64FMULD {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Block.Func.useFMA(v)) {
				continue
			}
			v.reset(OpARM64FMADDD)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (FADDD a (FNMULD x y))
	// cond: a.Block.Func.useFMA(v)
	// result: (FMSUBD a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != OpARM64FNMULD {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Block.Func.useFMA(v)) {
				continue
			}
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
	// cond: a.Block.Func.useFMA(v)
	// result: (FMADDS a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != OpARM64FMULS {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Block.Func.useFMA(v)) {
				continue
			}
			v.reset(OpARM64FMADDS)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (FADDS a (FNMULS x y))
	// cond: a.Block.Func.useFMA(v)
	// result: (FMSUBS a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != OpARM64FNMULS {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Block.Func.useFMA(v)) {
				continue
			}
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// match: (FMOVDload [off] {sym} (ADDshiftLL [3] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVDloadidx8 ptr idx mem)
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
		v.reset(OpARM64FMOVDloadidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (FMOVDload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// match: (FMOVDloadidx ptr (SLLconst [3] idx) mem)
	// result: (FMOVDloadidx8 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 3 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64FMOVDloadidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (FMOVDloadidx (SLLconst [3] idx) ptr mem)
	// result: (FMOVDloadidx8 ptr idx mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.reset(OpARM64FMOVDloadidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDloadidx8(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVDloadidx8 ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c<<3)
	// result: (FMOVDload ptr [int32(c)<<3] mem)
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
		v.reset(OpARM64FMOVDload)
		v.AuxInt = int32ToAuxInt(int32(c) << 3)
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// match: (FMOVDstore [off] {sym} (ADDshiftLL [3] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVDstoreidx8 ptr idx val mem)
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
		v.reset(OpARM64FMOVDstoreidx8)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (FMOVDstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// match: (FMOVDstoreidx ptr (SLLconst [3] idx) val mem)
	// result: (FMOVDstoreidx8 ptr idx val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 3 {
			break
		}
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.reset(OpARM64FMOVDstoreidx8)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (FMOVDstoreidx (SLLconst [3] idx) ptr val mem)
	// result: (FMOVDstoreidx8 ptr idx val mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.reset(OpARM64FMOVDstoreidx8)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDstoreidx8(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVDstoreidx8 ptr (MOVDconst [c]) val mem)
	// cond: is32Bit(c<<3)
	// result: (FMOVDstore [int32(c)<<3] ptr val mem)
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
		v.reset(OpARM64FMOVDstore)
		v.AuxInt = int32ToAuxInt(int32(c) << 3)
		v.AddArg3(ptr, val, mem)
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// match: (FMOVSload [off] {sym} (ADDshiftLL [2] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVSloadidx4 ptr idx mem)
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
		v.reset(OpARM64FMOVSloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (FMOVSload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// match: (FMOVSloadidx ptr (SLLconst [2] idx) mem)
	// result: (FMOVSloadidx4 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.reset(OpARM64FMOVSloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (FMOVSloadidx (SLLconst [2] idx) ptr mem)
	// result: (FMOVSloadidx4 ptr idx mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.reset(OpARM64FMOVSloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVSloadidx4(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVSloadidx4 ptr (MOVDconst [c]) mem)
	// cond: is32Bit(c<<2)
	// result: (FMOVSload ptr [int32(c)<<2] mem)
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
		v.reset(OpARM64FMOVSload)
		v.AuxInt = int32ToAuxInt(int32(c) << 2)
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// match: (FMOVSstore [off] {sym} (ADDshiftLL [2] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVSstoreidx4 ptr idx val mem)
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
		v.reset(OpARM64FMOVSstoreidx4)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (FMOVSstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// match: (FMOVSstoreidx ptr (SLLconst [2] idx) val mem)
	// result: (FMOVSstoreidx4 ptr idx val mem)
	for {
		ptr := v_0
		if v_1.Op != OpARM64SLLconst || auxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.reset(OpARM64FMOVSstoreidx4)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (FMOVSstoreidx (SLLconst [2] idx) ptr val mem)
	// result: (FMOVSstoreidx4 ptr idx val mem)
	for {
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.reset(OpARM64FMOVSstoreidx4)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVSstoreidx4(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVSstoreidx4 ptr (MOVDconst [c]) val mem)
	// cond: is32Bit(c<<2)
	// result: (FMOVSstore [int32(c)<<2] ptr val mem)
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
		v.reset(OpARM64FMOVSstore)
		v.AuxInt = int32ToAuxInt(int32(c) << 2)
		v.AddArg3(ptr, val, mem)
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
	// cond: a.Block.Func.useFMA(v)
	// result: (FMSUBD a x y)
	for {
		a := v_0
		if v_1.Op != OpARM64FMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Block.Func.useFMA(v)) {
			break
		}
		v.reset(OpARM64FMSUBD)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBD (FMULD x y) a)
	// cond: a.Block.Func.useFMA(v)
	// result: (FNMSUBD a x y)
	for {
		if v_0.Op != OpARM64FMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		if !(a.Block.Func.useFMA(v)) {
			break
		}
		v.reset(OpARM64FNMSUBD)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBD a (FNMULD x y))
	// cond: a.Block.Func.useFMA(v)
	// result: (FMADDD a x y)
	for {
		a := v_0
		if v_1.Op != OpARM64FNMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Block.Func.useFMA(v)) {
			break
		}
		v.reset(OpARM64FMADDD)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBD (FNMULD x y) a)
	// cond: a.Block.Func.useFMA(v)
	// result: (FNMADDD a x y)
	for {
		if v_0.Op != OpARM64FNMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		if !(a.Block.Func.useFMA(v)) {
			break
		}
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
	// cond: a.Block.Func.useFMA(v)
	// result: (FMSUBS a x y)
	for {
		a := v_0
		if v_1.Op != OpARM64FMULS {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Block.Func.useFMA(v)) {
			break
		}
		v.reset(OpARM64FMSUBS)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBS (FMULS x y) a)
	// cond: a.Block.Func.useFMA(v)
	// result: (FNMSUBS a x y)
	for {
		if v_0.Op != OpARM64FMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		if !(a.Block.Func.useFMA(v)) {
			break
		}
		v.reset(OpARM64FNMSUBS)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBS a (FNMULS x y))
	// cond: a.Block.Func.useFMA(v)
	// result: (FMADDS a x y)
	for {
		a := v_0
		if v_1.Op != OpARM64FNMULS {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Block.Func.useFMA(v)) {
			break
		}
		v.reset(OpARM64FMADDS)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBS (FNMULS x y) a)
	// cond: a.Block.Func.useFMA(v)
	// result: (FNMADDS a x y)
	for {
		if v_0.Op != OpARM64FNMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		if !(a.Block.Func.useFMA(v)) {
			break
		}
		v.reset(OpARM64FNMADDS)
		v.AddArg3(a, x, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64GreaterEqual(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (GreaterEqual (CMPconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqual (TST x y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TST, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPWconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (GreaterEqual (TSTWconst [int32(c)] y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPWconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqual (TSTW x y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TSTW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (GreaterEqual (TSTconst [c] y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TSTconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (GreaterEqualNoov (CMNconst [c] y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMNconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPWconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (GreaterEqualNoov (CMNWconst [int32(c)] y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMNWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqualNoov (CMN x y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPWconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqualNoov (CMNW x y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPconst [0] z:(MADD a x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqualNoov (CMN a (MUL <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPconst [0] z:(MSUB a x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqualNoov (CMP a (MUL <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPWconst [0] z:(MADDW a x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqualNoov (CMNW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPWconst [0] z:(MSUBW a x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqualNoov (CMPW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
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
func rewriteValueARM64_OpARM64GreaterEqualNoov(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterEqualNoov (FlagConstant [fc]))
	// result: (MOVDconst [b2i(fc.geNoov())])
	for {
		if v_0.Op != OpARM64FlagConstant {
			break
		}
		fc := auxIntToFlagConstant(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(fc.geNoov()))
		return true
	}
	// match: (GreaterEqualNoov (InvertFlags x))
	// result: (CSINC [OpARM64NotEqual] (LessThanNoov <typ.Bool> x) (MOVDconst [0]) x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64CSINC)
		v.AuxInt = opToAuxInt(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64LessThanNoov, typ.Bool)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v1.AuxInt = int64ToAuxInt(0)
		v.AddArg3(v0, v1, x)
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
	b := v.Block
	// match: (GreaterThan (CMPconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (GreaterThan (TST x y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, OpARM64TST, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterThan (CMPWconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (GreaterThan (TSTWconst [int32(c)] y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterThan (CMPWconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (GreaterThan (TSTW x y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, OpARM64TSTW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterThan (CMPconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (GreaterThan (TSTconst [c] y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, OpARM64TSTconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
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
func rewriteValueARM64_OpARM64LDP(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (LDP [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (LDP [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := auxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64LDP)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (LDP [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
	// result: (LDP [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64LDP)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64LessEqual(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (LessEqual (CMPconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (LessEqual (TST x y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TST, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (LessEqual (CMPWconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (LessEqual (TSTWconst [int32(c)] y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (LessEqual (CMPWconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (LessEqual (TSTW x y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TSTW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (LessEqual (CMPconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (LessEqual (TSTconst [c] y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TSTconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
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
	b := v.Block
	// match: (LessThan (CMPconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (LessThan (TST x y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, OpARM64TST, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPWconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (LessThan (TSTWconst [int32(c)] y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPWconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (LessThan (TSTW x y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, OpARM64TSTW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (LessThan (TSTconst [c] y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, OpARM64TSTconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (LessThanNoov (CMNconst [c] y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMNconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPWconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (LessThanNoov (CMNWconst [int32(c)] y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMNWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (LessThanNoov (CMN x y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPWconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (LessThanNoov (CMNW x y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPconst [0] z:(MADD a x y)))
	// cond: z.Uses == 1
	// result: (LessThanNoov (CMN a (MUL <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPconst [0] z:(MSUB a x y)))
	// cond: z.Uses == 1
	// result: (LessThanNoov (CMP a (MUL <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPWconst [0] z:(MADDW a x y)))
	// cond: z.Uses == 1
	// result: (LessThanNoov (CMNW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPWconst [0] z:(MSUBW a x y)))
	// cond: z.Uses == 1
	// result: (LessThanNoov (CMPW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
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
func rewriteValueARM64_OpARM64LessThanNoov(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LessThanNoov (FlagConstant [fc]))
	// result: (MOVDconst [b2i(fc.ltNoov())])
	for {
		if v_0.Op != OpARM64FlagConstant {
			break
		}
		fc := auxIntToFlagConstant(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(fc.ltNoov()))
		return true
	}
	// match: (LessThanNoov (InvertFlags x))
	// result: (CSEL0 [OpARM64NotEqual] (GreaterEqualNoov <typ.Bool> x) x)
	for {
		if v_0.Op != OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64CSEL0)
		v.AuxInt = opToAuxInt(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64GreaterEqualNoov, typ.Bool)
		v0.AddArg(x)
		v.AddArg2(v0, x)
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
func rewriteValueARM64_OpARM64LoweredPanicBoundsCR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsCR [kind] {p} (MOVDconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {PanicBoundsCC{Cx:p.C, Cy:c}} mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		p := auxToPanicBoundsC(v.Aux)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.reset(OpARM64LoweredPanicBoundsCC)
		v.AuxInt = int64ToAuxInt(kind)
		v.Aux = panicBoundsCCToAux(PanicBoundsCC{Cx: p.C, Cy: c})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64LoweredPanicBoundsRC(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRC [kind] {p} (MOVDconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {PanicBoundsCC{Cx:c, Cy:p.C}} mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		p := auxToPanicBoundsC(v.Aux)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.reset(OpARM64LoweredPanicBoundsCC)
		v.AuxInt = int64ToAuxInt(kind)
		v.Aux = panicBoundsCCToAux(PanicBoundsCC{Cx: c, Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64LoweredPanicBoundsRR(v *Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRR [kind] x (MOVDconst [c]) mem)
	// result: (LoweredPanicBoundsRC [kind] x {PanicBoundsC{C:c}} mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.reset(OpARM64LoweredPanicBoundsRC)
		v.AuxInt = int64ToAuxInt(kind)
		v.Aux = panicBoundsCToAux(PanicBoundsC{C: c})
		v.AddArg2(x, mem)
		return true
	}
	// match: (LoweredPanicBoundsRR [kind] (MOVDconst [c]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {PanicBoundsC{C:c}} y mem)
	for {
		kind := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		y := v_1
		mem := v_2
		v.reset(OpARM64LoweredPanicBoundsCR)
		v.AuxInt = int64ToAuxInt(kind)
		v.Aux = panicBoundsCToAux(PanicBoundsC{C: c})
		v.AddArg2(y, mem)
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
	// cond: isPowerOfTwo(c)
	// result: (ADDshiftLL a x [log64(c)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c-1) && c>=3
	// result: (ADD a (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo(c-1) && c >= 3) {
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
	// cond: isPowerOfTwo(c+1) && c>=7
	// result: (SUB a (SUBshiftLL <x.Type> x x [log64(c+1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo(c+1) && c >= 7) {
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
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
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
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
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
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%7 == 0 && isPowerOfTwo(c/7)) {
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
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
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
	// cond: isPowerOfTwo(c)
	// result: (ADDshiftLL a x [log64(c)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c-1) && c>=3
	// result: (ADD a (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo(c-1) && c >= 3) {
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
	// cond: isPowerOfTwo(c+1) && c>=7
	// result: (SUB a (SUBshiftLL <x.Type> x x [log64(c+1)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo(c+1) && c >= 7) {
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
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
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
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
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
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%7 == 0 && isPowerOfTwo(c/7)) {
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
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
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
	// result: (MOVWUreg (SUB <a.Type> a x))
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
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a _ (MOVDconst [c]))
	// cond: int32(c)==0
	// result: (MOVWUreg a)
	for {
		a := v_0
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v.AddArg(a)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: int32(c)==1
	// result: (MOVWUreg (ADD <a.Type> a x))
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
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a x [log64(c)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c))
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c-1) && int32(c)>=3
	// result: (MOVWUreg (ADD <a.Type> a (ADDshiftLL <x.Type> x x [log64(c-1)])))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, a.Type)
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(log64(c - 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c+1) && int32(c)>=7
	// result: (MOVWUreg (SUB <a.Type> a (SUBshiftLL <x.Type> x x [log64(c+1)])))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, a.Type)
		v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(log64(c + 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 3))
		v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 5))
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 7))
		v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 9))
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: int32(c)==-1
	// result: (MOVWUreg (SUB <a.Type> a x))
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
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) _)
	// cond: int32(c)==0
	// result: (MOVWUreg a)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v.AddArg(a)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: int32(c)==1
	// result: (MOVWUreg (ADD <a.Type> a x))
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
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a x [log64(c)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c))
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c-1) && int32(c)>=3
	// result: (MOVWUreg (ADD <a.Type> a (ADDshiftLL <x.Type> x x [log64(c-1)])))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, a.Type)
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(log64(c - 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c+1) && int32(c)>=7
	// result: (MOVWUreg (SUB <a.Type> a (SUBshiftLL <x.Type> x x [log64(c+1)])))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, a.Type)
		v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(log64(c + 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 3))
		v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 5))
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 7))
		v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 9))
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW (MOVDconst [c]) x y)
	// result: (MOVWUreg (ADDconst <x.Type> [c] (MULW <x.Type> x y)))
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		y := v_2
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDconst, x.Type)
		v0.AuxInt = int64ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVWUreg (ADDconst <a.Type> [c*d] a))
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
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDconst, a.Type)
		v0.AuxInt = int64ToAuxInt(c * d)
		v0.AddArg(a)
		v.AddArg(v0)
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
	// cond: isPowerOfTwo(c)
	// result: (NEG (SLLconst <x.Type> [log64(c)] x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo(c)) {
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
	// cond: isPowerOfTwo(c-1) && c >= 3
	// result: (NEG (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo(c-1) && c >= 3) {
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
	// cond: isPowerOfTwo(c+1) && c >= 7
	// result: (NEG (ADDshiftLL <x.Type> (NEG <x.Type> x) x [log64(c+1)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo(c+1) && c >= 7) {
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
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (SLLconst <x.Type> [log64(c/3)] (SUBshiftLL <x.Type> x x [2]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%3 == 0 && isPowerOfTwo(c/3)) {
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
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (NEG (SLLconst <x.Type> [log64(c/5)] (ADDshiftLL <x.Type> x x [2])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%5 == 0 && isPowerOfTwo(c/5)) {
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
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (SLLconst <x.Type> [log64(c/7)] (SUBshiftLL <x.Type> x x [3]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%7 == 0 && isPowerOfTwo(c/7)) {
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
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (NEG (SLLconst <x.Type> [log64(c/9)] (ADDshiftLL <x.Type> x x [3])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%9 == 0 && isPowerOfTwo(c/9)) {
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
	// result: (MOVWUreg x)
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
			v.reset(OpARM64MOVWUreg)
			v.AddArg(x)
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
	// result: (MOVWUreg (NEG <x.Type> x))
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
			v.reset(OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (NEG (SLLconst <x.Type> [log64(c)] x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo(c)) {
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
	// cond: isPowerOfTwo(c-1) && int32(c) >= 3
	// result: (MOVWUreg (NEG <x.Type> (ADDshiftLL <x.Type> x x [log64(c-1)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
				continue
			}
			v.reset(OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
			v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v1.AuxInt = int64ToAuxInt(log64(c - 1))
			v1.AddArg2(x, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: isPowerOfTwo(c+1) && int32(c) >= 7
	// result: (MOVWUreg (NEG <x.Type> (ADDshiftLL <x.Type> (NEG <x.Type> x) x [log64(c+1)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
				continue
			}
			v.reset(OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
			v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v1.AuxInt = int64ToAuxInt(log64(c + 1))
			v2 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
			v2.AddArg(x)
			v1.AddArg2(v2, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (MOVWUreg (SLLconst <x.Type> [log64(c/3)] (SUBshiftLL <x.Type> x x [2])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
				continue
			}
			v.reset(OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
			v0.AuxInt = int64ToAuxInt(log64(c / 3))
			v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
			v1.AuxInt = int64ToAuxInt(2)
			v1.AddArg2(x, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (MOVWUreg (NEG <x.Type> (SLLconst <x.Type> [log64(c/5)] (ADDshiftLL <x.Type> x x [2]))))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
				continue
			}
			v.reset(OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
			v1 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
			v1.AuxInt = int64ToAuxInt(log64(c / 5))
			v2 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v2.AuxInt = int64ToAuxInt(2)
			v2.AddArg2(x, x)
			v1.AddArg(v2)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (MOVWUreg (SLLconst <x.Type> [log64(c/7)] (SUBshiftLL <x.Type> x x [3])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
				continue
			}
			v.reset(OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
			v0.AuxInt = int64ToAuxInt(log64(c / 7))
			v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
			v1.AuxInt = int64ToAuxInt(3)
			v1.AddArg2(x, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (MOVWUreg (NEG <x.Type> (SLLconst <x.Type> [log64(c/9)] (ADDshiftLL <x.Type> x x [3]))))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
				continue
			}
			v.reset(OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
			v1 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
			v1.AuxInt = int64ToAuxInt(log64(c / 9))
			v2 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
			v2.AuxInt = int64ToAuxInt(3)
			v2.AddArg2(x, x)
			v1.AddArg(v2)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [int64(uint32(-c*d))])
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
			v.AuxInt = int64ToAuxInt(int64(uint32(-c * d)))
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
	// result: (MOVDconst [int64(uint32(int32(c)%int32(d)))])
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
		v.AuxInt = int64ToAuxInt(int64(uint32(int32(c) % int32(d))))
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64MOVBUload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
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
	return false
}
func rewriteValueARM64_OpARM64MOVBUreg(v *Value) bool {
	v_0 := v.Args[0]
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
	// cond: v.Type.Size() <= 1
	// result: x
	for {
		x := v_0
		if !(v.Type.Size() <= 1) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBUreg (SLLconst [lc] x))
	// cond: lc >= 8
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		if !(lc >= 8) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MOVBUreg (SLLconst [lc] x))
	// cond: lc < 8
	// result: (UBFIZ [armBFAuxInt(lc, 8-lc)] x)
	for {
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < 8) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc, 8-lc))
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (SRLconst [rc] x))
	// cond: rc < 8
	// result: (UBFX [armBFAuxInt(rc, 8)] x)
	for {
		if v_0.Op != OpARM64SRLconst {
			break
		}
		rc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(rc < 8) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(rc, 8))
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (UBFX [bfc] x))
	// cond: bfc.width() <= 8
	// result: (UBFX [bfc] x)
	for {
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(bfc.width() <= 8) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(bfc)
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64MOVBload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVDconst [int64(int8(read8(sym, int64(off))))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int8(read8(sym, int64(off)))))
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
	return false
}
func rewriteValueARM64_OpARM64MOVBreg(v *Value) bool {
	v_0 := v.Args[0]
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
	// match: (MOVBreg x)
	// cond: v.Type.Size() <= 1
	// result: x
	for {
		x := v_0
		if !(v.Type.Size() <= 1) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVBreg <t> (ANDconst x [c]))
	// cond: uint64(c) & uint64(0xffffffffffffff80) == 0
	// result: (ANDconst <t> x [c])
	for {
		t := v.Type
		if v_0.Op != OpARM64ANDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(uint64(c)&uint64(0xffffffffffffff80) == 0) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.Type = t
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
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
	// match: (MOVBreg (SBFX [bfc] x))
	// cond: bfc.width() <= 8
	// result: (SBFX [bfc] x)
	for {
		if v_0.Op != OpARM64SBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(bfc.width() <= 8) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = arm64BitFieldToAuxInt(bfc)
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
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
	return false
}
func rewriteValueARM64_OpARM64MOVBstoreidx(v *Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64MOVDload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
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
	return false
}
func rewriteValueARM64_OpARM64MOVDnop(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDnop (MOVDconst [c]))
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
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
	return false
}
func rewriteValueARM64_OpARM64MOVHUload(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64MOVHUload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
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
	return false
}
func rewriteValueARM64_OpARM64MOVHUreg(v *Value) bool {
	v_0 := v.Args[0]
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
	// match: (MOVHUreg x)
	// cond: v.Type.Size() <= 2
	// result: x
	for {
		x := v_0
		if !(v.Type.Size() <= 2) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVHUreg (SLLconst [lc] x))
	// cond: lc >= 16
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		if !(lc >= 16) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MOVHUreg (SLLconst [lc] x))
	// cond: lc < 16
	// result: (UBFIZ [armBFAuxInt(lc, 16-lc)] x)
	for {
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < 16) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc, 16-lc))
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (SRLconst [rc] x))
	// cond: rc < 16
	// result: (UBFX [armBFAuxInt(rc, 16)] x)
	for {
		if v_0.Op != OpARM64SRLconst {
			break
		}
		rc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(rc < 16) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(rc, 16))
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (UBFX [bfc] x))
	// cond: bfc.width() <= 16
	// result: (UBFX [bfc] x)
	for {
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(bfc.width() <= 16) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(bfc)
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64MOVHload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVDconst [int64(int16(read16(sym, int64(off), config.ctxt.Arch.ByteOrder)))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int16(read16(sym, int64(off), config.ctxt.Arch.ByteOrder))))
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
	return false
}
func rewriteValueARM64_OpARM64MOVHreg(v *Value) bool {
	v_0 := v.Args[0]
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
	// match: (MOVHreg x)
	// cond: v.Type.Size() <= 2
	// result: x
	for {
		x := v_0
		if !(v.Type.Size() <= 2) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVHreg <t> (ANDconst x [c]))
	// cond: uint64(c) & uint64(0xffffffffffff8000) == 0
	// result: (ANDconst <t> x [c])
	for {
		t := v.Type
		if v_0.Op != OpARM64ANDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(uint64(c)&uint64(0xffffffffffff8000) == 0) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.Type = t
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
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
	// match: (MOVHreg (SBFX [bfc] x))
	// cond: bfc.width() <= 16
	// result: (SBFX [bfc] x)
	for {
		if v_0.Op != OpARM64SBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(bfc.width() <= 16) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = arm64BitFieldToAuxInt(bfc)
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64MOVWUload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
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
	return false
}
func rewriteValueARM64_OpARM64MOVWUreg(v *Value) bool {
	v_0 := v.Args[0]
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
	// match: (MOVWUreg x)
	// cond: v.Type.Size() <= 4
	// result: x
	for {
		x := v_0
		if !(v.Type.Size() <= 4) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWUreg (SLLconst [lc] x))
	// cond: lc >= 32
	// result: (MOVDconst [0])
	for {
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		if !(lc >= 32) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (MOVWUreg (SLLconst [lc] x))
	// cond: lc < 32
	// result: (UBFIZ [armBFAuxInt(lc, 32-lc)] x)
	for {
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < 32) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc, 32-lc))
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (SRLconst [rc] x))
	// cond: rc < 32
	// result: (UBFX [armBFAuxInt(rc, 32)] x)
	for {
		if v_0.Op != OpARM64SRLconst {
			break
		}
		rc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(rc < 32) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(rc, 32))
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (UBFX [bfc] x))
	// cond: bfc.width() <= 32
	// result: (UBFX [bfc] x)
	for {
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(bfc.width() <= 32) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(bfc)
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64MOVWload)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVDconst [int64(int32(read32(sym, int64(off), config.ctxt.Arch.ByteOrder)))])
	for {
		off := auxIntToInt32(v.AuxInt)
		sym := auxToSym(v.Aux)
		if v_0.Op != OpSB || !(symIsRO(sym)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(int64(int32(read32(sym, int64(off), config.ctxt.Arch.ByteOrder))))
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
	return false
}
func rewriteValueARM64_OpARM64MOVWreg(v *Value) bool {
	v_0 := v.Args[0]
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
	// match: (MOVWreg x)
	// cond: v.Type.Size() <= 4
	// result: x
	for {
		x := v_0
		if !(v.Type.Size() <= 4) {
			break
		}
		v.copyOf(x)
		return true
	}
	// match: (MOVWreg <t> (ANDconst x [c]))
	// cond: uint64(c) & uint64(0xffffffff80000000) == 0
	// result: (ANDconst <t> x [c])
	for {
		t := v.Type
		if v_0.Op != OpARM64ANDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(uint64(c)&uint64(0xffffffff80000000) == 0) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.Type = t
		v.AuxInt = int64ToAuxInt(c)
		v.AddArg(x)
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
	// match: (MOVWreg (SBFX [bfc] x))
	// cond: bfc.width() <= 32
	// result: (SBFX [bfc] x)
	for {
		if v_0.Op != OpARM64SBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(bfc.width() <= 32) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = arm64BitFieldToAuxInt(bfc)
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(off1 + off2)
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
	// cond: isPowerOfTwo(c)
	// result: (SUBshiftLL a x [log64(c)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c-1) && c>=3
	// result: (SUB a (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo(c-1) && c >= 3) {
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
	// cond: isPowerOfTwo(c+1) && c>=7
	// result: (ADD a (SUBshiftLL <x.Type> x x [log64(c+1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo(c+1) && c >= 7) {
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
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
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
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
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
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%7 == 0 && isPowerOfTwo(c/7)) {
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
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
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
	// cond: isPowerOfTwo(c)
	// result: (SUBshiftLL a x [log64(c)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = int64ToAuxInt(log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c-1) && c>=3
	// result: (SUB a (ADDshiftLL <x.Type> x x [log64(c-1)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo(c-1) && c >= 3) {
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
	// cond: isPowerOfTwo(c+1) && c>=7
	// result: (ADD a (SUBshiftLL <x.Type> x x [log64(c+1)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo(c+1) && c >= 7) {
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
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
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
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
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
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%7 == 0 && isPowerOfTwo(c/7)) {
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
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)])
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
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
	// result: (MOVWUreg (ADD <a.Type> a x))
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
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a _ (MOVDconst [c]))
	// cond: int32(c)==0
	// result: (MOVWUreg a)
	for {
		a := v_0
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v.AddArg(a)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: int32(c)==1
	// result: (MOVWUreg (SUB <a.Type> a x))
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
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a x [log64(c)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c))
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c-1) && int32(c)>=3
	// result: (MOVWUreg (SUB <a.Type> a (ADDshiftLL <x.Type> x x [log64(c-1)])))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, a.Type)
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(log64(c - 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c+1) && int32(c)>=7
	// result: (MOVWUreg (ADD <a.Type> a (SUBshiftLL <x.Type> x x [log64(c+1)])))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, a.Type)
		v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(log64(c + 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 3))
		v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 5))
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 7))
		v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_2.AuxInt)
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 9))
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: int32(c)==-1
	// result: (MOVWUreg (ADD <a.Type> a x))
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
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) _)
	// cond: int32(c)==0
	// result: (MOVWUreg a)
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v.AddArg(a)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: int32(c)==1
	// result: (MOVWUreg (SUB <a.Type> a x))
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
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a x [log64(c)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c))
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c-1) && int32(c)>=3
	// result: (MOVWUreg (SUB <a.Type> a (ADDshiftLL <x.Type> x x [log64(c-1)])))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, a.Type)
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(log64(c - 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c+1) && int32(c)>=7
	// result: (MOVWUreg (ADD <a.Type> a (SUBshiftLL <x.Type> x x [log64(c+1)])))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, a.Type)
		v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(log64(c + 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [2]) [log64(c/3)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 3))
		v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [2]) [log64(c/5)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 5))
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [3]) [log64(c/7)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 7))
		v1 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [3]) [log64(c/9)]))
	for {
		a := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = int64ToAuxInt(log64(c / 9))
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW (MOVDconst [c]) x y)
	// result: (MOVWUreg (ADDconst <x.Type> [c] (MNEGW <x.Type> x y)))
	for {
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		y := v_2
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64ADDconst, x.Type)
		v0.AuxInt = int64ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, OpARM64MNEGW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVWUreg (SUBconst <a.Type> [c*d] a))
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
		v.reset(OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, OpARM64SUBconst, a.Type)
		v0.AuxInt = int64ToAuxInt(c * d)
		v0.AddArg(a)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MUL(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
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
	// cond: canMulStrengthReduce(config, c)
	// result: {mulStrengthReduce(v, x, c)}
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(canMulStrengthReduce(config, c)) {
				continue
			}
			v.copyOf(mulStrengthReduce(v, x, c))
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
	// match: (MUL r:(MOVWUreg x) s:(MOVWUreg y))
	// cond: r.Uses == 1 && s.Uses == 1
	// result: (UMULL x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			r := v_0
			if r.Op != OpARM64MOVWUreg {
				continue
			}
			x := r.Args[0]
			s := v_1
			if s.Op != OpARM64MOVWUreg {
				continue
			}
			y := s.Args[0]
			if !(r.Uses == 1 && s.Uses == 1) {
				continue
			}
			v.reset(OpARM64UMULL)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (MUL r:(MOVWreg x) s:(MOVWreg y))
	// cond: r.Uses == 1 && s.Uses == 1
	// result: (MULL x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			r := v_0
			if r.Op != OpARM64MOVWreg {
				continue
			}
			x := r.Args[0]
			s := v_1
			if s.Op != OpARM64MOVWreg {
				continue
			}
			y := s.Args[0]
			if !(r.Uses == 1 && s.Uses == 1) {
				continue
			}
			v.reset(OpARM64MULL)
			v.AddArg2(x, y)
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
	config := b.Func.Config
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
	// result: (MOVWUreg x)
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
			v.reset(OpARM64MOVWUreg)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: v.Type.Size() <= 4 && canMulStrengthReduce32(config, int32(c))
	// result: {mulStrengthReduce32(v, x, int32(c))}
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != OpARM64MOVDconst {
				continue
			}
			c := auxIntToInt64(v_1.AuxInt)
			if !(v.Type.Size() <= 4 && canMulStrengthReduce32(config, int32(c))) {
				continue
			}
			v.copyOf(mulStrengthReduce32(v, x, int32(c)))
			return true
		}
		break
	}
	// match: (MULW (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [int64(uint32(c*d))])
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
			v.AuxInt = int64ToAuxInt(int64(uint32(c * d)))
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
	// match: (MVN x:(RORconst [c] y))
	// cond: clobberIfDead(x)
	// result: (MVNshiftRO [c] y)
	for {
		x := v_0
		if x.Op != OpARM64RORconst {
			break
		}
		c := auxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64MVNshiftRO)
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
func rewriteValueARM64_OpARM64MVNshiftRO(v *Value) bool {
	v_0 := v.Args[0]
	// match: (MVNshiftRO (MOVDconst [c]) [d])
	// result: (MOVDconst [^rotateRight64(c, d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(^rotateRight64(c, d))
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
	// cond: v.Type.Size() <= 4
	// result: (MNEGW x y)
	for {
		if v_0.Op != OpARM64MULW {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(v.Type.Size() <= 4) {
			break
		}
		v.reset(OpARM64MNEGW)
		v.AddArg2(x, y)
		return true
	}
	// match: (NEG (SUB x y))
	// result: (SUB y x)
	for {
		if v_0.Op != OpARM64SUB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64SUB)
		v.AddArg2(y, x)
		return true
	}
	// match: (NEG (NEG x))
	// result: x
	for {
		if v_0.Op != OpARM64NEG {
			break
		}
		x := v_0.Args[0]
		v.copyOf(x)
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
	b := v.Block
	// match: (NotEqual (CMPconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (TST x y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TST, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPWconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (NotEqual (TSTWconst [int32(c)] y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPWconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (TSTW x y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TSTW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (NotEqual (TSTconst [c] y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TSTconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMP x z:(NEG y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMN x y))
	for {
		if v_0.Op != OpARM64CMP {
			break
		}
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
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPW x z:(NEG y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMNW x y))
	for {
		if v_0.Op != OpARM64CMPW {
			break
		}
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
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (NotEqual (CMNconst [c] y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMNconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPWconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (NotEqual (CMNWconst [int32(c)] y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMNWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMN x y))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPWconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMNW x y))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPconst [0] z:(MADD a x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMN a (MUL <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPconst [0] z:(MSUB a x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMP a (MUL <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPconst || auxIntToInt64(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPWconst [0] z:(MADDW a x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMNW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPWconst [0] z:(MSUBW a x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMPW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != OpARM64CMPWconst || auxIntToInt32(v_0.AuxInt) != 0 {
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
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
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
	// match: (OR x0 x1:(RORconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORshiftRO x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64RORconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64ORshiftRO)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (OR (UBFIZ [bfc] x) (ANDconst [ac] y))
	// cond: ac == ^((1<<uint(bfc.width())-1) << uint(bfc.lsb()))
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
			if !(ac == ^((1<<uint(bfc.width()) - 1) << uint(bfc.lsb()))) {
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
	// cond: ac == ^(1<<uint(bfc.width())-1)
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
			if !(ac == ^(1<<uint(bfc.width()) - 1)) {
				continue
			}
			v.reset(OpARM64BFXIL)
			v.AuxInt = arm64BitFieldToAuxInt(bfc)
			v.AddArg2(y, x)
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
	// match: (ORN x0 x1:(RORconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORNshiftRO x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != OpARM64RORconst {
			break
		}
		c := auxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORNshiftRO)
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
	// match: (ORNshiftLL (SLLconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	// match: (ORNshiftRA (SRAconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRAconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	// match: (ORNshiftRL (SRLconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORNshiftRO(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORNshiftRO x (MOVDconst [c]) [d])
	// result: (ORconst x [^rotateRight64(c, d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(^rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	// match: (ORNshiftRO (RORconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64RORconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	// match: (ORshiftLL y:(SLLconst x [c]) x [c])
	// result: y
	for {
		c := auxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != OpARM64SLLconst || auxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
			break
		}
		v.copyOf(y)
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
	// match: (ORshiftLL [8] (UBFX [armBFAuxInt(8, 24)] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff
	// result: (REV16W x)
	for {
		if auxIntToInt64(v.AuxInt) != 8 || v_0.Op != OpARM64UBFX || auxIntToArm64BitField(v_0.AuxInt) != armBFAuxInt(8, 24) {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64ANDconst {
			break
		}
		c1 := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != OpARM64ANDconst {
			break
		}
		c2 := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff) {
			break
		}
		v.reset(OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL [8] (SRLconst [8] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: (uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff)
	// result: (REV16 x)
	for {
		if auxIntToInt64(v.AuxInt) != 8 || v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != 8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64ANDconst {
			break
		}
		c1 := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != OpARM64ANDconst {
			break
		}
		c2 := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff) {
			break
		}
		v.reset(OpARM64REV16)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL [8] (SRLconst [8] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: (uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff)
	// result: (REV16 (ANDconst <x.Type> [0xffffffff] x))
	for {
		if auxIntToInt64(v.AuxInt) != 8 || v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != 8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64ANDconst {
			break
		}
		c1 := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != OpARM64ANDconst {
			break
		}
		c2 := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff) {
			break
		}
		v.reset(OpARM64REV16)
		v0 := b.NewValue0(v.Pos, OpARM64ANDconst, x.Type)
		v0.AuxInt = int64ToAuxInt(0xffffffff)
		v0.AddArg(x)
		v.AddArg(v0)
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
	// match: (ORshiftLL [s] (ANDconst [xc] x) (ANDconst [yc] y))
	// cond: xc == ^(yc << s) && yc & (yc+1) == 0 && yc > 0 && s+log64(yc+1) <= 64
	// result: (BFI [armBFAuxInt(s, log64(yc+1))] x y)
	for {
		s := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64ANDconst {
			break
		}
		xc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if v_1.Op != OpARM64ANDconst {
			break
		}
		yc := auxIntToInt64(v_1.AuxInt)
		y := v_1.Args[0]
		if !(xc == ^(yc<<s) && yc&(yc+1) == 0 && yc > 0 && s+log64(yc+1) <= 64) {
			break
		}
		v.reset(OpARM64BFI)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(s, log64(yc+1)))
		v.AddArg2(x, y)
		return true
	}
	// match: (ORshiftLL [sc] (UBFX [bfc] x) (SRLconst [sc] y))
	// cond: sc == bfc.width()
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
		if !(sc == bfc.width()) {
			break
		}
		v.reset(OpARM64BFXIL)
		v.AuxInt = arm64BitFieldToAuxInt(bfc)
		v.AddArg2(y, x)
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
	// match: (ORshiftRA y:(SRAconst x [c]) x [c])
	// result: y
	for {
		c := auxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != OpARM64SRAconst || auxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
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
	// match: (ORshiftRL y:(SRLconst x [c]) x [c])
	// result: y
	for {
		c := auxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != OpARM64SRLconst || auxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
			break
		}
		v.copyOf(y)
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
func rewriteValueARM64_OpARM64ORshiftRO(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ORshiftRO (MOVDconst [c]) x [d])
	// result: (ORconst [c] (RORconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64RORconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRO x (MOVDconst [c]) [d])
	// result: (ORconst x [rotateRight64(c, d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64ORconst)
		v.AuxInt = int64ToAuxInt(rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRO y:(RORconst x [c]) x [c])
	// result: y
	for {
		c := auxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != OpARM64RORconst || auxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
			break
		}
		v.copyOf(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64REV(v *Value) bool {
	v_0 := v.Args[0]
	// match: (REV (REV p))
	// result: p
	for {
		if v_0.Op != OpARM64REV {
			break
		}
		p := v_0.Args[0]
		v.copyOf(p)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64REVW(v *Value) bool {
	v_0 := v.Args[0]
	// match: (REVW (REVW p))
	// result: p
	for {
		if v_0.Op != OpARM64REVW {
			break
		}
		p := v_0.Args[0]
		v.copyOf(p)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ROR(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROR x (MOVDconst [c]))
	// result: (RORconst x [c&63])
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64RORconst)
		v.AuxInt = int64ToAuxInt(c & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64RORW(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RORW x (MOVDconst [c]))
	// result: (RORWconst x [c&31])
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64RORWconst)
		v.AuxInt = int64ToAuxInt(c & 31)
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
func rewriteValueARM64_OpARM64SBFX(v *Value) bool {
	v_0 := v.Args[0]
	// match: (SBFX [bfc] s:(SLLconst [sc] x))
	// cond: s.Uses == 1 && sc <= bfc.lsb()
	// result: (SBFX [armBFAuxInt(bfc.lsb() - sc, bfc.width())] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		s := v_0
		if s.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(s.AuxInt)
		x := s.Args[0]
		if !(s.Uses == 1 && sc <= bfc.lsb()) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.lsb()-sc, bfc.width()))
		v.AddArg(x)
		return true
	}
	// match: (SBFX [bfc] s:(SLLconst [sc] x))
	// cond: s.Uses == 1 && sc > bfc.lsb()
	// result: (SBFIZ [armBFAuxInt(sc - bfc.lsb(), bfc.width() - (sc-bfc.lsb()))] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		s := v_0
		if s.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(s.AuxInt)
		x := s.Args[0]
		if !(s.Uses == 1 && sc > bfc.lsb()) {
			break
		}
		v.reset(OpARM64SBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc-bfc.lsb(), bfc.width()-(sc-bfc.lsb())))
		v.AddArg(x)
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
	// match: (SLL x (ANDconst [63] y))
	// result: (SLL x y)
	for {
		x := v_0
		if v_1.Op != OpARM64ANDconst || auxIntToInt64(v_1.AuxInt) != 63 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpARM64SLL)
		v.AddArg2(x, y)
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
	// match: (SLLconst [lc] (MOVWreg x))
	// result: (SBFIZ [armBFAuxInt(lc, min(32, 64-lc))] x)
	for {
		lc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVWreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64SBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc, min(32, 64-lc)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [lc] (MOVHreg x))
	// result: (SBFIZ [armBFAuxInt(lc, min(16, 64-lc))] x)
	for {
		lc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVHreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64SBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc, min(16, 64-lc)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [lc] (MOVBreg x))
	// result: (SBFIZ [armBFAuxInt(lc, min(8, 64-lc))] x)
	for {
		lc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVBreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64SBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc, min(8, 64-lc)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [lc] (MOVWUreg x))
	// result: (UBFIZ [armBFAuxInt(lc, min(32, 64-lc))] x)
	for {
		lc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc, min(32, 64-lc)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [lc] (MOVHUreg x))
	// result: (UBFIZ [armBFAuxInt(lc, min(16, 64-lc))] x)
	for {
		lc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc, min(16, 64-lc)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [lc] (MOVBUreg x))
	// result: (UBFIZ [armBFAuxInt(lc, min(8, 64-lc))] x)
	for {
		lc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(lc, min(8, 64-lc)))
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
	// match: (SLLconst [sc] (UBFIZ [bfc] x))
	// cond: sc+bfc.width()+bfc.lsb() < 64
	// result: (UBFIZ [armBFAuxInt(bfc.lsb()+sc, bfc.width())] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc+bfc.width()+bfc.lsb() < 64) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.lsb()+sc, bfc.width()))
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
	// match: (SRA x (ANDconst [63] y))
	// result: (SRA x y)
	for {
		x := v_0
		if v_1.Op != OpARM64ANDconst || auxIntToInt64(v_1.AuxInt) != 63 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpARM64SRA)
		v.AddArg2(x, y)
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
	// cond: sc < bfc.lsb()
	// result: (SBFIZ [armBFAuxInt(bfc.lsb()-sc, bfc.width())] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SBFIZ {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.lsb()) {
			break
		}
		v.reset(OpARM64SBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.lsb()-sc, bfc.width()))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [sc] (SBFIZ [bfc] x))
	// cond: sc >= bfc.lsb() && sc < bfc.lsb()+bfc.width()
	// result: (SBFX [armBFAuxInt(sc-bfc.lsb(), bfc.lsb()+bfc.width()-sc)] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SBFIZ {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc >= bfc.lsb() && sc < bfc.lsb()+bfc.width()) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc-bfc.lsb(), bfc.lsb()+bfc.width()-sc))
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
	// match: (SRL x (ANDconst [63] y))
	// result: (SRL x y)
	for {
		x := v_0
		if v_1.Op != OpARM64ANDconst || auxIntToInt64(v_1.AuxInt) != 63 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpARM64SRL)
		v.AddArg2(x, y)
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
	// match: (SRLconst [rc] (MOVWUreg x))
	// cond: rc >= 32
	// result: (MOVDconst [0])
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVWUreg {
			break
		}
		if !(rc >= 32) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SRLconst [rc] (MOVHUreg x))
	// cond: rc >= 16
	// result: (MOVDconst [0])
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVHUreg {
			break
		}
		if !(rc >= 16) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	// match: (SRLconst [rc] (MOVBUreg x))
	// cond: rc >= 8
	// result: (MOVDconst [0])
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVBUreg {
			break
		}
		if !(rc >= 8) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
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
	// match: (SRLconst [rc] (MOVWUreg x))
	// cond: rc < 32
	// result: (UBFX [armBFAuxInt(rc, 32-rc)] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 32) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(rc, 32-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [rc] (MOVHUreg x))
	// cond: rc < 16
	// result: (UBFX [armBFAuxInt(rc, 16-rc)] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 16) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(rc, 16-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [rc] (MOVBUreg x))
	// cond: rc < 8
	// result: (UBFX [armBFAuxInt(rc, 8-rc)] x)
	for {
		rc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 8) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(rc, 8-rc))
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
	// match: (SRLconst [sc] (UBFX [bfc] x))
	// cond: sc < bfc.width()
	// result: (UBFX [armBFAuxInt(bfc.lsb()+sc, bfc.width()-sc)] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.width()) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.lsb()+sc, bfc.width()-sc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFIZ [bfc] x))
	// cond: sc == bfc.lsb()
	// result: (ANDconst [1<<uint(bfc.width())-1] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc == bfc.lsb()) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(1<<uint(bfc.width()) - 1)
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFIZ [bfc] x))
	// cond: sc < bfc.lsb()
	// result: (UBFIZ [armBFAuxInt(bfc.lsb()-sc, bfc.width())] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.lsb()) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.lsb()-sc, bfc.width()))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFIZ [bfc] x))
	// cond: sc > bfc.lsb() && sc < bfc.lsb()+bfc.width()
	// result: (UBFX [armBFAuxInt(sc-bfc.lsb(), bfc.lsb()+bfc.width()-sc)] x)
	for {
		sc := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := auxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc > bfc.lsb() && sc < bfc.lsb()+bfc.width()) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc-bfc.lsb(), bfc.lsb()+bfc.width()-sc))
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
	// cond: is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(is32Bit(int64(off1)+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64STP)
		v.AuxInt = int32ToAuxInt(off1 + int32(off2))
		v.Aux = symToAux(sym)
		v.AddArg4(ptr, val1, val2, mem)
		return true
	}
	// match: (STP [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val1 val2 mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)
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
		if !(canMergeSym(sym1, sym2) && is32Bit(int64(off1)+int64(off2)) && (ptr.Op != OpSB || !config.ctxt.Flag_dynlink)) {
			break
		}
		v.reset(OpARM64STP)
		v.AuxInt = int32ToAuxInt(off1 + off2)
		v.Aux = symToAux(mergeSym(sym1, sym2))
		v.AddArg4(ptr, val1, val2, mem)
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
	// cond: v.Type.Size() <= 4 && l.Uses==1 && clobber(l)
	// result: (MSUBW a x y)
	for {
		a := v_0
		l := v_1
		if l.Op != OpARM64MULW {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(v.Type.Size() <= 4 && l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MSUBW)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (SUB a l:(MNEGW x y))
	// cond: v.Type.Size() <= 4 && l.Uses==1 && clobber(l)
	// result: (MADDW a x y)
	for {
		a := v_0
		l := v_1
		if l.Op != OpARM64MNEGW {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(v.Type.Size() <= 4 && l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MADDW)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (SUB <t> a p:(ADDconst [c] m:(MUL _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (SUBconst [c] (SUB <v.Type> a m))
	for {
		t := v.Type
		a := v_0
		p := v_1
		if p.Op != OpARM64ADDconst {
			break
		}
		c := auxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != OpARM64MUL || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, v.Type)
		v0.AddArg2(a, m)
		v.AddArg(v0)
		return true
	}
	// match: (SUB <t> a p:(ADDconst [c] m:(MULW _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (SUBconst [c] (SUB <v.Type> a m))
	for {
		t := v.Type
		a := v_0
		p := v_1
		if p.Op != OpARM64ADDconst {
			break
		}
		c := auxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != OpARM64MULW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, v.Type)
		v0.AddArg2(a, m)
		v.AddArg(v0)
		return true
	}
	// match: (SUB <t> a p:(ADDconst [c] m:(MNEG _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (SUBconst [c] (SUB <v.Type> a m))
	for {
		t := v.Type
		a := v_0
		p := v_1
		if p.Op != OpARM64ADDconst {
			break
		}
		c := auxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != OpARM64MNEG || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, v.Type)
		v0.AddArg2(a, m)
		v.AddArg(v0)
		return true
	}
	// match: (SUB <t> a p:(ADDconst [c] m:(MNEGW _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (SUBconst [c] (SUB <v.Type> a m))
	for {
		t := v.Type
		a := v_0
		p := v_1
		if p.Op != OpARM64ADDconst {
			break
		}
		c := auxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != OpARM64MNEGW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, v.Type)
		v0.AddArg2(a, m)
		v.AddArg(v0)
		return true
	}
	// match: (SUB <t> a p:(SUBconst [c] m:(MUL _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (ADDconst [c] (SUB <v.Type> a m))
	for {
		t := v.Type
		a := v_0
		p := v_1
		if p.Op != OpARM64SUBconst {
			break
		}
		c := auxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != OpARM64MUL || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, v.Type)
		v0.AddArg2(a, m)
		v.AddArg(v0)
		return true
	}
	// match: (SUB <t> a p:(SUBconst [c] m:(MULW _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (ADDconst [c] (SUB <v.Type> a m))
	for {
		t := v.Type
		a := v_0
		p := v_1
		if p.Op != OpARM64SUBconst {
			break
		}
		c := auxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != OpARM64MULW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, v.Type)
		v0.AddArg2(a, m)
		v.AddArg(v0)
		return true
	}
	// match: (SUB <t> a p:(SUBconst [c] m:(MNEG _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (ADDconst [c] (SUB <v.Type> a m))
	for {
		t := v.Type
		a := v_0
		p := v_1
		if p.Op != OpARM64SUBconst {
			break
		}
		c := auxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != OpARM64MNEG || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, v.Type)
		v0.AddArg2(a, m)
		v.AddArg(v0)
		return true
	}
	// match: (SUB <t> a p:(SUBconst [c] m:(MNEGW _ _)))
	// cond: p.Uses==1 && m.Uses==1 && !t.IsPtrShaped()
	// result: (ADDconst [c] (SUB <v.Type> a m))
	for {
		t := v.Type
		a := v_0
		p := v_1
		if p.Op != OpARM64SUBconst {
			break
		}
		c := auxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != OpARM64MNEGW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64SUB, v.Type)
		v0.AddArg2(a, m)
		v.AddArg(v0)
		return true
	}
	// match: (SUB x (NEG y))
	// result: (ADD x y)
	for {
		x := v_0
		if v_1.Op != OpARM64NEG {
			break
		}
		y := v_1.Args[0]
		v.reset(OpARM64ADD)
		v.AddArg2(x, y)
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
	// match: (SUBshiftLL (SLLconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	// match: (SUBshiftRA (SRAconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRAconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	// match: (SUBshiftRL (SRLconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	// match: (TST x0 x1:(RORconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (TSTshiftRO x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64RORconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64TSTshiftRO)
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
func rewriteValueARM64_OpARM64TSTshiftRO(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftRO (MOVDconst [c]) x [d])
	// result: (TSTconst [c] (RORconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64TSTconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64RORconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRO x (MOVDconst [c]) [d])
	// result: (TSTconst x [rotateRight64(c, d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64TSTconst)
		v.AuxInt = int64ToAuxInt(rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UBFIZ(v *Value) bool {
	v_0 := v.Args[0]
	// match: (UBFIZ [bfc] (SLLconst [sc] x))
	// cond: sc < bfc.width()
	// result: (UBFIZ [armBFAuxInt(bfc.lsb()+sc, bfc.width()-sc)] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.width()) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.lsb()+sc, bfc.width()-sc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UBFX(v *Value) bool {
	v_0 := v.Args[0]
	// match: (UBFX [bfc] (ANDconst [c] x))
	// cond: isARM64BFMask(0, c, 0) && bfc.lsb() + bfc.width() <= arm64BFWidth(c, 0)
	// result: (UBFX [bfc] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		if v_0.Op != OpARM64ANDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(0, c, 0) && bfc.lsb()+bfc.width() <= arm64BFWidth(c, 0)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(bfc)
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] e:(MOVWUreg x))
	// cond: e.Uses == 1 && bfc.lsb() < 32
	// result: (UBFX [armBFAuxInt(bfc.lsb(), min(bfc.width(), 32-bfc.lsb()))] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		e := v_0
		if e.Op != OpARM64MOVWUreg {
			break
		}
		x := e.Args[0]
		if !(e.Uses == 1 && bfc.lsb() < 32) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.lsb(), min(bfc.width(), 32-bfc.lsb())))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] e:(MOVHUreg x))
	// cond: e.Uses == 1 && bfc.lsb() < 16
	// result: (UBFX [armBFAuxInt(bfc.lsb(), min(bfc.width(), 16-bfc.lsb()))] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		e := v_0
		if e.Op != OpARM64MOVHUreg {
			break
		}
		x := e.Args[0]
		if !(e.Uses == 1 && bfc.lsb() < 16) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.lsb(), min(bfc.width(), 16-bfc.lsb())))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] e:(MOVBUreg x))
	// cond: e.Uses == 1 && bfc.lsb() < 8
	// result: (UBFX [armBFAuxInt(bfc.lsb(), min(bfc.width(), 8-bfc.lsb()))] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		e := v_0
		if e.Op != OpARM64MOVBUreg {
			break
		}
		x := e.Args[0]
		if !(e.Uses == 1 && bfc.lsb() < 8) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.lsb(), min(bfc.width(), 8-bfc.lsb())))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SRLconst [sc] x))
	// cond: sc+bfc.width()+bfc.lsb() < 64
	// result: (UBFX [armBFAuxInt(bfc.lsb()+sc, bfc.width())] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		if v_0.Op != OpARM64SRLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc+bfc.width()+bfc.lsb() < 64) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.lsb()+sc, bfc.width()))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SLLconst [sc] x))
	// cond: sc == bfc.lsb()
	// result: (ANDconst [1<<uint(bfc.width())-1] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc == bfc.lsb()) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64ToAuxInt(1<<uint(bfc.width()) - 1)
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SLLconst [sc] x))
	// cond: sc < bfc.lsb()
	// result: (UBFX [armBFAuxInt(bfc.lsb()-sc, bfc.width())] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.lsb()) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(bfc.lsb()-sc, bfc.width()))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SLLconst [sc] x))
	// cond: sc > bfc.lsb() && sc < bfc.lsb()+bfc.width()
	// result: (UBFIZ [armBFAuxInt(sc-bfc.lsb(), bfc.lsb()+bfc.width()-sc)] x)
	for {
		bfc := auxIntToArm64BitField(v.AuxInt)
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := auxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc > bfc.lsb() && sc < bfc.lsb()+bfc.width()) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = arm64BitFieldToAuxInt(armBFAuxInt(sc-bfc.lsb(), bfc.lsb()+bfc.width()-sc))
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
	// cond: isPowerOfTwo(c)
	// result: (SRLconst [log64(c)] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
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
	b := v.Block
	// match: (UDIVW x (MOVDconst [c]))
	// cond: uint32(c)==1
	// result: (MOVWUreg x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(uint32(c) == 1) {
			break
		}
		v.reset(OpARM64MOVWUreg)
		v.AddArg(x)
		return true
	}
	// match: (UDIVW x (MOVDconst [c]))
	// cond: isPowerOfTwo(c) && is32Bit(c)
	// result: (SRLconst [log64(c)] (MOVWUreg <v.Type> x))
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo(c) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SRLconst)
		v.AuxInt = int64ToAuxInt(log64(c))
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUreg, v.Type)
		v0.AddArg(x)
		v.AddArg(v0)
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
	// cond: isPowerOfTwo(c)
	// result: (ANDconst [c-1] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo(c)) {
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
	// cond: isPowerOfTwo(c) && is32Bit(c)
	// result: (ANDconst [c-1] x)
	for {
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		if !(isPowerOfTwo(c) && is32Bit(c)) {
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
	// match: (XOR x0 x1:(RORconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (XORshiftRO x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != OpARM64RORconst {
				continue
			}
			c := auxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(clobberIfDead(x1)) {
				continue
			}
			v.reset(OpARM64XORshiftRO)
			v.AuxInt = int64ToAuxInt(c)
			v.AddArg2(x0, y)
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
	// match: (XORshiftLL (SLLconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SLLconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
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
	// match: (XORshiftLL [8] (UBFX [armBFAuxInt(8, 24)] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff
	// result: (REV16W x)
	for {
		if auxIntToInt64(v.AuxInt) != 8 || v_0.Op != OpARM64UBFX || auxIntToArm64BitField(v_0.AuxInt) != armBFAuxInt(8, 24) {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64ANDconst {
			break
		}
		c1 := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != OpARM64ANDconst {
			break
		}
		c2 := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff) {
			break
		}
		v.reset(OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL [8] (SRLconst [8] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: (uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff)
	// result: (REV16 x)
	for {
		if auxIntToInt64(v.AuxInt) != 8 || v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != 8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64ANDconst {
			break
		}
		c1 := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != OpARM64ANDconst {
			break
		}
		c2 := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff) {
			break
		}
		v.reset(OpARM64REV16)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL [8] (SRLconst [8] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: (uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff)
	// result: (REV16 (ANDconst <x.Type> [0xffffffff] x))
	for {
		if auxIntToInt64(v.AuxInt) != 8 || v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != 8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64ANDconst {
			break
		}
		c1 := auxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != OpARM64ANDconst {
			break
		}
		c2 := auxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff) {
			break
		}
		v.reset(OpARM64REV16)
		v0 := b.NewValue0(v.Pos, OpARM64ANDconst, x.Type)
		v0.AuxInt = int64ToAuxInt(0xffffffff)
		v0.AddArg(x)
		v.AddArg(v0)
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
	// match: (XORshiftRA (SRAconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRAconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
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
	// match: (XORshiftRL (SRLconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64SRLconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64XORshiftRO(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XORshiftRO (MOVDconst [c]) x [d])
	// result: (XORconst [c] (RORconst <x.Type> x [d]))
	for {
		d := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_0.AuxInt)
		x := v_1
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, OpARM64RORconst, x.Type)
		v0.AuxInt = int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRO x (MOVDconst [c]) [d])
	// result: (XORconst x [rotateRight64(c, d)])
	for {
		d := auxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := auxIntToInt64(v_1.AuxInt)
		v.reset(OpARM64XORconst)
		v.AuxInt = int64ToAuxInt(rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRO (RORconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := auxIntToInt64(v.AuxInt)
		if v_0.Op != OpARM64RORconst || auxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(0)
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
func rewriteValueARM64_OpBitLen16(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen16 x)
	// result: (BitLen64 (ZeroExt16to64 x))
	for {
		x := v_0
		v.reset(OpBitLen64)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
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
func rewriteValueARM64_OpBitLen8(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen8 x)
	// result: (BitLen64 (ZeroExt8to64 x))
	for {
		x := v_0
		v.reset(OpBitLen64)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
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
	// result: (CSEL [OpARM64NotEqual] x y (TSTWconst [1] boolval))
	for {
		x := v_0
		y := v_1
		boolval := v_2
		if !(flagArg(boolval) == nil) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = int32ToAuxInt(1)
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
	// match: (ConstBool [t])
	// result: (MOVDconst [b2i(t)])
	for {
		t := auxIntToBool(v.AuxInt)
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64ToAuxInt(b2i(t))
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
	// cond: (is8BitInt(t) && t.IsSigned())
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(is8BitInt(t) && t.IsSigned()) {
			break
		}
		v.reset(OpARM64MOVBload)
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
		v.reset(OpARM64MOVBUload)
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
		v.reset(OpARM64MOVHload)
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
		v.reset(OpARM64MOVHUload)
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
		v.reset(OpARM64MOVWload)
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
		v.reset(OpARM64MOVDaddr)
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
		v.reset(OpARM64MOVDaddr)
		v.Aux = symToAux(sym)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValueARM64_OpLsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpLsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpLsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x64 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
	return false
}
func rewriteValueARM64_OpLsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh16x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpLsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpLsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpLsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x64 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
	return false
}
func rewriteValueARM64_OpLsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh32x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpLsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpLsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpLsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x64 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
	return false
}
func rewriteValueARM64_OpLsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpLsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpLsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpLsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x64 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
	return false
}
func rewriteValueARM64_OpLsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh8x8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
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
	// result: (MOVWstore [3] dst (MOVWUload [3] src mem) (MOVWstore dst (MOVWUload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 7 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUload, typ.UInt32)
		v0.AuxInt = int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpARM64MOVWUload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
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
	// match: (Move [9] dst src mem)
	// result: (MOVBstore [8] dst (MOVBUload [8] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 9 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64MOVBstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpARM64MOVBUload, typ.UInt8)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [10] dst src mem)
	// result: (MOVHstore [8] dst (MOVHUload [8] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 10 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64MOVHstore)
		v.AuxInt = int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, OpARM64MOVHUload, typ.UInt16)
		v0.AuxInt = int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [11] dst src mem)
	// result: (MOVDstore [3] dst (MOVDload [3] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 11 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
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
	// match: (Move [13] dst src mem)
	// result: (MOVDstore [5] dst (MOVDload [5] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 13 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(5)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(5)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [14] dst src mem)
	// result: (MOVDstore [6] dst (MOVDload [6] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 14 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(6)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [15] dst src mem)
	// result: (MOVDstore [7] dst (MOVDload [7] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 15 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(7)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [16] dst src mem)
	// result: (STP dst (Select0 <typ.UInt64> (LDP src mem)) (Select1 <typ.UInt64> (LDP src mem)) mem)
	for {
		if auxIntToInt64(v.AuxInt) != 16 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.reset(OpARM64STP)
		v0 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v1.AddArg2(src, mem)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v2.AddArg(v1)
		v.AddArg4(dst, v0, v2, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 16 && s <= 24
	// result: (MOVDstore [int32(s-8)] dst (MOVDload [int32(s-8)] src mem) (STP dst (Select0 <typ.UInt64> (LDP src mem)) (Select1 <typ.UInt64> (LDP src mem)) mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 16 && s <= 24) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(int32(s - 8))
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(int32(s - 8))
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v2 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v3 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v3.AddArg2(src, mem)
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v4.AddArg(v3)
		v1.AddArg4(dst, v2, v4, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 24 && s <= 32
	// result: (STP [int32(s-16)] dst (Select0 <typ.UInt64> (LDP [int32(s-16)] src mem)) (Select1 <typ.UInt64> (LDP [int32(s-16)] src mem)) (STP dst (Select0 <typ.UInt64> (LDP src mem)) (Select1 <typ.UInt64> (LDP src mem)) mem))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 24 && s <= 32) {
			break
		}
		v.reset(OpARM64STP)
		v.AuxInt = int32ToAuxInt(int32(s - 16))
		v0 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v1.AuxInt = int32ToAuxInt(int32(s - 16))
		v1.AddArg2(src, mem)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v4 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v5 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v5.AddArg2(src, mem)
		v4.AddArg(v5)
		v6 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v6.AddArg(v5)
		v3.AddArg4(dst, v4, v6, mem)
		v.AddArg4(dst, v0, v2, v3)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 32 && s <= 40
	// result: (MOVDstore [int32(s-8)] dst (MOVDload [int32(s-8)] src mem) (STP [16] dst (Select0 <typ.UInt64> (LDP [16] src mem)) (Select1 <typ.UInt64> (LDP [16] src mem)) (STP dst (Select0 <typ.UInt64> (LDP src mem)) (Select1 <typ.UInt64> (LDP src mem)) mem)))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 32 && s <= 40) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(int32(s - 8))
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(int32(s - 8))
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(16)
		v2 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v3 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v3.AuxInt = int32ToAuxInt(16)
		v3.AddArg2(src, mem)
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v4.AddArg(v3)
		v5 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v6 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v7 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v7.AddArg2(src, mem)
		v6.AddArg(v7)
		v8 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v8.AddArg(v7)
		v5.AddArg4(dst, v6, v8, mem)
		v1.AddArg4(dst, v2, v4, v5)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 40 && s <= 48
	// result: (STP [int32(s-16)] dst (Select0 <typ.UInt64> (LDP [int32(s-16)] src mem)) (Select1 <typ.UInt64> (LDP [int32(s-16)] src mem)) (STP [16] dst (Select0 <typ.UInt64> (LDP [16] src mem)) (Select1 <typ.UInt64> (LDP [16] src mem)) (STP dst (Select0 <typ.UInt64> (LDP src mem)) (Select1 <typ.UInt64> (LDP src mem)) mem)))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 40 && s <= 48) {
			break
		}
		v.reset(OpARM64STP)
		v.AuxInt = int32ToAuxInt(int32(s - 16))
		v0 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v1.AuxInt = int32ToAuxInt(int32(s - 16))
		v1.AddArg2(src, mem)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v3.AuxInt = int32ToAuxInt(16)
		v4 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v5 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v5.AuxInt = int32ToAuxInt(16)
		v5.AddArg2(src, mem)
		v4.AddArg(v5)
		v6 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v6.AddArg(v5)
		v7 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v8 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v9 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v9.AddArg2(src, mem)
		v8.AddArg(v9)
		v10 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v10.AddArg(v9)
		v7.AddArg4(dst, v8, v10, mem)
		v3.AddArg4(dst, v4, v6, v7)
		v.AddArg4(dst, v0, v2, v3)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 48 && s <= 56
	// result: (MOVDstore [int32(s-8)] dst (MOVDload [int32(s-8)] src mem) (STP [32] dst (Select0 <typ.UInt64> (LDP [32] src mem)) (Select1 <typ.UInt64> (LDP [32] src mem)) (STP [16] dst (Select0 <typ.UInt64> (LDP [16] src mem)) (Select1 <typ.UInt64> (LDP [16] src mem)) (STP dst (Select0 <typ.UInt64> (LDP src mem)) (Select1 <typ.UInt64> (LDP src mem)) mem))))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 48 && s <= 56) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(int32(s - 8))
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = int32ToAuxInt(int32(s - 8))
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v1.AuxInt = int32ToAuxInt(32)
		v2 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v3 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v3.AuxInt = int32ToAuxInt(32)
		v3.AddArg2(src, mem)
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v4.AddArg(v3)
		v5 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v5.AuxInt = int32ToAuxInt(16)
		v6 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v7 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v7.AuxInt = int32ToAuxInt(16)
		v7.AddArg2(src, mem)
		v6.AddArg(v7)
		v8 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v8.AddArg(v7)
		v9 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v10 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v11 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v11.AddArg2(src, mem)
		v10.AddArg(v11)
		v12 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v12.AddArg(v11)
		v9.AddArg4(dst, v10, v12, mem)
		v5.AddArg4(dst, v6, v8, v9)
		v1.AddArg4(dst, v2, v4, v5)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 56 && s <= 64
	// result: (STP [int32(s-16)] dst (Select0 <typ.UInt64> (LDP [int32(s-16)] src mem)) (Select1 <typ.UInt64> (LDP [int32(s-16)] src mem)) (STP [32] dst (Select0 <typ.UInt64> (LDP [32] src mem)) (Select1 <typ.UInt64> (LDP [32] src mem)) (STP [16] dst (Select0 <typ.UInt64> (LDP [16] src mem)) (Select1 <typ.UInt64> (LDP [16] src mem)) (STP dst (Select0 <typ.UInt64> (LDP src mem)) (Select1 <typ.UInt64> (LDP src mem)) mem))))
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 56 && s <= 64) {
			break
		}
		v.reset(OpARM64STP)
		v.AuxInt = int32ToAuxInt(int32(s - 16))
		v0 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v1.AuxInt = int32ToAuxInt(int32(s - 16))
		v1.AddArg2(src, mem)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v3.AuxInt = int32ToAuxInt(32)
		v4 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v5 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v5.AuxInt = int32ToAuxInt(32)
		v5.AddArg2(src, mem)
		v4.AddArg(v5)
		v6 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v6.AddArg(v5)
		v7 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v7.AuxInt = int32ToAuxInt(16)
		v8 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v9 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v9.AuxInt = int32ToAuxInt(16)
		v9.AddArg2(src, mem)
		v8.AddArg(v9)
		v10 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v10.AddArg(v9)
		v11 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v12 := b.NewValue0(v.Pos, OpSelect0, typ.UInt64)
		v13 := b.NewValue0(v.Pos, OpARM64LDP, types.NewTuple(typ.UInt64, typ.UInt64))
		v13.AddArg2(src, mem)
		v12.AddArg(v13)
		v14 := b.NewValue0(v.Pos, OpSelect1, typ.UInt64)
		v14.AddArg(v13)
		v11.AddArg4(dst, v12, v14, mem)
		v7.AddArg4(dst, v8, v10, v11)
		v3.AddArg4(dst, v4, v6, v7)
		v.AddArg4(dst, v0, v2, v3)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 64 && s < 192 && logLargeCopy(v, s)
	// result: (LoweredMove [s] dst src mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 64 && s < 192 && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpARM64LoweredMove)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s >= 192 && logLargeCopy(v, s)
	// result: (LoweredMoveLoop [s] dst src mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s >= 192 && logLargeCopy(v, s)) {
			break
		}
		v.reset(OpARM64LoweredMoveLoop)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
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
func rewriteValueARM64_OpPrefetchCache(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PrefetchCache addr mem)
	// result: (PRFM [0] addr mem)
	for {
		addr := v_0
		mem := v_1
		v.reset(OpARM64PRFM)
		v.AuxInt = int64ToAuxInt(0)
		v.AddArg2(addr, mem)
		return true
	}
}
func rewriteValueARM64_OpPrefetchCacheStreamed(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PrefetchCacheStreamed addr mem)
	// result: (PRFM [1] addr mem)
	for {
		addr := v_0
		mem := v_1
		v.reset(OpARM64PRFM)
		v.AuxInt = int64ToAuxInt(1)
		v.AddArg2(addr, mem)
		return true
	}
}
func rewriteValueARM64_OpPubBarrier(v *Value) bool {
	v_0 := v.Args[0]
	// match: (PubBarrier mem)
	// result: (DMB [0xe] mem)
	for {
		mem := v_0
		v.reset(OpARM64DMB)
		v.AuxInt = int64ToAuxInt(0xe)
		v.AddArg(mem)
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
	// match: (RotateLeft16 <t> x y)
	// result: (RORW <t> (ORshiftLL <typ.UInt32> (ZeroExt16to32 x) (ZeroExt16to32 x) [16]) (NEG <typ.Int64> y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64RORW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARM64ORshiftLL, typ.UInt32)
		v0.AuxInt = int64ToAuxInt(16)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg2(v1, v1)
		v2 := b.NewValue0(v.Pos, OpARM64NEG, typ.Int64)
		v2.AddArg(y)
		v.AddArg2(v0, v2)
		return true
	}
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
	// match: (RotateLeft8 <t> x y)
	// result: (OR <t> (SLL <t> x (ANDconst <typ.Int64> [7] y)) (SRL <t> (ZeroExt8to64 x) (ANDconst <typ.Int64> [7] (NEG <typ.Int64> y))))
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.reset(OpARM64OR)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, OpARM64ANDconst, typ.Int64)
		v1.AuxInt = int64ToAuxInt(7)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, OpARM64ANDconst, typ.Int64)
		v4.AuxInt = int64ToAuxInt(7)
		v5 := b.NewValue0(v.Pos, OpARM64NEG, typ.Int64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v2.AddArg2(v3, v4)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueARM64_OpRsh16Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> (ZeroExt16to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt16to64 x) y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh16Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> (ZeroExt16to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt16to64 x) y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh16Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> (ZeroExt16to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt16to64 x) y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
	return false
}
func rewriteValueARM64_OpRsh16Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> (ZeroExt16to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16Ux8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt16to64 x) y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh16x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> (SignExt16to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x16 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA (SignExt16to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh16x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> (SignExt16to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x32 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA (SignExt16to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh16x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> (SignExt16to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x64 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA (SignExt16to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
	return false
}
func rewriteValueARM64_OpRsh16x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> (SignExt16to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh16x8 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA (SignExt16to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64 y))))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh32Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> (ZeroExt32to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh32Ux16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt32to64 x) y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh32Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> (ZeroExt32to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh32Ux32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt32to64 x) y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh32Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> (ZeroExt32to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh32Ux64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt32to64 x) y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
	return false
}
func rewriteValueARM64_OpRsh32Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> (ZeroExt32to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh32Ux8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt32to64 x) y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh32x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> (SignExt32to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh32x16 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA (SignExt32to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh32x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> (SignExt32to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh32x32 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA (SignExt32to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh32x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x64 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> (SignExt32to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh32x64 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA (SignExt32to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
	return false
}
func rewriteValueARM64_OpRsh32x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> (SignExt32to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh32x8 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA (SignExt32to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64 y))))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh64Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh64Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh64Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux64 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
	return false
}
func rewriteValueARM64_OpRsh64Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> x y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64CSEL)
		v.AuxInt = opToAuxInt(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh64x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x16 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA x (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v0.AuxInt = opToAuxInt(OpARM64LessThanU)
		v1 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v1.AuxInt = int64ToAuxInt(63)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v0.AddArg3(y, v1, v2)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh64x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x32 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA x (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v0.AuxInt = opToAuxInt(OpARM64LessThanU)
		v1 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v1.AuxInt = int64ToAuxInt(63)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v0.AddArg3(y, v1, v2)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh64x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x64 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x64 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA x (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
	return false
}
func rewriteValueARM64_OpRsh64x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x8 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA x (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64 y))))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v0.AuxInt = opToAuxInt(OpARM64LessThanU)
		v1 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v1.AuxInt = int64ToAuxInt(63)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = int64ToAuxInt(64)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v0.AddArg3(y, v1, v2)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh8Ux16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> (ZeroExt8to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux16 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt8to64 x) y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh8Ux32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> (ZeroExt8to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux32 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt8to64 x) y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh8Ux64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> (ZeroExt8to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux64 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt8to64 x) y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
	return false
}
func rewriteValueARM64_OpRsh8Ux8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRL <t> (ZeroExt8to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRL)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8Ux8 <t> x y)
	// cond: !shiftIsBounded(v)
	// result: (CSEL [OpARM64LessThanU] (SRL <t> (ZeroExt8to64 x) y) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg3(v0, v2, v3)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh8x16(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> (SignExt8to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x16 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA (SignExt8to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh8x32(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> (SignExt8to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x32 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA (SignExt8to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpRsh8x64(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> (SignExt8to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x64 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA (SignExt8to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
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
	return false
}
func rewriteValueARM64_OpRsh8x8(v *Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 <t> x y)
	// cond: shiftIsBounded(v)
	// result: (SRA <t> (SignExt8to64 x) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
	// match: (Rsh8x8 x y)
	// cond: !shiftIsBounded(v)
	// result: (SRA (SignExt8to64 x) (CSEL [OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64 y))))
	for {
		x := v_0
		y := v_1
		if !(!shiftIsBounded(v)) {
			break
		}
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.AuxInt = opToAuxInt(OpARM64LessThanU)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = int64ToAuxInt(63)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = int64ToAuxInt(64)
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v1.AddArg3(y, v2, v3)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpSelect0(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Mul64uhilo x y))
	// result: (UMULH x y)
	for {
		if v_0.Op != OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64UMULH)
		v.AddArg2(x, y)
		return true
	}
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
	// match: (Select0 (Mul64uover x y))
	// result: (MUL x y)
	for {
		if v_0.Op != OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64MUL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM64_OpSelect1(v *Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Mul64uhilo x y))
	// result: (MUL x y)
	for {
		if v_0.Op != OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64MUL)
		v.AddArg2(x, y)
		return true
	}
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
	// match: (Select1 (Mul64uover x y))
	// result: (NotEqual (CMPconst (UMULH <typ.UInt64> x y) [0]))
	for {
		if v_0.Op != OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64UMULH, typ.UInt64)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpSelectN(v *Value) bool {
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
		if call.Op != OpARM64CALLstatic || len(call.Args) != 1 {
			break
		}
		sym := auxToCall(call.Aux)
		s1 := call.Args[0]
		if s1.Op != OpARM64MOVDstore {
			break
		}
		_ = s1.Args[2]
		s1_1 := s1.Args[1]
		if s1_1.Op != OpARM64MOVDconst {
			break
		}
		sz := auxIntToInt64(s1_1.AuxInt)
		s2 := s1.Args[2]
		if s2.Op != OpARM64MOVDstore {
			break
		}
		_ = s2.Args[2]
		src := s2.Args[1]
		s3 := s2.Args[2]
		if s3.Op != OpARM64MOVDstore {
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
		if call.Op != OpARM64CALLstatic || len(call.Args) != 4 {
			break
		}
		sym := auxToCall(call.Aux)
		mem := call.Args[3]
		dst := call.Args[0]
		src := call.Args[1]
		call_2 := call.Args[2]
		if call_2.Op != OpARM64MOVDconst {
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
		v.reset(OpARM64MOVWstore)
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
		v.reset(OpARM64MOVDstore)
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
		v.reset(OpARM64FMOVSstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
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
	// result: (MOVWstore [3] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 7 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVWstore)
		v.AuxInt = int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
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
	// result: (MOVDstore [3] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 11 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
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
	// result: (MOVDstore [5] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 13 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(5)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [14] ptr mem)
	// result: (MOVDstore [6] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 14 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [15] ptr mem)
	// result: (MOVDstore [7] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if auxIntToInt64(v.AuxInt) != 15 {
			break
		}
		ptr := v_0
		mem := v_1
		v.reset(OpARM64MOVDstore)
		v.AuxInt = int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
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
	// match: (Zero [s] ptr mem)
	// cond: s > 16 && s < 192
	// result: (LoweredZero [s] ptr mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s > 16 && s < 192) {
			break
		}
		v.reset(OpARM64LoweredZero)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s >= 192
	// result: (LoweredZeroLoop [s] ptr mem)
	for {
		s := auxIntToInt64(v.AuxInt)
		ptr := v_0
		mem := v_1
		if !(s >= 192) {
			break
		}
		v.reset(OpARM64LoweredZeroLoop)
		v.AuxInt = int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteBlockARM64(b *Block) bool {
	typ := &b.Func.Config.Types
	switch b.Kind {
	case BlockARM64EQ:
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
		// result: (TBNZ [0] cond yes no)
		for {
			cond := b.Controls[0]
			b.resetWithControl(BlockARM64TBNZ, cond)
			b.AuxInt = int64ToAuxInt(0)
			return true
		}
	case BlockJumpTable:
		// match: (JumpTable idx)
		// result: (JUMPTABLE {makeJumpTableSym(b)} idx (MOVDaddr <typ.Uintptr> {makeJumpTableSym(b)} (SB)))
		for {
			idx := b.Controls[0]
			v0 := b.NewValue0(b.Pos, OpARM64MOVDaddr, typ.Uintptr)
			v0.Aux = symToAux(makeJumpTableSym(b))
			v1 := b.NewValue0(b.Pos, OpSB, typ.Uintptr)
			v0.AddArg(v1)
			b.resetWithControl2(BlockARM64JUMPTABLE, idx, v0)
			b.Aux = symToAux(makeJumpTableSym(b))
			return true
		}
	case BlockARM64LE:
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
		// match: (NZ sub:(SUB x y))
		// cond: sub.Uses == 1
		// result: (NE (CMP x y))
		for b.Controls[0].Op == OpARM64SUB {
			sub := b.Controls[0]
			y := sub.Args[1]
			x := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, OpARM64CMP, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (NZ sub:(SUBconst [c] y))
		// cond: sub.Uses == 1
		// result: (NE (CMPconst [c] y))
		for b.Controls[0].Op == OpARM64SUBconst {
			sub := b.Controls[0]
			c := auxIntToInt64(sub.AuxInt)
			y := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, OpARM64CMPconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64NE, v0)
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
		// match: (NZW sub:(SUB x y))
		// cond: sub.Uses == 1
		// result: (NE (CMPW x y))
		for b.Controls[0].Op == OpARM64SUB {
			sub := b.Controls[0]
			y := sub.Args[1]
			x := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, OpARM64CMPW, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (NZW sub:(SUBconst [c] y))
		// cond: sub.Uses == 1
		// result: (NE (CMPWconst [int32(c)] y))
		for b.Controls[0].Op == OpARM64SUBconst {
			sub := b.Controls[0]
			c := auxIntToInt64(sub.AuxInt)
			y := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, OpARM64CMPWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
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
	case BlockARM64TBNZ:
		// match: (TBNZ [0] (Equal cc) yes no)
		// result: (EQ cc yes no)
		for b.Controls[0].Op == OpARM64Equal {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64EQ, cc)
			return true
		}
		// match: (TBNZ [0] (NotEqual cc) yes no)
		// result: (NE cc yes no)
		for b.Controls[0].Op == OpARM64NotEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64NE, cc)
			return true
		}
		// match: (TBNZ [0] (LessThan cc) yes no)
		// result: (LT cc yes no)
		for b.Controls[0].Op == OpARM64LessThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64LT, cc)
			return true
		}
		// match: (TBNZ [0] (LessThanU cc) yes no)
		// result: (ULT cc yes no)
		for b.Controls[0].Op == OpARM64LessThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64ULT, cc)
			return true
		}
		// match: (TBNZ [0] (LessEqual cc) yes no)
		// result: (LE cc yes no)
		for b.Controls[0].Op == OpARM64LessEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64LE, cc)
			return true
		}
		// match: (TBNZ [0] (LessEqualU cc) yes no)
		// result: (ULE cc yes no)
		for b.Controls[0].Op == OpARM64LessEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64ULE, cc)
			return true
		}
		// match: (TBNZ [0] (GreaterThan cc) yes no)
		// result: (GT cc yes no)
		for b.Controls[0].Op == OpARM64GreaterThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64GT, cc)
			return true
		}
		// match: (TBNZ [0] (GreaterThanU cc) yes no)
		// result: (UGT cc yes no)
		for b.Controls[0].Op == OpARM64GreaterThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64UGT, cc)
			return true
		}
		// match: (TBNZ [0] (GreaterEqual cc) yes no)
		// result: (GE cc yes no)
		for b.Controls[0].Op == OpARM64GreaterEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64GE, cc)
			return true
		}
		// match: (TBNZ [0] (GreaterEqualU cc) yes no)
		// result: (UGE cc yes no)
		for b.Controls[0].Op == OpARM64GreaterEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64UGE, cc)
			return true
		}
		// match: (TBNZ [0] (LessThanF cc) yes no)
		// result: (FLT cc yes no)
		for b.Controls[0].Op == OpARM64LessThanF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64FLT, cc)
			return true
		}
		// match: (TBNZ [0] (LessEqualF cc) yes no)
		// result: (FLE cc yes no)
		for b.Controls[0].Op == OpARM64LessEqualF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64FLE, cc)
			return true
		}
		// match: (TBNZ [0] (GreaterThanF cc) yes no)
		// result: (FGT cc yes no)
		for b.Controls[0].Op == OpARM64GreaterThanF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64FGT, cc)
			return true
		}
		// match: (TBNZ [0] (GreaterEqualF cc) yes no)
		// result: (FGE cc yes no)
		for b.Controls[0].Op == OpARM64GreaterEqualF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64FGE, cc)
			return true
		}
		// match: (TBNZ [0] (XORconst [1] x) yes no)
		// result: (TBZ [0] x yes no)
		for b.Controls[0].Op == OpARM64XORconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			x := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64TBZ, x)
			b.AuxInt = int64ToAuxInt(0)
			return true
		}
	case BlockARM64TBZ:
		// match: (TBZ [0] (XORconst [1] x) yes no)
		// result: (TBNZ [0] x yes no)
		for b.Controls[0].Op == OpARM64XORconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			x := v_0.Args[0]
			if auxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.resetWithControl(BlockARM64TBNZ, x)
			b.AuxInt = int64ToAuxInt(0)
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
		// match: (UGT (CMPconst [0] x))
		// result: (NE (CMPconst [0] x))
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			v0 := b.NewValue0(v_0.Pos, OpARM64CMPconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(0)
			v0.AddArg(x)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
		// match: (UGT (CMPWconst [0] x))
		// result: (NE (CMPWconst [0] x))
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			v0 := b.NewValue0(v_0.Pos, OpARM64CMPWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(0)
			v0.AddArg(x)
			b.resetWithControl(BlockARM64NE, v0)
			return true
		}
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
		// match: (ULE (CMPconst [0] x))
		// result: (EQ (CMPconst [0] x))
		for b.Controls[0].Op == OpARM64CMPconst {
			v_0 := b.Controls[0]
			if auxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			v0 := b.NewValue0(v_0.Pos, OpARM64CMPconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(0)
			v0.AddArg(x)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (ULE (CMPWconst [0] x))
		// result: (EQ (CMPWconst [0] x))
		for b.Controls[0].Op == OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if auxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			v0 := b.NewValue0(v_0.Pos, OpARM64CMPWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(0)
			v0.AddArg(x)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
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
		// match: (Z sub:(SUB x y))
		// cond: sub.Uses == 1
		// result: (EQ (CMP x y))
		for b.Controls[0].Op == OpARM64SUB {
			sub := b.Controls[0]
			y := sub.Args[1]
			x := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, OpARM64CMP, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (Z sub:(SUBconst [c] y))
		// cond: sub.Uses == 1
		// result: (EQ (CMPconst [c] y))
		for b.Controls[0].Op == OpARM64SUBconst {
			sub := b.Controls[0]
			c := auxIntToInt64(sub.AuxInt)
			y := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, OpARM64CMPconst, types.TypeFlags)
			v0.AuxInt = int64ToAuxInt(c)
			v0.AddArg(y)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
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
		// match: (ZW sub:(SUB x y))
		// cond: sub.Uses == 1
		// result: (EQ (CMPW x y))
		for b.Controls[0].Op == OpARM64SUB {
			sub := b.Controls[0]
			y := sub.Args[1]
			x := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, OpARM64CMPW, types.TypeFlags)
			v0.AddArg2(x, y)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
		// match: (ZW sub:(SUBconst [c] y))
		// cond: sub.Uses == 1
		// result: (EQ (CMPWconst [int32(c)] y))
		for b.Controls[0].Op == OpARM64SUBconst {
			sub := b.Controls[0]
			c := auxIntToInt64(sub.AuxInt)
			y := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, OpARM64CMPWconst, types.TypeFlags)
			v0.AuxInt = int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.resetWithControl(BlockARM64EQ, v0)
			return true
		}
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
