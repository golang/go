// Code generated from _gen/ARM64.rules using 'go generate'; DO NOT EDIT.

package rewritearm64

import "cmd/compile/internal/types"
import "cmd/compile/internal/ssa/block"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func RewriteValue(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpARM64ADCSflags:
		return rewriteValue_OpARM64ADCSflags(v)
	case ssaop.OpARM64ADD:
		return rewriteValue_OpARM64ADD(v)
	case ssaop.OpARM64ADDSflags:
		return rewriteValue_OpARM64ADDSflags(v)
	case ssaop.OpARM64ADDconst:
		return rewriteValue_OpARM64ADDconst(v)
	case ssaop.OpARM64ADDshiftLL:
		return rewriteValue_OpARM64ADDshiftLL(v)
	case ssaop.OpARM64ADDshiftRA:
		return rewriteValue_OpARM64ADDshiftRA(v)
	case ssaop.OpARM64ADDshiftRL:
		return rewriteValue_OpARM64ADDshiftRL(v)
	case ssaop.OpARM64AND:
		return rewriteValue_OpARM64AND(v)
	case ssaop.OpARM64ANDconst:
		return rewriteValue_OpARM64ANDconst(v)
	case ssaop.OpARM64ANDshiftLL:
		return rewriteValue_OpARM64ANDshiftLL(v)
	case ssaop.OpARM64ANDshiftRA:
		return rewriteValue_OpARM64ANDshiftRA(v)
	case ssaop.OpARM64ANDshiftRL:
		return rewriteValue_OpARM64ANDshiftRL(v)
	case ssaop.OpARM64ANDshiftRO:
		return rewriteValue_OpARM64ANDshiftRO(v)
	case ssaop.OpARM64BIC:
		return rewriteValue_OpARM64BIC(v)
	case ssaop.OpARM64BICshiftLL:
		return rewriteValue_OpARM64BICshiftLL(v)
	case ssaop.OpARM64BICshiftRA:
		return rewriteValue_OpARM64BICshiftRA(v)
	case ssaop.OpARM64BICshiftRL:
		return rewriteValue_OpARM64BICshiftRL(v)
	case ssaop.OpARM64BICshiftRO:
		return rewriteValue_OpARM64BICshiftRO(v)
	case ssaop.OpARM64CMN:
		return rewriteValue_OpARM64CMN(v)
	case ssaop.OpARM64CMNW:
		return rewriteValue_OpARM64CMNW(v)
	case ssaop.OpARM64CMNWconst:
		return rewriteValue_OpARM64CMNWconst(v)
	case ssaop.OpARM64CMNconst:
		return rewriteValue_OpARM64CMNconst(v)
	case ssaop.OpARM64CMNshiftLL:
		return rewriteValue_OpARM64CMNshiftLL(v)
	case ssaop.OpARM64CMNshiftRA:
		return rewriteValue_OpARM64CMNshiftRA(v)
	case ssaop.OpARM64CMNshiftRL:
		return rewriteValue_OpARM64CMNshiftRL(v)
	case ssaop.OpARM64CMP:
		return rewriteValue_OpARM64CMP(v)
	case ssaop.OpARM64CMPW:
		return rewriteValue_OpARM64CMPW(v)
	case ssaop.OpARM64CMPWconst:
		return rewriteValue_OpARM64CMPWconst(v)
	case ssaop.OpARM64CMPconst:
		return rewriteValue_OpARM64CMPconst(v)
	case ssaop.OpARM64CMPshiftLL:
		return rewriteValue_OpARM64CMPshiftLL(v)
	case ssaop.OpARM64CMPshiftRA:
		return rewriteValue_OpARM64CMPshiftRA(v)
	case ssaop.OpARM64CMPshiftRL:
		return rewriteValue_OpARM64CMPshiftRL(v)
	case ssaop.OpARM64CSEL:
		return rewriteValue_OpARM64CSEL(v)
	case ssaop.OpARM64CSEL0:
		return rewriteValue_OpARM64CSEL0(v)
	case ssaop.OpARM64CSETM:
		return rewriteValue_OpARM64CSETM(v)
	case ssaop.OpARM64CSINC:
		return rewriteValue_OpARM64CSINC(v)
	case ssaop.OpARM64CSINV:
		return rewriteValue_OpARM64CSINV(v)
	case ssaop.OpARM64CSNEG:
		return rewriteValue_OpARM64CSNEG(v)
	case ssaop.OpARM64DIV:
		return rewriteValue_OpARM64DIV(v)
	case ssaop.OpARM64DIVW:
		return rewriteValue_OpARM64DIVW(v)
	case ssaop.OpARM64EON:
		return rewriteValue_OpARM64EON(v)
	case ssaop.OpARM64EONshiftLL:
		return rewriteValue_OpARM64EONshiftLL(v)
	case ssaop.OpARM64EONshiftRA:
		return rewriteValue_OpARM64EONshiftRA(v)
	case ssaop.OpARM64EONshiftRL:
		return rewriteValue_OpARM64EONshiftRL(v)
	case ssaop.OpARM64EONshiftRO:
		return rewriteValue_OpARM64EONshiftRO(v)
	case ssaop.OpARM64Equal:
		return rewriteValue_OpARM64Equal(v)
	case ssaop.OpARM64FADDD:
		return rewriteValue_OpARM64FADDD(v)
	case ssaop.OpARM64FADDS:
		return rewriteValue_OpARM64FADDS(v)
	case ssaop.OpARM64FCMPD:
		return rewriteValue_OpARM64FCMPD(v)
	case ssaop.OpARM64FCMPS:
		return rewriteValue_OpARM64FCMPS(v)
	case ssaop.OpARM64FCSELD:
		return rewriteValue_OpARM64FCSELD(v)
	case ssaop.OpARM64FCSELS:
		return rewriteValue_OpARM64FCSELS(v)
	case ssaop.OpARM64FCVTDS:
		return rewriteValue_OpARM64FCVTDS(v)
	case ssaop.OpARM64FLDPQ:
		return rewriteValue_OpARM64FLDPQ(v)
	case ssaop.OpARM64FMOVDfpgp:
		return rewriteValue_OpARM64FMOVDfpgp(v)
	case ssaop.OpARM64FMOVDgpfp:
		return rewriteValue_OpARM64FMOVDgpfp(v)
	case ssaop.OpARM64FMOVDload:
		return rewriteValue_OpARM64FMOVDload(v)
	case ssaop.OpARM64FMOVDloadidx:
		return rewriteValue_OpARM64FMOVDloadidx(v)
	case ssaop.OpARM64FMOVDloadidx8:
		return rewriteValue_OpARM64FMOVDloadidx8(v)
	case ssaop.OpARM64FMOVDstore:
		return rewriteValue_OpARM64FMOVDstore(v)
	case ssaop.OpARM64FMOVDstoreidx:
		return rewriteValue_OpARM64FMOVDstoreidx(v)
	case ssaop.OpARM64FMOVDstoreidx8:
		return rewriteValue_OpARM64FMOVDstoreidx8(v)
	case ssaop.OpARM64FMOVQload:
		return rewriteValue_OpARM64FMOVQload(v)
	case ssaop.OpARM64FMOVQstore:
		return rewriteValue_OpARM64FMOVQstore(v)
	case ssaop.OpARM64FMOVSload:
		return rewriteValue_OpARM64FMOVSload(v)
	case ssaop.OpARM64FMOVSloadidx:
		return rewriteValue_OpARM64FMOVSloadidx(v)
	case ssaop.OpARM64FMOVSloadidx4:
		return rewriteValue_OpARM64FMOVSloadidx4(v)
	case ssaop.OpARM64FMOVSstore:
		return rewriteValue_OpARM64FMOVSstore(v)
	case ssaop.OpARM64FMOVSstoreidx:
		return rewriteValue_OpARM64FMOVSstoreidx(v)
	case ssaop.OpARM64FMOVSstoreidx4:
		return rewriteValue_OpARM64FMOVSstoreidx4(v)
	case ssaop.OpARM64FMULD:
		return rewriteValue_OpARM64FMULD(v)
	case ssaop.OpARM64FMULS:
		return rewriteValue_OpARM64FMULS(v)
	case ssaop.OpARM64FNEGD:
		return rewriteValue_OpARM64FNEGD(v)
	case ssaop.OpARM64FNEGS:
		return rewriteValue_OpARM64FNEGS(v)
	case ssaop.OpARM64FNMULD:
		return rewriteValue_OpARM64FNMULD(v)
	case ssaop.OpARM64FNMULS:
		return rewriteValue_OpARM64FNMULS(v)
	case ssaop.OpARM64FSTPQ:
		return rewriteValue_OpARM64FSTPQ(v)
	case ssaop.OpARM64FSUBD:
		return rewriteValue_OpARM64FSUBD(v)
	case ssaop.OpARM64FSUBS:
		return rewriteValue_OpARM64FSUBS(v)
	case ssaop.OpARM64GreaterEqual:
		return rewriteValue_OpARM64GreaterEqual(v)
	case ssaop.OpARM64GreaterEqualF:
		return rewriteValue_OpARM64GreaterEqualF(v)
	case ssaop.OpARM64GreaterEqualNoov:
		return rewriteValue_OpARM64GreaterEqualNoov(v)
	case ssaop.OpARM64GreaterEqualU:
		return rewriteValue_OpARM64GreaterEqualU(v)
	case ssaop.OpARM64GreaterThan:
		return rewriteValue_OpARM64GreaterThan(v)
	case ssaop.OpARM64GreaterThanF:
		return rewriteValue_OpARM64GreaterThanF(v)
	case ssaop.OpARM64GreaterThanU:
		return rewriteValue_OpARM64GreaterThanU(v)
	case ssaop.OpARM64LDP:
		return rewriteValue_OpARM64LDP(v)
	case ssaop.OpARM64LessEqual:
		return rewriteValue_OpARM64LessEqual(v)
	case ssaop.OpARM64LessEqualF:
		return rewriteValue_OpARM64LessEqualF(v)
	case ssaop.OpARM64LessEqualU:
		return rewriteValue_OpARM64LessEqualU(v)
	case ssaop.OpARM64LessThan:
		return rewriteValue_OpARM64LessThan(v)
	case ssaop.OpARM64LessThanF:
		return rewriteValue_OpARM64LessThanF(v)
	case ssaop.OpARM64LessThanNoov:
		return rewriteValue_OpARM64LessThanNoov(v)
	case ssaop.OpARM64LessThanU:
		return rewriteValue_OpARM64LessThanU(v)
	case ssaop.OpARM64LoweredPanicBoundsCR:
		return rewriteValue_OpARM64LoweredPanicBoundsCR(v)
	case ssaop.OpARM64LoweredPanicBoundsRC:
		return rewriteValue_OpARM64LoweredPanicBoundsRC(v)
	case ssaop.OpARM64LoweredPanicBoundsRR:
		return rewriteValue_OpARM64LoweredPanicBoundsRR(v)
	case ssaop.OpARM64MADD:
		return rewriteValue_OpARM64MADD(v)
	case ssaop.OpARM64MADDW:
		return rewriteValue_OpARM64MADDW(v)
	case ssaop.OpARM64MNEG:
		return rewriteValue_OpARM64MNEG(v)
	case ssaop.OpARM64MNEGW:
		return rewriteValue_OpARM64MNEGW(v)
	case ssaop.OpARM64MOD:
		return rewriteValue_OpARM64MOD(v)
	case ssaop.OpARM64MODW:
		return rewriteValue_OpARM64MODW(v)
	case ssaop.OpARM64MOVBUload:
		return rewriteValue_OpARM64MOVBUload(v)
	case ssaop.OpARM64MOVBUloadidx:
		return rewriteValue_OpARM64MOVBUloadidx(v)
	case ssaop.OpARM64MOVBUreg:
		return rewriteValue_OpARM64MOVBUreg(v)
	case ssaop.OpARM64MOVBload:
		return rewriteValue_OpARM64MOVBload(v)
	case ssaop.OpARM64MOVBloadidx:
		return rewriteValue_OpARM64MOVBloadidx(v)
	case ssaop.OpARM64MOVBreg:
		return rewriteValue_OpARM64MOVBreg(v)
	case ssaop.OpARM64MOVBstore:
		return rewriteValue_OpARM64MOVBstore(v)
	case ssaop.OpARM64MOVBstoreidx:
		return rewriteValue_OpARM64MOVBstoreidx(v)
	case ssaop.OpARM64MOVDload:
		return rewriteValue_OpARM64MOVDload(v)
	case ssaop.OpARM64MOVDloadidx:
		return rewriteValue_OpARM64MOVDloadidx(v)
	case ssaop.OpARM64MOVDloadidx8:
		return rewriteValue_OpARM64MOVDloadidx8(v)
	case ssaop.OpARM64MOVDnop:
		return rewriteValue_OpARM64MOVDnop(v)
	case ssaop.OpARM64MOVDreg:
		return rewriteValue_OpARM64MOVDreg(v)
	case ssaop.OpARM64MOVDstore:
		return rewriteValue_OpARM64MOVDstore(v)
	case ssaop.OpARM64MOVDstoreidx:
		return rewriteValue_OpARM64MOVDstoreidx(v)
	case ssaop.OpARM64MOVDstoreidx8:
		return rewriteValue_OpARM64MOVDstoreidx8(v)
	case ssaop.OpARM64MOVHUload:
		return rewriteValue_OpARM64MOVHUload(v)
	case ssaop.OpARM64MOVHUloadidx:
		return rewriteValue_OpARM64MOVHUloadidx(v)
	case ssaop.OpARM64MOVHUloadidx2:
		return rewriteValue_OpARM64MOVHUloadidx2(v)
	case ssaop.OpARM64MOVHUreg:
		return rewriteValue_OpARM64MOVHUreg(v)
	case ssaop.OpARM64MOVHload:
		return rewriteValue_OpARM64MOVHload(v)
	case ssaop.OpARM64MOVHloadidx:
		return rewriteValue_OpARM64MOVHloadidx(v)
	case ssaop.OpARM64MOVHloadidx2:
		return rewriteValue_OpARM64MOVHloadidx2(v)
	case ssaop.OpARM64MOVHreg:
		return rewriteValue_OpARM64MOVHreg(v)
	case ssaop.OpARM64MOVHstore:
		return rewriteValue_OpARM64MOVHstore(v)
	case ssaop.OpARM64MOVHstoreidx:
		return rewriteValue_OpARM64MOVHstoreidx(v)
	case ssaop.OpARM64MOVHstoreidx2:
		return rewriteValue_OpARM64MOVHstoreidx2(v)
	case ssaop.OpARM64MOVWUload:
		return rewriteValue_OpARM64MOVWUload(v)
	case ssaop.OpARM64MOVWUloadidx:
		return rewriteValue_OpARM64MOVWUloadidx(v)
	case ssaop.OpARM64MOVWUloadidx4:
		return rewriteValue_OpARM64MOVWUloadidx4(v)
	case ssaop.OpARM64MOVWUreg:
		return rewriteValue_OpARM64MOVWUreg(v)
	case ssaop.OpARM64MOVWload:
		return rewriteValue_OpARM64MOVWload(v)
	case ssaop.OpARM64MOVWloadidx:
		return rewriteValue_OpARM64MOVWloadidx(v)
	case ssaop.OpARM64MOVWloadidx4:
		return rewriteValue_OpARM64MOVWloadidx4(v)
	case ssaop.OpARM64MOVWreg:
		return rewriteValue_OpARM64MOVWreg(v)
	case ssaop.OpARM64MOVWstore:
		return rewriteValue_OpARM64MOVWstore(v)
	case ssaop.OpARM64MOVWstoreidx:
		return rewriteValue_OpARM64MOVWstoreidx(v)
	case ssaop.OpARM64MOVWstoreidx4:
		return rewriteValue_OpARM64MOVWstoreidx4(v)
	case ssaop.OpARM64MSUB:
		return rewriteValue_OpARM64MSUB(v)
	case ssaop.OpARM64MSUBW:
		return rewriteValue_OpARM64MSUBW(v)
	case ssaop.OpARM64MUL:
		return rewriteValue_OpARM64MUL(v)
	case ssaop.OpARM64MULW:
		return rewriteValue_OpARM64MULW(v)
	case ssaop.OpARM64MVN:
		return rewriteValue_OpARM64MVN(v)
	case ssaop.OpARM64MVNshiftLL:
		return rewriteValue_OpARM64MVNshiftLL(v)
	case ssaop.OpARM64MVNshiftRA:
		return rewriteValue_OpARM64MVNshiftRA(v)
	case ssaop.OpARM64MVNshiftRL:
		return rewriteValue_OpARM64MVNshiftRL(v)
	case ssaop.OpARM64MVNshiftRO:
		return rewriteValue_OpARM64MVNshiftRO(v)
	case ssaop.OpARM64NEG:
		return rewriteValue_OpARM64NEG(v)
	case ssaop.OpARM64NEGshiftLL:
		return rewriteValue_OpARM64NEGshiftLL(v)
	case ssaop.OpARM64NEGshiftRA:
		return rewriteValue_OpARM64NEGshiftRA(v)
	case ssaop.OpARM64NEGshiftRL:
		return rewriteValue_OpARM64NEGshiftRL(v)
	case ssaop.OpARM64NotEqual:
		return rewriteValue_OpARM64NotEqual(v)
	case ssaop.OpARM64OR:
		return rewriteValue_OpARM64OR(v)
	case ssaop.OpARM64ORN:
		return rewriteValue_OpARM64ORN(v)
	case ssaop.OpARM64ORNshiftLL:
		return rewriteValue_OpARM64ORNshiftLL(v)
	case ssaop.OpARM64ORNshiftRA:
		return rewriteValue_OpARM64ORNshiftRA(v)
	case ssaop.OpARM64ORNshiftRL:
		return rewriteValue_OpARM64ORNshiftRL(v)
	case ssaop.OpARM64ORNshiftRO:
		return rewriteValue_OpARM64ORNshiftRO(v)
	case ssaop.OpARM64ORconst:
		return rewriteValue_OpARM64ORconst(v)
	case ssaop.OpARM64ORshiftLL:
		return rewriteValue_OpARM64ORshiftLL(v)
	case ssaop.OpARM64ORshiftRA:
		return rewriteValue_OpARM64ORshiftRA(v)
	case ssaop.OpARM64ORshiftRL:
		return rewriteValue_OpARM64ORshiftRL(v)
	case ssaop.OpARM64ORshiftRO:
		return rewriteValue_OpARM64ORshiftRO(v)
	case ssaop.OpARM64REV:
		return rewriteValue_OpARM64REV(v)
	case ssaop.OpARM64REV16:
		return rewriteValue_OpARM64REV16(v)
	case ssaop.OpARM64REVW:
		return rewriteValue_OpARM64REVW(v)
	case ssaop.OpARM64ROR:
		return rewriteValue_OpARM64ROR(v)
	case ssaop.OpARM64RORW:
		return rewriteValue_OpARM64RORW(v)
	case ssaop.OpARM64SBCSflags:
		return rewriteValue_OpARM64SBCSflags(v)
	case ssaop.OpARM64SBFX:
		return rewriteValue_OpARM64SBFX(v)
	case ssaop.OpARM64SLL:
		return rewriteValue_OpARM64SLL(v)
	case ssaop.OpARM64SLLconst:
		return rewriteValue_OpARM64SLLconst(v)
	case ssaop.OpARM64SRA:
		return rewriteValue_OpARM64SRA(v)
	case ssaop.OpARM64SRAconst:
		return rewriteValue_OpARM64SRAconst(v)
	case ssaop.OpARM64SRL:
		return rewriteValue_OpARM64SRL(v)
	case ssaop.OpARM64SRLconst:
		return rewriteValue_OpARM64SRLconst(v)
	case ssaop.OpARM64STP:
		return rewriteValue_OpARM64STP(v)
	case ssaop.OpARM64SUB:
		return rewriteValue_OpARM64SUB(v)
	case ssaop.OpARM64SUBconst:
		return rewriteValue_OpARM64SUBconst(v)
	case ssaop.OpARM64SUBshiftLL:
		return rewriteValue_OpARM64SUBshiftLL(v)
	case ssaop.OpARM64SUBshiftRA:
		return rewriteValue_OpARM64SUBshiftRA(v)
	case ssaop.OpARM64SUBshiftRL:
		return rewriteValue_OpARM64SUBshiftRL(v)
	case ssaop.OpARM64TST:
		return rewriteValue_OpARM64TST(v)
	case ssaop.OpARM64TSTW:
		return rewriteValue_OpARM64TSTW(v)
	case ssaop.OpARM64TSTWconst:
		return rewriteValue_OpARM64TSTWconst(v)
	case ssaop.OpARM64TSTconst:
		return rewriteValue_OpARM64TSTconst(v)
	case ssaop.OpARM64TSTshiftLL:
		return rewriteValue_OpARM64TSTshiftLL(v)
	case ssaop.OpARM64TSTshiftRA:
		return rewriteValue_OpARM64TSTshiftRA(v)
	case ssaop.OpARM64TSTshiftRL:
		return rewriteValue_OpARM64TSTshiftRL(v)
	case ssaop.OpARM64TSTshiftRO:
		return rewriteValue_OpARM64TSTshiftRO(v)
	case ssaop.OpARM64UBFIZ:
		return rewriteValue_OpARM64UBFIZ(v)
	case ssaop.OpARM64UBFX:
		return rewriteValue_OpARM64UBFX(v)
	case ssaop.OpARM64UDIV:
		return rewriteValue_OpARM64UDIV(v)
	case ssaop.OpARM64UDIVW:
		return rewriteValue_OpARM64UDIVW(v)
	case ssaop.OpARM64UMOD:
		return rewriteValue_OpARM64UMOD(v)
	case ssaop.OpARM64UMODW:
		return rewriteValue_OpARM64UMODW(v)
	case ssaop.OpARM64VBIF16B:
		return rewriteValue_OpARM64VBIF16B(v)
	case ssaop.OpARM64VBIT16B:
		return rewriteValue_OpARM64VBIT16B(v)
	case ssaop.OpARM64VDUPBbcast:
		return rewriteValue_OpARM64VDUPBbcast(v)
	case ssaop.OpARM64VEOR16B:
		return rewriteValue_OpARM64VEOR16B(v)
	case ssaop.OpARM64VFCVTL4S:
		return rewriteValue_OpARM64VFCVTL4S(v)
	case ssaop.OpARM64VMOVDins0:
		return rewriteValue_OpARM64VMOVDins0(v)
	case ssaop.OpARM64VMOVSins0:
		return rewriteValue_OpARM64VMOVSins0(v)
	case ssaop.OpARM64VNOT16B:
		return rewriteValue_OpARM64VNOT16B(v)
	case ssaop.OpARM64VPMULL2D:
		return rewriteValue_OpARM64VPMULL2D(v)
	case ssaop.OpARM64VSHL16B:
		return rewriteValue_OpARM64VSHL16B(v)
	case ssaop.OpARM64VSHL2D:
		return rewriteValue_OpARM64VSHL2D(v)
	case ssaop.OpARM64VSHL4S:
		return rewriteValue_OpARM64VSHL4S(v)
	case ssaop.OpARM64VSHL8H:
		return rewriteValue_OpARM64VSHL8H(v)
	case ssaop.OpARM64VSHRN2D:
		return rewriteValue_OpARM64VSHRN2D(v)
	case ssaop.OpARM64VSHRN4S:
		return rewriteValue_OpARM64VSHRN4S(v)
	case ssaop.OpARM64VSHRN8H:
		return rewriteValue_OpARM64VSHRN8H(v)
	case ssaop.OpARM64VSMULL16B:
		return rewriteValue_OpARM64VSMULL16B(v)
	case ssaop.OpARM64VSMULL4S:
		return rewriteValue_OpARM64VSMULL4S(v)
	case ssaop.OpARM64VSMULL8H:
		return rewriteValue_OpARM64VSMULL8H(v)
	case ssaop.OpARM64VSQSHL16Bconst:
		return rewriteValue_OpARM64VSQSHL16Bconst(v)
	case ssaop.OpARM64VSQSHL2Dconst:
		return rewriteValue_OpARM64VSQSHL2Dconst(v)
	case ssaop.OpARM64VSQSHL4Sconst:
		return rewriteValue_OpARM64VSQSHL4Sconst(v)
	case ssaop.OpARM64VSQSHL8Hconst:
		return rewriteValue_OpARM64VSQSHL8Hconst(v)
	case ssaop.OpARM64VSSHLL16B:
		return rewriteValue_OpARM64VSSHLL16B(v)
	case ssaop.OpARM64VSSHLL4S:
		return rewriteValue_OpARM64VSSHLL4S(v)
	case ssaop.OpARM64VSSHLL8H:
		return rewriteValue_OpARM64VSSHLL8H(v)
	case ssaop.OpARM64VSSHR16B:
		return rewriteValue_OpARM64VSSHR16B(v)
	case ssaop.OpARM64VSSHR2D:
		return rewriteValue_OpARM64VSSHR2D(v)
	case ssaop.OpARM64VSSHR4S:
		return rewriteValue_OpARM64VSSHR4S(v)
	case ssaop.OpARM64VSSHR8H:
		return rewriteValue_OpARM64VSSHR8H(v)
	case ssaop.OpARM64VSXTL16B:
		return rewriteValue_OpARM64VSXTL16B(v)
	case ssaop.OpARM64VSXTL4S:
		return rewriteValue_OpARM64VSXTL4S(v)
	case ssaop.OpARM64VSXTL8H:
		return rewriteValue_OpARM64VSXTL8H(v)
	case ssaop.OpARM64VUMULL16B:
		return rewriteValue_OpARM64VUMULL16B(v)
	case ssaop.OpARM64VUMULL4S:
		return rewriteValue_OpARM64VUMULL4S(v)
	case ssaop.OpARM64VUMULL8H:
		return rewriteValue_OpARM64VUMULL8H(v)
	case ssaop.OpARM64VUQSHL16Bconst:
		return rewriteValue_OpARM64VUQSHL16Bconst(v)
	case ssaop.OpARM64VUQSHL2Dconst:
		return rewriteValue_OpARM64VUQSHL2Dconst(v)
	case ssaop.OpARM64VUQSHL4Sconst:
		return rewriteValue_OpARM64VUQSHL4Sconst(v)
	case ssaop.OpARM64VUQSHL8Hconst:
		return rewriteValue_OpARM64VUQSHL8Hconst(v)
	case ssaop.OpARM64VUSHLL16B:
		return rewriteValue_OpARM64VUSHLL16B(v)
	case ssaop.OpARM64VUSHLL4S:
		return rewriteValue_OpARM64VUSHLL4S(v)
	case ssaop.OpARM64VUSHLL8H:
		return rewriteValue_OpARM64VUSHLL8H(v)
	case ssaop.OpARM64VUSHR16B:
		return rewriteValue_OpARM64VUSHR16B(v)
	case ssaop.OpARM64VUSHR2D:
		return rewriteValue_OpARM64VUSHR2D(v)
	case ssaop.OpARM64VUSHR4S:
		return rewriteValue_OpARM64VUSHR4S(v)
	case ssaop.OpARM64VUSHR8H:
		return rewriteValue_OpARM64VUSHR8H(v)
	case ssaop.OpARM64VUXTL16B:
		return rewriteValue_OpARM64VUXTL16B(v)
	case ssaop.OpARM64VUXTL4S:
		return rewriteValue_OpARM64VUXTL4S(v)
	case ssaop.OpARM64VUXTL8H:
		return rewriteValue_OpARM64VUXTL8H(v)
	case ssaop.OpARM64XOR:
		return rewriteValue_OpARM64XOR(v)
	case ssaop.OpARM64XORconst:
		return rewriteValue_OpARM64XORconst(v)
	case ssaop.OpARM64XORshiftLL:
		return rewriteValue_OpARM64XORshiftLL(v)
	case ssaop.OpARM64XORshiftRA:
		return rewriteValue_OpARM64XORshiftRA(v)
	case ssaop.OpARM64XORshiftRL:
		return rewriteValue_OpARM64XORshiftRL(v)
	case ssaop.OpARM64XORshiftRO:
		return rewriteValue_OpARM64XORshiftRO(v)
	case ssaop.OpARM64ZSELB:
		return rewriteValue_OpARM64ZSELB(v)
	case ssaop.OpARM64ZSELD:
		return rewriteValue_OpARM64ZSELD(v)
	case ssaop.OpARM64ZSELH:
		return rewriteValue_OpARM64ZSELH(v)
	case ssaop.OpARM64ZSELS:
		return rewriteValue_OpARM64ZSELS(v)
	case ssaop.OpAbs:
		v.Op = ssaop.OpARM64FABSD
		return true
	case ssaop.OpAbsFloat32x4:
		v.Op = ssaop.OpARM64VFABS4S
		return true
	case ssaop.OpAbsFloat64x2:
		v.Op = ssaop.OpARM64VFABS2D
		return true
	case ssaop.OpAbsInt16s:
		return rewriteValue_OpAbsInt16s(v)
	case ssaop.OpAbsInt16x8:
		v.Op = ssaop.OpARM64VABS8H
		return true
	case ssaop.OpAbsInt32s:
		return rewriteValue_OpAbsInt32s(v)
	case ssaop.OpAbsInt32x4:
		v.Op = ssaop.OpARM64VABS4S
		return true
	case ssaop.OpAbsInt64s:
		return rewriteValue_OpAbsInt64s(v)
	case ssaop.OpAbsInt64x2:
		v.Op = ssaop.OpARM64VABS2D
		return true
	case ssaop.OpAbsInt8s:
		return rewriteValue_OpAbsInt8s(v)
	case ssaop.OpAbsInt8x16:
		v.Op = ssaop.OpARM64VABS16B
		return true
	case ssaop.OpAdd16:
		v.Op = ssaop.OpARM64ADD
		return true
	case ssaop.OpAdd32:
		v.Op = ssaop.OpARM64ADD
		return true
	case ssaop.OpAdd32F:
		v.Op = ssaop.OpARM64FADDS
		return true
	case ssaop.OpAdd64:
		v.Op = ssaop.OpARM64ADD
		return true
	case ssaop.OpAdd64F:
		v.Op = ssaop.OpARM64FADDD
		return true
	case ssaop.OpAdd8:
		v.Op = ssaop.OpARM64ADD
		return true
	case ssaop.OpAddFloat32s:
		v.Op = ssaop.OpARM64ZFADDS
		return true
	case ssaop.OpAddFloat32x4:
		v.Op = ssaop.OpARM64VFADD4S
		return true
	case ssaop.OpAddFloat64s:
		v.Op = ssaop.OpARM64ZFADDD
		return true
	case ssaop.OpAddFloat64x2:
		v.Op = ssaop.OpARM64VFADD2D
		return true
	case ssaop.OpAddInt16s:
		v.Op = ssaop.OpARM64ZADDH
		return true
	case ssaop.OpAddInt16x8:
		v.Op = ssaop.OpARM64VADD8H
		return true
	case ssaop.OpAddInt32s:
		v.Op = ssaop.OpARM64ZADDS
		return true
	case ssaop.OpAddInt32x4:
		v.Op = ssaop.OpARM64VADD4S
		return true
	case ssaop.OpAddInt64s:
		v.Op = ssaop.OpARM64ZADDD
		return true
	case ssaop.OpAddInt64x2:
		v.Op = ssaop.OpARM64VADD2D
		return true
	case ssaop.OpAddInt8s:
		v.Op = ssaop.OpARM64ZADDB
		return true
	case ssaop.OpAddInt8x16:
		v.Op = ssaop.OpARM64VADD16B
		return true
	case ssaop.OpAddPtr:
		v.Op = ssaop.OpARM64ADD
		return true
	case ssaop.OpAddSaturatedInt16s:
		v.Op = ssaop.OpARM64ZSQADDH
		return true
	case ssaop.OpAddSaturatedInt16x8:
		v.Op = ssaop.OpARM64VSQADD8H
		return true
	case ssaop.OpAddSaturatedInt32s:
		v.Op = ssaop.OpARM64ZSQADDS
		return true
	case ssaop.OpAddSaturatedInt32x4:
		v.Op = ssaop.OpARM64VSQADD4S
		return true
	case ssaop.OpAddSaturatedInt64s:
		v.Op = ssaop.OpARM64ZSQADDD
		return true
	case ssaop.OpAddSaturatedInt64x2:
		v.Op = ssaop.OpARM64VSQADD2D
		return true
	case ssaop.OpAddSaturatedInt8s:
		v.Op = ssaop.OpARM64ZSQADDB
		return true
	case ssaop.OpAddSaturatedInt8x16:
		v.Op = ssaop.OpARM64VSQADD16B
		return true
	case ssaop.OpAddSaturatedUint16s:
		v.Op = ssaop.OpARM64ZUQADDH
		return true
	case ssaop.OpAddSaturatedUint16x8:
		v.Op = ssaop.OpARM64VUQADD8H
		return true
	case ssaop.OpAddSaturatedUint32s:
		v.Op = ssaop.OpARM64ZUQADDS
		return true
	case ssaop.OpAddSaturatedUint32x4:
		v.Op = ssaop.OpARM64VUQADD4S
		return true
	case ssaop.OpAddSaturatedUint64s:
		v.Op = ssaop.OpARM64ZUQADDD
		return true
	case ssaop.OpAddSaturatedUint64x2:
		v.Op = ssaop.OpARM64VUQADD2D
		return true
	case ssaop.OpAddSaturatedUint8s:
		v.Op = ssaop.OpARM64ZUQADDB
		return true
	case ssaop.OpAddSaturatedUint8x16:
		v.Op = ssaop.OpARM64VUQADD16B
		return true
	case ssaop.OpAddUint16s:
		v.Op = ssaop.OpARM64ZADDH
		return true
	case ssaop.OpAddUint16x8:
		v.Op = ssaop.OpARM64VADD8H
		return true
	case ssaop.OpAddUint32s:
		v.Op = ssaop.OpARM64ZADDS
		return true
	case ssaop.OpAddUint32x4:
		v.Op = ssaop.OpARM64VADD4S
		return true
	case ssaop.OpAddUint64s:
		v.Op = ssaop.OpARM64ZADDD
		return true
	case ssaop.OpAddUint64x2:
		v.Op = ssaop.OpARM64VADD2D
		return true
	case ssaop.OpAddUint8s:
		v.Op = ssaop.OpARM64ZADDB
		return true
	case ssaop.OpAddUint8x16:
		v.Op = ssaop.OpARM64VADD16B
		return true
	case ssaop.OpAddr:
		return rewriteValue_OpAddr(v)
	case ssaop.OpAnd16:
		v.Op = ssaop.OpARM64AND
		return true
	case ssaop.OpAnd32:
		v.Op = ssaop.OpARM64AND
		return true
	case ssaop.OpAnd64:
		v.Op = ssaop.OpARM64AND
		return true
	case ssaop.OpAnd8:
		v.Op = ssaop.OpARM64AND
		return true
	case ssaop.OpAndB:
		v.Op = ssaop.OpARM64AND
		return true
	case ssaop.OpAndInt16x8:
		v.Op = ssaop.OpARM64VAND16B
		return true
	case ssaop.OpAndInt32x4:
		v.Op = ssaop.OpARM64VAND16B
		return true
	case ssaop.OpAndInt64x2:
		v.Op = ssaop.OpARM64VAND16B
		return true
	case ssaop.OpAndInt8x16:
		v.Op = ssaop.OpARM64VAND16B
		return true
	case ssaop.OpAndNotInt16x8:
		v.Op = ssaop.OpARM64VBIC16B
		return true
	case ssaop.OpAndNotInt32x4:
		v.Op = ssaop.OpARM64VBIC16B
		return true
	case ssaop.OpAndNotInt64x2:
		v.Op = ssaop.OpARM64VBIC16B
		return true
	case ssaop.OpAndNotInt8x16:
		v.Op = ssaop.OpARM64VBIC16B
		return true
	case ssaop.OpAndNotUint16x8:
		v.Op = ssaop.OpARM64VBIC16B
		return true
	case ssaop.OpAndNotUint32x4:
		v.Op = ssaop.OpARM64VBIC16B
		return true
	case ssaop.OpAndNotUint64x2:
		v.Op = ssaop.OpARM64VBIC16B
		return true
	case ssaop.OpAndNotUint8x16:
		v.Op = ssaop.OpARM64VBIC16B
		return true
	case ssaop.OpAndUint16x8:
		v.Op = ssaop.OpARM64VAND16B
		return true
	case ssaop.OpAndUint32x4:
		v.Op = ssaop.OpARM64VAND16B
		return true
	case ssaop.OpAndUint64x2:
		v.Op = ssaop.OpARM64VAND16B
		return true
	case ssaop.OpAndUint8x16:
		v.Op = ssaop.OpARM64VAND16B
		return true
	case ssaop.OpAtomicAdd32:
		v.Op = ssaop.OpARM64LoweredAtomicAdd32
		return true
	case ssaop.OpAtomicAdd32Variant:
		v.Op = ssaop.OpARM64LoweredAtomicAdd32Variant
		return true
	case ssaop.OpAtomicAdd64:
		v.Op = ssaop.OpARM64LoweredAtomicAdd64
		return true
	case ssaop.OpAtomicAdd64Variant:
		v.Op = ssaop.OpARM64LoweredAtomicAdd64Variant
		return true
	case ssaop.OpAtomicAnd32value:
		v.Op = ssaop.OpARM64LoweredAtomicAnd32
		return true
	case ssaop.OpAtomicAnd32valueVariant:
		v.Op = ssaop.OpARM64LoweredAtomicAnd32Variant
		return true
	case ssaop.OpAtomicAnd64value:
		v.Op = ssaop.OpARM64LoweredAtomicAnd64
		return true
	case ssaop.OpAtomicAnd64valueVariant:
		v.Op = ssaop.OpARM64LoweredAtomicAnd64Variant
		return true
	case ssaop.OpAtomicAnd8value:
		v.Op = ssaop.OpARM64LoweredAtomicAnd8
		return true
	case ssaop.OpAtomicAnd8valueVariant:
		v.Op = ssaop.OpARM64LoweredAtomicAnd8Variant
		return true
	case ssaop.OpAtomicCompareAndSwap32:
		v.Op = ssaop.OpARM64LoweredAtomicCas32
		return true
	case ssaop.OpAtomicCompareAndSwap32Variant:
		v.Op = ssaop.OpARM64LoweredAtomicCas32Variant
		return true
	case ssaop.OpAtomicCompareAndSwap64:
		v.Op = ssaop.OpARM64LoweredAtomicCas64
		return true
	case ssaop.OpAtomicCompareAndSwap64Variant:
		v.Op = ssaop.OpARM64LoweredAtomicCas64Variant
		return true
	case ssaop.OpAtomicExchange32:
		v.Op = ssaop.OpARM64LoweredAtomicExchange32
		return true
	case ssaop.OpAtomicExchange32Variant:
		v.Op = ssaop.OpARM64LoweredAtomicExchange32Variant
		return true
	case ssaop.OpAtomicExchange64:
		v.Op = ssaop.OpARM64LoweredAtomicExchange64
		return true
	case ssaop.OpAtomicExchange64Variant:
		v.Op = ssaop.OpARM64LoweredAtomicExchange64Variant
		return true
	case ssaop.OpAtomicExchange8:
		v.Op = ssaop.OpARM64LoweredAtomicExchange8
		return true
	case ssaop.OpAtomicExchange8Variant:
		v.Op = ssaop.OpARM64LoweredAtomicExchange8Variant
		return true
	case ssaop.OpAtomicLoad32:
		v.Op = ssaop.OpARM64LDARW
		return true
	case ssaop.OpAtomicLoad64:
		v.Op = ssaop.OpARM64LDAR
		return true
	case ssaop.OpAtomicLoad8:
		v.Op = ssaop.OpARM64LDARB
		return true
	case ssaop.OpAtomicLoadPtr:
		v.Op = ssaop.OpARM64LDAR
		return true
	case ssaop.OpAtomicOr32value:
		v.Op = ssaop.OpARM64LoweredAtomicOr32
		return true
	case ssaop.OpAtomicOr32valueVariant:
		v.Op = ssaop.OpARM64LoweredAtomicOr32Variant
		return true
	case ssaop.OpAtomicOr64value:
		v.Op = ssaop.OpARM64LoweredAtomicOr64
		return true
	case ssaop.OpAtomicOr64valueVariant:
		v.Op = ssaop.OpARM64LoweredAtomicOr64Variant
		return true
	case ssaop.OpAtomicOr8value:
		v.Op = ssaop.OpARM64LoweredAtomicOr8
		return true
	case ssaop.OpAtomicOr8valueVariant:
		v.Op = ssaop.OpARM64LoweredAtomicOr8Variant
		return true
	case ssaop.OpAtomicStore32:
		v.Op = ssaop.OpARM64STLRW
		return true
	case ssaop.OpAtomicStore64:
		v.Op = ssaop.OpARM64STLR
		return true
	case ssaop.OpAtomicStore8:
		v.Op = ssaop.OpARM64STLRB
		return true
	case ssaop.OpAtomicStorePtrNoWB:
		v.Op = ssaop.OpARM64STLR
		return true
	case ssaop.OpAverageInt16x8:
		v.Op = ssaop.OpARM64VSRHADD8H
		return true
	case ssaop.OpAverageInt32x4:
		v.Op = ssaop.OpARM64VSRHADD4S
		return true
	case ssaop.OpAverageInt8x16:
		v.Op = ssaop.OpARM64VSRHADD16B
		return true
	case ssaop.OpAverageUint16x8:
		v.Op = ssaop.OpARM64VURHADD8H
		return true
	case ssaop.OpAverageUint32x4:
		v.Op = ssaop.OpARM64VURHADD4S
		return true
	case ssaop.OpAverageUint8x16:
		v.Op = ssaop.OpARM64VURHADD16B
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
		v.Op = ssaop.OpARM64RBITW
		return true
	case ssaop.OpBitRev64:
		v.Op = ssaop.OpARM64RBIT
		return true
	case ssaop.OpBitRev8:
		return rewriteValue_OpBitRev8(v)
	case ssaop.OpBswap16:
		v.Op = ssaop.OpARM64REV16W
		return true
	case ssaop.OpBswap32:
		v.Op = ssaop.OpARM64REVW
		return true
	case ssaop.OpBswap64:
		v.Op = ssaop.OpARM64REV
		return true
	case ssaop.OpCeil:
		v.Op = ssaop.OpARM64FRINTPD
		return true
	case ssaop.OpCeilFloat32x4:
		v.Op = ssaop.OpARM64VFRINTP4S
		return true
	case ssaop.OpCeilFloat64x2:
		v.Op = ssaop.OpARM64VFRINTP2D
		return true
	case ssaop.OpClosureCall:
		v.Op = ssaop.OpARM64CALLclosure
		return true
	case ssaop.OpCom16:
		v.Op = ssaop.OpARM64MVN
		return true
	case ssaop.OpCom32:
		v.Op = ssaop.OpARM64MVN
		return true
	case ssaop.OpCom64:
		v.Op = ssaop.OpARM64MVN
		return true
	case ssaop.OpCom8:
		v.Op = ssaop.OpARM64MVN
		return true
	case ssaop.OpConcatAddPairsFloat32x4:
		v.Op = ssaop.OpARM64VFADDP4S
		return true
	case ssaop.OpConcatAddPairsFloat64x2:
		v.Op = ssaop.OpARM64VFADDP2D
		return true
	case ssaop.OpConcatAddPairsInt16x8:
		v.Op = ssaop.OpARM64VADDP8H
		return true
	case ssaop.OpConcatAddPairsInt32x4:
		v.Op = ssaop.OpARM64VADDP4S
		return true
	case ssaop.OpConcatAddPairsInt64x2:
		v.Op = ssaop.OpARM64VADDP2D
		return true
	case ssaop.OpConcatAddPairsUint16x8:
		v.Op = ssaop.OpARM64VADDP8H
		return true
	case ssaop.OpConcatAddPairsUint32x4:
		v.Op = ssaop.OpARM64VADDP4S
		return true
	case ssaop.OpConcatAddPairsUint64x2:
		v.Op = ssaop.OpARM64VADDP2D
		return true
	case ssaop.OpConcatEvenInt16x8:
		v.Op = ssaop.OpARM64VUZP18H
		return true
	case ssaop.OpConcatEvenInt32x4:
		v.Op = ssaop.OpARM64VUZP14S
		return true
	case ssaop.OpConcatEvenInt64x2:
		v.Op = ssaop.OpARM64VUZP12D
		return true
	case ssaop.OpConcatEvenInt8x16:
		v.Op = ssaop.OpARM64VUZP116B
		return true
	case ssaop.OpConcatEvenUint16x8:
		v.Op = ssaop.OpARM64VUZP18H
		return true
	case ssaop.OpConcatEvenUint32x4:
		v.Op = ssaop.OpARM64VUZP14S
		return true
	case ssaop.OpConcatEvenUint64x2:
		v.Op = ssaop.OpARM64VUZP12D
		return true
	case ssaop.OpConcatEvenUint8x16:
		v.Op = ssaop.OpARM64VUZP116B
		return true
	case ssaop.OpConcatOddInt16x8:
		v.Op = ssaop.OpARM64VUZP28H
		return true
	case ssaop.OpConcatOddInt32x4:
		v.Op = ssaop.OpARM64VUZP24S
		return true
	case ssaop.OpConcatOddInt64x2:
		v.Op = ssaop.OpARM64VUZP22D
		return true
	case ssaop.OpConcatOddInt8x16:
		v.Op = ssaop.OpARM64VUZP216B
		return true
	case ssaop.OpConcatOddUint16x8:
		v.Op = ssaop.OpARM64VUZP28H
		return true
	case ssaop.OpConcatOddUint32x4:
		v.Op = ssaop.OpARM64VUZP24S
		return true
	case ssaop.OpConcatOddUint64x2:
		v.Op = ssaop.OpARM64VUZP22D
		return true
	case ssaop.OpConcatOddUint8x16:
		v.Op = ssaop.OpARM64VUZP216B
		return true
	case ssaop.OpConcatShiftBytesRightUint8x16:
		v.Op = ssaop.OpARM64VEXT16B
		return true
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
	case ssaop.OpConvertLo2ToFloat64Float32x4:
		v.Op = ssaop.OpARM64VFCVTL4S
		return true
	case ssaop.OpConvertToFloat32Float64x2:
		v.Op = ssaop.OpARM64VFCVTN2D
		return true
	case ssaop.OpConvertToFloat32Int32x4:
		v.Op = ssaop.OpARM64VSCVTF4S
		return true
	case ssaop.OpConvertToFloat32Uint32x4:
		v.Op = ssaop.OpARM64VUCVTF4S
		return true
	case ssaop.OpConvertToFloat64Int64x2:
		v.Op = ssaop.OpARM64VSCVTF2D
		return true
	case ssaop.OpConvertToFloat64Uint64x2:
		v.Op = ssaop.OpARM64VUCVTF2D
		return true
	case ssaop.OpConvertToInt32Float32x4:
		v.Op = ssaop.OpARM64VFCVTZS4S
		return true
	case ssaop.OpConvertToInt64Float64x2:
		v.Op = ssaop.OpARM64VFCVTZS2D
		return true
	case ssaop.OpConvertToUint32Float32x4:
		v.Op = ssaop.OpARM64VFCVTZU4S
		return true
	case ssaop.OpConvertToUint64Float64x2:
		v.Op = ssaop.OpARM64VFCVTZU2D
		return true
	case ssaop.OpCount8s:
		return rewriteValue_OpCount8s(v)
	case ssaop.OpCtz16:
		return rewriteValue_OpCtz16(v)
	case ssaop.OpCtz16NonZero:
		v.Op = ssaop.OpCtz32
		return true
	case ssaop.OpCtz32:
		return rewriteValue_OpCtz32(v)
	case ssaop.OpCtz32NonZero:
		v.Op = ssaop.OpCtz32
		return true
	case ssaop.OpCtz64:
		return rewriteValue_OpCtz64(v)
	case ssaop.OpCtz64NonZero:
		v.Op = ssaop.OpCtz64
		return true
	case ssaop.OpCtz8:
		return rewriteValue_OpCtz8(v)
	case ssaop.OpCtz8NonZero:
		v.Op = ssaop.OpCtz32
		return true
	case ssaop.OpCvt32Fto32:
		v.Op = ssaop.OpARM64FCVTZSSW
		return true
	case ssaop.OpCvt32Fto32U:
		v.Op = ssaop.OpARM64FCVTZUSW
		return true
	case ssaop.OpCvt32Fto64:
		v.Op = ssaop.OpARM64FCVTZSS
		return true
	case ssaop.OpCvt32Fto64F:
		v.Op = ssaop.OpARM64FCVTSD
		return true
	case ssaop.OpCvt32Fto64U:
		v.Op = ssaop.OpARM64FCVTZUS
		return true
	case ssaop.OpCvt32Uto32F:
		v.Op = ssaop.OpARM64UCVTFWS
		return true
	case ssaop.OpCvt32Uto64F:
		v.Op = ssaop.OpARM64UCVTFWD
		return true
	case ssaop.OpCvt32to32F:
		v.Op = ssaop.OpARM64SCVTFWS
		return true
	case ssaop.OpCvt32to64F:
		v.Op = ssaop.OpARM64SCVTFWD
		return true
	case ssaop.OpCvt64Fto32:
		v.Op = ssaop.OpARM64FCVTZSDW
		return true
	case ssaop.OpCvt64Fto32F:
		v.Op = ssaop.OpARM64FCVTDS
		return true
	case ssaop.OpCvt64Fto32U:
		v.Op = ssaop.OpARM64FCVTZUDW
		return true
	case ssaop.OpCvt64Fto64:
		v.Op = ssaop.OpARM64FCVTZSD
		return true
	case ssaop.OpCvt64Fto64U:
		v.Op = ssaop.OpARM64FCVTZUD
		return true
	case ssaop.OpCvt64Uto32F:
		v.Op = ssaop.OpARM64UCVTFS
		return true
	case ssaop.OpCvt64Uto64F:
		v.Op = ssaop.OpARM64UCVTFD
		return true
	case ssaop.OpCvt64to32F:
		v.Op = ssaop.OpARM64SCVTFS
		return true
	case ssaop.OpCvt64to64F:
		v.Op = ssaop.OpARM64SCVTFD
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
		v.Op = ssaop.OpARM64FDIVS
		return true
	case ssaop.OpDiv32u:
		v.Op = ssaop.OpARM64UDIVW
		return true
	case ssaop.OpDiv64:
		return rewriteValue_OpDiv64(v)
	case ssaop.OpDiv64F:
		v.Op = ssaop.OpARM64FDIVD
		return true
	case ssaop.OpDiv64u:
		v.Op = ssaop.OpARM64UDIV
		return true
	case ssaop.OpDiv8:
		return rewriteValue_OpDiv8(v)
	case ssaop.OpDiv8u:
		return rewriteValue_OpDiv8u(v)
	case ssaop.OpDivFloat32x4:
		v.Op = ssaop.OpARM64VFDIV4S
		return true
	case ssaop.OpDivFloat64x2:
		v.Op = ssaop.OpARM64VFDIV2D
		return true
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
	case ssaop.OpEqualFloat32x4:
		v.Op = ssaop.OpARM64VFCMEQ4S
		return true
	case ssaop.OpEqualFloat64x2:
		v.Op = ssaop.OpARM64VFCMEQ2D
		return true
	case ssaop.OpEqualInt16x8:
		v.Op = ssaop.OpARM64VCMEQ8H
		return true
	case ssaop.OpEqualInt32x4:
		v.Op = ssaop.OpARM64VCMEQ4S
		return true
	case ssaop.OpEqualInt64x2:
		v.Op = ssaop.OpARM64VCMEQ2D
		return true
	case ssaop.OpEqualInt8x16:
		v.Op = ssaop.OpARM64VCMEQ16B
		return true
	case ssaop.OpEqualUint16x8:
		v.Op = ssaop.OpARM64VCMEQ8H
		return true
	case ssaop.OpEqualUint32x4:
		v.Op = ssaop.OpARM64VCMEQ4S
		return true
	case ssaop.OpEqualUint64x2:
		v.Op = ssaop.OpARM64VCMEQ2D
		return true
	case ssaop.OpEqualUint8x16:
		v.Op = ssaop.OpARM64VCMEQ16B
		return true
	case ssaop.OpExtendLo2ToInt64Int32x4:
		v.Op = ssaop.OpARM64VSXTL4S
		return true
	case ssaop.OpExtendLo2ToUint64Uint32x4:
		v.Op = ssaop.OpARM64VUXTL4S
		return true
	case ssaop.OpExtendLo4ToInt32Int16x8:
		v.Op = ssaop.OpARM64VSXTL8H
		return true
	case ssaop.OpExtendLo4ToUint32Uint16x8:
		v.Op = ssaop.OpARM64VUXTL8H
		return true
	case ssaop.OpExtendLo8ToInt16Int8x16:
		v.Op = ssaop.OpARM64VSXTL16B
		return true
	case ssaop.OpExtendLo8ToUint16Uint8x16:
		v.Op = ssaop.OpARM64VUXTL16B
		return true
	case ssaop.OpFMA:
		return rewriteValue_OpFMA(v)
	case ssaop.OpFloor:
		v.Op = ssaop.OpARM64FRINTMD
		return true
	case ssaop.OpFloorFloat32x4:
		v.Op = ssaop.OpARM64VFRINTM4S
		return true
	case ssaop.OpFloorFloat64x2:
		v.Op = ssaop.OpARM64VFRINTM2D
		return true
	case ssaop.OpGetCallerPC:
		v.Op = ssaop.OpARM64LoweredGetCallerPC
		return true
	case ssaop.OpGetCallerSP:
		v.Op = ssaop.OpARM64LoweredGetCallerSP
		return true
	case ssaop.OpGetClosurePtr:
		v.Op = ssaop.OpARM64LoweredGetClosurePtr
		return true
	case ssaop.OpGetElemFloat32x4:
		v.Op = ssaop.OpARM64VDUPSextr
		return true
	case ssaop.OpGetElemFloat64x2:
		v.Op = ssaop.OpARM64VDUPDextr
		return true
	case ssaop.OpGetElemInt16x8:
		v.Op = ssaop.OpARM64VMOVHextr
		return true
	case ssaop.OpGetElemInt32x4:
		v.Op = ssaop.OpARM64VMOVSextr
		return true
	case ssaop.OpGetElemInt64x2:
		v.Op = ssaop.OpARM64VMOVDextr
		return true
	case ssaop.OpGetElemInt8x16:
		v.Op = ssaop.OpARM64VMOVBextr
		return true
	case ssaop.OpGetElemUint16x8:
		v.Op = ssaop.OpARM64VMOVHextr
		return true
	case ssaop.OpGetElemUint32x4:
		v.Op = ssaop.OpARM64VMOVSextr
		return true
	case ssaop.OpGetElemUint64x2:
		v.Op = ssaop.OpARM64VMOVDextr
		return true
	case ssaop.OpGetElemUint8x16:
		v.Op = ssaop.OpARM64VMOVBextr
		return true
	case ssaop.OpGreaterEqualFloat32x4:
		v.Op = ssaop.OpARM64VFCMGE4S
		return true
	case ssaop.OpGreaterEqualFloat64x2:
		v.Op = ssaop.OpARM64VFCMGE2D
		return true
	case ssaop.OpGreaterEqualInt16x8:
		v.Op = ssaop.OpARM64VCMGE8H
		return true
	case ssaop.OpGreaterEqualInt32x4:
		v.Op = ssaop.OpARM64VCMGE4S
		return true
	case ssaop.OpGreaterEqualInt64x2:
		v.Op = ssaop.OpARM64VCMGE2D
		return true
	case ssaop.OpGreaterEqualInt8x16:
		v.Op = ssaop.OpARM64VCMGE16B
		return true
	case ssaop.OpGreaterEqualUint16x8:
		v.Op = ssaop.OpARM64VCMHS8H
		return true
	case ssaop.OpGreaterEqualUint32x4:
		v.Op = ssaop.OpARM64VCMHS4S
		return true
	case ssaop.OpGreaterEqualUint64x2:
		v.Op = ssaop.OpARM64VCMHS2D
		return true
	case ssaop.OpGreaterEqualUint8x16:
		v.Op = ssaop.OpARM64VCMHS16B
		return true
	case ssaop.OpGreaterFloat32x4:
		v.Op = ssaop.OpARM64VFCMGT4S
		return true
	case ssaop.OpGreaterFloat64x2:
		v.Op = ssaop.OpARM64VFCMGT2D
		return true
	case ssaop.OpGreaterInt16s:
		return rewriteValue_OpGreaterInt16s(v)
	case ssaop.OpGreaterInt16x8:
		v.Op = ssaop.OpARM64VCMGT8H
		return true
	case ssaop.OpGreaterInt32s:
		return rewriteValue_OpGreaterInt32s(v)
	case ssaop.OpGreaterInt32x4:
		v.Op = ssaop.OpARM64VCMGT4S
		return true
	case ssaop.OpGreaterInt64s:
		return rewriteValue_OpGreaterInt64s(v)
	case ssaop.OpGreaterInt64x2:
		v.Op = ssaop.OpARM64VCMGT2D
		return true
	case ssaop.OpGreaterInt8s:
		return rewriteValue_OpGreaterInt8s(v)
	case ssaop.OpGreaterInt8x16:
		v.Op = ssaop.OpARM64VCMGT16B
		return true
	case ssaop.OpGreaterUint16x8:
		v.Op = ssaop.OpARM64VCMHI8H
		return true
	case ssaop.OpGreaterUint32x4:
		v.Op = ssaop.OpARM64VCMHI4S
		return true
	case ssaop.OpGreaterUint64x2:
		v.Op = ssaop.OpARM64VCMHI2D
		return true
	case ssaop.OpGreaterUint8x16:
		v.Op = ssaop.OpARM64VCMHI16B
		return true
	case ssaop.OpHmul32:
		return rewriteValue_OpHmul32(v)
	case ssaop.OpHmul32u:
		return rewriteValue_OpHmul32u(v)
	case ssaop.OpHmul64:
		v.Op = ssaop.OpARM64MULH
		return true
	case ssaop.OpHmul64u:
		v.Op = ssaop.OpARM64UMULH
		return true
	case ssaop.OpIfElseFloat32s:
		return rewriteValue_OpIfElseFloat32s(v)
	case ssaop.OpIfElseFloat64s:
		return rewriteValue_OpIfElseFloat64s(v)
	case ssaop.OpIfElseInt16s:
		return rewriteValue_OpIfElseInt16s(v)
	case ssaop.OpIfElseInt32s:
		return rewriteValue_OpIfElseInt32s(v)
	case ssaop.OpIfElseInt64s:
		return rewriteValue_OpIfElseInt64s(v)
	case ssaop.OpIfElseInt8s:
		return rewriteValue_OpIfElseInt8s(v)
	case ssaop.OpIfElseUint16s:
		return rewriteValue_OpIfElseUint16s(v)
	case ssaop.OpIfElseUint32s:
		return rewriteValue_OpIfElseUint32s(v)
	case ssaop.OpIfElseUint64s:
		return rewriteValue_OpIfElseUint64s(v)
	case ssaop.OpIfElseUint8s:
		return rewriteValue_OpIfElseUint8s(v)
	case ssaop.OpInterCall:
		v.Op = ssaop.OpARM64CALLinter
		return true
	case ssaop.OpInterleaveEvenInt16x8:
		v.Op = ssaop.OpARM64VTRN18H
		return true
	case ssaop.OpInterleaveEvenInt32x4:
		v.Op = ssaop.OpARM64VTRN14S
		return true
	case ssaop.OpInterleaveEvenInt64x2:
		v.Op = ssaop.OpARM64VTRN12D
		return true
	case ssaop.OpInterleaveEvenInt8x16:
		v.Op = ssaop.OpARM64VTRN116B
		return true
	case ssaop.OpInterleaveEvenUint16x8:
		v.Op = ssaop.OpARM64VTRN18H
		return true
	case ssaop.OpInterleaveEvenUint32x4:
		v.Op = ssaop.OpARM64VTRN14S
		return true
	case ssaop.OpInterleaveEvenUint64x2:
		v.Op = ssaop.OpARM64VTRN12D
		return true
	case ssaop.OpInterleaveEvenUint8x16:
		v.Op = ssaop.OpARM64VTRN116B
		return true
	case ssaop.OpInterleaveHiInt16x8:
		v.Op = ssaop.OpARM64VZIP28H
		return true
	case ssaop.OpInterleaveHiInt32x4:
		v.Op = ssaop.OpARM64VZIP24S
		return true
	case ssaop.OpInterleaveHiInt64x2:
		v.Op = ssaop.OpARM64VZIP22D
		return true
	case ssaop.OpInterleaveHiInt8x16:
		v.Op = ssaop.OpARM64VZIP216B
		return true
	case ssaop.OpInterleaveHiUint16x8:
		v.Op = ssaop.OpARM64VZIP28H
		return true
	case ssaop.OpInterleaveHiUint32x4:
		v.Op = ssaop.OpARM64VZIP24S
		return true
	case ssaop.OpInterleaveHiUint64x2:
		v.Op = ssaop.OpARM64VZIP22D
		return true
	case ssaop.OpInterleaveHiUint8x16:
		v.Op = ssaop.OpARM64VZIP216B
		return true
	case ssaop.OpInterleaveLoInt16x8:
		v.Op = ssaop.OpARM64VZIP18H
		return true
	case ssaop.OpInterleaveLoInt32x4:
		v.Op = ssaop.OpARM64VZIP14S
		return true
	case ssaop.OpInterleaveLoInt64x2:
		v.Op = ssaop.OpARM64VZIP12D
		return true
	case ssaop.OpInterleaveLoInt8x16:
		v.Op = ssaop.OpARM64VZIP116B
		return true
	case ssaop.OpInterleaveLoUint16x8:
		v.Op = ssaop.OpARM64VZIP18H
		return true
	case ssaop.OpInterleaveLoUint32x4:
		v.Op = ssaop.OpARM64VZIP14S
		return true
	case ssaop.OpInterleaveLoUint64x2:
		v.Op = ssaop.OpARM64VZIP12D
		return true
	case ssaop.OpInterleaveLoUint8x16:
		v.Op = ssaop.OpARM64VZIP116B
		return true
	case ssaop.OpInterleaveOddInt16x8:
		v.Op = ssaop.OpARM64VTRN28H
		return true
	case ssaop.OpInterleaveOddInt32x4:
		v.Op = ssaop.OpARM64VTRN24S
		return true
	case ssaop.OpInterleaveOddInt64x2:
		v.Op = ssaop.OpARM64VTRN22D
		return true
	case ssaop.OpInterleaveOddInt8x16:
		v.Op = ssaop.OpARM64VTRN216B
		return true
	case ssaop.OpInterleaveOddUint16x8:
		v.Op = ssaop.OpARM64VTRN28H
		return true
	case ssaop.OpInterleaveOddUint32x4:
		v.Op = ssaop.OpARM64VTRN24S
		return true
	case ssaop.OpInterleaveOddUint64x2:
		v.Op = ssaop.OpARM64VTRN22D
		return true
	case ssaop.OpInterleaveOddUint8x16:
		v.Op = ssaop.OpARM64VTRN216B
		return true
	case ssaop.OpIsInBounds:
		return rewriteValue_OpIsInBounds(v)
	case ssaop.OpIsNonNil:
		return rewriteValue_OpIsNonNil(v)
	case ssaop.OpIsSliceInBounds:
		return rewriteValue_OpIsSliceInBounds(v)
	case ssaop.OpLeadingSignBitsInt16x8:
		v.Op = ssaop.OpARM64VCLS8H
		return true
	case ssaop.OpLeadingSignBitsInt32x4:
		v.Op = ssaop.OpARM64VCLS4S
		return true
	case ssaop.OpLeadingSignBitsInt8x16:
		v.Op = ssaop.OpARM64VCLS16B
		return true
	case ssaop.OpLeadingSignBitsUint16x8:
		v.Op = ssaop.OpARM64VCLS8H
		return true
	case ssaop.OpLeadingSignBitsUint32x4:
		v.Op = ssaop.OpARM64VCLS4S
		return true
	case ssaop.OpLeadingSignBitsUint8x16:
		v.Op = ssaop.OpARM64VCLS16B
		return true
	case ssaop.OpLeadingZerosInt16x8:
		v.Op = ssaop.OpARM64VCLZ8H
		return true
	case ssaop.OpLeadingZerosInt32x4:
		v.Op = ssaop.OpARM64VCLZ4S
		return true
	case ssaop.OpLeadingZerosInt8x16:
		v.Op = ssaop.OpARM64VCLZ16B
		return true
	case ssaop.OpLeadingZerosUint16x8:
		v.Op = ssaop.OpARM64VCLZ8H
		return true
	case ssaop.OpLeadingZerosUint32x4:
		v.Op = ssaop.OpARM64VCLZ4S
		return true
	case ssaop.OpLeadingZerosUint8x16:
		v.Op = ssaop.OpARM64VCLZ16B
		return true
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
	case ssaop.OpLoadMasked8:
		return rewriteValue_OpLoadMasked8(v)
	case ssaop.OpLocalAddr:
		return rewriteValue_OpLocalAddr(v)
	case ssaop.OpLookupOrZeroInt8x16:
		v.Op = ssaop.OpARM64VTBL16B
		return true
	case ssaop.OpLookupOrZeroUint8x16:
		v.Op = ssaop.OpARM64VTBL16B
		return true
	case ssaop.OpLsh16x16:
		v.Op = ssaop.OpLsh64x16
		return true
	case ssaop.OpLsh16x32:
		v.Op = ssaop.OpLsh64x32
		return true
	case ssaop.OpLsh16x64:
		v.Op = ssaop.OpLsh64x64
		return true
	case ssaop.OpLsh16x8:
		v.Op = ssaop.OpLsh64x8
		return true
	case ssaop.OpLsh32x16:
		v.Op = ssaop.OpLsh64x16
		return true
	case ssaop.OpLsh32x32:
		v.Op = ssaop.OpLsh64x32
		return true
	case ssaop.OpLsh32x64:
		v.Op = ssaop.OpLsh64x64
		return true
	case ssaop.OpLsh32x8:
		v.Op = ssaop.OpLsh64x8
		return true
	case ssaop.OpLsh64x16:
		return rewriteValue_OpLsh64x16(v)
	case ssaop.OpLsh64x32:
		return rewriteValue_OpLsh64x32(v)
	case ssaop.OpLsh64x64:
		return rewriteValue_OpLsh64x64(v)
	case ssaop.OpLsh64x8:
		return rewriteValue_OpLsh64x8(v)
	case ssaop.OpLsh8x16:
		v.Op = ssaop.OpLsh64x16
		return true
	case ssaop.OpLsh8x32:
		v.Op = ssaop.OpLsh64x32
		return true
	case ssaop.OpLsh8x64:
		v.Op = ssaop.OpLsh64x64
		return true
	case ssaop.OpLsh8x8:
		v.Op = ssaop.OpLsh64x8
		return true
	case ssaop.OpMax32F:
		v.Op = ssaop.OpARM64FMAXS
		return true
	case ssaop.OpMax32FSel:
		return rewriteValue_OpMax32FSel(v)
	case ssaop.OpMax64F:
		v.Op = ssaop.OpARM64FMAXD
		return true
	case ssaop.OpMax64FSel:
		return rewriteValue_OpMax64FSel(v)
	case ssaop.OpMaxFloat32x4:
		v.Op = ssaop.OpARM64VFMAX4S
		return true
	case ssaop.OpMaxFloat64x2:
		v.Op = ssaop.OpARM64VFMAX2D
		return true
	case ssaop.OpMaxInt16x8:
		v.Op = ssaop.OpARM64VSMAX8H
		return true
	case ssaop.OpMaxInt32x4:
		v.Op = ssaop.OpARM64VSMAX4S
		return true
	case ssaop.OpMaxInt8x16:
		v.Op = ssaop.OpARM64VSMAX16B
		return true
	case ssaop.OpMaxUint16x8:
		v.Op = ssaop.OpARM64VUMAX8H
		return true
	case ssaop.OpMaxUint32x4:
		v.Op = ssaop.OpARM64VUMAX4S
		return true
	case ssaop.OpMaxUint8x16:
		v.Op = ssaop.OpARM64VUMAX16B
		return true
	case ssaop.OpMemEq:
		v.Op = ssaop.OpARM64LoweredMemEq
		return true
	case ssaop.OpMin32F:
		v.Op = ssaop.OpARM64FMINS
		return true
	case ssaop.OpMin32FSel:
		return rewriteValue_OpMin32FSel(v)
	case ssaop.OpMin64F:
		v.Op = ssaop.OpARM64FMIND
		return true
	case ssaop.OpMin64FSel:
		return rewriteValue_OpMin64FSel(v)
	case ssaop.OpMinFloat32x4:
		v.Op = ssaop.OpARM64VFMIN4S
		return true
	case ssaop.OpMinFloat64x2:
		v.Op = ssaop.OpARM64VFMIN2D
		return true
	case ssaop.OpMinInt16x8:
		v.Op = ssaop.OpARM64VSMIN8H
		return true
	case ssaop.OpMinInt32x4:
		v.Op = ssaop.OpARM64VSMIN4S
		return true
	case ssaop.OpMinInt8x16:
		v.Op = ssaop.OpARM64VSMIN16B
		return true
	case ssaop.OpMinUint16x8:
		v.Op = ssaop.OpARM64VUMIN8H
		return true
	case ssaop.OpMinUint32x4:
		v.Op = ssaop.OpARM64VUMIN4S
		return true
	case ssaop.OpMinUint8x16:
		v.Op = ssaop.OpARM64VUMIN16B
		return true
	case ssaop.OpMod16:
		return rewriteValue_OpMod16(v)
	case ssaop.OpMod16u:
		return rewriteValue_OpMod16u(v)
	case ssaop.OpMod32:
		return rewriteValue_OpMod32(v)
	case ssaop.OpMod32u:
		v.Op = ssaop.OpARM64UMODW
		return true
	case ssaop.OpMod64:
		return rewriteValue_OpMod64(v)
	case ssaop.OpMod64u:
		v.Op = ssaop.OpARM64UMOD
		return true
	case ssaop.OpMod8:
		return rewriteValue_OpMod8(v)
	case ssaop.OpMod8u:
		return rewriteValue_OpMod8u(v)
	case ssaop.OpMove:
		return rewriteValue_OpMove(v)
	case ssaop.OpMul16:
		v.Op = ssaop.OpARM64MULW
		return true
	case ssaop.OpMul32:
		v.Op = ssaop.OpARM64MULW
		return true
	case ssaop.OpMul32F:
		v.Op = ssaop.OpARM64FMULS
		return true
	case ssaop.OpMul64:
		v.Op = ssaop.OpARM64MUL
		return true
	case ssaop.OpMul64F:
		v.Op = ssaop.OpARM64FMULD
		return true
	case ssaop.OpMul8:
		v.Op = ssaop.OpARM64MULW
		return true
	case ssaop.OpMulAddFloat32x4:
		return rewriteValue_OpMulAddFloat32x4(v)
	case ssaop.OpMulAddFloat64x2:
		return rewriteValue_OpMulAddFloat64x2(v)
	case ssaop.OpMulAddInt16x8:
		return rewriteValue_OpMulAddInt16x8(v)
	case ssaop.OpMulAddInt32x4:
		return rewriteValue_OpMulAddInt32x4(v)
	case ssaop.OpMulAddInt8x16:
		return rewriteValue_OpMulAddInt8x16(v)
	case ssaop.OpMulAddUint16x8:
		return rewriteValue_OpMulAddUint16x8(v)
	case ssaop.OpMulAddUint32x4:
		return rewriteValue_OpMulAddUint32x4(v)
	case ssaop.OpMulAddUint8x16:
		return rewriteValue_OpMulAddUint8x16(v)
	case ssaop.OpMulFloat32x4:
		v.Op = ssaop.OpARM64VFMUL4S
		return true
	case ssaop.OpMulFloat64x2:
		v.Op = ssaop.OpARM64VFMUL2D
		return true
	case ssaop.OpMulInt16x8:
		v.Op = ssaop.OpARM64VMUL8H
		return true
	case ssaop.OpMulInt32x4:
		v.Op = ssaop.OpARM64VMUL4S
		return true
	case ssaop.OpMulInt8x16:
		v.Op = ssaop.OpARM64VMUL16B
		return true
	case ssaop.OpMulUint16x8:
		v.Op = ssaop.OpARM64VMUL8H
		return true
	case ssaop.OpMulUint32x4:
		v.Op = ssaop.OpARM64VMUL4S
		return true
	case ssaop.OpMulUint8x16:
		v.Op = ssaop.OpARM64VMUL16B
		return true
	case ssaop.OpMulWidenLoInt16x8:
		v.Op = ssaop.OpARM64VSMULL8H
		return true
	case ssaop.OpMulWidenLoInt32x4:
		v.Op = ssaop.OpARM64VSMULL4S
		return true
	case ssaop.OpMulWidenLoInt8x16:
		v.Op = ssaop.OpARM64VSMULL16B
		return true
	case ssaop.OpMulWidenLoUint16x8:
		v.Op = ssaop.OpARM64VUMULL8H
		return true
	case ssaop.OpMulWidenLoUint32x4:
		v.Op = ssaop.OpARM64VUMULL4S
		return true
	case ssaop.OpMulWidenLoUint8x16:
		v.Op = ssaop.OpARM64VUMULL16B
		return true
	case ssaop.OpNeg16:
		v.Op = ssaop.OpARM64NEG
		return true
	case ssaop.OpNeg32:
		v.Op = ssaop.OpARM64NEG
		return true
	case ssaop.OpNeg32F:
		v.Op = ssaop.OpARM64FNEGS
		return true
	case ssaop.OpNeg64:
		v.Op = ssaop.OpARM64NEG
		return true
	case ssaop.OpNeg64F:
		v.Op = ssaop.OpARM64FNEGD
		return true
	case ssaop.OpNeg8:
		v.Op = ssaop.OpARM64NEG
		return true
	case ssaop.OpNegFloat32s:
		return rewriteValue_OpNegFloat32s(v)
	case ssaop.OpNegFloat32x4:
		v.Op = ssaop.OpARM64VFNEG4S
		return true
	case ssaop.OpNegFloat64s:
		return rewriteValue_OpNegFloat64s(v)
	case ssaop.OpNegFloat64x2:
		v.Op = ssaop.OpARM64VFNEG2D
		return true
	case ssaop.OpNegInt16s:
		return rewriteValue_OpNegInt16s(v)
	case ssaop.OpNegInt16x8:
		v.Op = ssaop.OpARM64VNEG8H
		return true
	case ssaop.OpNegInt32s:
		return rewriteValue_OpNegInt32s(v)
	case ssaop.OpNegInt32x4:
		v.Op = ssaop.OpARM64VNEG4S
		return true
	case ssaop.OpNegInt64s:
		return rewriteValue_OpNegInt64s(v)
	case ssaop.OpNegInt64x2:
		v.Op = ssaop.OpARM64VNEG2D
		return true
	case ssaop.OpNegInt8s:
		return rewriteValue_OpNegInt8s(v)
	case ssaop.OpNegInt8x16:
		v.Op = ssaop.OpARM64VNEG16B
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
		v.Op = ssaop.OpARM64XOR
		return true
	case ssaop.OpNeqPtr:
		return rewriteValue_OpNeqPtr(v)
	case ssaop.OpNilCheck:
		v.Op = ssaop.OpARM64LoweredNilCheck
		return true
	case ssaop.OpNot:
		return rewriteValue_OpNot(v)
	case ssaop.OpNotInt16x8:
		v.Op = ssaop.OpARM64VNOT16B
		return true
	case ssaop.OpNotInt32x4:
		v.Op = ssaop.OpARM64VNOT16B
		return true
	case ssaop.OpNotInt64x2:
		v.Op = ssaop.OpARM64VNOT16B
		return true
	case ssaop.OpNotInt8x16:
		v.Op = ssaop.OpARM64VNOT16B
		return true
	case ssaop.OpNotUint16x8:
		v.Op = ssaop.OpARM64VNOT16B
		return true
	case ssaop.OpNotUint32x4:
		v.Op = ssaop.OpARM64VNOT16B
		return true
	case ssaop.OpNotUint64x2:
		v.Op = ssaop.OpARM64VNOT16B
		return true
	case ssaop.OpNotUint8x16:
		v.Op = ssaop.OpARM64VNOT16B
		return true
	case ssaop.OpOffPtr:
		return rewriteValue_OpOffPtr(v)
	case ssaop.OpOnesCountInt8x16:
		v.Op = ssaop.OpARM64VCNT16B
		return true
	case ssaop.OpOnesCountUint8x16:
		v.Op = ssaop.OpARM64VCNT16B
		return true
	case ssaop.OpOr16:
		v.Op = ssaop.OpARM64OR
		return true
	case ssaop.OpOr32:
		v.Op = ssaop.OpARM64OR
		return true
	case ssaop.OpOr64:
		v.Op = ssaop.OpARM64OR
		return true
	case ssaop.OpOr8:
		v.Op = ssaop.OpARM64OR
		return true
	case ssaop.OpOrB:
		v.Op = ssaop.OpARM64OR
		return true
	case ssaop.OpOrInt16x8:
		v.Op = ssaop.OpARM64VORR16B
		return true
	case ssaop.OpOrInt32x4:
		v.Op = ssaop.OpARM64VORR16B
		return true
	case ssaop.OpOrInt64x2:
		v.Op = ssaop.OpARM64VORR16B
		return true
	case ssaop.OpOrInt8x16:
		v.Op = ssaop.OpARM64VORR16B
		return true
	case ssaop.OpOrNotInt16x8:
		v.Op = ssaop.OpARM64VORN16B
		return true
	case ssaop.OpOrNotInt32x4:
		v.Op = ssaop.OpARM64VORN16B
		return true
	case ssaop.OpOrNotInt64x2:
		v.Op = ssaop.OpARM64VORN16B
		return true
	case ssaop.OpOrNotInt8x16:
		v.Op = ssaop.OpARM64VORN16B
		return true
	case ssaop.OpOrNotUint16x8:
		v.Op = ssaop.OpARM64VORN16B
		return true
	case ssaop.OpOrNotUint32x4:
		v.Op = ssaop.OpARM64VORN16B
		return true
	case ssaop.OpOrNotUint64x2:
		v.Op = ssaop.OpARM64VORN16B
		return true
	case ssaop.OpOrNotUint8x16:
		v.Op = ssaop.OpARM64VORN16B
		return true
	case ssaop.OpOrUint16x8:
		v.Op = ssaop.OpARM64VORR16B
		return true
	case ssaop.OpOrUint32x4:
		v.Op = ssaop.OpARM64VORR16B
		return true
	case ssaop.OpOrUint64x2:
		v.Op = ssaop.OpARM64VORR16B
		return true
	case ssaop.OpOrUint8x16:
		v.Op = ssaop.OpARM64VORR16B
		return true
	case ssaop.OpPanicBounds:
		v.Op = ssaop.OpARM64LoweredPanicBoundsRR
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
		return rewriteValue_OpPubBarrier(v)
	case ssaop.OpRotateLeft16:
		return rewriteValue_OpRotateLeft16(v)
	case ssaop.OpRotateLeft32:
		return rewriteValue_OpRotateLeft32(v)
	case ssaop.OpRotateLeft64:
		return rewriteValue_OpRotateLeft64(v)
	case ssaop.OpRotateLeft8:
		return rewriteValue_OpRotateLeft8(v)
	case ssaop.OpRound:
		v.Op = ssaop.OpARM64FRINTAD
		return true
	case ssaop.OpRound32F:
		v.Op = ssaop.OpARM64LoweredRound32F
		return true
	case ssaop.OpRound64F:
		v.Op = ssaop.OpARM64LoweredRound64F
		return true
	case ssaop.OpRoundFloat32x4:
		v.Op = ssaop.OpARM64VFRINTN4S
		return true
	case ssaop.OpRoundFloat64x2:
		v.Op = ssaop.OpARM64VFRINTN2D
		return true
	case ssaop.OpRoundToEven:
		v.Op = ssaop.OpARM64FRINTND
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
	case ssaop.OpSaturateToInt16Int32x4:
		v.Op = ssaop.OpARM64VSQXTN4S
		return true
	case ssaop.OpSaturateToInt32Int64x2:
		v.Op = ssaop.OpARM64VSQXTN2D
		return true
	case ssaop.OpSaturateToInt8Int16x8:
		v.Op = ssaop.OpARM64VSQXTN8H
		return true
	case ssaop.OpSaturateToUint16Int32x4:
		v.Op = ssaop.OpARM64VSQXTUN4S
		return true
	case ssaop.OpSaturateToUint16Uint32x4:
		v.Op = ssaop.OpARM64VUQXTN4S
		return true
	case ssaop.OpSaturateToUint32Int64x2:
		v.Op = ssaop.OpARM64VSQXTUN2D
		return true
	case ssaop.OpSaturateToUint32Uint64x2:
		v.Op = ssaop.OpARM64VUQXTN2D
		return true
	case ssaop.OpSaturateToUint8Int16x8:
		v.Op = ssaop.OpARM64VSQXTUN8H
		return true
	case ssaop.OpSaturateToUint8Uint16x8:
		v.Op = ssaop.OpARM64VUQXTN8H
		return true
	case ssaop.OpScalableVectorLen:
		return rewriteValue_OpScalableVectorLen(v)
	case ssaop.OpSelect0:
		return rewriteValue_OpSelect0(v)
	case ssaop.OpSelect1:
		return rewriteValue_OpSelect1(v)
	case ssaop.OpSelectN:
		return rewriteValue_OpSelectN(v)
	case ssaop.OpSetElemFloat32x4:
		v.Op = ssaop.OpARM64VMOVSins0
		return true
	case ssaop.OpSetElemFloat64x2:
		v.Op = ssaop.OpARM64VMOVDins0
		return true
	case ssaop.OpSetElemInt16x8:
		v.Op = ssaop.OpARM64VMOVHins
		return true
	case ssaop.OpSetElemInt32x4:
		v.Op = ssaop.OpARM64VMOVSins
		return true
	case ssaop.OpSetElemInt64x2:
		v.Op = ssaop.OpARM64VMOVDins
		return true
	case ssaop.OpSetElemInt8x16:
		v.Op = ssaop.OpARM64VMOVBins
		return true
	case ssaop.OpSetElemUint16x8:
		v.Op = ssaop.OpARM64VMOVHins
		return true
	case ssaop.OpSetElemUint32x4:
		v.Op = ssaop.OpARM64VMOVSins
		return true
	case ssaop.OpSetElemUint64x2:
		v.Op = ssaop.OpARM64VMOVDins
		return true
	case ssaop.OpSetElemUint8x16:
		v.Op = ssaop.OpARM64VMOVBins
		return true
	case ssaop.OpShiftAllLeftInt16x8:
		return rewriteValue_OpShiftAllLeftInt16x8(v)
	case ssaop.OpShiftAllLeftInt32x4:
		return rewriteValue_OpShiftAllLeftInt32x4(v)
	case ssaop.OpShiftAllLeftInt64x2:
		return rewriteValue_OpShiftAllLeftInt64x2(v)
	case ssaop.OpShiftAllLeftInt8x16:
		return rewriteValue_OpShiftAllLeftInt8x16(v)
	case ssaop.OpShiftAllLeftUint16x8:
		return rewriteValue_OpShiftAllLeftUint16x8(v)
	case ssaop.OpShiftAllLeftUint32x4:
		return rewriteValue_OpShiftAllLeftUint32x4(v)
	case ssaop.OpShiftAllLeftUint64x2:
		return rewriteValue_OpShiftAllLeftUint64x2(v)
	case ssaop.OpShiftAllLeftUint8x16:
		return rewriteValue_OpShiftAllLeftUint8x16(v)
	case ssaop.OpShiftAllRightInt16x8:
		return rewriteValue_OpShiftAllRightInt16x8(v)
	case ssaop.OpShiftAllRightInt32x4:
		return rewriteValue_OpShiftAllRightInt32x4(v)
	case ssaop.OpShiftAllRightInt64x2:
		return rewriteValue_OpShiftAllRightInt64x2(v)
	case ssaop.OpShiftAllRightInt8x16:
		return rewriteValue_OpShiftAllRightInt8x16(v)
	case ssaop.OpShiftAllRightUint16x8:
		return rewriteValue_OpShiftAllRightUint16x8(v)
	case ssaop.OpShiftAllRightUint32x4:
		return rewriteValue_OpShiftAllRightUint32x4(v)
	case ssaop.OpShiftAllRightUint64x2:
		return rewriteValue_OpShiftAllRightUint64x2(v)
	case ssaop.OpShiftAllRightUint8x16:
		return rewriteValue_OpShiftAllRightUint8x16(v)
	case ssaop.OpShiftInt16x8:
		v.Op = ssaop.OpARM64VSSHL8H
		return true
	case ssaop.OpShiftInt32x4:
		v.Op = ssaop.OpARM64VSSHL4S
		return true
	case ssaop.OpShiftInt64x2:
		v.Op = ssaop.OpARM64VSSHL2D
		return true
	case ssaop.OpShiftInt8x16:
		v.Op = ssaop.OpARM64VSSHL16B
		return true
	case ssaop.OpShiftSaturatedInt16x8:
		v.Op = ssaop.OpARM64VSQSHL8H
		return true
	case ssaop.OpShiftSaturatedInt32x4:
		v.Op = ssaop.OpARM64VSQSHL4S
		return true
	case ssaop.OpShiftSaturatedInt64x2:
		v.Op = ssaop.OpARM64VSQSHL2D
		return true
	case ssaop.OpShiftSaturatedInt8x16:
		v.Op = ssaop.OpARM64VSQSHL16B
		return true
	case ssaop.OpShiftSaturatedUint16x8:
		v.Op = ssaop.OpARM64VUQSHL8H
		return true
	case ssaop.OpShiftSaturatedUint32x4:
		v.Op = ssaop.OpARM64VUQSHL4S
		return true
	case ssaop.OpShiftSaturatedUint64x2:
		v.Op = ssaop.OpARM64VUQSHL2D
		return true
	case ssaop.OpShiftSaturatedUint8x16:
		v.Op = ssaop.OpARM64VUQSHL16B
		return true
	case ssaop.OpShiftUint16x8:
		v.Op = ssaop.OpARM64VUSHL8H
		return true
	case ssaop.OpShiftUint32x4:
		v.Op = ssaop.OpARM64VUSHL4S
		return true
	case ssaop.OpShiftUint64x2:
		v.Op = ssaop.OpARM64VUSHL2D
		return true
	case ssaop.OpShiftUint8x16:
		v.Op = ssaop.OpARM64VUSHL16B
		return true
	case ssaop.OpSignExt16to32:
		v.Op = ssaop.OpARM64MOVHreg
		return true
	case ssaop.OpSignExt16to64:
		v.Op = ssaop.OpARM64MOVHreg
		return true
	case ssaop.OpSignExt32to64:
		v.Op = ssaop.OpARM64MOVWreg
		return true
	case ssaop.OpSignExt8to16:
		v.Op = ssaop.OpARM64MOVBreg
		return true
	case ssaop.OpSignExt8to32:
		v.Op = ssaop.OpARM64MOVBreg
		return true
	case ssaop.OpSignExt8to64:
		v.Op = ssaop.OpARM64MOVBreg
		return true
	case ssaop.OpSlicemask:
		return rewriteValue_OpSlicemask(v)
	case ssaop.OpSqrt:
		v.Op = ssaop.OpARM64FSQRTD
		return true
	case ssaop.OpSqrt32:
		v.Op = ssaop.OpARM64FSQRTS
		return true
	case ssaop.OpSqrtFloat32x4:
		v.Op = ssaop.OpARM64VFSQRT4S
		return true
	case ssaop.OpSqrtFloat64x2:
		v.Op = ssaop.OpARM64VFSQRT2D
		return true
	case ssaop.OpStaticCall:
		v.Op = ssaop.OpARM64CALLstatic
		return true
	case ssaop.OpStore:
		return rewriteValue_OpStore(v)
	case ssaop.OpStoreMasked8:
		return rewriteValue_OpStoreMasked8(v)
	case ssaop.OpSub16:
		v.Op = ssaop.OpARM64SUB
		return true
	case ssaop.OpSub32:
		v.Op = ssaop.OpARM64SUB
		return true
	case ssaop.OpSub32F:
		v.Op = ssaop.OpARM64FSUBS
		return true
	case ssaop.OpSub64:
		v.Op = ssaop.OpARM64SUB
		return true
	case ssaop.OpSub64F:
		v.Op = ssaop.OpARM64FSUBD
		return true
	case ssaop.OpSub8:
		v.Op = ssaop.OpARM64SUB
		return true
	case ssaop.OpSubFloat32s:
		v.Op = ssaop.OpARM64ZFSUBS
		return true
	case ssaop.OpSubFloat32x4:
		v.Op = ssaop.OpARM64VFSUB4S
		return true
	case ssaop.OpSubFloat64s:
		v.Op = ssaop.OpARM64ZFSUBD
		return true
	case ssaop.OpSubFloat64x2:
		v.Op = ssaop.OpARM64VFSUB2D
		return true
	case ssaop.OpSubInt16s:
		v.Op = ssaop.OpARM64ZSUBH
		return true
	case ssaop.OpSubInt16x8:
		v.Op = ssaop.OpARM64VSUB8H
		return true
	case ssaop.OpSubInt32s:
		v.Op = ssaop.OpARM64ZSUBS
		return true
	case ssaop.OpSubInt32x4:
		v.Op = ssaop.OpARM64VSUB4S
		return true
	case ssaop.OpSubInt64s:
		v.Op = ssaop.OpARM64ZSUBD
		return true
	case ssaop.OpSubInt64x2:
		v.Op = ssaop.OpARM64VSUB2D
		return true
	case ssaop.OpSubInt8s:
		v.Op = ssaop.OpARM64ZSUBB
		return true
	case ssaop.OpSubInt8x16:
		v.Op = ssaop.OpARM64VSUB16B
		return true
	case ssaop.OpSubPtr:
		v.Op = ssaop.OpARM64SUB
		return true
	case ssaop.OpSubSaturatedInt16s:
		v.Op = ssaop.OpARM64ZSQSUBH
		return true
	case ssaop.OpSubSaturatedInt16x8:
		v.Op = ssaop.OpARM64VSQSUB8H
		return true
	case ssaop.OpSubSaturatedInt32s:
		v.Op = ssaop.OpARM64ZSQSUBS
		return true
	case ssaop.OpSubSaturatedInt32x4:
		v.Op = ssaop.OpARM64VSQSUB4S
		return true
	case ssaop.OpSubSaturatedInt64s:
		v.Op = ssaop.OpARM64ZSQSUBD
		return true
	case ssaop.OpSubSaturatedInt64x2:
		v.Op = ssaop.OpARM64VSQSUB2D
		return true
	case ssaop.OpSubSaturatedInt8s:
		v.Op = ssaop.OpARM64ZSQSUBB
		return true
	case ssaop.OpSubSaturatedInt8x16:
		v.Op = ssaop.OpARM64VSQSUB16B
		return true
	case ssaop.OpSubSaturatedUint16s:
		v.Op = ssaop.OpARM64ZUQSUBH
		return true
	case ssaop.OpSubSaturatedUint16x8:
		v.Op = ssaop.OpARM64VUQSUB8H
		return true
	case ssaop.OpSubSaturatedUint32s:
		v.Op = ssaop.OpARM64ZUQSUBS
		return true
	case ssaop.OpSubSaturatedUint32x4:
		v.Op = ssaop.OpARM64VUQSUB4S
		return true
	case ssaop.OpSubSaturatedUint64s:
		v.Op = ssaop.OpARM64ZUQSUBD
		return true
	case ssaop.OpSubSaturatedUint64x2:
		v.Op = ssaop.OpARM64VUQSUB2D
		return true
	case ssaop.OpSubSaturatedUint8s:
		v.Op = ssaop.OpARM64ZUQSUBB
		return true
	case ssaop.OpSubSaturatedUint8x16:
		v.Op = ssaop.OpARM64VUQSUB16B
		return true
	case ssaop.OpSubUint16s:
		v.Op = ssaop.OpARM64ZSUBH
		return true
	case ssaop.OpSubUint16x8:
		v.Op = ssaop.OpARM64VSUB8H
		return true
	case ssaop.OpSubUint32s:
		v.Op = ssaop.OpARM64ZSUBS
		return true
	case ssaop.OpSubUint32x4:
		v.Op = ssaop.OpARM64VSUB4S
		return true
	case ssaop.OpSubUint64s:
		v.Op = ssaop.OpARM64ZSUBD
		return true
	case ssaop.OpSubUint64x2:
		v.Op = ssaop.OpARM64VSUB2D
		return true
	case ssaop.OpSubUint8s:
		v.Op = ssaop.OpARM64ZSUBB
		return true
	case ssaop.OpSubUint8x16:
		v.Op = ssaop.OpARM64VSUB16B
		return true
	case ssaop.OpTailCall:
		v.Op = ssaop.OpARM64CALLtail
		return true
	case ssaop.OpTailCallInter:
		v.Op = ssaop.OpARM64CALLtailinter
		return true
	case ssaop.OpTrunc:
		v.Op = ssaop.OpARM64FRINTZD
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
		v.Op = ssaop.OpARM64VFRINTZ4S
		return true
	case ssaop.OpTruncFloat64x2:
		v.Op = ssaop.OpARM64VFRINTZ2D
		return true
	case ssaop.OpTruncToInt16Int32x4:
		v.Op = ssaop.OpARM64VXTN4S
		return true
	case ssaop.OpTruncToInt32Int64x2:
		v.Op = ssaop.OpARM64VXTN2D
		return true
	case ssaop.OpTruncToInt8Int16x8:
		v.Op = ssaop.OpARM64VXTN8H
		return true
	case ssaop.OpTruncToUint16Uint32x4:
		v.Op = ssaop.OpARM64VXTN4S
		return true
	case ssaop.OpTruncToUint32Uint64x2:
		v.Op = ssaop.OpARM64VXTN2D
		return true
	case ssaop.OpTruncToUint8Uint16x8:
		v.Op = ssaop.OpARM64VXTN8H
		return true
	case ssaop.OpWB:
		v.Op = ssaop.OpARM64LoweredWB
		return true
	case ssaop.OpXor16:
		v.Op = ssaop.OpARM64XOR
		return true
	case ssaop.OpXor32:
		v.Op = ssaop.OpARM64XOR
		return true
	case ssaop.OpXor64:
		v.Op = ssaop.OpARM64XOR
		return true
	case ssaop.OpXor8:
		v.Op = ssaop.OpARM64XOR
		return true
	case ssaop.OpXorInt16x8:
		v.Op = ssaop.OpARM64VEOR16B
		return true
	case ssaop.OpXorInt32x4:
		v.Op = ssaop.OpARM64VEOR16B
		return true
	case ssaop.OpXorInt64x2:
		v.Op = ssaop.OpARM64VEOR16B
		return true
	case ssaop.OpXorInt8x16:
		v.Op = ssaop.OpARM64VEOR16B
		return true
	case ssaop.OpXorUint16x8:
		v.Op = ssaop.OpARM64VEOR16B
		return true
	case ssaop.OpXorUint32x4:
		v.Op = ssaop.OpARM64VEOR16B
		return true
	case ssaop.OpXorUint64x2:
		v.Op = ssaop.OpARM64VEOR16B
		return true
	case ssaop.OpXorUint8x16:
		v.Op = ssaop.OpARM64VEOR16B
		return true
	case ssaop.OpZero:
		return rewriteValue_OpZero(v)
	case ssaop.OpZeroExt16to32:
		v.Op = ssaop.OpARM64MOVHUreg
		return true
	case ssaop.OpZeroExt16to64:
		v.Op = ssaop.OpARM64MOVHUreg
		return true
	case ssaop.OpZeroExt32to64:
		v.Op = ssaop.OpARM64MOVWUreg
		return true
	case ssaop.OpZeroExt8to16:
		v.Op = ssaop.OpARM64MOVBUreg
		return true
	case ssaop.OpZeroExt8to32:
		v.Op = ssaop.OpARM64MOVBUreg
		return true
	case ssaop.OpZeroExt8to64:
		v.Op = ssaop.OpARM64MOVBUreg
		return true
	case ssaop.OpZeroSIMD:
		return rewriteValue_OpZeroSIMD(v)
	case ssaop.OpbitSelectInt8x16:
		v.Op = ssaop.OpARM64VBIF16B
		return true
	case ssaop.OpbitSelectNotInt8x16:
		v.Op = ssaop.OpARM64VBIT16B
		return true
	case ssaop.Opbroadcast1To16Int8x16:
		return rewriteValue_Opbroadcast1To16Int8x16(v)
	case ssaop.Opbroadcast1To16Uint8x16:
		return rewriteValue_Opbroadcast1To16Uint8x16(v)
	case ssaop.Opbroadcast1To2Float64x2:
		return rewriteValue_Opbroadcast1To2Float64x2(v)
	case ssaop.Opbroadcast1To2Int64x2:
		return rewriteValue_Opbroadcast1To2Int64x2(v)
	case ssaop.Opbroadcast1To2Uint64x2:
		return rewriteValue_Opbroadcast1To2Uint64x2(v)
	case ssaop.Opbroadcast1To4Float32x4:
		return rewriteValue_Opbroadcast1To4Float32x4(v)
	case ssaop.Opbroadcast1To4Int32x4:
		return rewriteValue_Opbroadcast1To4Int32x4(v)
	case ssaop.Opbroadcast1To4Uint32x4:
		return rewriteValue_Opbroadcast1To4Uint32x4(v)
	case ssaop.Opbroadcast1To8Int16x8:
		return rewriteValue_Opbroadcast1To8Int16x8(v)
	case ssaop.Opbroadcast1To8Uint16x8:
		return rewriteValue_Opbroadcast1To8Uint16x8(v)
	case ssaop.OpcarrylessMultiplyWidenLoUint64x2:
		v.Op = ssaop.OpARM64VPMULL2D
		return true
	case ssaop.OpreduceMaxFloat32x4:
		v.Op = ssaop.OpARM64VFMAXV4S
		return true
	case ssaop.OpreduceMaxInt16x8:
		v.Op = ssaop.OpARM64VSMAXV8H
		return true
	case ssaop.OpreduceMaxInt32x4:
		v.Op = ssaop.OpARM64VSMAXV4S
		return true
	case ssaop.OpreduceMaxInt8x16:
		v.Op = ssaop.OpARM64VSMAXV16B
		return true
	case ssaop.OpreduceMaxUint16x8:
		v.Op = ssaop.OpARM64VUMAXV8H
		return true
	case ssaop.OpreduceMaxUint32x4:
		v.Op = ssaop.OpARM64VUMAXV4S
		return true
	case ssaop.OpreduceMaxUint8x16:
		v.Op = ssaop.OpARM64VUMAXV16B
		return true
	case ssaop.OpreduceMinFloat32x4:
		v.Op = ssaop.OpARM64VFMINV4S
		return true
	case ssaop.OpreduceMinInt16x8:
		v.Op = ssaop.OpARM64VSMINV8H
		return true
	case ssaop.OpreduceMinInt32x4:
		v.Op = ssaop.OpARM64VSMINV4S
		return true
	case ssaop.OpreduceMinInt8x16:
		v.Op = ssaop.OpARM64VSMINV16B
		return true
	case ssaop.OpreduceMinUint16x8:
		v.Op = ssaop.OpARM64VUMINV8H
		return true
	case ssaop.OpreduceMinUint32x4:
		v.Op = ssaop.OpARM64VUMINV4S
		return true
	case ssaop.OpreduceMinUint8x16:
		v.Op = ssaop.OpARM64VUMINV16B
		return true
	case ssaop.OpreduceSumInt16x8:
		v.Op = ssaop.OpARM64VADDV8H
		return true
	case ssaop.OpreduceSumInt32x4:
		v.Op = ssaop.OpARM64VADDV4S
		return true
	case ssaop.OpreduceSumInt8x16:
		v.Op = ssaop.OpARM64VADDV16B
		return true
	case ssaop.OpreduceSumUint16x8:
		v.Op = ssaop.OpARM64VADDV8H
		return true
	case ssaop.OpreduceSumUint32x4:
		v.Op = ssaop.OpARM64VADDV4S
		return true
	case ssaop.OpreduceSumUint8x16:
		v.Op = ssaop.OpARM64VADDV16B
		return true
	}
	return false
}
func rewriteValue_OpARM64ADCSflags(v *ssa.Value) bool {
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
		if v_2.Op != ssaop.OpSelect1 || v_2.Type != types.TypeFlags {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpARM64ADDSconstflags || ssa.AuxIntToInt64(v_2_0.AuxInt) != -1 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != ssaop.OpARM64ADCzerocarry || v_2_0_0.Type != typ.UInt64 {
			break
		}
		c := v_2_0_0.Args[0]
		v.Reset(ssaop.OpARM64ADCSflags)
		v.AddArg3(x, y, c)
		return true
	}
	// match: (ADCSflags x y (Select1 <types.TypeFlags> (ADDSconstflags [-1] (MOVDconst [0]))))
	// result: (ADDSflags x y)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpSelect1 || v_2.Type != types.TypeFlags {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpARM64ADDSconstflags || ssa.AuxIntToInt64(v_2_0.AuxInt) != -1 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_2_0_0.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpARM64ADDSflags)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64ADD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADD x (MOVDconst <t> [c]))
	// cond: !t.IsPtr()
	// result: (ADDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			t := v_1.Type
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(!t.IsPtr()) {
				continue
			}
			v.Reset(ssaop.OpARM64ADDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADD a l:(MUL x y))
	// cond: l.Uses==1 && ssa.Clobber(l)
	// result: (MADD a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			l := v_1
			if l.Op != ssaop.OpARM64MUL {
				continue
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1 && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.OpARM64MADD)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (ADD a l:(MNEG x y))
	// cond: l.Uses==1 && ssa.Clobber(l)
	// result: (MSUB a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			l := v_1
			if l.Op != ssaop.OpARM64MNEG {
				continue
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1 && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.OpARM64MSUB)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (ADD a l:(MULW x y))
	// cond: v.Type.Size() <= 4 && l.Uses==1 && ssa.Clobber(l)
	// result: (MADDW a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			l := v_1
			if l.Op != ssaop.OpARM64MULW {
				continue
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(v.Type.Size() <= 4 && l.Uses == 1 && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.OpARM64MADDW)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (ADD a l:(MNEGW x y))
	// cond: v.Type.Size() <= 4 && l.Uses==1 && ssa.Clobber(l)
	// result: (MSUBW a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			l := v_1
			if l.Op != ssaop.OpARM64MNEGW {
				continue
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(v.Type.Size() <= 4 && l.Uses == 1 && ssa.Clobber(l)) {
				continue
			}
			v.Reset(ssaop.OpARM64MSUBW)
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
			if p.Op != ssaop.OpARM64ADDconst {
				continue
			}
			c := ssa.AuxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != ssaop.OpARM64MUL || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.Reset(ssaop.OpARM64ADDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, v.Type)
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
			if p.Op != ssaop.OpARM64ADDconst {
				continue
			}
			c := ssa.AuxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != ssaop.OpARM64MULW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.Reset(ssaop.OpARM64ADDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, v.Type)
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
			if p.Op != ssaop.OpARM64ADDconst {
				continue
			}
			c := ssa.AuxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != ssaop.OpARM64MNEG || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.Reset(ssaop.OpARM64ADDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, v.Type)
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
			if p.Op != ssaop.OpARM64ADDconst {
				continue
			}
			c := ssa.AuxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != ssaop.OpARM64MNEGW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.Reset(ssaop.OpARM64ADDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, v.Type)
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
			if p.Op != ssaop.OpARM64SUBconst {
				continue
			}
			c := ssa.AuxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != ssaop.OpARM64MUL || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.Reset(ssaop.OpARM64SUBconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, v.Type)
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
			if p.Op != ssaop.OpARM64SUBconst {
				continue
			}
			c := ssa.AuxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != ssaop.OpARM64MULW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.Reset(ssaop.OpARM64SUBconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, v.Type)
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
			if p.Op != ssaop.OpARM64SUBconst {
				continue
			}
			c := ssa.AuxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != ssaop.OpARM64MNEG || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.Reset(ssaop.OpARM64SUBconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, v.Type)
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
			if p.Op != ssaop.OpARM64SUBconst {
				continue
			}
			c := ssa.AuxIntToInt64(p.AuxInt)
			m := p.Args[0]
			if m.Op != ssaop.OpARM64MNEGW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
				continue
			}
			v.Reset(ssaop.OpARM64SUBconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, v.Type)
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
			if v_1.Op != ssaop.OpARM64NEG {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARM64SUB)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADD x0 x1:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ADDshiftLL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SLLconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64ADDshiftLL)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (ADD x0 x1:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ADDshiftRL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SRLconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64ADDshiftRL)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (ADD x0 x1:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ADDshiftRA x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SRAconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64ADDshiftRA)
			v.AuxInt = ssa.Int64ToAuxInt(c)
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
			if x1.Op != ssaop.OpARM64ANDshiftRA || ssa.AuxIntToInt64(x1.AuxInt) != 63 {
				continue
			}
			z := x1.Args[1]
			x2 := x1.Args[0]
			if x2.Op != ssaop.OpARM64SLLconst {
				continue
			}
			sl := ssa.AuxIntToInt64(x2.AuxInt)
			y := x2.Args[0]
			if !(x1.Uses == 1 && x2.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpARM64ADDshiftLL)
			v.AuxInt = ssa.Int64ToAuxInt(sl)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64ANDshiftRA, y.Type)
			v0.AuxInt = ssa.Int64ToAuxInt(63)
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
			if x1.Op != ssaop.OpARM64ANDshiftLL {
				continue
			}
			sl := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[1]
			x2 := x1.Args[0]
			if x2.Op != ssaop.OpARM64SRAconst || ssa.AuxIntToInt64(x2.AuxInt) != 63 {
				continue
			}
			z := x2.Args[0]
			if !(x1.Uses == 1 && x2.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpARM64ADDshiftLL)
			v.AuxInt = ssa.Int64ToAuxInt(sl)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64ANDshiftRA, y.Type)
			v0.AuxInt = ssa.Int64ToAuxInt(63)
			v0.AddArg2(y, z)
			v.AddArg2(x0, v0)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64ADDSflags(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDSflags x (MOVDconst [c]))
	// result: (ADDSconstflags [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpARM64ADDSconstflags)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64ADDconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ADDconst [off1] (MOVDaddr [off2] {sym} ptr))
	// cond: ssa.Is32Bit(off1+int64(off2))
	// result: (MOVDaddr [int32(off1)+off2] {sym} ptr)
	for {
		off1 := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		if !(ssa.Is32Bit(off1 + int64(off2))) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDaddr)
		v.AuxInt = ssa.Int32ToAuxInt(int32(off1) + off2)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(ptr)
		return true
	}
	// match: (ADDconst [c] y)
	// cond: c < 0
	// result: (SUBconst [-c] y)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if !(c < 0) {
			break
		}
		v.Reset(ssaop.OpARM64SUBconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c)
		v.AddArg(y)
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
	// match: (ADDconst [c] (MOVDconst [d]))
	// result: (MOVDconst [c+d])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c + d)
		return true
	}
	// match: (ADDconst [c] (ADDconst [d] x))
	// result: (ADDconst [c+d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (SUBconst [d] x))
	// result: (ADDconst [c-d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SUBconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64ADDshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDshiftLL (MOVDconst [c]) x [d])
	// result: (ADDconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftLL x (MOVDconst [c]) [d])
	// result: (ADDconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL <typ.UInt16> [8] (UBFX <typ.UInt16> [ssa.ArmBFAuxInt(8, 8)] x) x)
	// result: (REV16W x)
	for {
		if v.Type != typ.UInt16 || ssa.AuxIntToInt64(v.AuxInt) != 8 || v_0.Op != ssaop.OpARM64UBFX || v_0.Type != typ.UInt16 || ssa.AuxIntToArm64BitField(v_0.AuxInt) != ssa.ArmBFAuxInt(8, 8) {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL [8] (UBFX [ssa.ArmBFAuxInt(8, 24)] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff
	// result: (REV16W x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 || v_0.Op != ssaop.OpARM64UBFX || ssa.AuxIntToArm64BitField(v_0.AuxInt) != ssa.ArmBFAuxInt(8, 24) {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c1 := ssa.AuxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != ssaop.OpARM64ANDconst {
			break
		}
		c2 := ssa.AuxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff) {
			break
		}
		v.Reset(ssaop.OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL [8] (SRLconst [8] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: (uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff)
	// result: (REV16 x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 || v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c1 := ssa.AuxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != ssaop.OpARM64ANDconst {
			break
		}
		c2 := ssa.AuxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff) {
			break
		}
		v.Reset(ssaop.OpARM64REV16)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL [8] (SRLconst [8] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: (uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff)
	// result: (REV16 (ANDconst <x.Type> [0xffffffff] x))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 || v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c1 := ssa.AuxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != ssaop.OpARM64ANDconst {
			break
		}
		c2 := ssa.AuxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff) {
			break
		}
		v.Reset(ssaop.OpARM64REV16)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ANDconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(0xffffffff)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftLL [c] (SRLconst x [64-c]) x2)
	// result: (EXTRconst [64-c] x2 x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 64-c {
			break
		}
		x := v_0.Args[0]
		x2 := v_1
		v.Reset(ssaop.OpARM64EXTRconst)
		v.AuxInt = ssa.Int64ToAuxInt(64 - c)
		v.AddArg2(x2, x)
		return true
	}
	// match: (ADDshiftLL <t> [c] (UBFX [bfc] x) x2)
	// cond: c < 32 && t.Size() == 4 && bfc == ssa.ArmBFAuxInt(32-c, c)
	// result: (EXTRWconst [32-c] x2 x)
	for {
		t := v.Type
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64UBFX {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		x2 := v_1
		if !(c < 32 && t.Size() == 4 && bfc == ssa.ArmBFAuxInt(32-c, c)) {
			break
		}
		v.Reset(ssaop.OpARM64EXTRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(32 - c)
		v.AddArg2(x2, x)
		return true
	}
	return false
}
func rewriteValue_OpARM64ADDshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDshiftRA (MOVDconst [c]) x [d])
	// result: (ADDconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRAconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRA x (MOVDconst [c]) [d])
	// result: (ADDconst x [c>>uint64(d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64ADDshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDshiftRL (MOVDconst [c]) x [d])
	// result: (ADDconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRLconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRL x (MOVDconst [c]) [d])
	// result: (ADDconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64AND(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AND x (MOVDconst [c]))
	// result: (ANDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpARM64ANDconst)
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
	// match: (AND x (MVN y))
	// result: (BIC x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MVN {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARM64BIC)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (AND x0 x1:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ANDshiftLL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SLLconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64ANDshiftLL)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (AND x0 x1:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ANDshiftRL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SRLconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64ANDshiftRL)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (AND x0 x1:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ANDshiftRA x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SRAconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64ANDshiftRA)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (AND x0 x1:(RORconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ANDshiftRO x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64RORconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64ANDshiftRO)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64ANDconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ANDconst [0] _)
	// result: (MOVDconst [0])
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
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
	// match: (ANDconst [c] (MOVDconst [d]))
	// result: (MOVDconst [c&d])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & d)
		return true
	}
	// match: (ANDconst [c] (ANDconst [d] x))
	// result: (ANDconst [c&d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVWUreg x))
	// result: (ANDconst [c&(1<<32-1)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & (1<<32 - 1))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVHUreg x))
	// result: (ANDconst [c&(1<<16-1)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & (1<<16 - 1))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVBUreg x))
	// result: (ANDconst [c&(1<<8-1)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & (1<<8 - 1))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [ac] (SLLconst [sc] x))
	// cond: isARM64BFMask(sc, ac, sc)
	// result: (UBFIZ [ssa.ArmBFAuxInt(sc, arm64BFWidth(ac, sc))] x)
	for {
		ac := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		sc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, ac, sc)) {
			break
		}
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(sc, arm64BFWidth(ac, sc)))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [ac] (SRLconst [sc] x))
	// cond: isARM64BFMask(sc, ac, 0)
	// result: (UBFX [ssa.ArmBFAuxInt(sc, arm64BFWidth(ac, 0))] x)
	for {
		ac := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRLconst {
			break
		}
		sc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, ac, 0)) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(sc, arm64BFWidth(ac, 0)))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (UBFX [bfc] x))
	// cond: isARM64BFMask(0, c, 0)
	// result: (UBFX [ssa.ArmBFAuxInt(bfc.Lsb(), min(bfc.Width(), arm64BFWidth(c, 0)))] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64UBFX {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(0, c, 0)) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(bfc.Lsb(), min(bfc.Width(), arm64BFWidth(c, 0))))
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
		v.Reset(ssaop.OpARM64MOVWUreg)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [0xffff ] x)
	// result: (MOVHUreg x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0xffff {
			break
		}
		x := v_0
		v.Reset(ssaop.OpARM64MOVHUreg)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [0xff ] x)
	// result: (MOVBUreg x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0xff {
			break
		}
		x := v_0
		v.Reset(ssaop.OpARM64MOVBUreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64ANDshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftLL (MOVDconst [c]) x [d])
	// result: (ANDconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftLL x (MOVDconst [c]) [d])
	// result: (ANDconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftLL y:(SLLconst x [c]) x [c])
	// result: y
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
			break
		}
		v.CopyOf(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64ANDshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftRA (MOVDconst [c]) x [d])
	// result: (ANDconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRAconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRA x (MOVDconst [c]) [d])
	// result: (ANDconst x [c>>uint64(d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftRA y:(SRAconst x [c]) x [c])
	// result: y
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARM64SRAconst || ssa.AuxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
			break
		}
		v.CopyOf(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64ANDshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftRL (MOVDconst [c]) x [d])
	// result: (ANDconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRLconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRL x (MOVDconst [c]) [d])
	// result: (ANDconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftRL y:(SRLconst x [c]) x [c])
	// result: y
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
			break
		}
		v.CopyOf(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64ANDshiftRO(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftRO (MOVDconst [c]) x [d])
	// result: (ANDconst [c] (RORconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64RORconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRO x (MOVDconst [c]) [d])
	// result: (ANDconst x [rotateRight64(c, d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftRO y:(RORconst x [c]) x [c])
	// result: y
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARM64RORconst || ssa.AuxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
			break
		}
		v.CopyOf(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64BIC(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BIC x (MOVDconst [c]))
	// result: (ANDconst [^c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(^c)
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
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (BIC x0 x1:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (BICshiftLL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SLLconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64BICshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (BIC x0 x1:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (BICshiftRL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SRLconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64BICshiftRL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (BIC x0 x1:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (BICshiftRA x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SRAconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64BICshiftRA)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (BIC x0 x1:(RORconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (BICshiftRO x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64RORconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64BICshiftRO)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64BICshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftLL x (MOVDconst [c]) [d])
	// result: (ANDconst x [^int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(^int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftLL (SLLconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpARM64BICshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftRA x (MOVDconst [c]) [d])
	// result: (ANDconst x [^(c>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(^(c >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftRA (SRAconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRAconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpARM64BICshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftRL x (MOVDconst [c]) [d])
	// result: (ANDconst x [^int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(^int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftRL (SRLconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpARM64BICshiftRO(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftRO x (MOVDconst [c]) [d])
	// result: (ANDconst x [^rotateRight64(c, d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(^rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftRO (RORconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64RORconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpARM64CMN(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMN x (MOVDconst [c]))
	// result: (CMNconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpARM64CMNconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (CMN x0 x1:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (CMNshiftLL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SLLconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64CMNshiftLL)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (CMN x0 x1:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (CMNshiftRL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SRLconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64CMNshiftRL)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (CMN x0 x1:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (CMNshiftRA x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SRAconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64CMNshiftRA)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64CMNW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMNW x (MOVDconst [c]))
	// result: (CMNWconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpARM64CMNWconst)
			v.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64CMNWconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (CMNWconst [c] y)
	// cond: c < 0 && c != -1<<31
	// result: (CMPWconst [-c] y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		y := v_0
		if !(c < 0 && c != -1<<31) {
			break
		}
		v.Reset(ssaop.OpARM64CMPWconst)
		v.AuxInt = ssa.Int32ToAuxInt(-c)
		v.AddArg(y)
		return true
	}
	// match: (CMNWconst (MOVDconst [x]) [y])
	// result: (FlagConstant [ssa.AddFlags32(int32(x),y)])
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.AddFlags32(int32(x), y))
		return true
	}
	return false
}
func rewriteValue_OpARM64CMNconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (CMNconst [c] y)
	// cond: c < 0 && c != -1<<63
	// result: (CMPconst [-c] y)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if !(c < 0 && c != -1<<63) {
			break
		}
		v.Reset(ssaop.OpARM64CMPconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c)
		v.AddArg(y)
		return true
	}
	// match: (CMNconst (MOVDconst [x]) [y])
	// result: (FlagConstant [ssa.AddFlags64(x,y)])
	for {
		y := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.AddFlags64(x, y))
		return true
	}
	return false
}
func rewriteValue_OpARM64CMNshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMNshiftLL (MOVDconst [c]) x [d])
	// result: (CMNconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64CMNconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftLL x (MOVDconst [c]) [d])
	// result: (CMNconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64CMNconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64CMNshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMNshiftRA (MOVDconst [c]) x [d])
	// result: (CMNconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64CMNconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRAconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRA x (MOVDconst [c]) [d])
	// result: (CMNconst x [c>>uint64(d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64CMNconst)
		v.AuxInt = ssa.Int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64CMNshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMNshiftRL (MOVDconst [c]) x [d])
	// result: (CMNconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64CMNconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRLconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRL x (MOVDconst [c]) [d])
	// result: (CMNconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64CMNconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64CMP(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMP x (MOVDconst [c]))
	// result: (CMPconst [c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64CMPconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMP (MOVDconst [c]) x)
	// result: (InvertFlags (CMPconst [c] x))
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(x)
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
		v.Reset(ssaop.OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x0 x1:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (CMPshiftLL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SLLconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64CMPshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (CMP x0:(SLLconst [c] y) x1)
	// cond: ssa.ClobberIfDead(x0)
	// result: (InvertFlags (CMPshiftLL x1 y [c]))
	for {
		x0 := v_0
		if x0.Op != ssaop.OpARM64SLLconst {
			break
		}
		c := ssa.AuxIntToInt64(x0.AuxInt)
		y := x0.Args[0]
		x1 := v_1
		if !(ssa.ClobberIfDead(x0)) {
			break
		}
		v.Reset(ssaop.OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPshiftLL, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg2(x1, y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x0 x1:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (CMPshiftRL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SRLconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64CMPshiftRL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (CMP x0:(SRLconst [c] y) x1)
	// cond: ssa.ClobberIfDead(x0)
	// result: (InvertFlags (CMPshiftRL x1 y [c]))
	for {
		x0 := v_0
		if x0.Op != ssaop.OpARM64SRLconst {
			break
		}
		c := ssa.AuxIntToInt64(x0.AuxInt)
		y := x0.Args[0]
		x1 := v_1
		if !(ssa.ClobberIfDead(x0)) {
			break
		}
		v.Reset(ssaop.OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPshiftRL, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg2(x1, y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x0 x1:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (CMPshiftRA x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SRAconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64CMPshiftRA)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (CMP x0:(SRAconst [c] y) x1)
	// cond: ssa.ClobberIfDead(x0)
	// result: (InvertFlags (CMPshiftRA x1 y [c]))
	for {
		x0 := v_0
		if x0.Op != ssaop.OpARM64SRAconst {
			break
		}
		c := ssa.AuxIntToInt64(x0.AuxInt)
		y := x0.Args[0]
		x1 := v_1
		if !(ssa.ClobberIfDead(x0)) {
			break
		}
		v.Reset(ssaop.OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPshiftRA, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg2(x1, y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpARM64CMPW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPW x (MOVDconst [c]))
	// result: (CMPWconst [int32(c)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64CMPWconst)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPW (MOVDconst [c]) x)
	// result: (InvertFlags (CMPWconst [int32(c)] x))
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(x)
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
		v.Reset(ssaop.OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpARM64CMPWconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (CMPWconst [c] y)
	// cond: c < 0 && c != -1<<31
	// result: (CMNWconst [-c] y)
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		y := v_0
		if !(c < 0 && c != -1<<31) {
			break
		}
		v.Reset(ssaop.OpARM64CMNWconst)
		v.AuxInt = ssa.Int32ToAuxInt(-c)
		v.AddArg(y)
		return true
	}
	// match: (CMPWconst (MOVDconst [x]) [y])
	// result: (FlagConstant [ssa.SubFlags32(int32(x),y)])
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.SubFlags32(int32(x), y))
		return true
	}
	// match: (CMPWconst (MOVBUreg _) [c])
	// cond: 0xff < c
	// result: (FlagConstant [ssa.SubFlags64(0,1)])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVBUreg || !(0xff < c) {
			break
		}
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.SubFlags64(0, 1))
		return true
	}
	// match: (CMPWconst (MOVHUreg _) [c])
	// cond: 0xffff < c
	// result: (FlagConstant [ssa.SubFlags64(0,1)])
	for {
		c := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVHUreg || !(0xffff < c) {
			break
		}
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.SubFlags64(0, 1))
		return true
	}
	return false
}
func rewriteValue_OpARM64CMPconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (CMPconst [c] y)
	// cond: c < 0 && c != -1<<63
	// result: (CMNconst [-c] y)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if !(c < 0 && c != -1<<63) {
			break
		}
		v.Reset(ssaop.OpARM64CMNconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c)
		v.AddArg(y)
		return true
	}
	// match: (CMPconst (MOVDconst [x]) [y])
	// result: (FlagConstant [ssa.SubFlags64(x,y)])
	for {
		y := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.SubFlags64(x, y))
		return true
	}
	// match: (CMPconst (MOVBUreg _) [c])
	// cond: 0xff < c
	// result: (FlagConstant [ssa.SubFlags64(0,1)])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVBUreg || !(0xff < c) {
			break
		}
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.SubFlags64(0, 1))
		return true
	}
	// match: (CMPconst (MOVHUreg _) [c])
	// cond: 0xffff < c
	// result: (FlagConstant [ssa.SubFlags64(0,1)])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVHUreg || !(0xffff < c) {
			break
		}
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.SubFlags64(0, 1))
		return true
	}
	// match: (CMPconst (MOVWUreg _) [c])
	// cond: 0xffffffff < c
	// result: (FlagConstant [ssa.SubFlags64(0,1)])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVWUreg || !(0xffffffff < c) {
			break
		}
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.SubFlags64(0, 1))
		return true
	}
	// match: (CMPconst (ANDconst _ [m]) [n])
	// cond: 0 <= m && m < n
	// result: (FlagConstant [ssa.SubFlags64(0,1)])
	for {
		n := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		m := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(0 <= m && m < n) {
			break
		}
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.SubFlags64(0, 1))
		return true
	}
	// match: (CMPconst (SRLconst _ [c]) [n])
	// cond: 0 <= n && 0 < c && c <= 63 && (1<<uint64(64-c)) <= uint64(n)
	// result: (FlagConstant [ssa.SubFlags64(0,1)])
	for {
		n := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRLconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(0 <= n && 0 < c && c <= 63 && (1<<uint64(64-c)) <= uint64(n)) {
			break
		}
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.SubFlags64(0, 1))
		return true
	}
	return false
}
func rewriteValue_OpARM64CMPshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPshiftLL (MOVDconst [c]) x [d])
	// result: (InvertFlags (CMPconst [c] (SLLconst <x.Type> x [d])))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(d)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftLL x (MOVDconst [c]) [d])
	// result: (CMPconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64CMPconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64CMPshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPshiftRA (MOVDconst [c]) x [d])
	// result: (InvertFlags (CMPconst [c] (SRAconst <x.Type> x [d])))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SRAconst, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(d)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRA x (MOVDconst [c]) [d])
	// result: (CMPconst x [c>>uint64(d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64CMPconst)
		v.AuxInt = ssa.Int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64CMPshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPshiftRL (MOVDconst [c]) x [d])
	// result: (InvertFlags (CMPconst [c] (SRLconst <x.Type> x [d])))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SRLconst, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(d)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRL x (MOVDconst [c]) [d])
	// result: (CMPconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64CMPconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64CSEL(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CSEL [cc] (MOVDconst [-1]) (MOVDconst [0]) flag)
	// result: (CSETM [cc] flag)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != -1 || v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		flag := v_2
		v.Reset(ssaop.OpARM64CSETM)
		v.AuxInt = ssa.OpToAuxInt(cc)
		v.AddArg(flag)
		return true
	}
	// match: (CSEL [cc] (MOVDconst [0]) (MOVDconst [-1]) flag)
	// result: (CSETM [arm64Negate(cc)] flag)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 || v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		flag := v_2
		v.Reset(ssaop.OpARM64CSETM)
		v.AuxInt = ssa.OpToAuxInt(arm64Negate(cc))
		v.AddArg(flag)
		return true
	}
	// match: (CSEL [cc] x (MOVDconst [0]) flag)
	// result: (CSEL0 [cc] x flag)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		flag := v_2
		v.Reset(ssaop.OpARM64CSEL0)
		v.AuxInt = ssa.OpToAuxInt(cc)
		v.AddArg2(x, flag)
		return true
	}
	// match: (CSEL [cc] (MOVDconst [0]) y flag)
	// result: (CSEL0 [arm64Negate(cc)] y flag)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		y := v_1
		flag := v_2
		v.Reset(ssaop.OpARM64CSEL0)
		v.AuxInt = ssa.OpToAuxInt(arm64Negate(cc))
		v.AddArg2(y, flag)
		return true
	}
	// match: (CSEL [cc] x (ADDconst [1] a) flag)
	// result: (CSINC [cc] x a flag)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64ADDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		a := v_1.Args[0]
		flag := v_2
		v.Reset(ssaop.OpARM64CSINC)
		v.AuxInt = ssa.OpToAuxInt(cc)
		v.AddArg3(x, a, flag)
		return true
	}
	// match: (CSEL [cc] (ADDconst [1] a) x flag)
	// result: (CSINC [arm64Negate(cc)] x a flag)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ADDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		a := v_0.Args[0]
		x := v_1
		flag := v_2
		v.Reset(ssaop.OpARM64CSINC)
		v.AuxInt = ssa.OpToAuxInt(arm64Negate(cc))
		v.AddArg3(x, a, flag)
		return true
	}
	// match: (CSEL [cc] x (MVN a) flag)
	// result: (CSINV [cc] x a flag)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MVN {
			break
		}
		a := v_1.Args[0]
		flag := v_2
		v.Reset(ssaop.OpARM64CSINV)
		v.AuxInt = ssa.OpToAuxInt(cc)
		v.AddArg3(x, a, flag)
		return true
	}
	// match: (CSEL [cc] (MVN a) x flag)
	// result: (CSINV [arm64Negate(cc)] x a flag)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MVN {
			break
		}
		a := v_0.Args[0]
		x := v_1
		flag := v_2
		v.Reset(ssaop.OpARM64CSINV)
		v.AuxInt = ssa.OpToAuxInt(arm64Negate(cc))
		v.AddArg3(x, a, flag)
		return true
	}
	// match: (CSEL [cc] x (NEG a) flag)
	// result: (CSNEG [cc] x a flag)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64NEG {
			break
		}
		a := v_1.Args[0]
		flag := v_2
		v.Reset(ssaop.OpARM64CSNEG)
		v.AuxInt = ssa.OpToAuxInt(cc)
		v.AddArg3(x, a, flag)
		return true
	}
	// match: (CSEL [cc] (NEG a) x flag)
	// result: (CSNEG [arm64Negate(cc)] x a flag)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		if v_0.Op != ssaop.OpARM64NEG {
			break
		}
		a := v_0.Args[0]
		x := v_1
		flag := v_2
		v.Reset(ssaop.OpARM64CSNEG)
		v.AuxInt = ssa.OpToAuxInt(arm64Negate(cc))
		v.AddArg3(x, a, flag)
		return true
	}
	// match: (CSEL [cc] x y (InvertFlags cmp))
	// result: (CSEL [arm64Invert(cc)] x y cmp)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARM64InvertFlags {
			break
		}
		cmp := v_2.Args[0]
		v.Reset(ssaop.OpARM64CSEL)
		v.AuxInt = ssa.OpToAuxInt(arm64Invert(cc))
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (CSEL [cc] x _ flag)
	// cond: ccARM64Eval(cc, flag) > 0
	// result: x
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		flag := v_2
		if !(ccARM64Eval(cc, flag) > 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (CSEL [cc] _ y flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: y
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		y := v_1
		flag := v_2
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (CSEL [cc] x y (CMPWconst [0] boolval))
	// cond: cc == ssaop.OpARM64NotEqual && ssa.FlagArg(boolval) != nil
	// result: (CSEL [boolval.Op] x y ssa.FlagArg(boolval))
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_2.AuxInt) != 0 {
			break
		}
		boolval := v_2.Args[0]
		if !(cc == ssaop.OpARM64NotEqual && ssa.FlagArg(boolval) != nil) {
			break
		}
		v.Reset(ssaop.OpARM64CSEL)
		v.AuxInt = ssa.OpToAuxInt(boolval.Op)
		v.AddArg3(x, y, ssa.FlagArg(boolval))
		return true
	}
	// match: (CSEL [cc] x y (CMPWconst [0] boolval))
	// cond: cc == ssaop.OpARM64Equal && ssa.FlagArg(boolval) != nil
	// result: (CSEL [arm64Negate(boolval.Op)] x y ssa.FlagArg(boolval))
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_2.AuxInt) != 0 {
			break
		}
		boolval := v_2.Args[0]
		if !(cc == ssaop.OpARM64Equal && ssa.FlagArg(boolval) != nil) {
			break
		}
		v.Reset(ssaop.OpARM64CSEL)
		v.AuxInt = ssa.OpToAuxInt(arm64Negate(boolval.Op))
		v.AddArg3(x, y, ssa.FlagArg(boolval))
		return true
	}
	return false
}
func rewriteValue_OpARM64CSEL0(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CSEL0 [cc] x (InvertFlags cmp))
	// result: (CSEL0 [arm64Invert(cc)] x cmp)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64InvertFlags {
			break
		}
		cmp := v_1.Args[0]
		v.Reset(ssaop.OpARM64CSEL0)
		v.AuxInt = ssa.OpToAuxInt(arm64Invert(cc))
		v.AddArg2(x, cmp)
		return true
	}
	// match: (CSEL0 [cc] x flag)
	// cond: ccARM64Eval(cc, flag) > 0
	// result: x
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		flag := v_1
		if !(ccARM64Eval(cc, flag) > 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (CSEL0 [cc] _ flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: (MOVDconst [0])
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		flag := v_1
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (CSEL0 [cc] x (CMPWconst [0] boolval))
	// cond: cc == ssaop.OpARM64NotEqual && ssa.FlagArg(boolval) != nil
	// result: (CSEL0 [boolval.Op] x ssa.FlagArg(boolval))
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		boolval := v_1.Args[0]
		if !(cc == ssaop.OpARM64NotEqual && ssa.FlagArg(boolval) != nil) {
			break
		}
		v.Reset(ssaop.OpARM64CSEL0)
		v.AuxInt = ssa.OpToAuxInt(boolval.Op)
		v.AddArg2(x, ssa.FlagArg(boolval))
		return true
	}
	// match: (CSEL0 [cc] x (CMPWconst [0] boolval))
	// cond: cc == ssaop.OpARM64Equal && ssa.FlagArg(boolval) != nil
	// result: (CSEL0 [arm64Negate(boolval.Op)] x ssa.FlagArg(boolval))
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		boolval := v_1.Args[0]
		if !(cc == ssaop.OpARM64Equal && ssa.FlagArg(boolval) != nil) {
			break
		}
		v.Reset(ssaop.OpARM64CSEL0)
		v.AuxInt = ssa.OpToAuxInt(arm64Negate(boolval.Op))
		v.AddArg2(x, ssa.FlagArg(boolval))
		return true
	}
	return false
}
func rewriteValue_OpARM64CSETM(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (CSETM [cc] (InvertFlags cmp))
	// result: (CSETM [arm64Invert(cc)] cmp)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		cmp := v_0.Args[0]
		v.Reset(ssaop.OpARM64CSETM)
		v.AuxInt = ssa.OpToAuxInt(arm64Invert(cc))
		v.AddArg(cmp)
		return true
	}
	// match: (CSETM [cc] flag)
	// cond: ccARM64Eval(cc, flag) > 0
	// result: (MOVDconst [-1])
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		flag := v_0
		if !(ccARM64Eval(cc, flag) > 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	// match: (CSETM [cc] flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: (MOVDconst [0])
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		flag := v_0
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpARM64CSINC(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CSINC [cc] x y (InvertFlags cmp))
	// result: (CSINC [arm64Invert(cc)] x y cmp)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARM64InvertFlags {
			break
		}
		cmp := v_2.Args[0]
		v.Reset(ssaop.OpARM64CSINC)
		v.AuxInt = ssa.OpToAuxInt(arm64Invert(cc))
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (CSINC [cc] x _ flag)
	// cond: ccARM64Eval(cc, flag) > 0
	// result: x
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		flag := v_2
		if !(ccARM64Eval(cc, flag) > 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (CSINC [cc] _ y flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: (ADDconst [1] y)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		y := v_1
		flag := v_2
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64CSINV(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CSINV [cc] x y (InvertFlags cmp))
	// result: (CSINV [arm64Invert(cc)] x y cmp)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARM64InvertFlags {
			break
		}
		cmp := v_2.Args[0]
		v.Reset(ssaop.OpARM64CSINV)
		v.AuxInt = ssa.OpToAuxInt(arm64Invert(cc))
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (CSINV [cc] x _ flag)
	// cond: ccARM64Eval(cc, flag) > 0
	// result: x
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		flag := v_2
		if !(ccARM64Eval(cc, flag) > 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (CSINV [cc] _ y flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: (Not y)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		y := v_1
		flag := v_2
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.Reset(ssaop.OpNot)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64CSNEG(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CSNEG [cc] x y (InvertFlags cmp))
	// result: (CSNEG [arm64Invert(cc)] x y cmp)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARM64InvertFlags {
			break
		}
		cmp := v_2.Args[0]
		v.Reset(ssaop.OpARM64CSNEG)
		v.AuxInt = ssa.OpToAuxInt(arm64Invert(cc))
		v.AddArg3(x, y, cmp)
		return true
	}
	// match: (CSNEG [cc] x _ flag)
	// cond: ccARM64Eval(cc, flag) > 0
	// result: x
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		flag := v_2
		if !(ccARM64Eval(cc, flag) > 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (CSNEG [cc] _ y flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: (NEG y)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		y := v_1
		flag := v_2
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.Reset(ssaop.OpARM64NEG)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64DIV(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIV (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [c/d])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c / d)
		return true
	}
	return false
}
func rewriteValue_OpARM64DIVW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (DIVW (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [int64(uint32(int32(c)/int32(d)))])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint32(int32(c) / int32(d))))
		return true
	}
	return false
}
func rewriteValue_OpARM64EON(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EON x (MOVDconst [c]))
	// result: (XORconst [^c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(^c)
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
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	// match: (EON x0 x1:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (EONshiftLL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SLLconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64EONshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (EON x0 x1:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (EONshiftRL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SRLconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64EONshiftRL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (EON x0 x1:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (EONshiftRA x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SRAconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64EONshiftRA)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (EON x0 x1:(RORconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (EONshiftRO x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64RORconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64EONshiftRO)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64EONshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EONshiftLL x (MOVDconst [c]) [d])
	// result: (XORconst x [^int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(^int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (EONshiftLL (SLLconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValue_OpARM64EONshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EONshiftRA x (MOVDconst [c]) [d])
	// result: (XORconst x [^(c>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(^(c >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (EONshiftRA (SRAconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRAconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValue_OpARM64EONshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EONshiftRL x (MOVDconst [c]) [d])
	// result: (XORconst x [^int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(^int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (EONshiftRL (SRLconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValue_OpARM64EONshiftRO(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (EONshiftRO x (MOVDconst [c]) [d])
	// result: (XORconst x [^rotateRight64(c, d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(^rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	// match: (EONshiftRO (RORconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64RORconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValue_OpARM64Equal(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Equal (CMPconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (Equal (TST x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TST, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPWconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (Equal (TSTWconst [int32(c)] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPWconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (Equal (TSTW x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (Equal (TSTconst [c] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMP x z:(NEG y)))
	// cond: z.Uses == 1
	// result: (Equal (CMN x y))
	for {
		if v_0.Op != ssaop.OpARM64CMP {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		z := v_0.Args[1]
		if z.Op != ssaop.OpARM64NEG {
			break
		}
		y := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMN, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPW x z:(NEG y)))
	// cond: z.Uses == 1
	// result: (Equal (CMNW x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPW {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		z := v_0.Args[1]
		if z.Op != ssaop.OpARM64NEG {
			break
		}
		y := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (Equal (CMNconst [c] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPWconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (Equal (CMNWconst [int32(c)] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (Equal (CMN x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMN, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPWconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (Equal (CMNW x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPconst [0] z:(MADD a x y)))
	// cond: z.Uses == 1
	// result: (Equal (CMN a (MUL <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64MADD {
			break
		}
		y := z.Args[2]
		a := z.Args[0]
		x := z.Args[1]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMN, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPWconst [0] z:(MADDW a x y)))
	// cond: z.Uses == 1
	// result: (Equal (CMNW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64MADDW {
			break
		}
		y := z.Args[2]
		a := z.Args[0]
		x := z.Args[1]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPconst [0] z:(MSUB a x y)))
	// cond: z.Uses == 1
	// result: (Equal (CMP a (MUL <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64MSUB {
			break
		}
		y := z.Args[2]
		a := z.Args[0]
		x := z.Args[1]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (CMPWconst [0] z:(MSUBW a x y)))
	// cond: z.Uses == 1
	// result: (Equal (CMPW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64MSUBW {
			break
		}
		y := z.Args[2]
		a := z.Args[0]
		x := z.Args[1]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Equal (FlagConstant [fc]))
	// result: (MOVDconst [ssa.B2i(fc.Eq())])
	for {
		if v_0.Op != ssaop.OpARM64FlagConstant {
			break
		}
		fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(fc.Eq()))
		return true
	}
	// match: (Equal (InvertFlags x))
	// result: (Equal x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64Equal)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64FADDD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FADDD a (FMULD x y))
	// cond: a.Block.Func.UseFMA(v)
	// result: (FMADDD a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != ssaop.OpARM64FMULD {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Block.Func.UseFMA(v)) {
				continue
			}
			v.Reset(ssaop.OpARM64FMADDD)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (FADDD a (FNMULD x y))
	// cond: a.Block.Func.UseFMA(v)
	// result: (FMSUBD a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != ssaop.OpARM64FNMULD {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Block.Func.UseFMA(v)) {
				continue
			}
			v.Reset(ssaop.OpARM64FMSUBD)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64FADDS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FADDS a (FMULS x y))
	// cond: a.Block.Func.UseFMA(v)
	// result: (FMADDS a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != ssaop.OpARM64FMULS {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Block.Func.UseFMA(v)) {
				continue
			}
			v.Reset(ssaop.OpARM64FMADDS)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (FADDS a (FNMULS x y))
	// cond: a.Block.Func.UseFMA(v)
	// result: (FMSUBS a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != ssaop.OpARM64FNMULS {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Block.Func.UseFMA(v)) {
				continue
			}
			v.Reset(ssaop.OpARM64FMSUBS)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64FCMPD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (FCMPD x (FMOVDconst [0]))
	// result: (FCMPD0 x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64FMOVDconst || ssa.AuxIntToFloat64(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpARM64FCMPD0)
		v.AddArg(x)
		return true
	}
	// match: (FCMPD (FMOVDconst [0]) x)
	// result: (InvertFlags (FCMPD0 x))
	for {
		if v_0.Op != ssaop.OpARM64FMOVDconst || ssa.AuxIntToFloat64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPD0, types.TypeFlags)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpARM64FCMPS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (FCMPS x (FMOVSconst [0]))
	// result: (FCMPS0 x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64FMOVSconst || ssa.AuxIntToFloat64(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpARM64FCMPS0)
		v.AddArg(x)
		return true
	}
	// match: (FCMPS (FMOVSconst [0]) x)
	// result: (InvertFlags (FCMPS0 x))
	for {
		if v_0.Op != ssaop.OpARM64FMOVSconst || ssa.AuxIntToFloat64(v_0.AuxInt) != 0 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPS0, types.TypeFlags)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpARM64FCSELD(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FCSELD [cc] x y (InvertFlags cmp))
	// result: (FCSELD [arm64Invert(cc)] x y cmp)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARM64InvertFlags {
			break
		}
		cmp := v_2.Args[0]
		v.Reset(ssaop.OpARM64FCSELD)
		v.AuxInt = ssa.OpToAuxInt(arm64Invert(cc))
		v.AddArg3(x, y, cmp)
		return true
	}
	return false
}
func rewriteValue_OpARM64FCSELS(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FCSELS [cc] x y (InvertFlags cmp))
	// result: (FCSELS [arm64Invert(cc)] x y cmp)
	for {
		cc := ssa.AuxIntToOp(v.AuxInt)
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARM64InvertFlags {
			break
		}
		cmp := v_2.Args[0]
		v.Reset(ssaop.OpARM64FCSELS)
		v.AuxInt = ssa.OpToAuxInt(arm64Invert(cc))
		v.AddArg3(x, y, cmp)
		return true
	}
	return false
}
func rewriteValue_OpARM64FCVTDS(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (FCVTDS (FABSD (FCVTSD x)))
	// result: (FABSS x)
	for {
		if v_0.Op != ssaop.OpARM64FABSD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64FCVTSD {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpARM64FABSS)
		v.AddArg(x)
		return true
	}
	// match: (FCVTDS (FSQRTD (FCVTSD x)))
	// result: (FSQRTS x)
	for {
		if v_0.Op != ssaop.OpARM64FSQRTD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64FCVTSD {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpARM64FSQRTS)
		v.AddArg(x)
		return true
	}
	// match: (FCVTDS (FRINTPD (FCVTSD x)))
	// result: (FRINTPS x)
	for {
		if v_0.Op != ssaop.OpARM64FRINTPD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64FCVTSD {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpARM64FRINTPS)
		v.AddArg(x)
		return true
	}
	// match: (FCVTDS (FRINTMD (FCVTSD x)))
	// result: (FRINTMS x)
	for {
		if v_0.Op != ssaop.OpARM64FRINTMD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64FCVTSD {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpARM64FRINTMS)
		v.AddArg(x)
		return true
	}
	// match: (FCVTDS (FRINTAD (FCVTSD x)))
	// result: (FRINTAS x)
	for {
		if v_0.Op != ssaop.OpARM64FRINTAD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64FCVTSD {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpARM64FRINTAS)
		v.AddArg(x)
		return true
	}
	// match: (FCVTDS (FRINTND (FCVTSD x)))
	// result: (FRINTNS x)
	for {
		if v_0.Op != ssaop.OpARM64FRINTND {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64FCVTSD {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpARM64FRINTNS)
		v.AddArg(x)
		return true
	}
	// match: (FCVTDS (FRINTZD (FCVTSD x)))
	// result: (FRINTZS x)
	for {
		if v_0.Op != ssaop.OpARM64FRINTZD {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64FCVTSD {
			break
		}
		x := v_0_0.Args[0]
		v.Reset(ssaop.OpARM64FRINTZS)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64FLDPQ(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FLDPQ [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FLDPQ [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FLDPQ)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FLDPQ [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FLDPQ [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FLDPQ)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVDfpgp(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (FMOVDfpgp <t> (Arg [off] {sym}))
	// result: @b.Func.Entry (Arg <t> [off] {sym})
	for {
		t := v.Type
		if v_0.Op != ssaop.OpArg {
			break
		}
		off := ssa.AuxIntToInt32(v_0.AuxInt)
		sym := ssa.AuxToSym(v_0.Aux)
		b = b.Func.Entry
		v0 := b.NewValue0(v.Pos, ssaop.OpArg, t)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVDgpfp(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (FMOVDgpfp <t> (Arg [off] {sym}))
	// result: @b.Func.Entry (Arg <t> [off] {sym})
	for {
		t := v.Type
		if v_0.Op != ssaop.OpArg {
			break
		}
		off := ssa.AuxIntToInt32(v_0.AuxInt)
		sym := ssa.AuxToSym(v_0.Aux)
		b = b.Func.Entry
		v0 := b.NewValue0(v.Pos, ssaop.OpArg, t)
		v.CopyOf(v0)
		v0.AuxInt = ssa.Int32ToAuxInt(off)
		v0.Aux = ssa.SymToAux(sym)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVDload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVDload [off] {sym} ptr (MOVDstore [off] {sym} ptr val _))
	// result: (FMOVDgpfp val)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDgpfp)
		v.AddArg(val)
		return true
	}
	// match: (FMOVDload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVDload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVDload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVDloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (FMOVDload [off] {sym} (ADDshiftLL [3] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVDloadidx8 ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDshiftLL || ssa.AuxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDloadidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (FMOVDload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVDload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVDloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVDloadidx ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (FMOVDload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVDloadidx (MOVDconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (FMOVDload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVDloadidx ptr (SLLconst [3] idx) mem)
	// result: (FMOVDloadidx8 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 3 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64FMOVDloadidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (FMOVDloadidx (SLLconst [3] idx) ptr mem)
	// result: (FMOVDloadidx8 ptr idx mem)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64FMOVDloadidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVDloadidx8(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVDloadidx8 ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c<<3)
	// result: (FMOVDload ptr [int32(c)<<3] mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c << 3)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) << 3)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVDstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVDstore [off] {sym} ptr (FMOVDgpfp val) mem)
	// result: (MOVDstore [off] {sym} ptr val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARM64FMOVDgpfp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVDstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVDstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVDstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVDstoreidx ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (FMOVDstore [off] {sym} (ADDshiftLL [3] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVDstoreidx8 ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDshiftLL || ssa.AuxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDstoreidx8)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (FMOVDstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVDstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
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
		v.Reset(ssaop.OpARM64FMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVDstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVDstoreidx ptr (MOVDconst [c]) val mem)
	// cond: ssa.Is32Bit(c)
	// result: (FMOVDstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVDstoreidx (MOVDconst [c]) idx val mem)
	// cond: ssa.Is32Bit(c)
	// result: (FMOVDstore [int32(c)] idx val mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (FMOVDstoreidx ptr (SLLconst [3] idx) val mem)
	// result: (FMOVDstoreidx8 ptr idx val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 3 {
			break
		}
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARM64FMOVDstoreidx8)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (FMOVDstoreidx (SLLconst [3] idx) ptr val mem)
	// result: (FMOVDstoreidx8 ptr idx val mem)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARM64FMOVDstoreidx8)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVDstoreidx8(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVDstoreidx8 ptr (MOVDconst [c]) val mem)
	// cond: ssa.Is32Bit(c<<3)
	// result: (FMOVDstore [int32(c)<<3] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c << 3)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) << 3)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVQload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVQload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVQload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVQload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVQload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVQload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVQload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVQstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVQstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVQstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVQstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVQstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVQstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
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
		v.Reset(ssaop.OpARM64FMOVQstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVSload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVSload [off] {sym} ptr (MOVWstore [off] {sym} ptr val _))
	// result: (FMOVSgpfp val)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVWstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSgpfp)
		v.AddArg(val)
		return true
	}
	// match: (FMOVSload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVSload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVSload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVSloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (FMOVSload [off] {sym} (ADDshiftLL [2] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVSloadidx4 ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDshiftLL || ssa.AuxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (FMOVSload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVSload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVSloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVSloadidx ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (FMOVSload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVSloadidx (MOVDconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (FMOVSload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (FMOVSloadidx ptr (SLLconst [2] idx) mem)
	// result: (FMOVSloadidx4 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64FMOVSloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (FMOVSloadidx (SLLconst [2] idx) ptr mem)
	// result: (FMOVSloadidx4 ptr idx mem)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64FMOVSloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVSloadidx4(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVSloadidx4 ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c<<2)
	// result: (FMOVSload ptr [int32(c)<<2] mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c << 2)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) << 2)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVSstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FMOVSstore [off] {sym} ptr (FMOVSgpfp val) mem)
	// result: (MOVWstore [off] {sym} ptr val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARM64FMOVSgpfp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVSstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVSstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVSstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVSstoreidx ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (FMOVSstore [off] {sym} (ADDshiftLL [2] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVSstoreidx4 ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDshiftLL || ssa.AuxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSstoreidx4)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (FMOVSstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FMOVSstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
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
		v.Reset(ssaop.OpARM64FMOVSstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVSstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVSstoreidx ptr (MOVDconst [c]) val mem)
	// cond: ssa.Is32Bit(c)
	// result: (FMOVSstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (FMOVSstoreidx (MOVDconst [c]) idx val mem)
	// cond: ssa.Is32Bit(c)
	// result: (FMOVSstore [int32(c)] idx val mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (FMOVSstoreidx ptr (SLLconst [2] idx) val mem)
	// result: (FMOVSstoreidx4 ptr idx val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARM64FMOVSstoreidx4)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (FMOVSstoreidx (SLLconst [2] idx) ptr val mem)
	// result: (FMOVSstoreidx4 ptr idx val mem)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARM64FMOVSstoreidx4)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMOVSstoreidx4(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMOVSstoreidx4 ptr (MOVDconst [c]) val mem)
	// cond: ssa.Is32Bit(c<<2)
	// result: (FMOVSstore [int32(c)<<2] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c << 2)) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) << 2)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FMULD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMULD (FNEGD x) y)
	// result: (FNMULD x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64FNEGD {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.Reset(ssaop.OpARM64FNMULD)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64FMULS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMULS (FNEGS x) y)
	// result: (FNMULS x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64FNEGS {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.Reset(ssaop.OpARM64FNMULS)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64FNEGD(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (FNEGD (FMULD x y))
	// result: (FNMULD x y)
	for {
		if v_0.Op != ssaop.OpARM64FMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64FNMULD)
		v.AddArg2(x, y)
		return true
	}
	// match: (FNEGD (FNMULD x y))
	// result: (FMULD x y)
	for {
		if v_0.Op != ssaop.OpARM64FNMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64FMULD)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64FNEGS(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (FNEGS (FMULS x y))
	// result: (FNMULS x y)
	for {
		if v_0.Op != ssaop.OpARM64FMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64FNMULS)
		v.AddArg2(x, y)
		return true
	}
	// match: (FNEGS (FNMULS x y))
	// result: (FMULS x y)
	for {
		if v_0.Op != ssaop.OpARM64FNMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64FMULS)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64FNMULD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FNMULD (FNEGD x) y)
	// result: (FMULD x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64FNEGD {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.Reset(ssaop.OpARM64FMULD)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64FNMULS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FNMULS (FNEGS x) y)
	// result: (FMULS x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64FNEGS {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.Reset(ssaop.OpARM64FMULS)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64FSTPQ(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (FSTPQ [off1] {sym} (ADDconst [off2] ptr) val1 val2 mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FSTPQ [off1+int32(off2)] {sym} ptr val1 val2 mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val1 := v_1
		val2 := v_2
		mem := v_3
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FSTPQ)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg4(ptr, val1, val2, mem)
		return true
	}
	// match: (FSTPQ [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val1 val2 mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (FSTPQ [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val1 val2 mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val1 := v_1
		val2 := v_2
		mem := v_3
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64FSTPQ)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg4(ptr, val1, val2, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64FSUBD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FSUBD a (FMULD x y))
	// cond: a.Block.Func.UseFMA(v)
	// result: (FMSUBD a x y)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64FMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpARM64FMSUBD)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBD (FMULD x y) a)
	// cond: a.Block.Func.UseFMA(v)
	// result: (FNMSUBD a x y)
	for {
		if v_0.Op != ssaop.OpARM64FMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		if !(a.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpARM64FNMSUBD)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBD a (FNMULD x y))
	// cond: a.Block.Func.UseFMA(v)
	// result: (FMADDD a x y)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64FNMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpARM64FMADDD)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBD (FNMULD x y) a)
	// cond: a.Block.Func.UseFMA(v)
	// result: (FNMADDD a x y)
	for {
		if v_0.Op != ssaop.OpARM64FNMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		if !(a.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpARM64FNMADDD)
		v.AddArg3(a, x, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64FSUBS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FSUBS a (FMULS x y))
	// cond: a.Block.Func.UseFMA(v)
	// result: (FMSUBS a x y)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64FMULS {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpARM64FMSUBS)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBS (FMULS x y) a)
	// cond: a.Block.Func.UseFMA(v)
	// result: (FNMSUBS a x y)
	for {
		if v_0.Op != ssaop.OpARM64FMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		if !(a.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpARM64FNMSUBS)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBS a (FNMULS x y))
	// cond: a.Block.Func.UseFMA(v)
	// result: (FMADDS a x y)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64FNMULS {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpARM64FMADDS)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (FSUBS (FNMULS x y) a)
	// cond: a.Block.Func.UseFMA(v)
	// result: (FNMADDS a x y)
	for {
		if v_0.Op != ssaop.OpARM64FNMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		if !(a.Block.Func.UseFMA(v)) {
			break
		}
		v.Reset(ssaop.OpARM64FNMADDS)
		v.AddArg3(a, x, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64GreaterEqual(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (GreaterEqual (CMPconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqual (TST x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TST, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPWconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (GreaterEqual (TSTWconst [int32(c)] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPWconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqual (TSTW x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (GreaterEqual (TSTconst [c] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (GreaterEqualNoov (CMNconst [c] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPWconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (GreaterEqualNoov (CMNWconst [int32(c)] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqualNoov (CMN x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMN, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPWconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqualNoov (CMNW x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPconst [0] z:(MADD a x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqualNoov (CMN a (MUL <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64MADD {
			break
		}
		y := z.Args[2]
		a := z.Args[0]
		x := z.Args[1]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMN, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPWconst [0] z:(MADDW a x y)))
	// cond: z.Uses == 1
	// result: (GreaterEqualNoov (CMNW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64MADDW {
			break
		}
		y := z.Args[2]
		a := z.Args[0]
		x := z.Args[1]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterEqualNoov)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (FlagConstant [fc]))
	// result: (MOVDconst [ssa.B2i(fc.Ge())])
	for {
		if v_0.Op != ssaop.OpARM64FlagConstant {
			break
		}
		fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(fc.Ge()))
		return true
	}
	// match: (GreaterEqual (InvertFlags x))
	// result: (LessEqual x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64LessEqual)
		v.AddArg(x)
		return true
	}
	// match: (GreaterEqual (CMPconst x [0]))
	// result: (XORconst [1] (SRLconst <v.Type> [63] x))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRLconst, v.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(63)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterEqual (CMPWconst x [0]))
	// result: (XORconst [1] (UBFX <v.Type> [ssa.ArmBFAuxInt(31,1)] x))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64UBFX, v.Type)
		v0.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(31, 1))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpARM64GreaterEqualF(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterEqualF (InvertFlags x))
	// result: (LessEqualF x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64LessEqualF)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64GreaterEqualNoov(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterEqualNoov (FlagConstant [fc]))
	// result: (MOVDconst [ssa.B2i(fc.GeNoov())])
	for {
		if v_0.Op != ssaop.OpARM64FlagConstant {
			break
		}
		fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(fc.GeNoov()))
		return true
	}
	return false
}
func rewriteValue_OpARM64GreaterEqualU(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterEqualU (FlagConstant [fc]))
	// result: (MOVDconst [ssa.B2i(fc.Uge())])
	for {
		if v_0.Op != ssaop.OpARM64FlagConstant {
			break
		}
		fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(fc.Uge()))
		return true
	}
	// match: (GreaterEqualU (InvertFlags x))
	// result: (LessEqualU x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64LessEqualU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64GreaterThan(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (GreaterThan (CMPconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (GreaterThan (TST x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TST, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterThan (CMPWconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (GreaterThan (TSTWconst [int32(c)] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterThan (CMPWconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (GreaterThan (TSTW x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterThan (CMPconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (GreaterThan (TSTconst [c] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (GreaterThan (FlagConstant [fc]))
	// result: (MOVDconst [ssa.B2i(fc.Gt())])
	for {
		if v_0.Op != ssaop.OpARM64FlagConstant {
			break
		}
		fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(fc.Gt()))
		return true
	}
	// match: (GreaterThan (InvertFlags x))
	// result: (LessThan x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64LessThan)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64GreaterThanF(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterThanF (InvertFlags x))
	// result: (LessThanF x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64LessThanF)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64GreaterThanU(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterThanU (FlagConstant [fc]))
	// result: (MOVDconst [ssa.B2i(fc.Ugt())])
	for {
		if v_0.Op != ssaop.OpARM64FlagConstant {
			break
		}
		fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(fc.Ugt()))
		return true
	}
	// match: (GreaterThanU (InvertFlags x))
	// result: (LessThanU x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64LessThanU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64LDP(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (LDP [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (LDP [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64LDP)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (LDP [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (LDP [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64LDP)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64LessEqual(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (LessEqual (CMPconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (LessEqual (TST x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TST, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (LessEqual (CMPWconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (LessEqual (TSTWconst [int32(c)] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (LessEqual (CMPWconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (LessEqual (TSTW x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (LessEqual (CMPconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (LessEqual (TSTconst [c] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (LessEqual (FlagConstant [fc]))
	// result: (MOVDconst [ssa.B2i(fc.Le())])
	for {
		if v_0.Op != ssaop.OpARM64FlagConstant {
			break
		}
		fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(fc.Le()))
		return true
	}
	// match: (LessEqual (InvertFlags x))
	// result: (GreaterEqual x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64GreaterEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64LessEqualF(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (LessEqualF (InvertFlags x))
	// result: (GreaterEqualF x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64GreaterEqualF)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64LessEqualU(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (LessEqualU (FlagConstant [fc]))
	// result: (MOVDconst [ssa.B2i(fc.Ule())])
	for {
		if v_0.Op != ssaop.OpARM64FlagConstant {
			break
		}
		fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(fc.Ule()))
		return true
	}
	// match: (LessEqualU (InvertFlags x))
	// result: (GreaterEqualU x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64GreaterEqualU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64LessThan(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (LessThan (CMPconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (LessThan (TST x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TST, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPWconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (LessThan (TSTWconst [int32(c)] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPWconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (LessThan (TSTW x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (LessThan (TSTconst [c] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (LessThanNoov (CMNconst [c] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPWconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (LessThanNoov (CMNWconst [int32(c)] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (LessThanNoov (CMN x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMN, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPWconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (LessThanNoov (CMNW x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPconst [0] z:(MADD a x y)))
	// cond: z.Uses == 1
	// result: (LessThanNoov (CMN a (MUL <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64MADD {
			break
		}
		y := z.Args[2]
		a := z.Args[0]
		x := z.Args[1]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMN, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (CMPWconst [0] z:(MADDW a x y)))
	// cond: z.Uses == 1
	// result: (LessThanNoov (CMNW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64MADDW {
			break
		}
		y := z.Args[2]
		a := z.Args[0]
		x := z.Args[1]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64LessThanNoov)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (LessThan (FlagConstant [fc]))
	// result: (MOVDconst [ssa.B2i(fc.Lt())])
	for {
		if v_0.Op != ssaop.OpARM64FlagConstant {
			break
		}
		fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(fc.Lt()))
		return true
	}
	// match: (LessThan (InvertFlags x))
	// result: (GreaterThan x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64GreaterThan)
		v.AddArg(x)
		return true
	}
	// match: (LessThan (CMPconst x [0]))
	// result: (SRLconst [63] x)
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64SRLconst)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v.AddArg(x)
		return true
	}
	// match: (LessThan (CMPWconst x [0]))
	// result: (UBFX [ssa.ArmBFAuxInt(31,1)] x)
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(31, 1))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64LessThanF(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (LessThanF (InvertFlags x))
	// result: (GreaterThanF x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64GreaterThanF)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64LessThanNoov(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (LessThanNoov (FlagConstant [fc]))
	// result: (MOVDconst [ssa.B2i(fc.LtNoov())])
	for {
		if v_0.Op != ssaop.OpARM64FlagConstant {
			break
		}
		fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(fc.LtNoov()))
		return true
	}
	return false
}
func rewriteValue_OpARM64LessThanU(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (LessThanU (FlagConstant [fc]))
	// result: (MOVDconst [ssa.B2i(fc.Ult())])
	for {
		if v_0.Op != ssaop.OpARM64FlagConstant {
			break
		}
		fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(fc.Ult()))
		return true
	}
	// match: (LessThanU (InvertFlags x))
	// result: (GreaterThanU x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64GreaterThanU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64LoweredPanicBoundsCR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsCR [kind] {p} (MOVDconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:p.C, Cy:c}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		p := ssa.AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.OpARM64LoweredPanicBoundsCC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: p.C, Cy: c})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64LoweredPanicBoundsRC(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRC [kind] {p} (MOVDconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:c, Cy:p.C}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		p := ssa.AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.OpARM64LoweredPanicBoundsCC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: c, Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64LoweredPanicBoundsRR(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRR [kind] x (MOVDconst [c]) mem)
	// result: (LoweredPanicBoundsRC [kind] x {ssa.PanicBoundsC{C:c}} mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpARM64LoweredPanicBoundsRC)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: c})
		v.AddArg2(x, mem)
		return true
	}
	// match: (LoweredPanicBoundsRR [kind] (MOVDconst [c]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {ssa.PanicBoundsC{C:c}} y mem)
	for {
		kind := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64LoweredPanicBoundsCR)
		v.AuxInt = ssa.Int64ToAuxInt(kind)
		v.Aux = ssa.PanicBoundsCToAux(ssa.PanicBoundsC{C: c})
		v.AddArg2(y, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MADD(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MADD a x (MOVDconst [-1]))
	// result: (SUB a x)
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_2.AuxInt) != -1 {
			break
		}
		v.Reset(ssaop.OpARM64SUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a _ (MOVDconst [0]))
	// result: a
	for {
		a := v_0
		if v_2.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		v.CopyOf(a)
		return true
	}
	// match: (MADD a x (MOVDconst [1]))
	// result: (ADD a x)
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_2.AuxInt) != 1 {
			break
		}
		v.Reset(ssaop.OpARM64ADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (ADDshiftLL a x [ssa.Log64(c)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARM64ADDshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c-1) && c>=3
	// result: (ADD a (ADDshiftLL <x.Type> x x [ssa.Log64(c-1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(ssa.IsPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.Reset(ssaop.OpARM64ADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c - 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c+1) && c>=7
	// result: (SUB a (SUBshiftLL <x.Type> x x [ssa.Log64(c+1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(ssa.IsPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.Reset(ssaop.OpARM64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c + 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [2]) [ssa.Log64(c/3)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 3))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [2]) [ssa.Log64(c/5)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5)) {
			break
		}
		v.Reset(ssaop.OpARM64ADDshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 5))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: c%7 == 0 && ssa.IsPowerOfTwo(c/7)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [3]) [ssa.Log64(c/7)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%7 == 0 && ssa.IsPowerOfTwo(c/7)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 7))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [3]) [ssa.Log64(c/9)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9)) {
			break
		}
		v.Reset(ssaop.OpARM64ADDshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 9))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a (MOVDconst [-1]) x)
	// result: (SUB a x)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		x := v_2
		v.Reset(ssaop.OpARM64SUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a (MOVDconst [0]) _)
	// result: a
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.CopyOf(a)
		return true
	}
	// match: (MADD a (MOVDconst [1]) x)
	// result: (ADD a x)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		x := v_2
		v.Reset(ssaop.OpARM64ADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(c)
	// result: (ADDshiftLL a x [ssa.Log64(c)])
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARM64ADDshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(c-1) && c>=3
	// result: (ADD a (ADDshiftLL <x.Type> x x [ssa.Log64(c-1)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(ssa.IsPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.Reset(ssaop.OpARM64ADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c - 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(c+1) && c>=7
	// result: (SUB a (SUBshiftLL <x.Type> x x [ssa.Log64(c+1)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(ssa.IsPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.Reset(ssaop.OpARM64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c + 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [2]) [ssa.Log64(c/3)])
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 3))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [2]) [ssa.Log64(c/5)])
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5)) {
			break
		}
		v.Reset(ssaop.OpARM64ADDshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 5))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: c%7 == 0 && ssa.IsPowerOfTwo(c/7)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [3]) [ssa.Log64(c/7)])
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%7 == 0 && ssa.IsPowerOfTwo(c/7)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 7))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [3]) [ssa.Log64(c/9)])
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9)) {
			break
		}
		v.Reset(ssaop.OpARM64ADDshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 9))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MADD (MOVDconst [c]) x y)
	// result: (ADDconst [c] (MUL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MUL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) (MOVDconst [d]))
	// result: (ADDconst [c*d] a)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_2.AuxInt)
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c * d)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValue_OpARM64MADDW(v *ssa.Value) bool {
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
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(int32(c) == -1) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a _ (MOVDconst [c]))
	// cond: int32(c)==0
	// result: (MOVWUreg a)
	for {
		a := v_0
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(int32(c) == 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v.AddArg(a)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: int32(c)==1
	// result: (MOVWUreg (ADD <a.Type> a x))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(int32(c) == 1) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a x [ssa.Log64(c)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c-1) && int32(c)>=3
	// result: (MOVWUreg (ADD <a.Type> a (ADDshiftLL <x.Type> x x [ssa.Log64(c-1)])))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(ssa.IsPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, a.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c - 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c+1) && int32(c)>=7
	// result: (MOVWUreg (SUB <a.Type> a (SUBshiftLL <x.Type> x x [ssa.Log64(c+1)])))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(ssa.IsPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, a.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c + 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3) && ssa.Is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [2]) [ssa.Log64(c/3)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 3))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5) && ssa.Is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [2]) [ssa.Log64(c/5)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 5))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%7 == 0 && ssa.IsPowerOfTwo(c/7) && ssa.Is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [3]) [ssa.Log64(c/7)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%7 == 0 && ssa.IsPowerOfTwo(c/7) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 7))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9) && ssa.Is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [3]) [ssa.Log64(c/9)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 9))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(3)
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
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(int32(c) == -1) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) _)
	// cond: int32(c)==0
	// result: (MOVWUreg a)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(int32(c) == 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v.AddArg(a)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: int32(c)==1
	// result: (MOVWUreg (ADD <a.Type> a x))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(int32(c) == 1) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a x [ssa.Log64(c)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(c-1) && int32(c)>=3
	// result: (MOVWUreg (ADD <a.Type> a (ADDshiftLL <x.Type> x x [ssa.Log64(c-1)])))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(ssa.IsPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, a.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c - 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(c+1) && int32(c)>=7
	// result: (MOVWUreg (SUB <a.Type> a (SUBshiftLL <x.Type> x x [ssa.Log64(c+1)])))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(ssa.IsPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, a.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c + 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3) && ssa.Is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [2]) [ssa.Log64(c/3)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 3))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5) && ssa.Is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [2]) [ssa.Log64(c/5)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 5))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%7 == 0 && ssa.IsPowerOfTwo(c/7) && ssa.Is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [3]) [ssa.Log64(c/7)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%7 == 0 && ssa.IsPowerOfTwo(c/7) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 7))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9) && ssa.Is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [3]) [ssa.Log64(c/9)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 9))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW (MOVDconst [c]) x y)
	// result: (MOVWUreg (ADDconst <x.Type> [c] (MULW <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVWUreg (ADDconst <a.Type> [c*d] a))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_2.AuxInt)
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDconst, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(c * d)
		v0.AddArg(a)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpARM64MNEG(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MNEG x (MOVDconst [-1]))
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != -1 {
				continue
			}
			v.CopyOf(x)
			return true
		}
		break
	}
	// match: (MNEG _ (MOVDconst [0]))
	// result: (MOVDconst [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				continue
			}
			v.Reset(ssaop.OpARM64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [1]))
	// result: (NEG x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
				continue
			}
			v.Reset(ssaop.OpARM64NEG)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (NEG (SLLconst <x.Type> [ssa.Log64(c)] x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.IsPowerOfTwo(c)) {
				continue
			}
			v.Reset(ssaop.OpARM64NEG)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
			v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c-1) && c >= 3
	// result: (NEG (ADDshiftLL <x.Type> x x [ssa.Log64(c-1)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.IsPowerOfTwo(c-1) && c >= 3) {
				continue
			}
			v.Reset(ssaop.OpARM64NEG)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c - 1))
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c+1) && c >= 7
	// result: (NEG (ADDshiftLL <x.Type> (NEG <x.Type> x) x [ssa.Log64(c+1)]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.IsPowerOfTwo(c+1) && c >= 7) {
				continue
			}
			v.Reset(ssaop.OpARM64NEG)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
			v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c + 1))
			v1 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, x.Type)
			v1.AddArg(x)
			v0.AddArg2(v1, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3)
	// result: (SLLconst <x.Type> [ssa.Log64(c/3)] (SUBshiftLL <x.Type> x x [2]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3)) {
				continue
			}
			v.Reset(ssaop.OpARM64SLLconst)
			v.Type = x.Type
			v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 3))
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
			v0.AuxInt = ssa.Int64ToAuxInt(2)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5)
	// result: (NEG (SLLconst <x.Type> [ssa.Log64(c/5)] (ADDshiftLL <x.Type> x x [2])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5)) {
				continue
			}
			v.Reset(ssaop.OpARM64NEG)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
			v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 5))
			v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
			v1.AuxInt = ssa.Int64ToAuxInt(2)
			v1.AddArg2(x, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: c%7 == 0 && ssa.IsPowerOfTwo(c/7)
	// result: (SLLconst <x.Type> [ssa.Log64(c/7)] (SUBshiftLL <x.Type> x x [3]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%7 == 0 && ssa.IsPowerOfTwo(c/7)) {
				continue
			}
			v.Reset(ssaop.OpARM64SLLconst)
			v.Type = x.Type
			v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 7))
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
			v0.AuxInt = ssa.Int64ToAuxInt(3)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9)
	// result: (NEG (SLLconst <x.Type> [ssa.Log64(c/9)] (ADDshiftLL <x.Type> x x [3])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9)) {
				continue
			}
			v.Reset(ssaop.OpARM64NEG)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
			v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 9))
			v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
			v1.AuxInt = ssa.Int64ToAuxInt(3)
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
			if v_0.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			d := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpARM64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(-c * d)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64MNEGW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MNEGW x (MOVDconst [c]))
	// cond: int32(c)==-1
	// result: (MOVWUreg x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(int32(c) == -1) {
				continue
			}
			v.Reset(ssaop.OpARM64MOVWUreg)
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
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(int32(c) == 0) {
				continue
			}
			v.Reset(ssaop.OpARM64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(0)
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
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(int32(c) == 1) {
				continue
			}
			v.Reset(ssaop.OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, x.Type)
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (NEG (SLLconst <x.Type> [ssa.Log64(c)] x))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.IsPowerOfTwo(c)) {
				continue
			}
			v.Reset(ssaop.OpARM64NEG)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
			v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
			v0.AddArg(x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c-1) && int32(c) >= 3
	// result: (MOVWUreg (NEG <x.Type> (ADDshiftLL <x.Type> x x [ssa.Log64(c-1)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.IsPowerOfTwo(c-1) && int32(c) >= 3) {
				continue
			}
			v.Reset(ssaop.OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, x.Type)
			v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
			v1.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c - 1))
			v1.AddArg2(x, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c+1) && int32(c) >= 7
	// result: (MOVWUreg (NEG <x.Type> (ADDshiftLL <x.Type> (NEG <x.Type> x) x [ssa.Log64(c+1)])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(ssa.IsPowerOfTwo(c+1) && int32(c) >= 7) {
				continue
			}
			v.Reset(ssaop.OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, x.Type)
			v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
			v1.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c + 1))
			v2 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, x.Type)
			v2.AddArg(x)
			v1.AddArg2(v2, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3) && ssa.Is32Bit(c)
	// result: (MOVWUreg (SLLconst <x.Type> [ssa.Log64(c/3)] (SUBshiftLL <x.Type> x x [2])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3) && ssa.Is32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
			v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 3))
			v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
			v1.AuxInt = ssa.Int64ToAuxInt(2)
			v1.AddArg2(x, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5) && ssa.Is32Bit(c)
	// result: (MOVWUreg (NEG <x.Type> (SLLconst <x.Type> [ssa.Log64(c/5)] (ADDshiftLL <x.Type> x x [2]))))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5) && ssa.Is32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, x.Type)
			v1 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
			v1.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 5))
			v2 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
			v2.AuxInt = ssa.Int64ToAuxInt(2)
			v2.AddArg2(x, x)
			v1.AddArg(v2)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%7 == 0 && ssa.IsPowerOfTwo(c/7) && ssa.Is32Bit(c)
	// result: (MOVWUreg (SLLconst <x.Type> [ssa.Log64(c/7)] (SUBshiftLL <x.Type> x x [3])))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%7 == 0 && ssa.IsPowerOfTwo(c/7) && ssa.Is32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
			v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 7))
			v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
			v1.AuxInt = ssa.Int64ToAuxInt(3)
			v1.AddArg2(x, x)
			v0.AddArg(v1)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9) && ssa.Is32Bit(c)
	// result: (MOVWUreg (NEG <x.Type> (SLLconst <x.Type> [ssa.Log64(c/9)] (ADDshiftLL <x.Type> x x [3]))))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9) && ssa.Is32Bit(c)) {
				continue
			}
			v.Reset(ssaop.OpARM64MOVWUreg)
			v0 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, x.Type)
			v1 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
			v1.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 9))
			v2 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
			v2.AuxInt = ssa.Int64ToAuxInt(3)
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
			if v_0.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			d := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpARM64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(int64(uint32(-c * d)))
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64MOD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOD (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [c%d])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c % d)
		return true
	}
	return false
}
func rewriteValue_OpARM64MODW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MODW (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [int64(uint32(int32(c)%int32(d)))])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint32(int32(c) % int32(d))))
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVBUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBUloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVBUload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBUload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVDconst [int64(ssa.Read8(sym, int64(off)))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Read8(sym, int64(off))))
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVBUloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBUloadidx ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVBUload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUloadidx (MOVDconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVBUload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVBUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBUreg (ANDconst [c] x))
	// result: (ANDconst [c&(1<<8-1)] x)
	for {
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & (1<<8 - 1))
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint8(c))])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint8(c)))
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
		v.CopyOf(x)
		return true
	}
	// match: (MOVBUreg (SLLconst [lc] x))
	// cond: lc >= 8
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(lc >= 8) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (MOVBUreg (SLLconst [lc] x))
	// cond: lc < 8
	// result: (UBFIZ [ssa.ArmBFAuxInt(lc, 8-lc)] x)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < 8) {
			break
		}
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc, 8-lc))
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (SRLconst [rc] x))
	// cond: rc < 8
	// result: (UBFX [ssa.ArmBFAuxInt(rc, 8)] x)
	for {
		if v_0.Op != ssaop.OpARM64SRLconst {
			break
		}
		rc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(rc < 8) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(rc, 8))
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (UBFX [bfc] x))
	// cond: bfc.Width() <= 8
	// result: (UBFX [bfc] x)
	for {
		if v_0.Op != ssaop.OpARM64UBFX {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(bfc.Width() <= 8) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(bfc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVBload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVDconst [int64(int8(ssa.Read8(sym, int64(off))))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int8(ssa.Read8(sym, int64(off)))))
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVBloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBloadidx ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVBload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBloadidx (MOVDconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVBload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVBreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int8(c))])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int8(c)))
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
		v.CopyOf(x)
		return true
	}
	// match: (MOVBreg <t> (ANDconst x [c]))
	// cond: uint64(c) & uint64(0xffffffffffffff80) == 0
	// result: (ANDconst <t> x [c])
	for {
		t := v.Type
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(uint64(c)&uint64(0xffffffffffffff80) == 0) {
			break
		}
		v.Reset(ssaop.OpARM64ANDconst)
		v.Type = t
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SLLconst [lc] x))
	// cond: lc < 8
	// result: (SBFIZ [ssa.ArmBFAuxInt(lc, 8-lc)] x)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < 8) {
			break
		}
		v.Reset(ssaop.OpARM64SBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc, 8-lc))
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (SBFX [bfc] x))
	// cond: bfc.Width() <= 8
	// result: (SBFX [bfc] x)
	for {
		if v_0.Op != ssaop.OpARM64SBFX {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(bfc.Width() <= 8) {
			break
		}
		v.Reset(ssaop.OpARM64SBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(bfc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVBstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBstoreidx ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVBstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
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
		v.Reset(ssaop.OpARM64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
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
		if v_1.Op != ssaop.OpARM64MOVBreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVBstore)
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
		if v_1.Op != ssaop.OpARM64MOVBUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVBstore)
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
		if v_1.Op != ssaop.OpARM64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVBstore)
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
		if v_1.Op != ssaop.OpARM64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVBstore)
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
		if v_1.Op != ssaop.OpARM64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVBstore)
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
		if v_1.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVBstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstoreidx ptr (MOVDconst [c]) val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVBstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstoreidx (MOVDconst [c]) idx val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVBstore [int32(c)] idx val mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVBreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVBreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVBUreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVBUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVHreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVHreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVHUreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVHUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVWreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVWUreg x) mem)
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVBstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVDload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVDload [off] {sym} ptr (FMOVDstore [off] {sym} ptr val _))
	// result: (FMOVDfpgp val)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARM64FMOVDstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpARM64FMOVDfpgp)
		v.AddArg(val)
		return true
	}
	// match: (MOVDload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVDload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVDload [off] {sym} (ADDshiftLL [3] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDloadidx8 ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDshiftLL || ssa.AuxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDloadidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVDload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVDload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVDconst [int64(ssa.Read64(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Read64(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVDloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDloadidx ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVDload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDloadidx (MOVDconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVDload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDloadidx ptr (SLLconst [3] idx) mem)
	// result: (MOVDloadidx8 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 3 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVDloadidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVDloadidx (SLLconst [3] idx) ptr mem)
	// result: (MOVDloadidx8 ptr idx mem)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVDloadidx8)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVDloadidx8(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDloadidx8 ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c<<3)
	// result: (MOVDload [int32(c)<<3] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c << 3)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) << 3)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVDnop(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDnop (MOVDconst [c]))
	// result: (MOVDconst [c])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVDreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVDreg x)
	// cond: x.Uses == 1
	// result: (MOVDnop x)
	for {
		x := v_0
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDnop)
		v.AddArg(x)
		return true
	}
	// match: (MOVDreg (MOVDconst [c]))
	// result: (MOVDconst [c])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVDstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVDstore [off] {sym} ptr (FMOVDfpgp val) mem)
	// result: (FMOVDstore [off] {sym} ptr val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARM64FMOVDfpgp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64FMOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVDstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstoreidx ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVDstore [off] {sym} (ADDshiftLL [3] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstoreidx8 ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDshiftLL || ssa.AuxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDstoreidx8)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVDstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
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
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVDstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstoreidx ptr (MOVDconst [c]) val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVDstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstoreidx (MOVDconst [c]) idx val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVDstore [int32(c)] idx val mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (MOVDstoreidx ptr (SLLconst [3] idx) val mem)
	// result: (MOVDstoreidx8 ptr idx val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 3 {
			break
		}
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARM64MOVDstoreidx8)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVDstoreidx (SLLconst [3] idx) ptr val mem)
	// result: (MOVDstoreidx8 ptr idx val mem)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARM64MOVDstoreidx8)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVDstoreidx8(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstoreidx8 ptr (MOVDconst [c]) val mem)
	// cond: ssa.Is32Bit(c<<3)
	// result: (MOVDstore [int32(c)<<3] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c << 3)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) << 3)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVHUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHUloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (ADDshiftLL [1] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHUloadidx2 ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDshiftLL || ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHUloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHUload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHUload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVDconst [int64(ssa.Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVHUloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHUloadidx ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVHUload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUloadidx (MOVDconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVHUload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUloadidx ptr (SLLconst [1] idx) mem)
	// result: (MOVHUloadidx2 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVHUloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHUloadidx ptr (ADD idx idx) mem)
	// result: (MOVHUloadidx2 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_1.Args[1]
		if idx != v_1.Args[0] {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpARM64MOVHUloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHUloadidx (ADD idx idx) ptr mem)
	// result: (MOVHUloadidx2 ptr idx mem)
	for {
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		if idx != v_0.Args[0] {
			break
		}
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVHUloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVHUloadidx2(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHUloadidx2 ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c<<1)
	// result: (MOVHUload [int32(c)<<1] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c << 1)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) << 1)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVHUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHUreg (ANDconst [c] x))
	// result: (ANDconst [c&(1<<16-1)] x)
	for {
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & (1<<16 - 1))
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint16(c))])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint16(c)))
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
		v.CopyOf(x)
		return true
	}
	// match: (MOVHUreg (SLLconst [lc] x))
	// cond: lc >= 16
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(lc >= 16) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (MOVHUreg (SLLconst [lc] x))
	// cond: lc < 16
	// result: (UBFIZ [ssa.ArmBFAuxInt(lc, 16-lc)] x)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < 16) {
			break
		}
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc, 16-lc))
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (SRLconst [rc] x))
	// cond: rc < 16
	// result: (UBFX [ssa.ArmBFAuxInt(rc, 16)] x)
	for {
		if v_0.Op != ssaop.OpARM64SRLconst {
			break
		}
		rc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(rc < 16) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(rc, 16))
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (UBFX [bfc] x))
	// cond: bfc.Width() <= 16
	// result: (UBFX [bfc] x)
	for {
		if v_0.Op != ssaop.OpARM64UBFX {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(bfc.Width() <= 16) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(bfc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVHload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHload [off] {sym} (ADDshiftLL [1] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHloadidx2 ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDshiftLL || ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVDconst [int64(int16(ssa.Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder)))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int16(ssa.Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder))))
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVHloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHloadidx ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVHload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHloadidx (MOVDconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVHload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHloadidx ptr (SLLconst [1] idx) mem)
	// result: (MOVHloadidx2 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVHloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHloadidx ptr (ADD idx idx) mem)
	// result: (MOVHloadidx2 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_1.Args[1]
		if idx != v_1.Args[0] {
			break
		}
		mem := v_2
		v.Reset(ssaop.OpARM64MOVHloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHloadidx (ADD idx idx) ptr mem)
	// result: (MOVHloadidx2 ptr idx mem)
	for {
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		if idx != v_0.Args[0] {
			break
		}
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVHloadidx2)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVHloadidx2(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHloadidx2 ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c<<1)
	// result: (MOVHload [int32(c)<<1] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c << 1)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) << 1)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVHreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int16(c))])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int16(c)))
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
		v.CopyOf(x)
		return true
	}
	// match: (MOVHreg <t> (ANDconst x [c]))
	// cond: uint64(c) & uint64(0xffffffffffff8000) == 0
	// result: (ANDconst <t> x [c])
	for {
		t := v.Type
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(uint64(c)&uint64(0xffffffffffff8000) == 0) {
			break
		}
		v.Reset(ssaop.OpARM64ANDconst)
		v.Type = t
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SLLconst [lc] x))
	// cond: lc < 16
	// result: (SBFIZ [ssa.ArmBFAuxInt(lc, 16-lc)] x)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < 16) {
			break
		}
		v.Reset(ssaop.OpARM64SBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc, 16-lc))
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (SBFX [bfc] x))
	// cond: bfc.Width() <= 16
	// result: (SBFX [bfc] x)
	for {
		if v_0.Op != ssaop.OpARM64SBFX {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(bfc.Width() <= 16) {
			break
		}
		v.Reset(ssaop.OpARM64SBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(bfc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVHstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstoreidx ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstore [off] {sym} (ADDshiftLL [1] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDshiftLL || ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVHstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
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
		v.Reset(ssaop.OpARM64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
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
		if v_1.Op != ssaop.OpARM64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVHstore)
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
		if v_1.Op != ssaop.OpARM64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVHstore)
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
		if v_1.Op != ssaop.OpARM64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVHstore)
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
		if v_1.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVHstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstoreidx ptr (MOVDconst [c]) val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVHstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstoreidx (MOVDconst [c]) idx val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVHstore [int32(c)] idx val mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx ptr (SLLconst [1] idx) val mem)
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx ptr (ADD idx idx) val mem)
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_1.Args[1]
		if idx != v_1.Args[0] {
			break
		}
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx (SLLconst [1] idx) ptr val mem)
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx (ADD idx idx) ptr val mem)
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		if idx != v_0.Args[0] {
			break
		}
		ptr := v_1
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVHreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVHreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVHUreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVHUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVWreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVWUreg x) mem)
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVHstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVHstoreidx2(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstoreidx2 ptr (MOVDconst [c]) val mem)
	// cond: ssa.Is32Bit(c<<1)
	// result: (MOVHstore [int32(c)<<1] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c << 1)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) << 1)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVHreg x) mem)
	// result: (MOVHstoreidx2 ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVHreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVHUreg x) mem)
	// result: (MOVHstoreidx2 ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVHUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVWreg x) mem)
	// result: (MOVHstoreidx2 ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVWUreg x) mem)
	// result: (MOVHstoreidx2 ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVHstoreidx2)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVWUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWUload [off] {sym} ptr (FMOVSstore [off] {sym} ptr val _))
	// result: (FMOVSfpgp val)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARM64FMOVSstore || ssa.AuxIntToInt32(v_1.AuxInt) != off || ssa.AuxToSym(v_1.Aux) != sym {
			break
		}
		val := v_1.Args[1]
		if ptr != v_1.Args[0] {
			break
		}
		v.Reset(ssaop.OpARM64FMOVSfpgp)
		v.AddArg(val)
		return true
	}
	// match: (MOVWUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWUload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWUloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWUload [off] {sym} (ADDshiftLL [2] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWUloadidx4 ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDshiftLL || ssa.AuxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWUload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWUload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVDconst [int64(ssa.Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVWUloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWUloadidx ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVWUload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUloadidx (MOVDconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVWUload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWUloadidx ptr (SLLconst [2] idx) mem)
	// result: (MOVWUloadidx4 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVWUloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWUloadidx (SLLconst [2] idx) ptr mem)
	// result: (MOVWUloadidx4 ptr idx mem)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVWUloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVWUloadidx4(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWUloadidx4 ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c<<2)
	// result: (MOVWUload [int32(c)<<2] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c << 2)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) << 2)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVWUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWUreg (ANDconst [c] x))
	// result: (ANDconst [c&(1<<32-1)] x)
	for {
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & (1<<32 - 1))
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (MOVDconst [c]))
	// result: (MOVDconst [int64(uint32(c))])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint32(c)))
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
		v.CopyOf(x)
		return true
	}
	// match: (MOVWUreg (SLLconst [lc] x))
	// cond: lc >= 32
	// result: (MOVDconst [0])
	for {
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		if !(lc >= 32) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (MOVWUreg (SLLconst [lc] x))
	// cond: lc < 32
	// result: (UBFIZ [ssa.ArmBFAuxInt(lc, 32-lc)] x)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < 32) {
			break
		}
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc, 32-lc))
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (SRLconst [rc] x))
	// cond: rc < 32
	// result: (UBFX [ssa.ArmBFAuxInt(rc, 32)] x)
	for {
		if v_0.Op != ssaop.OpARM64SRLconst {
			break
		}
		rc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(rc < 32) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(rc, 32))
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (UBFX [bfc] x))
	// cond: bfc.Width() <= 32
	// result: (UBFX [bfc] x)
	for {
		if v_0.Op != ssaop.OpARM64UBFX {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(bfc.Width() <= 32) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(bfc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVWload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWload [off1+int32(off2)] {sym} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWloadidx ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (ADDshiftLL [2] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWloadidx4 ptr idx mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDshiftLL || ssa.AuxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWload [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (SB) _)
	// cond: ssa.SymIsRO(sym)
	// result: (MOVDconst [int64(int32(ssa.Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder)))])
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(ssa.SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int32(ssa.Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder))))
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVWloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWloadidx ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVWload [int32(c)] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWloadidx (MOVDconst [c]) ptr mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVWload [int32(c)] ptr mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWloadidx ptr (SLLconst [2] idx) mem)
	// result: (MOVWloadidx4 ptr idx mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVWloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWloadidx (SLLconst [2] idx) ptr mem)
	// result: (MOVWloadidx4 ptr idx mem)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVWloadidx4)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVWloadidx4(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWloadidx4 ptr (MOVDconst [c]) mem)
	// cond: ssa.Is32Bit(c<<2)
	// result: (MOVWload [int32(c)<<2] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		mem := v_2
		if !(ssa.Is32Bit(c << 2)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWload)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) << 2)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVWreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWreg (MOVDconst [c]))
	// result: (MOVDconst [int64(int32(c))])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(int32(c)))
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
		v.CopyOf(x)
		return true
	}
	// match: (MOVWreg <t> (ANDconst x [c]))
	// cond: uint64(c) & uint64(0xffffffff80000000) == 0
	// result: (ANDconst <t> x [c])
	for {
		t := v.Type
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(uint64(c)&uint64(0xffffffff80000000) == 0) {
			break
		}
		v.Reset(ssaop.OpARM64ANDconst)
		v.Type = t
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (SLLconst [lc] x))
	// cond: lc < 32
	// result: (SBFIZ [ssa.ArmBFAuxInt(lc, 32-lc)] x)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < 32) {
			break
		}
		v.Reset(ssaop.OpARM64SBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc, 32-lc))
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (SBFX [bfc] x))
	// cond: bfc.Width() <= 32
	// result: (SBFX [bfc] x)
	for {
		if v_0.Op != ssaop.OpARM64SBFX {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(bfc.Width() <= 32) {
			break
		}
		v.Reset(ssaop.OpARM64SBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(bfc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVWstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstore [off] {sym} ptr (FMOVSfpgp val) mem)
	// result: (FMOVSstore [off] {sym} ptr val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARM64FMOVSfpgp {
			break
		}
		val := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64FMOVSstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWstore [off1+int32(off2)] {sym} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstoreidx ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstore [off] {sym} (ADDshiftLL [2] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstoreidx4 ptr idx val mem)
	for {
		off := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDshiftLL || ssa.AuxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(off == 0 && sym == nil) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWstoreidx4)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (MOVWstore [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
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
		v.Reset(ssaop.OpARM64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
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
		if v_1.Op != ssaop.OpARM64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVWstore)
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
		if v_1.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARM64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(off)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVWstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreidx ptr (MOVDconst [c]) val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVWstore [int32(c)] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstoreidx (MOVDconst [c]) idx val mem)
	// cond: ssa.Is32Bit(c)
	// result: (MOVWstore [int32(c)] idx val mem)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		idx := v_1
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v.AddArg3(idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx ptr (SLLconst [2] idx) val mem)
	// result: (MOVWstoreidx4 ptr idx val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_1.AuxInt) != 2 {
			break
		}
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARM64MOVWstoreidx4)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx (SLLconst [2] idx) ptr val mem)
	// result: (MOVWstoreidx4 ptr idx val mem)
	for {
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARM64MOVWstoreidx4)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVWreg x) mem)
	// result: (MOVWstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVWstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVWUreg x) mem)
	// result: (MOVWstoreidx ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVWstoreidx)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MOVWstoreidx4(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreidx4 ptr (MOVDconst [c]) val mem)
	// cond: ssa.Is32Bit(c<<2)
	// result: (MOVWstore [int32(c)<<2] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		val := v_2
		mem := v_3
		if !(ssa.Is32Bit(c << 2)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(c) << 2)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstoreidx4 ptr idx (MOVWreg x) mem)
	// result: (MOVWstoreidx4 ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVWstoreidx4)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	// match: (MOVWstoreidx4 ptr idx (MOVWUreg x) mem)
	// result: (MOVWstoreidx4 ptr idx x mem)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		mem := v_3
		v.Reset(ssaop.OpARM64MOVWstoreidx4)
		v.AddArg4(ptr, idx, x, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64MSUB(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MSUB a x (MOVDconst [-1]))
	// result: (ADD a x)
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_2.AuxInt) != -1 {
			break
		}
		v.Reset(ssaop.OpARM64ADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a _ (MOVDconst [0]))
	// result: a
	for {
		a := v_0
		if v_2.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_2.AuxInt) != 0 {
			break
		}
		v.CopyOf(a)
		return true
	}
	// match: (MSUB a x (MOVDconst [1]))
	// result: (SUB a x)
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_2.AuxInt) != 1 {
			break
		}
		v.Reset(ssaop.OpARM64SUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (SUBshiftLL a x [ssa.Log64(c)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c-1) && c>=3
	// result: (SUB a (ADDshiftLL <x.Type> x x [ssa.Log64(c-1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(ssa.IsPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.Reset(ssaop.OpARM64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c - 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c+1) && c>=7
	// result: (ADD a (SUBshiftLL <x.Type> x x [ssa.Log64(c+1)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(ssa.IsPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.Reset(ssaop.OpARM64ADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c + 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [2]) [ssa.Log64(c/3)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3)) {
			break
		}
		v.Reset(ssaop.OpARM64ADDshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 3))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [2]) [ssa.Log64(c/5)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 5))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: c%7 == 0 && ssa.IsPowerOfTwo(c/7)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [3]) [ssa.Log64(c/7)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%7 == 0 && ssa.IsPowerOfTwo(c/7)) {
			break
		}
		v.Reset(ssaop.OpARM64ADDshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 7))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [3]) [ssa.Log64(c/9)])
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 9))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a (MOVDconst [-1]) x)
	// result: (ADD a x)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != -1 {
			break
		}
		x := v_2
		v.Reset(ssaop.OpARM64ADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a (MOVDconst [0]) _)
	// result: a
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
			break
		}
		v.CopyOf(a)
		return true
	}
	// match: (MSUB a (MOVDconst [1]) x)
	// result: (SUB a x)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		x := v_2
		v.Reset(ssaop.OpARM64SUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(c)
	// result: (SUBshiftLL a x [ssa.Log64(c)])
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
		v.AddArg2(a, x)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(c-1) && c>=3
	// result: (SUB a (ADDshiftLL <x.Type> x x [ssa.Log64(c-1)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(ssa.IsPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.Reset(ssaop.OpARM64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c - 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(c+1) && c>=7
	// result: (ADD a (SUBshiftLL <x.Type> x x [ssa.Log64(c+1)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(ssa.IsPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.Reset(ssaop.OpARM64ADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c + 1))
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [2]) [ssa.Log64(c/3)])
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3)) {
			break
		}
		v.Reset(ssaop.OpARM64ADDshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 3))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [2]) [ssa.Log64(c/5)])
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 5))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(2)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: c%7 == 0 && ssa.IsPowerOfTwo(c/7)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [3]) [ssa.Log64(c/7)])
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%7 == 0 && ssa.IsPowerOfTwo(c/7)) {
			break
		}
		v.Reset(ssaop.OpARM64ADDshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 7))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [3]) [ssa.Log64(c/9)])
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 9))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(3)
		v0.AddArg2(x, x)
		v.AddArg2(a, v0)
		return true
	}
	// match: (MSUB (MOVDconst [c]) x y)
	// result: (ADDconst [c] (MNEG <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MNEG, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) (MOVDconst [d]))
	// result: (SUBconst [c*d] a)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_2.AuxInt)
		v.Reset(ssaop.OpARM64SUBconst)
		v.AuxInt = ssa.Int64ToAuxInt(c * d)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValue_OpARM64MSUBW(v *ssa.Value) bool {
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
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(int32(c) == -1) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a _ (MOVDconst [c]))
	// cond: int32(c)==0
	// result: (MOVWUreg a)
	for {
		a := v_0
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(int32(c) == 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v.AddArg(a)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: int32(c)==1
	// result: (MOVWUreg (SUB <a.Type> a x))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(int32(c) == 1) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a x [ssa.Log64(c)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c-1) && int32(c)>=3
	// result: (MOVWUreg (SUB <a.Type> a (ADDshiftLL <x.Type> x x [ssa.Log64(c-1)])))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(ssa.IsPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, a.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c - 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c+1) && int32(c)>=7
	// result: (MOVWUreg (ADD <a.Type> a (SUBshiftLL <x.Type> x x [ssa.Log64(c+1)])))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(ssa.IsPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, a.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c + 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3) && ssa.Is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [2]) [ssa.Log64(c/3)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 3))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5) && ssa.Is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [2]) [ssa.Log64(c/5)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 5))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%7 == 0 && ssa.IsPowerOfTwo(c/7) && ssa.Is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [3]) [ssa.Log64(c/7)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%7 == 0 && ssa.IsPowerOfTwo(c/7) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 7))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9) && ssa.Is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [3]) [ssa.Log64(c/9)]))
	for {
		a := v_0
		x := v_1
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_2.AuxInt)
		if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 9))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(3)
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
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(int32(c) == -1) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) _)
	// cond: int32(c)==0
	// result: (MOVWUreg a)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(int32(c) == 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v.AddArg(a)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: int32(c)==1
	// result: (MOVWUreg (SUB <a.Type> a x))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(int32(c) == 1) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, a.Type)
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a x [ssa.Log64(c)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
		v0.AddArg2(a, x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(c-1) && int32(c)>=3
	// result: (MOVWUreg (SUB <a.Type> a (ADDshiftLL <x.Type> x x [ssa.Log64(c-1)])))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(ssa.IsPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, a.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c - 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: ssa.IsPowerOfTwo(c+1) && int32(c)>=7
	// result: (MOVWUreg (ADD <a.Type> a (SUBshiftLL <x.Type> x x [ssa.Log64(c+1)])))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(ssa.IsPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, a.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c + 1))
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%3 == 0 && ssa.IsPowerOfTwo(c/3) && ssa.Is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [2]) [ssa.Log64(c/3)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%3 == 0 && ssa.IsPowerOfTwo(c/3) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 3))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%5 == 0 && ssa.IsPowerOfTwo(c/5) && ssa.Is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [2]) [ssa.Log64(c/5)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%5 == 0 && ssa.IsPowerOfTwo(c/5) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 5))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%7 == 0 && ssa.IsPowerOfTwo(c/7) && ssa.Is32Bit(c)
	// result: (MOVWUreg (ADDshiftLL <a.Type> a (SUBshiftLL <x.Type> x x [3]) [ssa.Log64(c/7)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%7 == 0 && ssa.IsPowerOfTwo(c/7) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 7))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%9 == 0 && ssa.IsPowerOfTwo(c/9) && ssa.Is32Bit(c)
	// result: (MOVWUreg (SUBshiftLL <a.Type> a (ADDshiftLL <x.Type> x x [3]) [ssa.Log64(c/9)]))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_2
		if !(c%9 == 0 && ssa.IsPowerOfTwo(c/9) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBshiftLL, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c / 9))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW (MOVDconst [c]) x y)
	// result: (MOVWUreg (ADDconst <x.Type> [c] (MNEGW <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADDconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MNEGW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVWUreg (SUBconst <a.Type> [c*d] a))
	for {
		a := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if v_2.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_2.AuxInt)
		v.Reset(ssaop.OpARM64MOVWUreg)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUBconst, a.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(c * d)
		v0.AddArg(a)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpARM64MUL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MUL (NEG x) y)
	// result: (MNEG x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64NEG {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.Reset(ssaop.OpARM64MNEG)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (MUL _ (MOVDconst [0]))
	// result: (MOVDconst [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 0 {
				continue
			}
			v.Reset(ssaop.OpARM64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(0)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [1]))
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
				continue
			}
			v.CopyOf(x)
			return true
		}
		break
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: ssa.CanMulStrengthReduce(config, c)
	// result: {ssa.MulStrengthReduce(v, x, c)}
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
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
	// match: (MUL (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			d := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpARM64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(c * d)
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
			if r.Op != ssaop.OpARM64MOVWUreg {
				continue
			}
			x := r.Args[0]
			s := v_1
			if s.Op != ssaop.OpARM64MOVWUreg {
				continue
			}
			y := s.Args[0]
			if !(r.Uses == 1 && s.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpARM64UMULL)
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
			if r.Op != ssaop.OpARM64MOVWreg {
				continue
			}
			x := r.Args[0]
			s := v_1
			if s.Op != ssaop.OpARM64MOVWreg {
				continue
			}
			y := s.Args[0]
			if !(r.Uses == 1 && s.Uses == 1) {
				continue
			}
			v.Reset(ssaop.OpARM64MULL)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64MULW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MULW (NEG x) y)
	// result: (MNEGW x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64NEG {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.Reset(ssaop.OpARM64MNEGW)
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
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(int32(c) == 0) {
				continue
			}
			v.Reset(ssaop.OpARM64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(0)
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
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(int32(c) == 1) {
				continue
			}
			v.Reset(ssaop.OpARM64MOVWUreg)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: v.Type.Size() <= 4 && ssa.CanMulStrengthReduce32(config, int32(c))
	// result: {ssa.MulStrengthReduce32(v, x, int32(c))}
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			if !(v.Type.Size() <= 4 && ssa.CanMulStrengthReduce32(config, int32(c))) {
				continue
			}
			v.CopyOf(ssa.MulStrengthReduce32(v, x, int32(c)))
			return true
		}
		break
	}
	// match: (MULW (MOVDconst [c]) (MOVDconst [d]))
	// result: (MOVDconst [int64(uint32(c*d))])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			d := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpARM64MOVDconst)
			v.AuxInt = ssa.Int64ToAuxInt(int64(uint32(c * d)))
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64MVN(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MVN (XOR x y))
	// result: (EON x y)
	for {
		if v_0.Op != ssaop.OpARM64XOR {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64EON)
		v.AddArg2(x, y)
		return true
	}
	// match: (MVN (MOVDconst [c]))
	// result: (MOVDconst [^c])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(^c)
		return true
	}
	// match: (MVN x:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x)
	// result: (MVNshiftLL [c] y)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64SLLconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(ssa.ClobberIfDead(x)) {
			break
		}
		v.Reset(ssaop.OpARM64MVNshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	// match: (MVN x:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x)
	// result: (MVNshiftRL [c] y)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64SRLconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(ssa.ClobberIfDead(x)) {
			break
		}
		v.Reset(ssaop.OpARM64MVNshiftRL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	// match: (MVN x:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x)
	// result: (MVNshiftRA [c] y)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64SRAconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(ssa.ClobberIfDead(x)) {
			break
		}
		v.Reset(ssaop.OpARM64MVNshiftRA)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	// match: (MVN x:(RORconst [c] y))
	// cond: ssa.ClobberIfDead(x)
	// result: (MVNshiftRO [c] y)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64RORconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(ssa.ClobberIfDead(x)) {
			break
		}
		v.Reset(ssaop.OpARM64MVNshiftRO)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64MVNshiftLL(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MVNshiftLL (MOVDconst [c]) [d])
	// result: (MOVDconst [^int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(^int64(uint64(c) << uint64(d)))
		return true
	}
	return false
}
func rewriteValue_OpARM64MVNshiftRA(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MVNshiftRA (MOVDconst [c]) [d])
	// result: (MOVDconst [^(c>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(^(c >> uint64(d)))
		return true
	}
	return false
}
func rewriteValue_OpARM64MVNshiftRL(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MVNshiftRL (MOVDconst [c]) [d])
	// result: (MOVDconst [^int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(^int64(uint64(c) >> uint64(d)))
		return true
	}
	return false
}
func rewriteValue_OpARM64MVNshiftRO(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MVNshiftRO (MOVDconst [c]) [d])
	// result: (MOVDconst [^rotateRight64(c, d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(^rotateRight64(c, d))
		return true
	}
	return false
}
func rewriteValue_OpARM64NEG(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NEG (MUL x y))
	// result: (MNEG x y)
	for {
		if v_0.Op != ssaop.OpARM64MUL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64MNEG)
		v.AddArg2(x, y)
		return true
	}
	// match: (NEG (MULW x y))
	// cond: v.Type.Size() <= 4
	// result: (MNEGW x y)
	for {
		if v_0.Op != ssaop.OpARM64MULW {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(v.Type.Size() <= 4) {
			break
		}
		v.Reset(ssaop.OpARM64MNEGW)
		v.AddArg2(x, y)
		return true
	}
	// match: (NEG (SUB x y))
	// result: (SUB y x)
	for {
		if v_0.Op != ssaop.OpARM64SUB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64SUB)
		v.AddArg2(y, x)
		return true
	}
	// match: (NEG (NEG x))
	// result: x
	for {
		if v_0.Op != ssaop.OpARM64NEG {
			break
		}
		x := v_0.Args[0]
		v.CopyOf(x)
		return true
	}
	// match: (NEG (MOVDconst [c]))
	// result: (MOVDconst [-c])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c)
		return true
	}
	// match: (NEG x:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x)
	// result: (NEGshiftLL [c] y)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64SLLconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(ssa.ClobberIfDead(x)) {
			break
		}
		v.Reset(ssaop.OpARM64NEGshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	// match: (NEG x:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x)
	// result: (NEGshiftRL [c] y)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64SRLconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(ssa.ClobberIfDead(x)) {
			break
		}
		v.Reset(ssaop.OpARM64NEGshiftRL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	// match: (NEG x:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x)
	// result: (NEGshiftRA [c] y)
	for {
		x := v_0
		if x.Op != ssaop.OpARM64SRAconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(ssa.ClobberIfDead(x)) {
			break
		}
		v.Reset(ssaop.OpARM64NEGshiftRA)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64NEGshiftLL(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NEGshiftLL (MOVDconst [c]) [d])
	// result: (MOVDconst [-int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-int64(uint64(c) << uint64(d)))
		return true
	}
	return false
}
func rewriteValue_OpARM64NEGshiftRA(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NEGshiftRA (MOVDconst [c]) [d])
	// result: (MOVDconst [-(c>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-(c >> uint64(d)))
		return true
	}
	return false
}
func rewriteValue_OpARM64NEGshiftRL(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NEGshiftRL (MOVDconst [c]) [d])
	// result: (MOVDconst [-int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-int64(uint64(c) >> uint64(d)))
		return true
	}
	return false
}
func rewriteValue_OpARM64NotEqual(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (NotEqual (CMPconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (TST x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TST, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPWconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (NotEqual (TSTWconst [int32(c)] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPWconst [0] z:(AND x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (TSTW x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64AND {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPconst [0] x:(ANDconst [c] y)))
	// cond: x.Uses == 1
	// result: (NotEqual (TSTconst [c] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMP x z:(NEG y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMN x y))
	for {
		if v_0.Op != ssaop.OpARM64CMP {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		z := v_0.Args[1]
		if z.Op != ssaop.OpARM64NEG {
			break
		}
		y := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMN, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPW x z:(NEG y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMNW x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPW {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		z := v_0.Args[1]
		if z.Op != ssaop.OpARM64NEG {
			break
		}
		y := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (NotEqual (CMNconst [c] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(c)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPWconst [0] x:(ADDconst [c] y)))
	// cond: x.Uses == 1
	// result: (NotEqual (CMNWconst [int32(c)] y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		x := v_0.Args[0]
		if x.Op != ssaop.OpARM64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(x.AuxInt)
		y := x.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMN x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMN, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPWconst [0] z:(ADD x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMNW x y))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64ADD {
			break
		}
		y := z.Args[1]
		x := z.Args[0]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPconst [0] z:(MADD a x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMN a (MUL <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64MADD {
			break
		}
		y := z.Args[2]
		a := z.Args[0]
		x := z.Args[1]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMN, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPWconst [0] z:(MADDW a x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMNW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64MADDW {
			break
		}
		y := z.Args[2]
		a := z.Args[0]
		x := z.Args[1]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPconst [0] z:(MSUB a x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMP a (MUL <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64CMPconst || ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64MSUB {
			break
		}
		y := z.Args[2]
		a := z.Args[0]
		x := z.Args[1]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MUL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (CMPWconst [0] z:(MSUBW a x y)))
	// cond: z.Uses == 1
	// result: (NotEqual (CMPW a (MULW <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARM64CMPWconst || ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		z := v_0.Args[0]
		if z.Op != ssaop.OpARM64MSUBW {
			break
		}
		y := z.Args[2]
		a := z.Args[0]
		x := z.Args[1]
		if !(z.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MULW, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg2(a, v1)
		v.AddArg(v0)
		return true
	}
	// match: (NotEqual (FlagConstant [fc]))
	// result: (MOVDconst [ssa.B2i(fc.Ne())])
	for {
		if v_0.Op != ssaop.OpARM64FlagConstant {
			break
		}
		fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(fc.Ne()))
		return true
	}
	// match: (NotEqual (InvertFlags x))
	// result: (NotEqual x)
	for {
		if v_0.Op != ssaop.OpARM64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64NotEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64OR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (OR x (MOVDconst [c]))
	// result: (ORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpARM64ORconst)
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
	// match: (OR x (MVN y))
	// result: (ORN x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MVN {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARM64ORN)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR x0 x1:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ORshiftLL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SLLconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64ORshiftLL)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (OR x0 x1:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ORshiftRL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SRLconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64ORshiftRL)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (OR x0 x1:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ORshiftRA x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SRAconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64ORshiftRA)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (OR x0 x1:(RORconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ORshiftRO x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64RORconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64ORshiftRO)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (OR (UBFIZ [bfc] x) (ANDconst [ac] y))
	// cond: ac == ^((1<<uint(bfc.Width())-1) << uint(bfc.Lsb()))
	// result: (BFI [bfc] y x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64UBFIZ {
				continue
			}
			bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpARM64ANDconst {
				continue
			}
			ac := ssa.AuxIntToInt64(v_1.AuxInt)
			y := v_1.Args[0]
			if !(ac == ^((1<<uint(bfc.Width()) - 1) << uint(bfc.Lsb()))) {
				continue
			}
			v.Reset(ssaop.OpARM64BFI)
			v.AuxInt = ssa.Arm64BitFieldToAuxInt(bfc)
			v.AddArg2(y, x)
			return true
		}
		break
	}
	// match: (OR (UBFX [bfc] x) (ANDconst [ac] y))
	// cond: ac == ^(1<<uint(bfc.Width())-1)
	// result: (BFXIL [bfc] y x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64UBFX {
				continue
			}
			bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpARM64ANDconst {
				continue
			}
			ac := ssa.AuxIntToInt64(v_1.AuxInt)
			y := v_1.Args[0]
			if !(ac == ^(1<<uint(bfc.Width()) - 1)) {
				continue
			}
			v.Reset(ssaop.OpARM64BFXIL)
			v.AuxInt = ssa.Arm64BitFieldToAuxInt(bfc)
			v.AddArg2(y, x)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64ORN(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORN x (MOVDconst [c]))
	// result: (ORconst [^c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(^c)
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
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	// match: (ORN x0 x1:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ORNshiftLL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SLLconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64ORNshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (ORN x0 x1:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ORNshiftRL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SRLconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64ORNshiftRL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (ORN x0 x1:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ORNshiftRA x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SRAconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64ORNshiftRA)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (ORN x0 x1:(RORconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (ORNshiftRO x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64RORconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64ORNshiftRO)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64ORNshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORNshiftLL x (MOVDconst [c]) [d])
	// result: (ORconst x [^int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(^int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORNshiftLL (SLLconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValue_OpARM64ORNshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORNshiftRA x (MOVDconst [c]) [d])
	// result: (ORconst x [^(c>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(^(c >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORNshiftRA (SRAconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRAconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValue_OpARM64ORNshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORNshiftRL x (MOVDconst [c]) [d])
	// result: (ORconst x [^int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(^int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORNshiftRL (SRLconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValue_OpARM64ORNshiftRO(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ORNshiftRO x (MOVDconst [c]) [d])
	// result: (ORconst x [^rotateRight64(c, d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(^rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	// match: (ORNshiftRO (RORconst x [c]) x [c])
	// result: (MOVDconst [-1])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64RORconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	return false
}
func rewriteValue_OpARM64ORconst(v *ssa.Value) bool {
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
	// result: (MOVDconst [-1])
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != -1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-1)
		return true
	}
	// match: (ORconst [c] (MOVDconst [d]))
	// result: (MOVDconst [c|d])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c | d)
		return true
	}
	// match: (ORconst [c] (ORconst [d] x))
	// result: (ORconst [c|d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ORconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c | d)
		v.AddArg(x)
		return true
	}
	// match: (ORconst [c1] (ANDconst [c2] x))
	// cond: c2|c1 == ^0
	// result: (ORconst [c1] x)
	for {
		c1 := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c2 := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c2|c1 == ^0) {
			break
		}
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c1)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64ORshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ORshiftLL (MOVDconst [c]) x [d])
	// result: (ORconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftLL x (MOVDconst [c]) [d])
	// result: (ORconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL y:(SLLconst x [c]) x [c])
	// result: y
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ORshiftLL <typ.UInt16> [8] (UBFX <typ.UInt16> [ssa.ArmBFAuxInt(8, 8)] x) x)
	// result: (REV16W x)
	for {
		if v.Type != typ.UInt16 || ssa.AuxIntToInt64(v.AuxInt) != 8 || v_0.Op != ssaop.OpARM64UBFX || v_0.Type != typ.UInt16 || ssa.AuxIntToArm64BitField(v_0.AuxInt) != ssa.ArmBFAuxInt(8, 8) {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL [8] (UBFX [ssa.ArmBFAuxInt(8, 24)] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff
	// result: (REV16W x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 || v_0.Op != ssaop.OpARM64UBFX || ssa.AuxIntToArm64BitField(v_0.AuxInt) != ssa.ArmBFAuxInt(8, 24) {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c1 := ssa.AuxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != ssaop.OpARM64ANDconst {
			break
		}
		c2 := ssa.AuxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff) {
			break
		}
		v.Reset(ssaop.OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL [8] (SRLconst [8] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: (uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff)
	// result: (REV16 x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 || v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c1 := ssa.AuxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != ssaop.OpARM64ANDconst {
			break
		}
		c2 := ssa.AuxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff) {
			break
		}
		v.Reset(ssaop.OpARM64REV16)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL [8] (SRLconst [8] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: (uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff)
	// result: (REV16 (ANDconst <x.Type> [0xffffffff] x))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 || v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c1 := ssa.AuxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != ssaop.OpARM64ANDconst {
			break
		}
		c2 := ssa.AuxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff) {
			break
		}
		v.Reset(ssaop.OpARM64REV16)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ANDconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(0xffffffff)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: ( ORshiftLL [c] (SRLconst x [64-c]) x2)
	// result: (EXTRconst [64-c] x2 x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 64-c {
			break
		}
		x := v_0.Args[0]
		x2 := v_1
		v.Reset(ssaop.OpARM64EXTRconst)
		v.AuxInt = ssa.Int64ToAuxInt(64 - c)
		v.AddArg2(x2, x)
		return true
	}
	// match: ( ORshiftLL <t> [c] (UBFX [bfc] x) x2)
	// cond: c < 32 && t.Size() == 4 && bfc == ssa.ArmBFAuxInt(32-c, c)
	// result: (EXTRWconst [32-c] x2 x)
	for {
		t := v.Type
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64UBFX {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		x2 := v_1
		if !(c < 32 && t.Size() == 4 && bfc == ssa.ArmBFAuxInt(32-c, c)) {
			break
		}
		v.Reset(ssaop.OpARM64EXTRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(32 - c)
		v.AddArg2(x2, x)
		return true
	}
	// match: (ORshiftLL [s] (ANDconst [xc] x) (ANDconst [yc] y))
	// cond: xc == ^(yc << s) && yc & (yc+1) == 0 && yc > 0 && s+ssa.Log64(yc+1) <= 64
	// result: (BFI [ssa.ArmBFAuxInt(s, ssa.Log64(yc+1))] x y)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		xc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if v_1.Op != ssaop.OpARM64ANDconst {
			break
		}
		yc := ssa.AuxIntToInt64(v_1.AuxInt)
		y := v_1.Args[0]
		if !(xc == ^(yc<<s) && yc&(yc+1) == 0 && yc > 0 && s+ssa.Log64(yc+1) <= 64) {
			break
		}
		v.Reset(ssaop.OpARM64BFI)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(s, ssa.Log64(yc+1)))
		v.AddArg2(x, y)
		return true
	}
	// match: (ORshiftLL [sc] (UBFX [bfc] x) (SRLconst [sc] y))
	// cond: sc == bfc.Width()
	// result: (BFXIL [bfc] y x)
	for {
		sc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64UBFX {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if v_1.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_1.AuxInt) != sc {
			break
		}
		y := v_1.Args[0]
		if !(sc == bfc.Width()) {
			break
		}
		v.Reset(ssaop.OpARM64BFXIL)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(bfc)
		v.AddArg2(y, x)
		return true
	}
	return false
}
func rewriteValue_OpARM64ORshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ORshiftRA (MOVDconst [c]) x [d])
	// result: (ORconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRAconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRA x (MOVDconst [c]) [d])
	// result: (ORconst x [c>>uint64(d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRA y:(SRAconst x [c]) x [c])
	// result: y
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARM64SRAconst || ssa.AuxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
			break
		}
		v.CopyOf(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64ORshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ORshiftRL (MOVDconst [c]) x [d])
	// result: (ORconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRLconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRL x (MOVDconst [c]) [d])
	// result: (ORconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRL y:(SRLconst x [c]) x [c])
	// result: y
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
			break
		}
		v.CopyOf(y)
		return true
	}
	// match: (ORshiftRL [rc] (ANDconst [ac] x) (SLLconst [lc] y))
	// cond: lc > rc && ac == ^((1<<uint(64-lc)-1) << uint64(lc-rc))
	// result: (BFI [ssa.ArmBFAuxInt(lc-rc, 64-lc)] x y)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		ac := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if v_1.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_1.AuxInt)
		y := v_1.Args[0]
		if !(lc > rc && ac == ^((1<<uint(64-lc)-1)<<uint64(lc-rc))) {
			break
		}
		v.Reset(ssaop.OpARM64BFI)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc-rc, 64-lc))
		v.AddArg2(x, y)
		return true
	}
	// match: (ORshiftRL [rc] (ANDconst [ac] y) (SLLconst [lc] x))
	// cond: lc < rc && ac == ^((1<<uint(64-rc)-1))
	// result: (BFXIL [ssa.ArmBFAuxInt(rc-lc, 64-rc)] y x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		ac := ssa.AuxIntToInt64(v_0.AuxInt)
		y := v_0.Args[0]
		if v_1.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_1.AuxInt)
		x := v_1.Args[0]
		if !(lc < rc && ac == ^(1<<uint(64-rc)-1)) {
			break
		}
		v.Reset(ssaop.OpARM64BFXIL)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(rc-lc, 64-rc))
		v.AddArg2(y, x)
		return true
	}
	return false
}
func rewriteValue_OpARM64ORshiftRO(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ORshiftRO (MOVDconst [c]) x [d])
	// result: (ORconst [c] (RORconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64RORconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRO x (MOVDconst [c]) [d])
	// result: (ORconst x [rotateRight64(c, d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64ORconst)
		v.AuxInt = ssa.Int64ToAuxInt(rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRO y:(RORconst x [c]) x [c])
	// result: y
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARM64RORconst || ssa.AuxIntToInt64(y.AuxInt) != c {
			break
		}
		x := y.Args[0]
		if x != v_1 {
			break
		}
		v.CopyOf(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64REV(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (REV (REV p))
	// result: p
	for {
		if v_0.Op != ssaop.OpARM64REV {
			break
		}
		p := v_0.Args[0]
		v.CopyOf(p)
		return true
	}
	return false
}
func rewriteValue_OpARM64REV16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (REV16 (MOVWUreg x))
	// result: (REV16W x)
	for {
		if v_0.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64REVW(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (REVW (REVW p))
	// result: p
	for {
		if v_0.Op != ssaop.OpARM64REVW {
			break
		}
		p := v_0.Args[0]
		v.CopyOf(p)
		return true
	}
	return false
}
func rewriteValue_OpARM64ROR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ROR x (MOVDconst [c]))
	// result: (RORconst x [c&63])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64RORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 63)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64RORW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RORW x (MOVDconst [c]))
	// result: (RORWconst x [c&31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64RORWconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64SBCSflags(v *ssa.Value) bool {
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
		if v_2.Op != ssaop.OpSelect1 || v_2.Type != types.TypeFlags {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpARM64NEGSflags {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != ssaop.OpARM64NEG || v_2_0_0.Type != typ.UInt64 {
			break
		}
		v_2_0_0_0 := v_2_0_0.Args[0]
		if v_2_0_0_0.Op != ssaop.OpARM64NGCzerocarry || v_2_0_0_0.Type != typ.UInt64 {
			break
		}
		bo := v_2_0_0_0.Args[0]
		v.Reset(ssaop.OpARM64SBCSflags)
		v.AddArg3(x, y, bo)
		return true
	}
	// match: (SBCSflags x y (Select1 <types.TypeFlags> (NEGSflags (MOVDconst [0]))))
	// result: (SUBSflags x y)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpSelect1 || v_2.Type != types.TypeFlags {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != ssaop.OpARM64NEGSflags {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_2_0_0.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpARM64SUBSflags)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64SBFX(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SBFX [bfc] s:(SLLconst [sc] x))
	// cond: s.Uses == 1 && sc <= bfc.Lsb()
	// result: (SBFX [ssa.ArmBFAuxInt(bfc.Lsb() - sc, bfc.Width())] x)
	for {
		bfc := ssa.AuxIntToArm64BitField(v.AuxInt)
		s := v_0
		if s.Op != ssaop.OpARM64SLLconst {
			break
		}
		sc := ssa.AuxIntToInt64(s.AuxInt)
		x := s.Args[0]
		if !(s.Uses == 1 && sc <= bfc.Lsb()) {
			break
		}
		v.Reset(ssaop.OpARM64SBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(bfc.Lsb()-sc, bfc.Width()))
		v.AddArg(x)
		return true
	}
	// match: (SBFX [bfc] s:(SLLconst [sc] x))
	// cond: s.Uses == 1 && sc > bfc.Lsb() && sc < bfc.Lsb()+bfc.Width()
	// result: (SBFIZ [ssa.ArmBFAuxInt(sc - bfc.Lsb(), bfc.Width() - (sc-bfc.Lsb()))] x)
	for {
		bfc := ssa.AuxIntToArm64BitField(v.AuxInt)
		s := v_0
		if s.Op != ssaop.OpARM64SLLconst {
			break
		}
		sc := ssa.AuxIntToInt64(s.AuxInt)
		x := s.Args[0]
		if !(s.Uses == 1 && sc > bfc.Lsb() && sc < bfc.Lsb()+bfc.Width()) {
			break
		}
		v.Reset(ssaop.OpARM64SBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(sc-bfc.Lsb(), bfc.Width()-(sc-bfc.Lsb())))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64SLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLL x (MOVDconst [c]))
	// result: (SLLconst x [c&63])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64SLLconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 63)
		v.AddArg(x)
		return true
	}
	// match: (SLL x (ANDconst [63] y))
	// result: (SLL x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64ANDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 63 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpARM64SLL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64SLLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SLLconst [c] (MOVDconst [d]))
	// result: (MOVDconst [d<<uint64(c)])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(d << uint64(c))
		return true
	}
	// match: (SLLconst [c] (SRLconst [c] x))
	// cond: 0 < c && c < 64
	// result: (ANDconst [^(1<<uint(c)-1)] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if !(0 < c && c < 64) {
			break
		}
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(^(1<<uint(c) - 1))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [lc] (MOVWreg x))
	// result: (SBFIZ [ssa.ArmBFAuxInt(lc, min(32, 64-lc))] x)
	for {
		lc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVWreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64SBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc, min(32, 64-lc)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [lc] (MOVHreg x))
	// result: (SBFIZ [ssa.ArmBFAuxInt(lc, min(16, 64-lc))] x)
	for {
		lc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVHreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64SBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc, min(16, 64-lc)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [lc] (MOVBreg x))
	// result: (SBFIZ [ssa.ArmBFAuxInt(lc, min(8, 64-lc))] x)
	for {
		lc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVBreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64SBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc, min(8, 64-lc)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [lc] (MOVWUreg x))
	// result: (UBFIZ [ssa.ArmBFAuxInt(lc, min(32, 64-lc))] x)
	for {
		lc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc, min(32, 64-lc)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [lc] (MOVHUreg x))
	// result: (UBFIZ [ssa.ArmBFAuxInt(lc, min(16, 64-lc))] x)
	for {
		lc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc, min(16, 64-lc)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [lc] (MOVBUreg x))
	// result: (UBFIZ [ssa.ArmBFAuxInt(lc, min(8, 64-lc))] x)
	for {
		lc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc, min(8, 64-lc)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [sc] (ANDconst [ac] x))
	// cond: isARM64BFMask(sc, ac, 0)
	// result: (UBFIZ [ssa.ArmBFAuxInt(sc, arm64BFWidth(ac, 0))] x)
	for {
		sc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		ac := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, ac, 0)) {
			break
		}
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(sc, arm64BFWidth(ac, 0)))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [sc] (UBFIZ [bfc] x))
	// cond: sc+bfc.Width()+bfc.Lsb() < 64
	// result: (UBFIZ [ssa.ArmBFAuxInt(bfc.Lsb()+sc, bfc.Width())] x)
	for {
		sc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64UBFIZ {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc+bfc.Width()+bfc.Lsb() < 64) {
			break
		}
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(bfc.Lsb()+sc, bfc.Width()))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64SRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRA x (MOVDconst [c]))
	// result: (SRAconst x [c&63])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64SRAconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 63)
		v.AddArg(x)
		return true
	}
	// match: (SRA x (ANDconst [63] y))
	// result: (SRA x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64ANDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 63 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpARM64SRA)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64SRAconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SRAconst [c] (MOVDconst [d]))
	// result: (MOVDconst [d>>uint64(c)])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(d >> uint64(c))
		return true
	}
	// match: (SRAconst [rc] (SLLconst [lc] x))
	// cond: lc > rc
	// result: (SBFIZ [ssa.ArmBFAuxInt(lc-rc, 64-lc)] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc > rc) {
			break
		}
		v.Reset(ssaop.OpARM64SBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc-rc, 64-lc))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [rc] (SLLconst [lc] x))
	// cond: lc <= rc
	// result: (SBFX [ssa.ArmBFAuxInt(rc-lc, 64-rc)] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc <= rc) {
			break
		}
		v.Reset(ssaop.OpARM64SBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(rc-lc, 64-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [rc] (MOVWreg x))
	// cond: rc < 32
	// result: (SBFX [ssa.ArmBFAuxInt(rc, 32-rc)] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVWreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 32) {
			break
		}
		v.Reset(ssaop.OpARM64SBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(rc, 32-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [rc] (MOVHreg x))
	// cond: rc < 16
	// result: (SBFX [ssa.ArmBFAuxInt(rc, 16-rc)] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVHreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 16) {
			break
		}
		v.Reset(ssaop.OpARM64SBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(rc, 16-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [rc] (MOVBreg x))
	// cond: rc < 8
	// result: (SBFX [ssa.ArmBFAuxInt(rc, 8-rc)] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVBreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 8) {
			break
		}
		v.Reset(ssaop.OpARM64SBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(rc, 8-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [sc] (SBFIZ [bfc] x))
	// cond: sc < bfc.Lsb()
	// result: (SBFIZ [ssa.ArmBFAuxInt(bfc.Lsb()-sc, bfc.Width())] x)
	for {
		sc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SBFIZ {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.Lsb()) {
			break
		}
		v.Reset(ssaop.OpARM64SBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(bfc.Lsb()-sc, bfc.Width()))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [sc] (SBFIZ [bfc] x))
	// cond: sc >= bfc.Lsb() && sc < bfc.Lsb()+bfc.Width()
	// result: (SBFX [ssa.ArmBFAuxInt(sc-bfc.Lsb(), bfc.Lsb()+bfc.Width()-sc)] x)
	for {
		sc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SBFIZ {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc >= bfc.Lsb() && sc < bfc.Lsb()+bfc.Width()) {
			break
		}
		v.Reset(ssaop.OpARM64SBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(sc-bfc.Lsb(), bfc.Lsb()+bfc.Width()-sc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64SRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRL x (MOVDconst [c]))
	// result: (SRLconst x [c&63])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64SRLconst)
		v.AuxInt = ssa.Int64ToAuxInt(c & 63)
		v.AddArg(x)
		return true
	}
	// match: (SRL x (ANDconst [63] y))
	// result: (SRL x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64ANDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 63 {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpARM64SRL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64SRLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SRLconst [c] (MOVDconst [d]))
	// result: (MOVDconst [int64(uint64(d)>>uint64(c))])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(d) >> uint64(c)))
		return true
	}
	// match: (SRLconst [c] (SLLconst [c] x))
	// cond: 0 < c && c < 64
	// result: (ANDconst [1<<uint(64-c)-1] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if !(0 < c && c < 64) {
			break
		}
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1<<uint(64-c) - 1)
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [rc] (MOVWUreg x))
	// cond: rc >= 32
	// result: (MOVDconst [0])
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		if !(rc >= 32) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SRLconst [rc] (MOVHUreg x))
	// cond: rc >= 16
	// result: (MOVDconst [0])
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVHUreg {
			break
		}
		if !(rc >= 16) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SRLconst [rc] (MOVBUreg x))
	// cond: rc >= 8
	// result: (MOVDconst [0])
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVBUreg {
			break
		}
		if !(rc >= 8) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SRLconst [rc] (SLLconst [lc] x))
	// cond: lc > rc
	// result: (UBFIZ [ssa.ArmBFAuxInt(lc-rc, 64-lc)] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc > rc) {
			break
		}
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(lc-rc, 64-lc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [rc] (SLLconst [lc] x))
	// cond: lc < rc
	// result: (UBFX [ssa.ArmBFAuxInt(rc-lc, 64-rc)] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		lc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(lc < rc) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(rc-lc, 64-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [rc] (MOVWUreg x))
	// cond: rc < 32
	// result: (UBFX [ssa.ArmBFAuxInt(rc, 32-rc)] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 32) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(rc, 32-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [rc] (MOVHUreg x))
	// cond: rc < 16
	// result: (UBFX [ssa.ArmBFAuxInt(rc, 16-rc)] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 16) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(rc, 16-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [rc] (MOVBUreg x))
	// cond: rc < 8
	// result: (UBFX [ssa.ArmBFAuxInt(rc, 8-rc)] x)
	for {
		rc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 8) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(rc, 8-rc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (ANDconst [ac] x))
	// cond: isARM64BFMask(sc, ac, sc)
	// result: (UBFX [ssa.ArmBFAuxInt(sc, arm64BFWidth(ac, sc))] x)
	for {
		sc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		ac := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, ac, sc)) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(sc, arm64BFWidth(ac, sc)))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFX [bfc] x))
	// cond: sc < bfc.Width()
	// result: (UBFX [ssa.ArmBFAuxInt(bfc.Lsb()+sc, bfc.Width()-sc)] x)
	for {
		sc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64UBFX {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.Width()) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(bfc.Lsb()+sc, bfc.Width()-sc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFIZ [bfc] x))
	// cond: sc == bfc.Lsb()
	// result: (ANDconst [1<<uint(bfc.Width())-1] x)
	for {
		sc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64UBFIZ {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc == bfc.Lsb()) {
			break
		}
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1<<uint(bfc.Width()) - 1)
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFIZ [bfc] x))
	// cond: sc < bfc.Lsb()
	// result: (UBFIZ [ssa.ArmBFAuxInt(bfc.Lsb()-sc, bfc.Width())] x)
	for {
		sc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64UBFIZ {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.Lsb()) {
			break
		}
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(bfc.Lsb()-sc, bfc.Width()))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFIZ [bfc] x))
	// cond: sc > bfc.Lsb() && sc < bfc.Lsb()+bfc.Width()
	// result: (UBFX [ssa.ArmBFAuxInt(sc-bfc.Lsb(), bfc.Lsb()+bfc.Width()-sc)] x)
	for {
		sc := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64UBFIZ {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc > bfc.Lsb() && sc < bfc.Lsb()+bfc.Width()) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(sc-bfc.Lsb(), bfc.Lsb()+bfc.Width()-sc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64STP(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (STP [off1] {sym} (ADDconst [off2] ptr) val1 val2 mem)
	// cond: ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (STP [off1+int32(off2)] {sym} ptr val1 val2 mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		off2 := ssa.AuxIntToInt64(v_0.AuxInt)
		ptr := v_0.Args[0]
		val1 := v_1
		val2 := v_2
		mem := v_3
		if !(ssa.Is32Bit(int64(off1)+off2) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64STP)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + int32(off2))
		v.Aux = ssa.SymToAux(sym)
		v.AddArg4(ptr, val1, val2, mem)
		return true
	}
	// match: (STP [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val1 val2 mem)
	// cond: ssa.CanMergeSym(sym1,sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)
	// result: (STP [off1+off2] {ssa.MergeSym(sym1,sym2)} ptr val1 val2 mem)
	for {
		off1 := ssa.AuxIntToInt32(v.AuxInt)
		sym1 := ssa.AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARM64MOVDaddr {
			break
		}
		off2 := ssa.AuxIntToInt32(v_0.AuxInt)
		sym2 := ssa.AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val1 := v_1
		val2 := v_2
		mem := v_3
		if !(ssa.CanMergeSym(sym1, sym2) && ssa.Is32Bit(int64(off1)+int64(off2)) && (ptr.Op != ssaop.OpSB || !config.Ctxt.Flag_dynlink)) {
			break
		}
		v.Reset(ssaop.OpARM64STP)
		v.AuxInt = ssa.Int32ToAuxInt(off1 + off2)
		v.Aux = ssa.SymToAux(ssa.MergeSym(sym1, sym2))
		v.AddArg4(ptr, val1, val2, mem)
		return true
	}
	return false
}
func rewriteValue_OpARM64SUB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUB x (MOVDconst [c]))
	// result: (SUBconst [c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64SUBconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUB a l:(MUL x y))
	// cond: l.Uses==1 && ssa.Clobber(l)
	// result: (MSUB a x y)
	for {
		a := v_0
		l := v_1
		if l.Op != ssaop.OpARM64MUL {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1 && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.OpARM64MSUB)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (SUB a l:(MNEG x y))
	// cond: l.Uses==1 && ssa.Clobber(l)
	// result: (MADD a x y)
	for {
		a := v_0
		l := v_1
		if l.Op != ssaop.OpARM64MNEG {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1 && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.OpARM64MADD)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (SUB a l:(MULW x y))
	// cond: v.Type.Size() <= 4 && l.Uses==1 && ssa.Clobber(l)
	// result: (MSUBW a x y)
	for {
		a := v_0
		l := v_1
		if l.Op != ssaop.OpARM64MULW {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(v.Type.Size() <= 4 && l.Uses == 1 && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.OpARM64MSUBW)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (SUB a l:(MNEGW x y))
	// cond: v.Type.Size() <= 4 && l.Uses==1 && ssa.Clobber(l)
	// result: (MADDW a x y)
	for {
		a := v_0
		l := v_1
		if l.Op != ssaop.OpARM64MNEGW {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(v.Type.Size() <= 4 && l.Uses == 1 && ssa.Clobber(l)) {
			break
		}
		v.Reset(ssaop.OpARM64MADDW)
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
		if p.Op != ssaop.OpARM64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != ssaop.OpARM64MUL || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.Reset(ssaop.OpARM64SUBconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, v.Type)
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
		if p.Op != ssaop.OpARM64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != ssaop.OpARM64MULW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.Reset(ssaop.OpARM64SUBconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, v.Type)
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
		if p.Op != ssaop.OpARM64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != ssaop.OpARM64MNEG || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.Reset(ssaop.OpARM64SUBconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, v.Type)
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
		if p.Op != ssaop.OpARM64ADDconst {
			break
		}
		c := ssa.AuxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != ssaop.OpARM64MNEGW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.Reset(ssaop.OpARM64SUBconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, v.Type)
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
		if p.Op != ssaop.OpARM64SUBconst {
			break
		}
		c := ssa.AuxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != ssaop.OpARM64MUL || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, v.Type)
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
		if p.Op != ssaop.OpARM64SUBconst {
			break
		}
		c := ssa.AuxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != ssaop.OpARM64MULW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, v.Type)
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
		if p.Op != ssaop.OpARM64SUBconst {
			break
		}
		c := ssa.AuxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != ssaop.OpARM64MNEG || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, v.Type)
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
		if p.Op != ssaop.OpARM64SUBconst {
			break
		}
		c := ssa.AuxIntToInt64(p.AuxInt)
		m := p.Args[0]
		if m.Op != ssaop.OpARM64MNEGW || !(p.Uses == 1 && m.Uses == 1 && !t.IsPtrShaped()) {
			break
		}
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, v.Type)
		v0.AddArg2(a, m)
		v.AddArg(v0)
		return true
	}
	// match: (SUB x (NEG y))
	// result: (ADD x y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64NEG {
			break
		}
		y := v_1.Args[0]
		v.Reset(ssaop.OpARM64ADD)
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
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (SUB x (SUB y z))
	// result: (SUB (ADD <v.Type> x z) y)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64SUB {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARM64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, v.Type)
		v0.AddArg2(x, z)
		v.AddArg2(v0, y)
		return true
	}
	// match: (SUB (SUB x y) z)
	// result: (SUB x (ADD <y.Type> y z))
	for {
		if v_0.Op != ssaop.OpARM64SUB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		v.Reset(ssaop.OpARM64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADD, y.Type)
		v0.AddArg2(y, z)
		v.AddArg2(x, v0)
		return true
	}
	// match: (SUB x0 x1:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (SUBshiftLL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SLLconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftLL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (SUB x0 x1:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (SUBshiftRL x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SRLconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftRL)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	// match: (SUB x0 x1:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (SUBshiftRA x0 y [c])
	for {
		x0 := v_0
		x1 := v_1
		if x1.Op != ssaop.OpARM64SRAconst {
			break
		}
		c := ssa.AuxIntToInt64(x1.AuxInt)
		y := x1.Args[0]
		if !(ssa.ClobberIfDead(x1)) {
			break
		}
		v.Reset(ssaop.OpARM64SUBshiftRA)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v.AddArg2(x0, y)
		return true
	}
	return false
}
func rewriteValue_OpARM64SUBconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SUBconst [0] x)
	// result: x
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (SUBconst [c] (MOVDconst [d]))
	// result: (MOVDconst [d-c])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(d - c)
		return true
	}
	// match: (SUBconst [c] (SUBconst [d] x))
	// result: (ADDconst [-c-d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SUBconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c - d)
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] (ADDconst [d] x))
	// result: (ADDconst [-c+d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ADDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64ADDconst)
		v.AuxInt = ssa.Int64ToAuxInt(-c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64SUBshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBshiftLL x (MOVDconst [c]) [d])
	// result: (SUBconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64SUBconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftLL (SLLconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpARM64SUBshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBshiftRA x (MOVDconst [c]) [d])
	// result: (SUBconst x [c>>uint64(d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64SUBconst)
		v.AuxInt = ssa.Int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftRA (SRAconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRAconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpARM64SUBshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBshiftRL x (MOVDconst [c]) [d])
	// result: (SUBconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64SUBconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftRL (SRLconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpARM64TST(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (TST x (MOVDconst [c]))
	// result: (TSTconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpARM64TSTconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (TST x0 x1:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (TSTshiftLL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SLLconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64TSTshiftLL)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (TST x0 x1:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (TSTshiftRL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SRLconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64TSTshiftRL)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (TST x0 x1:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (TSTshiftRA x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SRAconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64TSTshiftRA)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (TST x0 x1:(RORconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (TSTshiftRO x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64RORconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64TSTshiftRO)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64TSTW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (TSTW x (MOVDconst [c]))
	// result: (TSTWconst [int32(c)] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpARM64TSTWconst)
			v.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v.AddArg(x)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64TSTWconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (TSTWconst (MOVDconst [x]) [y])
	// result: (FlagConstant [ssa.LogicFlags32(int32(x)&y)])
	for {
		y := ssa.AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.LogicFlags32(int32(x) & y))
		return true
	}
	return false
}
func rewriteValue_OpARM64TSTconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (TSTconst (MOVDconst [x]) [y])
	// result: (FlagConstant [ssa.LogicFlags64(x&y)])
	for {
		y := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		x := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64FlagConstant)
		v.AuxInt = ssa.FlagConstantToAuxInt(ssa.LogicFlags64(x & y))
		return true
	}
	return false
}
func rewriteValue_OpARM64TSTshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftLL (MOVDconst [c]) x [d])
	// result: (TSTconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64TSTconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftLL x (MOVDconst [c]) [d])
	// result: (TSTconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64TSTconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64TSTshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftRA (MOVDconst [c]) x [d])
	// result: (TSTconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64TSTconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRAconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRA x (MOVDconst [c]) [d])
	// result: (TSTconst x [c>>uint64(d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64TSTconst)
		v.AuxInt = ssa.Int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64TSTshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftRL (MOVDconst [c]) x [d])
	// result: (TSTconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64TSTconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRLconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRL x (MOVDconst [c]) [d])
	// result: (TSTconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64TSTconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64TSTshiftRO(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftRO (MOVDconst [c]) x [d])
	// result: (TSTconst [c] (RORconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64TSTconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64RORconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRO x (MOVDconst [c]) [d])
	// result: (TSTconst x [rotateRight64(c, d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64TSTconst)
		v.AuxInt = ssa.Int64ToAuxInt(rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64UBFIZ(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (UBFIZ [bfc] (SLLconst [sc] x))
	// cond: sc < bfc.Width()
	// result: (UBFIZ [ssa.ArmBFAuxInt(bfc.Lsb()+sc, bfc.Width()-sc)] x)
	for {
		bfc := ssa.AuxIntToArm64BitField(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		sc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.Width()) {
			break
		}
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(bfc.Lsb()+sc, bfc.Width()-sc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64UBFX(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (UBFX [bfc] (ANDconst [c] x))
	// cond: isARM64BFMask(0, c, 0) && bfc.Lsb() + bfc.Width() <= arm64BFWidth(c, 0)
	// result: (UBFX [bfc] x)
	for {
		bfc := ssa.AuxIntToArm64BitField(v.AuxInt)
		if v_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(isARM64BFMask(0, c, 0) && bfc.Lsb()+bfc.Width() <= arm64BFWidth(c, 0)) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(bfc)
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] e:(MOVWUreg x))
	// cond: e.Uses == 1 && bfc.Lsb() < 32
	// result: (UBFX [ssa.ArmBFAuxInt(bfc.Lsb(), min(bfc.Width(), 32-bfc.Lsb()))] x)
	for {
		bfc := ssa.AuxIntToArm64BitField(v.AuxInt)
		e := v_0
		if e.Op != ssaop.OpARM64MOVWUreg {
			break
		}
		x := e.Args[0]
		if !(e.Uses == 1 && bfc.Lsb() < 32) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(bfc.Lsb(), min(bfc.Width(), 32-bfc.Lsb())))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] e:(MOVHUreg x))
	// cond: e.Uses == 1 && bfc.Lsb() < 16
	// result: (UBFX [ssa.ArmBFAuxInt(bfc.Lsb(), min(bfc.Width(), 16-bfc.Lsb()))] x)
	for {
		bfc := ssa.AuxIntToArm64BitField(v.AuxInt)
		e := v_0
		if e.Op != ssaop.OpARM64MOVHUreg {
			break
		}
		x := e.Args[0]
		if !(e.Uses == 1 && bfc.Lsb() < 16) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(bfc.Lsb(), min(bfc.Width(), 16-bfc.Lsb())))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] e:(MOVBUreg x))
	// cond: e.Uses == 1 && bfc.Lsb() < 8
	// result: (UBFX [ssa.ArmBFAuxInt(bfc.Lsb(), min(bfc.Width(), 8-bfc.Lsb()))] x)
	for {
		bfc := ssa.AuxIntToArm64BitField(v.AuxInt)
		e := v_0
		if e.Op != ssaop.OpARM64MOVBUreg {
			break
		}
		x := e.Args[0]
		if !(e.Uses == 1 && bfc.Lsb() < 8) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(bfc.Lsb(), min(bfc.Width(), 8-bfc.Lsb())))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SRLconst [sc] x))
	// cond: sc+bfc.Width()+bfc.Lsb() < 64
	// result: (UBFX [ssa.ArmBFAuxInt(bfc.Lsb()+sc, bfc.Width())] x)
	for {
		bfc := ssa.AuxIntToArm64BitField(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRLconst {
			break
		}
		sc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc+bfc.Width()+bfc.Lsb() < 64) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(bfc.Lsb()+sc, bfc.Width()))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SLLconst [sc] x))
	// cond: sc == bfc.Lsb()
	// result: (ANDconst [1<<uint(bfc.Width())-1] x)
	for {
		bfc := ssa.AuxIntToArm64BitField(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		sc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc == bfc.Lsb()) {
			break
		}
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(1<<uint(bfc.Width()) - 1)
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SLLconst [sc] x))
	// cond: sc < bfc.Lsb()
	// result: (UBFX [ssa.ArmBFAuxInt(bfc.Lsb()-sc, bfc.Width())] x)
	for {
		bfc := ssa.AuxIntToArm64BitField(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		sc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc < bfc.Lsb()) {
			break
		}
		v.Reset(ssaop.OpARM64UBFX)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(bfc.Lsb()-sc, bfc.Width()))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SLLconst [sc] x))
	// cond: sc > bfc.Lsb() && sc < bfc.Lsb()+bfc.Width()
	// result: (UBFIZ [ssa.ArmBFAuxInt(sc-bfc.Lsb(), bfc.Lsb()+bfc.Width()-sc)] x)
	for {
		bfc := ssa.AuxIntToArm64BitField(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst {
			break
		}
		sc := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		if !(sc > bfc.Lsb() && sc < bfc.Lsb()+bfc.Width()) {
			break
		}
		v.Reset(ssaop.OpARM64UBFIZ)
		v.AuxInt = ssa.Arm64BitFieldToAuxInt(ssa.ArmBFAuxInt(sc-bfc.Lsb(), bfc.Lsb()+bfc.Width()-sc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64UDIV(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (UDIV x (MOVDconst [1]))
	// result: x
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (UDIV x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (SRLconst [ssa.Log64(c)] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARM64SRLconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
		v.AddArg(x)
		return true
	}
	// match: (UDIV (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [int64(uint64(c)/uint64(d))])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) / uint64(d)))
		return true
	}
	return false
}
func rewriteValue_OpARM64UDIVW(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (UDIVW x (MOVDconst [c]))
	// cond: uint32(c)==1
	// result: (MOVWUreg x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) == 1) {
			break
		}
		v.Reset(ssaop.OpARM64MOVWUreg)
		v.AddArg(x)
		return true
	}
	// match: (UDIVW x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c) && ssa.Is32Bit(c)
	// result: (SRLconst [ssa.Log64(c)] (MOVWUreg <v.Type> x))
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.IsPowerOfTwo(c) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64SRLconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.Log64(c))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWUreg, v.Type)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (UDIVW (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [int64(uint32(c)/uint32(d))])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint32(c) / uint32(d)))
		return true
	}
	return false
}
func rewriteValue_OpARM64UMOD(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpARM64MSUB)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64UDIV, typ.UInt64)
		v0.AddArg2(x, y)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (UMOD _ (MOVDconst [1]))
	// result: (MOVDconst [0])
	for {
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (UMOD x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c)
	// result: (ANDconst [c-1] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c - 1)
		v.AddArg(x)
		return true
	}
	// match: (UMOD (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [int64(uint64(c)%uint64(d))])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) % uint64(d)))
		return true
	}
	return false
}
func rewriteValue_OpARM64UMODW(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpARM64MSUBW)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64UDIVW, typ.UInt32)
		v0.AddArg2(x, y)
		v.AddArg3(x, y, v0)
		return true
	}
	// match: (UMODW _ (MOVDconst [c]))
	// cond: uint32(c)==1
	// result: (MOVDconst [0])
	for {
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(uint32(c) == 1) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (UMODW x (MOVDconst [c]))
	// cond: ssa.IsPowerOfTwo(c) && ssa.Is32Bit(c)
	// result: (ANDconst [c-1] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(ssa.IsPowerOfTwo(c) && ssa.Is32Bit(c)) {
			break
		}
		v.Reset(ssaop.OpARM64ANDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c - 1)
		v.AddArg(x)
		return true
	}
	// match: (UMODW (MOVDconst [c]) (MOVDconst [d]))
	// cond: d != 0
	// result: (MOVDconst [int64(uint32(c)%uint32(d))])
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint32(c) % uint32(d)))
		return true
	}
	return false
}
func rewriteValue_OpARM64VBIF16B(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VBIF16B x y (VNOT16B mask))
	// result: (VBIT16B x y mask)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARM64VNOT16B {
			break
		}
		mask := v_2.Args[0]
		v.Reset(ssaop.OpARM64VBIT16B)
		v.AddArg3(x, y, mask)
		return true
	}
	return false
}
func rewriteValue_OpARM64VBIT16B(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VBIT16B x y (VNOT16B mask))
	// result: (VBIF16B x y mask)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARM64VNOT16B {
			break
		}
		mask := v_2.Args[0]
		v.Reset(ssaop.OpARM64VBIF16B)
		v.AddArg3(x, y, mask)
		return true
	}
	return false
}
func rewriteValue_OpARM64VDUPBbcast(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VDUPBbcast [i] (VMOVBins [j] _ (MOVDconst [c])))
	// cond: i == j && c>=-128 && c<=255
	// result: (VMOVI16B [uint8(c)])
	for {
		i := ssa.AuxIntToUint8(v.AuxInt)
		if v_0.Op != ssaop.OpARM64VMOVBins {
			break
		}
		j := ssa.AuxIntToUint8(v_0.AuxInt)
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0_1.AuxInt)
		if !(i == j && c >= -128 && c <= 255) {
			break
		}
		v.Reset(ssaop.OpARM64VMOVI16B)
		v.AuxInt = ssa.Uint8ToAuxInt(uint8(c))
		return true
	}
	return false
}
func rewriteValue_OpARM64VEOR16B(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VEOR16B x x)
	// result: (VMOVI16B [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64VMOVI16B)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpARM64VFCVTL4S(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VFCVTL4S (VDUPDextr [1] x))
	// result: (VFCVTL2_4S x)
	for {
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VFCVTL2_4S)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VMOVDins0(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VFCVTN2D y)))
	// result: (VFCVTN2_2D dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VFCVTN2D {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VFCVTN2_2D)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VSHRN2D [c] y)))
	// result: (VSHRN2_2D dst [c] y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VSHRN2D {
			break
		}
		c := ssa.AuxIntToUint8(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VSHRN2_2D)
		v.AuxInt = ssa.Uint8ToAuxInt(c)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VSHRN4S [c] y)))
	// result: (VSHRN2_4S dst [c] y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VSHRN4S {
			break
		}
		c := ssa.AuxIntToUint8(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VSHRN2_4S)
		v.AuxInt = ssa.Uint8ToAuxInt(c)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VSHRN8H [c] y)))
	// result: (VSHRN2_8H dst [c] y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VSHRN8H {
			break
		}
		c := ssa.AuxIntToUint8(v_1_0.AuxInt)
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VSHRN2_8H)
		v.AuxInt = ssa.Uint8ToAuxInt(c)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VSQXTN2D y)))
	// result: (VSQXTN2_2D dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VSQXTN2D {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VSQXTN2_2D)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VSQXTN4S y)))
	// result: (VSQXTN2_4S dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VSQXTN4S {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VSQXTN2_4S)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VSQXTN8H y)))
	// result: (VSQXTN2_8H dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VSQXTN8H {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VSQXTN2_8H)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VSQXTUN2D y)))
	// result: (VSQXTUN2_2D dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VSQXTUN2D {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VSQXTUN2_2D)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VSQXTUN4S y)))
	// result: (VSQXTUN2_4S dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VSQXTUN4S {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VSQXTUN2_4S)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VSQXTUN8H y)))
	// result: (VSQXTUN2_8H dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VSQXTUN8H {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VSQXTUN2_8H)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VUQXTN2D y)))
	// result: (VUQXTN2_2D dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VUQXTN2D {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VUQXTN2_2D)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VUQXTN4S y)))
	// result: (VUQXTN2_4S dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VUQXTN4S {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VUQXTN2_4S)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VUQXTN8H y)))
	// result: (VUQXTN2_8H dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VUQXTN8H {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VUQXTN2_8H)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VXTN2D y)))
	// result: (VXTN2_2D dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VXTN2D {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VXTN2_2D)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VXTN4S y)))
	// result: (VXTN2_4S dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VXTN4S {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VXTN2_4S)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [1] dst (VDUPDextr [0] (VXTN8H y)))
	// result: (VXTN2_8H dst y)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 0 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != ssaop.OpARM64VXTN8H {
			break
		}
		y := v_1_0.Args[0]
		v.Reset(ssaop.OpARM64VXTN2_8H)
		v.AddArg2(dst, y)
		return true
	}
	// match: (VMOVDins0 [0] (VMOVI16B [0]) y:(VDUPDextr [i] _))
	// result: y
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 0 || v_0.Op != ssaop.OpARM64VMOVI16B || ssa.AuxIntToUint8(v_0.AuxInt) != 0 {
			break
		}
		y := v_1
		if y.Op != ssaop.OpARM64VDUPDextr {
			break
		}
		v.CopyOf(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64VMOVSins0(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VMOVSins0 [0] (VMOVI16B [0]) y:(VDUPSextr [i] _))
	// result: y
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 0 || v_0.Op != ssaop.OpARM64VMOVI16B || ssa.AuxIntToUint8(v_0.AuxInt) != 0 {
			break
		}
		y := v_1
		if y.Op != ssaop.OpARM64VDUPSextr {
			break
		}
		v.CopyOf(y)
		return true
	}
	return false
}
func rewriteValue_OpARM64VNOT16B(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VNOT16B (VCMEQ16B (VAND16B x y) (VMOVI16B [0])))
	// result: (VCMTST16B x y)
	for {
		if v_0.Op != ssaop.OpARM64VCMEQ16B {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpARM64VAND16B {
				continue
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			if v_0_1.Op != ssaop.OpARM64VMOVI16B || ssa.AuxIntToUint8(v_0_1.AuxInt) != 0 {
				continue
			}
			v.Reset(ssaop.OpARM64VCMTST16B)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (VNOT16B (VCMEQ8H (VAND16B x y) (VMOVI16B [0])))
	// result: (VCMTST8H x y)
	for {
		if v_0.Op != ssaop.OpARM64VCMEQ8H {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpARM64VAND16B {
				continue
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			if v_0_1.Op != ssaop.OpARM64VMOVI16B || ssa.AuxIntToUint8(v_0_1.AuxInt) != 0 {
				continue
			}
			v.Reset(ssaop.OpARM64VCMTST8H)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (VNOT16B (VCMEQ4S (VAND16B x y) (VMOVI16B [0])))
	// result: (VCMTST4S x y)
	for {
		if v_0.Op != ssaop.OpARM64VCMEQ4S {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpARM64VAND16B {
				continue
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			if v_0_1.Op != ssaop.OpARM64VMOVI16B || ssa.AuxIntToUint8(v_0_1.AuxInt) != 0 {
				continue
			}
			v.Reset(ssaop.OpARM64VCMTST4S)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (VNOT16B (VCMEQ2D (VAND16B x y) (VMOVI16B [0])))
	// result: (VCMTST2D x y)
	for {
		if v_0.Op != ssaop.OpARM64VCMEQ2D {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			if v_0_0.Op != ssaop.OpARM64VAND16B {
				continue
			}
			y := v_0_0.Args[1]
			x := v_0_0.Args[0]
			if v_0_1.Op != ssaop.OpARM64VMOVI16B || ssa.AuxIntToUint8(v_0_1.AuxInt) != 0 {
				continue
			}
			v.Reset(ssaop.OpARM64VCMTST2D)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64VPMULL2D(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VPMULL2D (VDUPDextr [1] x) (VDUPDextr [1] y))
	// result: (VPMULL2_2D x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARM64VPMULL2_2D)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64VSHL16B(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSHL16B [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSHL2D(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSHL2D [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSHL4S(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSHL4S [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSHL8H(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSHL8H [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSHRN2D(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSHRN2D [0] x)
	// result: (VXTN2D x)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.Reset(ssaop.OpARM64VXTN2D)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSHRN4S(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSHRN4S [0] x)
	// result: (VXTN4S x)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.Reset(ssaop.OpARM64VXTN4S)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSHRN8H(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSHRN8H [0] x)
	// result: (VXTN8H x)
	for {
		if ssa.AuxIntToUint8(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.Reset(ssaop.OpARM64VXTN8H)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSMULL16B(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VSMULL16B (VDUPDextr [1] x) (VDUPDextr [1] y))
	// result: (VSMULL2_16B x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARM64VSMULL2_16B)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64VSMULL4S(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VSMULL4S (VDUPDextr [1] x) (VDUPDextr [1] y))
	// result: (VSMULL2_4S x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARM64VSMULL2_4S)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64VSMULL8H(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VSMULL8H (VDUPDextr [1] x) (VDUPDextr [1] y))
	// result: (VSMULL2_8H x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARM64VSMULL2_8H)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64VSQSHL16Bconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSQSHL16Bconst [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSQSHL2Dconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSQSHL2Dconst [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSQSHL4Sconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSQSHL4Sconst [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSQSHL8Hconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSQSHL8Hconst [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSSHLL16B(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSSHLL16B [a] (VDUPDextr [1] x))
	// result: (VSSHLL2_16B [a] x)
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VSSHLL2_16B)
		v.AuxInt = ssa.Uint8ToAuxInt(a)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSSHLL4S(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSSHLL4S [a] (VDUPDextr [1] x))
	// result: (VSSHLL2_4S [a] x)
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VSSHLL2_4S)
		v.AuxInt = ssa.Uint8ToAuxInt(a)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSSHLL8H(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSSHLL8H [a] (VDUPDextr [1] x))
	// result: (VSSHLL2_8H [a] x)
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VSSHLL2_8H)
		v.AuxInt = ssa.Uint8ToAuxInt(a)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSSHR16B(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSSHR16B [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSSHR2D(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSSHR2D [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSSHR4S(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSSHR4S [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSSHR8H(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSSHR8H [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSXTL16B(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSXTL16B (VDUPDextr [1] x))
	// result: (VSXTL2_16B x)
	for {
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VSXTL2_16B)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSXTL4S(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSXTL4S (VDUPDextr [1] x))
	// result: (VSXTL2_4S x)
	for {
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VSXTL2_4S)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VSXTL8H(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VSXTL8H (VDUPDextr [1] x))
	// result: (VSXTL2_8H x)
	for {
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VSXTL2_8H)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUMULL16B(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VUMULL16B (VDUPDextr [1] x) (VDUPDextr [1] y))
	// result: (VUMULL2_16B x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARM64VUMULL2_16B)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64VUMULL4S(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VUMULL4S (VDUPDextr [1] x) (VDUPDextr [1] y))
	// result: (VUMULL2_4S x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARM64VUMULL2_4S)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64VUMULL8H(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (VUMULL8H (VDUPDextr [1] x) (VDUPDextr [1] y))
	// result: (VUMULL2_8H x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
				continue
			}
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_1.AuxInt) != 1 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARM64VUMULL2_8H)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64VUQSHL16Bconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUQSHL16Bconst [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUQSHL2Dconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUQSHL2Dconst [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUQSHL4Sconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUQSHL4Sconst [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUQSHL8Hconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUQSHL8Hconst [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUSHLL16B(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUSHLL16B [a] (VDUPDextr [1] x))
	// result: (VUSHLL2_16B [a] x)
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VUSHLL2_16B)
		v.AuxInt = ssa.Uint8ToAuxInt(a)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUSHLL4S(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUSHLL4S [a] (VDUPDextr [1] x))
	// result: (VUSHLL2_4S [a] x)
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VUSHLL2_4S)
		v.AuxInt = ssa.Uint8ToAuxInt(a)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUSHLL8H(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUSHLL8H [a] (VDUPDextr [1] x))
	// result: (VUSHLL2_8H [a] x)
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VUSHLL2_8H)
		v.AuxInt = ssa.Uint8ToAuxInt(a)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUSHR16B(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUSHR16B [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUSHR2D(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUSHR2D [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUSHR4S(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUSHR4S [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUSHR8H(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUSHR8H [a] x)
	// cond: a==0
	// result: x
	for {
		a := ssa.AuxIntToUint8(v.AuxInt)
		x := v_0
		if !(a == 0) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUXTL16B(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUXTL16B (VDUPDextr [1] x))
	// result: (VUXTL2_16B x)
	for {
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VUXTL2_16B)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUXTL4S(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUXTL4S (VDUPDextr [1] x))
	// result: (VUXTL2_4S x)
	for {
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VUXTL2_4S)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64VUXTL8H(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (VUXTL8H (VDUPDextr [1] x))
	// result: (VUXTL2_8H x)
	for {
		if v_0.Op != ssaop.OpARM64VDUPDextr || ssa.AuxIntToUint8(v_0.AuxInt) != 1 {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64VUXTL2_8H)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64XOR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XOR x (MOVDconst [c]))
	// result: (XORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MOVDconst {
				continue
			}
			c := ssa.AuxIntToInt64(v_1.AuxInt)
			v.Reset(ssaop.OpARM64XORconst)
			v.AuxInt = ssa.Int64ToAuxInt(c)
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
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (XOR x (MVN y))
	// result: (EON x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARM64MVN {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARM64EON)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (XOR x0 x1:(SLLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (XORshiftLL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SLLconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64XORshiftLL)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (XOR x0 x1:(SRLconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (XORshiftRL x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SRLconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64XORshiftRL)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (XOR x0 x1:(SRAconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (XORshiftRA x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64SRAconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64XORshiftRA)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	// match: (XOR x0 x1:(RORconst [c] y))
	// cond: ssa.ClobberIfDead(x1)
	// result: (XORshiftRO x0 y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x0 := v_0
			x1 := v_1
			if x1.Op != ssaop.OpARM64RORconst {
				continue
			}
			c := ssa.AuxIntToInt64(x1.AuxInt)
			y := x1.Args[0]
			if !(ssa.ClobberIfDead(x1)) {
				continue
			}
			v.Reset(ssaop.OpARM64XORshiftRO)
			v.AuxInt = ssa.Int64ToAuxInt(c)
			v.AddArg2(x0, y)
			return true
		}
		break
	}
	return false
}
func rewriteValue_OpARM64XORconst(v *ssa.Value) bool {
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
	// result: (MVN x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != -1 {
			break
		}
		x := v_0
		v.Reset(ssaop.OpARM64MVN)
		v.AddArg(x)
		return true
	}
	// match: (XORconst [c] (MOVDconst [d]))
	// result: (MOVDconst [c^d])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(c ^ d)
		return true
	}
	// match: (XORconst [c] (XORconst [d] x))
	// result: (XORconst [c^d] x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64XORconst {
			break
		}
		d := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c ^ d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValue_OpARM64XORshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (XORshiftLL (MOVDconst [c]) x [d])
	// result: (XORconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLLconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftLL x (MOVDconst [c]) [d])
	// result: (XORconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL (SLLconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SLLconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	// match: (XORshiftLL <typ.UInt16> [8] (UBFX <typ.UInt16> [ssa.ArmBFAuxInt(8, 8)] x) x)
	// result: (REV16W x)
	for {
		if v.Type != typ.UInt16 || ssa.AuxIntToInt64(v.AuxInt) != 8 || v_0.Op != ssaop.OpARM64UBFX || v_0.Type != typ.UInt16 || ssa.AuxIntToArm64BitField(v_0.AuxInt) != ssa.ArmBFAuxInt(8, 8) {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL [8] (UBFX [ssa.ArmBFAuxInt(8, 24)] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff
	// result: (REV16W x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 || v_0.Op != ssaop.OpARM64UBFX || ssa.AuxIntToArm64BitField(v_0.AuxInt) != ssa.ArmBFAuxInt(8, 24) {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c1 := ssa.AuxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != ssaop.OpARM64ANDconst {
			break
		}
		c2 := ssa.AuxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint32(c1) == 0xff00ff00 && uint32(c2) == 0x00ff00ff) {
			break
		}
		v.Reset(ssaop.OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL [8] (SRLconst [8] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: (uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff)
	// result: (REV16 x)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 || v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c1 := ssa.AuxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != ssaop.OpARM64ANDconst {
			break
		}
		c2 := ssa.AuxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint64(c1) == 0xff00ff00ff00ff00 && uint64(c2) == 0x00ff00ff00ff00ff) {
			break
		}
		v.Reset(ssaop.OpARM64REV16)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL [8] (SRLconst [8] (ANDconst [c1] x)) (ANDconst [c2] x))
	// cond: (uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff)
	// result: (REV16 (ANDconst <x.Type> [0xffffffff] x))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 || v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARM64ANDconst {
			break
		}
		c1 := ssa.AuxIntToInt64(v_0_0.AuxInt)
		x := v_0_0.Args[0]
		if v_1.Op != ssaop.OpARM64ANDconst {
			break
		}
		c2 := ssa.AuxIntToInt64(v_1.AuxInt)
		if x != v_1.Args[0] || !(uint64(c1) == 0xff00ff00 && uint64(c2) == 0x00ff00ff) {
			break
		}
		v.Reset(ssaop.OpARM64REV16)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ANDconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(0xffffffff)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftLL [c] (SRLconst x [64-c]) x2)
	// result: (EXTRconst [64-c] x2 x)
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != 64-c {
			break
		}
		x := v_0.Args[0]
		x2 := v_1
		v.Reset(ssaop.OpARM64EXTRconst)
		v.AuxInt = ssa.Int64ToAuxInt(64 - c)
		v.AddArg2(x2, x)
		return true
	}
	// match: (XORshiftLL <t> [c] (UBFX [bfc] x) x2)
	// cond: c < 32 && t.Size() == 4 && bfc == ssa.ArmBFAuxInt(32-c, c)
	// result: (EXTRWconst [32-c] x2 x)
	for {
		t := v.Type
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64UBFX {
			break
		}
		bfc := ssa.AuxIntToArm64BitField(v_0.AuxInt)
		x := v_0.Args[0]
		x2 := v_1
		if !(c < 32 && t.Size() == 4 && bfc == ssa.ArmBFAuxInt(32-c, c)) {
			break
		}
		v.Reset(ssaop.OpARM64EXTRWconst)
		v.AuxInt = ssa.Int64ToAuxInt(32 - c)
		v.AddArg2(x2, x)
		return true
	}
	return false
}
func rewriteValue_OpARM64XORshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XORshiftRA (MOVDconst [c]) x [d])
	// result: (XORconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRAconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRA x (MOVDconst [c]) [d])
	// result: (XORconst x [c>>uint64(d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRA (SRAconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRAconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpARM64XORshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XORshiftRL (MOVDconst [c]) x [d])
	// result: (XORconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRLconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRL x (MOVDconst [c]) [d])
	// result: (XORconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(uint64(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRL (SRLconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64SRLconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpARM64XORshiftRO(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XORshiftRO (MOVDconst [c]) x [d])
	// result: (XORconst [c] (RORconst <x.Type> x [d]))
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64RORconst, x.Type)
		v0.AuxInt = ssa.Int64ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRO x (MOVDconst [c]) [d])
	// result: (XORconst x [rotateRight64(c, d)])
	for {
		d := ssa.AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpARM64XORconst)
		v.AuxInt = ssa.Int64ToAuxInt(rotateRight64(c, d))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRO (RORconst x [c]) x [c])
	// result: (MOVDconst [0])
	for {
		c := ssa.AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARM64RORconst || ssa.AuxIntToInt64(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValue_OpARM64ZSELB(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ZSELB (ZABSB x (Select0 <types.TypeMask> (PWHILELTB (MOVDconst [0]) (MOVDconst [32])))) z mask)
	// result: (ZABSMergingB z x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZABSB {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpSelect0 || v_0_1.Type != types.TypeMask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != ssaop.OpARM64PWHILELTB {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_0.AuxInt) != 0 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_1.AuxInt) != 32 {
			break
		}
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZABSMergingB)
		v.AddArg3(z, x, mask)
		return true
	}
	// match: (ZSELB (ZADDB x y) x mask)
	// result: (ZADDMergingB x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZADDB {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZADDMergingB)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELB (ZADDB x y) y mask)
	// result: (ZADDMergingB y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZADDB {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZADDMergingB)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELB (ZADDB x y) z mask)
	// result: (ZADDMergingPrefixedB z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZADDB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZADDMergingPrefixedB)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELB (ZNEGB x (Select0 <types.TypeMask> (PWHILELTB (MOVDconst [0]) (MOVDconst [32])))) z mask)
	// result: (ZNEGMergingB z x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZNEGB {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpSelect0 || v_0_1.Type != types.TypeMask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != ssaop.OpARM64PWHILELTB {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_0.AuxInt) != 0 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_1.AuxInt) != 32 {
			break
		}
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZNEGMergingB)
		v.AddArg3(z, x, mask)
		return true
	}
	// match: (ZSELB (ZSQADDB x y) x mask)
	// result: (ZSQADDMergingB x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQADDB {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZSQADDMergingB)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELB (ZSQADDB x y) y mask)
	// result: (ZSQADDMergingB y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQADDB {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZSQADDMergingB)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELB (ZSQADDB x y) z mask)
	// result: (ZSQADDMergingPrefixedB z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQADDB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZSQADDMergingPrefixedB)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELB (ZSQSUBB x y) x mask)
	// result: (ZSQSUBMergingB x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQSUBB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZSQSUBMergingB)
		v.AddArg3(x, y, mask)
		return true
	}
	// match: (ZSELB (ZSUBB x y) x mask)
	// result: (ZSUBMergingB x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSUBB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZSUBMergingB)
		v.AddArg3(x, y, mask)
		return true
	}
	// match: (ZSELB (ZUQADDB x y) x mask)
	// result: (ZUQADDMergingB x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQADDB {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZUQADDMergingB)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELB (ZUQADDB x y) y mask)
	// result: (ZUQADDMergingB y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQADDB {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZUQADDMergingB)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELB (ZUQADDB x y) z mask)
	// result: (ZUQADDMergingPrefixedB z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQADDB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZUQADDMergingPrefixedB)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELB (ZUQSUBB x y) x mask)
	// result: (ZUQSUBMergingB x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQSUBB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZUQSUBMergingB)
		v.AddArg3(x, y, mask)
		return true
	}
	return false
}
func rewriteValue_OpARM64ZSELD(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ZSELD (ZABSD x (Select0 <types.TypeMask> (PWHILELTD (MOVDconst [0]) (MOVDconst [4])))) z mask)
	// result: (ZABSMergingD z x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZABSD {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpSelect0 || v_0_1.Type != types.TypeMask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != ssaop.OpARM64PWHILELTD {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_0.AuxInt) != 0 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_1.AuxInt) != 4 {
			break
		}
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZABSMergingD)
		v.AddArg3(z, x, mask)
		return true
	}
	// match: (ZSELD (ZADDD x y) x mask)
	// result: (ZADDMergingD x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZADDD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZADDMergingD)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELD (ZADDD x y) y mask)
	// result: (ZADDMergingD y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZADDD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZADDMergingD)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELD (ZADDD x y) z mask)
	// result: (ZADDMergingPrefixedD z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZADDD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZADDMergingPrefixedD)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELD (ZFADDD x y) x mask)
	// result: (ZFADDMergingD x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZFADDD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZFADDMergingD)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELD (ZFADDD x y) y mask)
	// result: (ZFADDMergingD y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZFADDD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZFADDMergingD)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELD (ZFADDD x y) z mask)
	// result: (ZFADDMergingPrefixedD z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZFADDD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZFADDMergingPrefixedD)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELD (ZFNEGD x (Select0 <types.TypeMask> (PWHILELTD (MOVDconst [0]) (MOVDconst [4])))) z mask)
	// result: (ZFNEGMergingD z x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZFNEGD {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpSelect0 || v_0_1.Type != types.TypeMask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != ssaop.OpARM64PWHILELTD {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_0.AuxInt) != 0 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_1.AuxInt) != 4 {
			break
		}
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZFNEGMergingD)
		v.AddArg3(z, x, mask)
		return true
	}
	// match: (ZSELD (ZFSUBD x y) x mask)
	// result: (ZFSUBMergingD x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZFSUBD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZFSUBMergingD)
		v.AddArg3(x, y, mask)
		return true
	}
	// match: (ZSELD (ZNEGD x (Select0 <types.TypeMask> (PWHILELTD (MOVDconst [0]) (MOVDconst [4])))) z mask)
	// result: (ZNEGMergingD z x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZNEGD {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpSelect0 || v_0_1.Type != types.TypeMask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != ssaop.OpARM64PWHILELTD {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_0.AuxInt) != 0 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_1.AuxInt) != 4 {
			break
		}
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZNEGMergingD)
		v.AddArg3(z, x, mask)
		return true
	}
	// match: (ZSELD (ZSQADDD x y) x mask)
	// result: (ZSQADDMergingD x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQADDD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZSQADDMergingD)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELD (ZSQADDD x y) y mask)
	// result: (ZSQADDMergingD y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQADDD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZSQADDMergingD)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELD (ZSQADDD x y) z mask)
	// result: (ZSQADDMergingPrefixedD z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQADDD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZSQADDMergingPrefixedD)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELD (ZSQSUBD x y) x mask)
	// result: (ZSQSUBMergingD x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQSUBD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZSQSUBMergingD)
		v.AddArg3(x, y, mask)
		return true
	}
	// match: (ZSELD (ZSUBD x y) x mask)
	// result: (ZSUBMergingD x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSUBD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZSUBMergingD)
		v.AddArg3(x, y, mask)
		return true
	}
	// match: (ZSELD (ZUQADDD x y) x mask)
	// result: (ZUQADDMergingD x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQADDD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZUQADDMergingD)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELD (ZUQADDD x y) y mask)
	// result: (ZUQADDMergingD y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQADDD {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZUQADDMergingD)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELD (ZUQADDD x y) z mask)
	// result: (ZUQADDMergingPrefixedD z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQADDD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZUQADDMergingPrefixedD)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELD (ZUQSUBD x y) x mask)
	// result: (ZUQSUBMergingD x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQSUBD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZUQSUBMergingD)
		v.AddArg3(x, y, mask)
		return true
	}
	return false
}
func rewriteValue_OpARM64ZSELH(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ZSELH (ZABSH x (Select0 <types.TypeMask> (PWHILELTH (MOVDconst [0]) (MOVDconst [16])))) z mask)
	// result: (ZABSMergingH z x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZABSH {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpSelect0 || v_0_1.Type != types.TypeMask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != ssaop.OpARM64PWHILELTH {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_0.AuxInt) != 0 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_1.AuxInt) != 16 {
			break
		}
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZABSMergingH)
		v.AddArg3(z, x, mask)
		return true
	}
	// match: (ZSELH (ZADDH x y) x mask)
	// result: (ZADDMergingH x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZADDH {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZADDMergingH)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELH (ZADDH x y) y mask)
	// result: (ZADDMergingH y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZADDH {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZADDMergingH)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELH (ZADDH x y) z mask)
	// result: (ZADDMergingPrefixedH z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZADDH {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZADDMergingPrefixedH)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELH (ZNEGH x (Select0 <types.TypeMask> (PWHILELTH (MOVDconst [0]) (MOVDconst [16])))) z mask)
	// result: (ZNEGMergingH z x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZNEGH {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpSelect0 || v_0_1.Type != types.TypeMask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != ssaop.OpARM64PWHILELTH {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_0.AuxInt) != 0 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_1.AuxInt) != 16 {
			break
		}
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZNEGMergingH)
		v.AddArg3(z, x, mask)
		return true
	}
	// match: (ZSELH (ZSQADDH x y) x mask)
	// result: (ZSQADDMergingH x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQADDH {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZSQADDMergingH)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELH (ZSQADDH x y) y mask)
	// result: (ZSQADDMergingH y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQADDH {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZSQADDMergingH)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELH (ZSQADDH x y) z mask)
	// result: (ZSQADDMergingPrefixedH z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQADDH {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZSQADDMergingPrefixedH)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELH (ZSQSUBH x y) x mask)
	// result: (ZSQSUBMergingH x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQSUBH {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZSQSUBMergingH)
		v.AddArg3(x, y, mask)
		return true
	}
	// match: (ZSELH (ZSUBH x y) x mask)
	// result: (ZSUBMergingH x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSUBH {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZSUBMergingH)
		v.AddArg3(x, y, mask)
		return true
	}
	// match: (ZSELH (ZUQADDH x y) x mask)
	// result: (ZUQADDMergingH x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQADDH {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZUQADDMergingH)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELH (ZUQADDH x y) y mask)
	// result: (ZUQADDMergingH y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQADDH {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZUQADDMergingH)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELH (ZUQADDH x y) z mask)
	// result: (ZUQADDMergingPrefixedH z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQADDH {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZUQADDMergingPrefixedH)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELH (ZUQSUBH x y) x mask)
	// result: (ZUQSUBMergingH x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQSUBH {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZUQSUBMergingH)
		v.AddArg3(x, y, mask)
		return true
	}
	return false
}
func rewriteValue_OpARM64ZSELS(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ZSELS (ZABSS x (Select0 <types.TypeMask> (PWHILELTS (MOVDconst [0]) (MOVDconst [8])))) z mask)
	// result: (ZABSMergingS z x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZABSS {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpSelect0 || v_0_1.Type != types.TypeMask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != ssaop.OpARM64PWHILELTS {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_0.AuxInt) != 0 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_1.AuxInt) != 8 {
			break
		}
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZABSMergingS)
		v.AddArg3(z, x, mask)
		return true
	}
	// match: (ZSELS (ZADDS x y) x mask)
	// result: (ZADDMergingS x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZADDS {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZADDMergingS)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELS (ZADDS x y) y mask)
	// result: (ZADDMergingS y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZADDS {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZADDMergingS)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELS (ZADDS x y) z mask)
	// result: (ZADDMergingPrefixedS z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZADDS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZADDMergingPrefixedS)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELS (ZFADDS x y) x mask)
	// result: (ZFADDMergingS x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZFADDS {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZFADDMergingS)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELS (ZFADDS x y) y mask)
	// result: (ZFADDMergingS y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZFADDS {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZFADDMergingS)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELS (ZFADDS x y) z mask)
	// result: (ZFADDMergingPrefixedS z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZFADDS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZFADDMergingPrefixedS)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELS (ZFNEGS x (Select0 <types.TypeMask> (PWHILELTS (MOVDconst [0]) (MOVDconst [8])))) z mask)
	// result: (ZFNEGMergingS z x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZFNEGS {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpSelect0 || v_0_1.Type != types.TypeMask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != ssaop.OpARM64PWHILELTS {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_0.AuxInt) != 0 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_1.AuxInt) != 8 {
			break
		}
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZFNEGMergingS)
		v.AddArg3(z, x, mask)
		return true
	}
	// match: (ZSELS (ZFSUBS x y) x mask)
	// result: (ZFSUBMergingS x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZFSUBS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZFSUBMergingS)
		v.AddArg3(x, y, mask)
		return true
	}
	// match: (ZSELS (ZNEGS x (Select0 <types.TypeMask> (PWHILELTS (MOVDconst [0]) (MOVDconst [8])))) z mask)
	// result: (ZNEGMergingS z x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZNEGS {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpSelect0 || v_0_1.Type != types.TypeMask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != ssaop.OpARM64PWHILELTS {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_0.AuxInt) != 0 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0_1_0_1.AuxInt) != 8 {
			break
		}
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZNEGMergingS)
		v.AddArg3(z, x, mask)
		return true
	}
	// match: (ZSELS (ZSQADDS x y) x mask)
	// result: (ZSQADDMergingS x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQADDS {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZSQADDMergingS)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELS (ZSQADDS x y) y mask)
	// result: (ZSQADDMergingS y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQADDS {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZSQADDMergingS)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELS (ZSQADDS x y) z mask)
	// result: (ZSQADDMergingPrefixedS z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQADDS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZSQADDMergingPrefixedS)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELS (ZSQSUBS x y) x mask)
	// result: (ZSQSUBMergingS x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSQSUBS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZSQSUBMergingS)
		v.AddArg3(x, y, mask)
		return true
	}
	// match: (ZSELS (ZSUBS x y) x mask)
	// result: (ZSUBMergingS x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZSUBS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZSUBMergingS)
		v.AddArg3(x, y, mask)
		return true
	}
	// match: (ZSELS (ZUQADDS x y) x mask)
	// result: (ZUQADDMergingS x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQADDS {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if x != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZUQADDMergingS)
			v.AddArg3(x, y, mask)
			return true
		}
		break
	}
	// match: (ZSELS (ZUQADDS x y) y mask)
	// result: (ZUQADDMergingS y x mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQADDS {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
			x := v_0_0
			y := v_0_1
			if y != v_1 {
				continue
			}
			mask := v_2
			v.Reset(ssaop.OpARM64ZUQADDMergingS)
			v.AddArg3(y, x, mask)
			return true
		}
		break
	}
	// match: (ZSELS (ZUQADDS x y) z mask)
	// result: (ZUQADDMergingPrefixedS z x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQADDS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		z := v_1
		mask := v_2
		v.Reset(ssaop.OpARM64ZUQADDMergingPrefixedS)
		v.AddArg4(z, x, y, mask)
		return true
	}
	// match: (ZSELS (ZUQSUBS x y) x mask)
	// result: (ZUQSUBMergingS x y mask)
	for {
		if v_0.Op != ssaop.OpARM64ZUQSUBS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		mask := v_2
		v.Reset(ssaop.OpARM64ZUQSUBMergingS)
		v.AddArg3(x, y, mask)
		return true
	}
	return false
}
func rewriteValue_OpAbsInt16s(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AbsInt16s x)
	// result: (ZABSH x (Select0 <types.TypeMask> (PWHILELTH (MOVDconst [0]) (MOVDconst [16]))))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64ZABSH)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTH, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(16)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpAbsInt32s(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AbsInt32s x)
	// result: (ZABSS x (Select0 <types.TypeMask> (PWHILELTS (MOVDconst [0]) (MOVDconst [8]))))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64ZABSS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTS, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(8)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpAbsInt64s(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AbsInt64s x)
	// result: (ZABSD x (Select0 <types.TypeMask> (PWHILELTD (MOVDconst [0]) (MOVDconst [4]))))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64ZABSD)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTD, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(4)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpAbsInt8s(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AbsInt8s x)
	// result: (ZABSB x (Select0 <types.TypeMask> (PWHILELTB (MOVDconst [0]) (MOVDconst [32]))))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64ZABSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTB, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(32)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpAddr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (MOVDaddr {sym} base)
	for {
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		v.Reset(ssaop.OpARM64MOVDaddr)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValue_OpAvg64u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Avg64u <t> x y)
	// result: (ADD (SRLconst <t> (SUB <t> x y) [1]) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64ADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRLconst, t)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64SUB, t)
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
	typ := &b.Func.Config.Types
	// match: (BitLen32 x)
	// result: (SUB (MOVDconst [32]) (CLZW <typ.Int> x))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(32)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64CLZW, typ.Int)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpBitLen64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen64 x)
	// result: (SUB (MOVDconst [64]) (CLZ <typ.Int> x))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64SUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(64)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64CLZ, typ.Int)
		v1.AddArg(x)
		v.AddArg2(v0, v1)
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
	typ := &b.Func.Config.Types
	// match: (BitRev16 x)
	// result: (SRLconst [48] (RBIT <typ.UInt64> x))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64SRLconst)
		v.AuxInt = ssa.Int64ToAuxInt(48)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64RBIT, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpBitRev8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitRev8 x)
	// result: (SRLconst [56] (RBIT <typ.UInt64> x))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64SRLconst)
		v.AuxInt = ssa.Int64ToAuxInt(56)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64RBIT, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpCondSelect(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CondSelect x y boolval)
	// cond: ssa.FlagArg(boolval) != nil
	// result: (CSEL [boolval.Op] x y ssa.FlagArg(boolval))
	for {
		x := v_0
		y := v_1
		boolval := v_2
		if !(ssa.FlagArg(boolval) != nil) {
			break
		}
		v.Reset(ssaop.OpARM64CSEL)
		v.AuxInt = ssa.OpToAuxInt(boolval.Op)
		v.AddArg3(x, y, ssa.FlagArg(boolval))
		return true
	}
	// match: (CondSelect x y boolval)
	// cond: ssa.FlagArg(boolval) == nil
	// result: (CSEL [ssaop.OpARM64NotEqual] x y (TSTWconst [1] boolval))
	for {
		x := v_0
		y := v_1
		boolval := v_2
		if !(ssa.FlagArg(boolval) == nil) {
			break
		}
		v.Reset(ssaop.OpARM64CSEL)
		v.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
		v0.AuxInt = ssa.Int32ToAuxInt(1)
		v0.AddArg(boolval)
		v.AddArg3(x, y, v0)
		return true
	}
	return false
}
func rewriteValue_OpConst16(v *ssa.Value) bool {
	// match: (Const16 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := ssa.AuxIntToInt16(v.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValue_OpConst32(v *ssa.Value) bool {
	// match: (Const32 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := ssa.AuxIntToInt32(v.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValue_OpConst32F(v *ssa.Value) bool {
	// match: (Const32F [val])
	// result: (FMOVSconst [float64(val)])
	for {
		val := ssa.AuxIntToFloat32(v.AuxInt)
		v.Reset(ssaop.OpARM64FMOVSconst)
		v.AuxInt = ssa.Float64ToAuxInt(float64(val))
		return true
	}
}
func rewriteValue_OpConst64(v *ssa.Value) bool {
	// match: (Const64 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := ssa.AuxIntToInt64(v.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValue_OpConst64F(v *ssa.Value) bool {
	// match: (Const64F [val])
	// result: (FMOVDconst [float64(val)])
	for {
		val := ssa.AuxIntToFloat64(v.AuxInt)
		v.Reset(ssaop.OpARM64FMOVDconst)
		v.AuxInt = ssa.Float64ToAuxInt(float64(val))
		return true
	}
}
func rewriteValue_OpConst8(v *ssa.Value) bool {
	// match: (Const8 [val])
	// result: (MOVDconst [int64(val)])
	for {
		val := ssa.AuxIntToInt8(v.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(int64(val))
		return true
	}
}
func rewriteValue_OpConstBool(v *ssa.Value) bool {
	// match: (ConstBool [t])
	// result: (MOVDconst [ssa.B2i(t)])
	for {
		t := ssa.AuxIntToBool(v.AuxInt)
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(ssa.B2i(t))
		return true
	}
}
func rewriteValue_OpConstNil(v *ssa.Value) bool {
	// match: (ConstNil)
	// result: (MOVDconst [0])
	for {
		v.Reset(ssaop.OpARM64MOVDconst)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		return true
	}
}
func rewriteValue_OpCount8s(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Count8s r)
	// result: (Select0 <types.TypeMask> (PWHILELTB (MOVDconst [0]) r))
	for {
		r := v_0
		v.Reset(ssaop.OpSelect0)
		v.Type = types.TypeMask
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTB, types.NewTuple(typ.Mask, types.TypeFlags))
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v0.AddArg2(v1, r)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpCtz16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 <t> x)
	// result: (CLZW <t> (RBITW <typ.UInt32> (ORconst <typ.UInt32> [0x10000] x)))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpARM64CLZW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64RBITW, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ORconst, typ.UInt32)
		v1.AuxInt = ssa.Int64ToAuxInt(0x10000)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpCtz32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Ctz32 <t> x)
	// result: (CLZW (RBITW <t> x))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpARM64CLZW)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64RBITW, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpCtz64(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Ctz64 <t> x)
	// result: (CLZ (RBIT <t> x))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpARM64CLZ)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64RBIT, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpCtz8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 <t> x)
	// result: (CLZW <t> (RBITW <typ.UInt32> (ORconst <typ.UInt32> [0x100] x)))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpARM64CLZW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64RBITW, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ORconst, typ.UInt32)
		v1.AuxInt = ssa.Int64ToAuxInt(0x100)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpDiv16(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpARM64DIVW)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
	return false
}
func rewriteValue_OpDiv16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (UDIVW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64UDIVW)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpDiv32(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpARM64DIVW)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpDiv64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Div64 [false] x y)
	// result: (DIV x y)
	for {
		if ssa.AuxIntToBool(v.AuxInt) != false {
			break
		}
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64DIV)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpDiv8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (DIVW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64DIVW)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
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
	// result: (UDIVW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64UDIVW)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
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
	// result: (Equal (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32 x y)
	// result: (Equal (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32F x y)
	// result: (Equal (FCMPS x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64 x y)
	// result: (Equal (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64F x y)
	// result: (Equal (FCMPD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPD, types.TypeFlags)
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
	// result: (Equal (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpEqB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// result: (XOR (MOVDconst [1]) (XOR <typ.Bool> x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64XOR, typ.Bool)
		v1.AddArg2(x, y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpEqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (EqPtr x y)
	// result: (Equal (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64Equal)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpFMA(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMA x y z)
	// result: (FMADDD z x y)
	for {
		x := v_0
		y := v_1
		z := v_2
		v.Reset(ssaop.OpARM64FMADDD)
		v.AddArg3(z, x, y)
		return true
	}
}
func rewriteValue_OpGreaterInt16s(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterInt16s x y)
	// result: (ZCMPGTH x y (Select0 <types.TypeMask> (PWHILELTH (MOVDconst [0]) (MOVDconst [16]))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64ZCMPGTH)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTH, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(16)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValue_OpGreaterInt32s(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterInt32s x y)
	// result: (ZCMPGTS x y (Select0 <types.TypeMask> (PWHILELTS (MOVDconst [0]) (MOVDconst [8]))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64ZCMPGTS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTS, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(8)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValue_OpGreaterInt64s(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterInt64s x y)
	// result: (ZCMPGTD x y (Select0 <types.TypeMask> (PWHILELTD (MOVDconst [0]) (MOVDconst [4]))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64ZCMPGTD)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTD, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(4)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValue_OpGreaterInt8s(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (GreaterInt8s x y)
	// result: (ZCMPGTB x y (Select0 <types.TypeMask> (PWHILELTB (MOVDconst [0]) (MOVDconst [32]))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64ZCMPGTB)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTB, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(32)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValue_OpHmul32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32 x y)
	// result: (SRAconst (MULL <typ.Int64> x y) [32])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64SRAconst)
		v.AuxInt = ssa.Int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MULL, typ.Int64)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpHmul32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32u x y)
	// result: (SRAconst (UMULL <typ.UInt64> x y) [32])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64SRAconst)
		v.AuxInt = ssa.Int64ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64UMULL, typ.UInt64)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpIfElseFloat32s(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IfElseFloat32s x mask y)
	// result: (ZSELS x y mask)
	for {
		x := v_0
		mask := v_1
		y := v_2
		v.Reset(ssaop.OpARM64ZSELS)
		v.AddArg3(x, y, mask)
		return true
	}
}
func rewriteValue_OpIfElseFloat64s(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IfElseFloat64s x mask y)
	// result: (ZSELD x y mask)
	for {
		x := v_0
		mask := v_1
		y := v_2
		v.Reset(ssaop.OpARM64ZSELD)
		v.AddArg3(x, y, mask)
		return true
	}
}
func rewriteValue_OpIfElseInt16s(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IfElseInt16s x mask y)
	// result: (ZSELH x y mask)
	for {
		x := v_0
		mask := v_1
		y := v_2
		v.Reset(ssaop.OpARM64ZSELH)
		v.AddArg3(x, y, mask)
		return true
	}
}
func rewriteValue_OpIfElseInt32s(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IfElseInt32s x mask y)
	// result: (ZSELS x y mask)
	for {
		x := v_0
		mask := v_1
		y := v_2
		v.Reset(ssaop.OpARM64ZSELS)
		v.AddArg3(x, y, mask)
		return true
	}
}
func rewriteValue_OpIfElseInt64s(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IfElseInt64s x mask y)
	// result: (ZSELD x y mask)
	for {
		x := v_0
		mask := v_1
		y := v_2
		v.Reset(ssaop.OpARM64ZSELD)
		v.AddArg3(x, y, mask)
		return true
	}
}
func rewriteValue_OpIfElseInt8s(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IfElseInt8s x mask y)
	// result: (ZSELB x y mask)
	for {
		x := v_0
		mask := v_1
		y := v_2
		v.Reset(ssaop.OpARM64ZSELB)
		v.AddArg3(x, y, mask)
		return true
	}
}
func rewriteValue_OpIfElseUint16s(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IfElseUint16s x mask y)
	// result: (ZSELH x y mask)
	for {
		x := v_0
		mask := v_1
		y := v_2
		v.Reset(ssaop.OpARM64ZSELH)
		v.AddArg3(x, y, mask)
		return true
	}
}
func rewriteValue_OpIfElseUint32s(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IfElseUint32s x mask y)
	// result: (ZSELS x y mask)
	for {
		x := v_0
		mask := v_1
		y := v_2
		v.Reset(ssaop.OpARM64ZSELS)
		v.AddArg3(x, y, mask)
		return true
	}
}
func rewriteValue_OpIfElseUint64s(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IfElseUint64s x mask y)
	// result: (ZSELD x y mask)
	for {
		x := v_0
		mask := v_1
		y := v_2
		v.Reset(ssaop.OpARM64ZSELD)
		v.AddArg3(x, y, mask)
		return true
	}
}
func rewriteValue_OpIfElseUint8s(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (IfElseUint8s x mask y)
	// result: (ZSELB x y mask)
	for {
		x := v_0
		mask := v_1
		y := v_2
		v.Reset(ssaop.OpARM64ZSELB)
		v.AddArg3(x, y, mask)
		return true
	}
}
func rewriteValue_OpIsInBounds(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsInBounds idx len)
	// result: (LessThanU (CMP idx len))
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpIsNonNil(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsNonNil ptr)
	// result: (NotEqual (CMPconst [0] ptr))
	for {
		ptr := v_0
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v0.AddArg(ptr)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpIsSliceInBounds(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsSliceInBounds idx len)
	// result: (LessEqualU (CMP idx len))
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (LessEqual (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x zero:(MOVDconst [0]))
	// result: (Eq16 x zero)
	for {
		x := v_0
		zero := v_1
		if zero.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpEq16)
		v.AddArg2(x, zero)
		return true
	}
	// match: (Leq16U (MOVDconst [1]) x)
	// result: (Neq16 (MOVDconst [0]) x)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpNeq16)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq16U x y)
	// result: (LessEqualU (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32 x y)
	// result: (LessEqual (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32F x y)
	// result: (LessEqualF (FCMPS x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessEqualF)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32U x zero:(MOVDconst [0]))
	// result: (Eq32 x zero)
	for {
		x := v_0
		zero := v_1
		if zero.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpEq32)
		v.AddArg2(x, zero)
		return true
	}
	// match: (Leq32U (MOVDconst [1]) x)
	// result: (Neq32 (MOVDconst [0]) x)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpNeq32)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq32U x y)
	// result: (LessEqualU (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64 x y)
	// result: (LessEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64F x y)
	// result: (LessEqualF (FCMPD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessEqualF)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq64U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64U x zero:(MOVDconst [0]))
	// result: (Eq64 x zero)
	for {
		x := v_0
		zero := v_1
		if zero.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpEq64)
		v.AddArg2(x, zero)
		return true
	}
	// match: (Leq64U (MOVDconst [1]) x)
	// result: (Neq64 (MOVDconst [0]) x)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpNeq64)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq64U x y)
	// result: (LessEqualU (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (LessEqual (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLeq8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x zero:(MOVDconst [0]))
	// result: (Eq8 x zero)
	for {
		x := v_0
		zero := v_1
		if zero.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpEq8)
		v.AddArg2(x, zero)
		return true
	}
	// match: (Leq8U (MOVDconst [1]) x)
	// result: (Neq8 (MOVDconst [0]) x)
	for {
		if v_0.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpNeq8)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(v0, x)
		return true
	}
	// match: (Leq8U x y)
	// result: (LessEqualU (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (LessThan (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U zero:(MOVDconst [0]) x)
	// result: (Neq16 zero x)
	for {
		zero := v_0
		if zero.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpNeq16)
		v.AddArg2(zero, x)
		return true
	}
	// match: (Less16U x (MOVDconst [1]))
	// result: (Eq16 x (MOVDconst [0]))
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.Reset(ssaop.OpEq16)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less16U x y)
	// result: (LessThanU (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32 x y)
	// result: (LessThan (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32F x y)
	// result: (LessThanF (FCMPS x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessThanF)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32U zero:(MOVDconst [0]) x)
	// result: (Neq32 zero x)
	for {
		zero := v_0
		if zero.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpNeq32)
		v.AddArg2(zero, x)
		return true
	}
	// match: (Less32U x (MOVDconst [1]))
	// result: (Eq32 x (MOVDconst [0]))
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.Reset(ssaop.OpEq32)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less32U x y)
	// result: (LessThanU (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64 x y)
	// result: (LessThan (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64F x y)
	// result: (LessThanF (FCMPD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessThanF)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess64U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less64U zero:(MOVDconst [0]) x)
	// result: (Neq64 zero x)
	for {
		zero := v_0
		if zero.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpNeq64)
		v.AddArg2(zero, x)
		return true
	}
	// match: (Less64U x (MOVDconst [1]))
	// result: (Eq64 x (MOVDconst [0]))
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.Reset(ssaop.OpEq64)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less64U x y)
	// result: (LessThanU (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (LessThan (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpLess8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U zero:(MOVDconst [0]) x)
	// result: (Neq8 zero x)
	for {
		zero := v_0
		if zero.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(zero.AuxInt) != 0 {
			break
		}
		x := v_1
		v.Reset(ssaop.OpNeq8)
		v.AddArg2(zero, x)
		return true
	}
	// match: (Less8U x (MOVDconst [1]))
	// result: (Eq8 x (MOVDconst [0]))
	for {
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst || ssa.AuxIntToInt64(v_1.AuxInt) != 1 {
			break
		}
		v.Reset(ssaop.OpEq8)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(x, v0)
		return true
	}
	// match: (Less8U x y)
	// result: (LessThanU (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
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
		v.Reset(ssaop.OpARM64MOVBUload)
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
		v.Reset(ssaop.OpARM64MOVBload)
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
		v.Reset(ssaop.OpARM64MOVBUload)
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
		v.Reset(ssaop.OpARM64MOVHload)
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
		v.Reset(ssaop.OpARM64MOVHUload)
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
		v.Reset(ssaop.OpARM64MOVWload)
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
		v.Reset(ssaop.OpARM64MOVWUload)
		v.AddArg2(ptr, mem)
		return true
	}
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
		v.Reset(ssaop.OpARM64MOVDload)
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
		v.Reset(ssaop.OpARM64FMOVSload)
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
		v.Reset(ssaop.OpARM64FMOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 16
	// result: (FMOVQload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 16) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVQload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 32 && t.IsSIMD()
	// result: (ZLDRload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 32 && t.IsSIMD()) {
			break
		}
		v.Reset(ssaop.OpARM64ZLDRload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.Size() == 8 && t.IsSIMD()
	// result: (PLDRload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(t.Size() == 8 && t.IsSIMD()) {
			break
		}
		v.Reset(ssaop.OpARM64PLDRload)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpLoadMasked8(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoadMasked8 <t> ptr mask mem)
	// cond: t.Size() == 32
	// result: (ZLD1BPredload ptr mask mem)
	for {
		t := v.Type
		ptr := v_0
		mask := v_1
		mem := v_2
		if !(t.Size() == 32) {
			break
		}
		v.Reset(ssaop.OpARM64ZLD1BPredload)
		v.AddArg3(ptr, mask, mem)
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
	// result: (MOVDaddr {sym} (SPanchored base mem))
	for {
		t := v.Type
		sym := ssa.AuxToSym(v.Aux)
		base := v_0
		mem := v_1
		if !(t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDaddr)
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
		v.Reset(ssaop.OpARM64MOVDaddr)
		v.Aux = ssa.SymToAux(sym)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValue_OpLsh64x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x16 <t> [bounded] x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (Lsh64x32 <t> [bounded] x (ZeroExt16to32 y))
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLsh64x32)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpLsh64x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x32 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (CSEL [ssaop.OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPWconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64CSEL)
		v.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CMPWconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh64x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh64x64 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (CSEL [ssaop.OpARM64LessThanU] (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64CSEL)
		v.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValue_OpLsh64x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SLL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SLL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Lsh64x8 <t> [bounded] x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (Lsh64x32 <t> [bounded] x (ZeroExt8to32 y))
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpLsh64x32)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpMax32FSel(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Max32FSel x y)
	// result: (FCSELS [ssaop.OpARM64GreaterThanF] x y (FCMPS x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64FCSELS)
		v.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64GreaterThanF)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValue_OpMax64FSel(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Max64FSel x y)
	// result: (FCSELD [ssaop.OpARM64GreaterThanF] x y (FCMPD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64FCSELD)
		v.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64GreaterThanF)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValue_OpMin32FSel(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Min32FSel x y)
	// result: (FCSELS [ssaop.OpARM64LessThanF] x y (FCMPS x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64FCSELS)
		v.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanF)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValue_OpMin64FSel(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Min64FSel x y)
	// result: (FCSELD [ssaop.OpARM64LessThanF] x y (FCMPD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64FCSELD)
		v.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanF)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValue_OpMod16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// result: (MODW (SignExt16to32 x) (SignExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64MODW)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
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
	// result: (UMODW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64UMODW)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValue_OpMod32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod32 x y)
	// result: (MODW x y)
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64MODW)
		v.AddArg2(x, y)
		return true
	}
}
func rewriteValue_OpMod64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Mod64 x y)
	// result: (MOD x y)
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64MOD)
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
	// result: (MODW (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64MODW)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
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
	// result: (UMODW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64UMODW)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
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
		v.Reset(ssaop.OpARM64MOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVBUload, typ.UInt8)
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
		v.Reset(ssaop.OpARM64MOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVHUload, typ.UInt16)
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
		v.Reset(ssaop.OpARM64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVBUload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVHstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVHUload, typ.UInt16)
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
		v.Reset(ssaop.OpARM64MOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWUload, typ.UInt32)
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
		v.Reset(ssaop.OpARM64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVBUload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWUload, typ.UInt32)
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
		v.Reset(ssaop.OpARM64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVHUload, typ.UInt16)
		v0.AuxInt = ssa.Int32ToAuxInt(4)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWUload, typ.UInt32)
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
		v.Reset(ssaop.OpARM64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWUload, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWUload, typ.UInt32)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [8] dst src mem)
	// result: (MOVDstore dst (MOVDload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVDstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [9] dst src mem)
	// result: (MOVBstore [8] dst (MOVBUload [8] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 9 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVBUload, typ.UInt8)
		v0.AuxInt = ssa.Int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [10] dst src mem)
	// result: (MOVHstore [8] dst (MOVHUload [8] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 10 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVHUload, typ.UInt16)
		v0.AuxInt = ssa.Int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [11] dst src mem)
	// result: (MOVDstore [3] dst (MOVDload [3] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 11 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = ssa.Int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [12] dst src mem)
	// result: (MOVWstore [8] dst (MOVWUload [8] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 12 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWUload, typ.UInt32)
		v0.AuxInt = ssa.Int32ToAuxInt(8)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [13] dst src mem)
	// result: (MOVDstore [5] dst (MOVDload [5] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 13 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(5)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = ssa.Int32ToAuxInt(5)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [14] dst src mem)
	// result: (MOVDstore [6] dst (MOVDload [6] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 14 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = ssa.Int32ToAuxInt(6)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [15] dst src mem)
	// result: (MOVDstore [7] dst (MOVDload [7] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 15 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = ssa.Int32ToAuxInt(7)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [16] dst src mem)
	// result: (FMOVQstore dst (FMOVQload src mem) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 16 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64FMOVQstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FMOVQload, typ.Vec128)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 16 && s <= 24
	// result: (MOVDstore [int32(s-8)] dst (MOVDload [int32(s-8)] src mem) (FMOVQstore dst (FMOVQload src mem) mem))
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 16 && s <= 24) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(s - 8))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(s - 8))
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64FMOVQstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64FMOVQload, typ.Vec128)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 24 && s < 32
	// result: (FMOVQstore [int32(s-16)] dst (FMOVQload [int32(s-16)] src mem) (FMOVQstore dst (FMOVQload src mem) mem))
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 24 && s < 32) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVQstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(s - 16))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FMOVQload, typ.Vec128)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(s - 16))
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64FMOVQstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64FMOVQload, typ.Vec128)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [32] dst src mem)
	// result: (FSTPQ dst (Select0 <typ.Vec128> (FLDPQ src mem)) (Select1 <typ.Vec128> (FLDPQ src mem)) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 32 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARM64FSTPQ)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, typ.Vec128)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64FLDPQ, types.NewTuple(typ.Vec128, typ.Vec128))
		v1.AddArg2(src, mem)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.Vec128)
		v2.AddArg(v1)
		v.AddArg4(dst, v0, v2, mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 32 && s <= 40
	// result: (MOVDstore [int32(s-8)] dst (MOVDload [int32(s-8)] src mem) (FSTPQ dst (Select0 <typ.Vec128> (FLDPQ src mem)) (Select1 <typ.Vec128> (FLDPQ src mem)) mem))
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 32 && s <= 40) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(s - 8))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(s - 8))
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64FSTPQ, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpSelect0, typ.Vec128)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64FLDPQ, types.NewTuple(typ.Vec128, typ.Vec128))
		v3.AddArg2(src, mem)
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.Vec128)
		v4.AddArg(v3)
		v1.AddArg4(dst, v2, v4, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 40 && s <= 48
	// result: (FMOVQstore [int32(s-16)] dst (FMOVQload [int32(s-16)] src mem) (FSTPQ dst (Select0 <typ.Vec128> (FLDPQ src mem)) (Select1 <typ.Vec128> (FLDPQ src mem)) mem))
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 40 && s <= 48) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVQstore)
		v.AuxInt = ssa.Int32ToAuxInt(int32(s - 16))
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FMOVQload, typ.Vec128)
		v0.AuxInt = ssa.Int32ToAuxInt(int32(s - 16))
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64FSTPQ, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpSelect0, typ.Vec128)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64FLDPQ, types.NewTuple(typ.Vec128, typ.Vec128))
		v3.AddArg2(src, mem)
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.Vec128)
		v4.AddArg(v3)
		v1.AddArg4(dst, v2, v4, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 48 && s <= 64
	// result: (FSTPQ [int32(s-32)] dst (Select0 <typ.Vec128> (FLDPQ [int32(s-32)] src mem)) (Select1 <typ.Vec128> (FLDPQ [int32(s-32)] src mem)) (FSTPQ dst (Select0 <typ.Vec128> (FLDPQ src mem)) (Select1 <typ.Vec128> (FLDPQ src mem)) mem))
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 48 && s <= 64) {
			break
		}
		v.Reset(ssaop.OpARM64FSTPQ)
		v.AuxInt = ssa.Int32ToAuxInt(int32(s - 32))
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, typ.Vec128)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64FLDPQ, types.NewTuple(typ.Vec128, typ.Vec128))
		v1.AuxInt = ssa.Int32ToAuxInt(int32(s - 32))
		v1.AddArg2(src, mem)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.Vec128)
		v2.AddArg(v1)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64FSTPQ, types.TypeMem)
		v4 := b.NewValue0(v.Pos, ssaop.OpSelect0, typ.Vec128)
		v5 := b.NewValue0(v.Pos, ssaop.OpARM64FLDPQ, types.NewTuple(typ.Vec128, typ.Vec128))
		v5.AddArg2(src, mem)
		v4.AddArg(v5)
		v6 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.Vec128)
		v6.AddArg(v5)
		v3.AddArg4(dst, v4, v6, mem)
		v.AddArg4(dst, v0, v2, v3)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 64 && s < 192 && ssa.LogLargeCopyValue(v, s)
	// result: (LoweredMove [s] dst src mem)
	for {
		s := ssa.AuxIntToInt64(v.AuxInt)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s > 64 && s < 192 && ssa.LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.OpARM64LoweredMove)
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
		v.Reset(ssaop.OpARM64LoweredMoveLoop)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg3(dst, src, mem)
		return true
	}
	return false
}
func rewriteValue_OpMulAddFloat32x4(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MulAddFloat32x4 x y z)
	// result: (VFMLA4S z x y)
	for {
		x := v_0
		y := v_1
		z := v_2
		v.Reset(ssaop.OpARM64VFMLA4S)
		v.AddArg3(z, x, y)
		return true
	}
}
func rewriteValue_OpMulAddFloat64x2(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MulAddFloat64x2 x y z)
	// result: (VFMLA2D z x y)
	for {
		x := v_0
		y := v_1
		z := v_2
		v.Reset(ssaop.OpARM64VFMLA2D)
		v.AddArg3(z, x, y)
		return true
	}
}
func rewriteValue_OpMulAddInt16x8(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MulAddInt16x8 x y z)
	// result: (VMLA8H z x y)
	for {
		x := v_0
		y := v_1
		z := v_2
		v.Reset(ssaop.OpARM64VMLA8H)
		v.AddArg3(z, x, y)
		return true
	}
}
func rewriteValue_OpMulAddInt32x4(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MulAddInt32x4 x y z)
	// result: (VMLA4S z x y)
	for {
		x := v_0
		y := v_1
		z := v_2
		v.Reset(ssaop.OpARM64VMLA4S)
		v.AddArg3(z, x, y)
		return true
	}
}
func rewriteValue_OpMulAddInt8x16(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MulAddInt8x16 x y z)
	// result: (VMLA16B z x y)
	for {
		x := v_0
		y := v_1
		z := v_2
		v.Reset(ssaop.OpARM64VMLA16B)
		v.AddArg3(z, x, y)
		return true
	}
}
func rewriteValue_OpMulAddUint16x8(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MulAddUint16x8 x y z)
	// result: (VMLA8H z x y)
	for {
		x := v_0
		y := v_1
		z := v_2
		v.Reset(ssaop.OpARM64VMLA8H)
		v.AddArg3(z, x, y)
		return true
	}
}
func rewriteValue_OpMulAddUint32x4(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MulAddUint32x4 x y z)
	// result: (VMLA4S z x y)
	for {
		x := v_0
		y := v_1
		z := v_2
		v.Reset(ssaop.OpARM64VMLA4S)
		v.AddArg3(z, x, y)
		return true
	}
}
func rewriteValue_OpMulAddUint8x16(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MulAddUint8x16 x y z)
	// result: (VMLA16B z x y)
	for {
		x := v_0
		y := v_1
		z := v_2
		v.Reset(ssaop.OpARM64VMLA16B)
		v.AddArg3(z, x, y)
		return true
	}
}
func rewriteValue_OpNegFloat32s(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NegFloat32s x)
	// result: (ZFNEGS x (Select0 <types.TypeMask> (PWHILELTS (MOVDconst [0]) (MOVDconst [8]))))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64ZFNEGS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTS, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(8)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpNegFloat64s(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NegFloat64s x)
	// result: (ZFNEGD x (Select0 <types.TypeMask> (PWHILELTD (MOVDconst [0]) (MOVDconst [4]))))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64ZFNEGD)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTD, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(4)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpNegInt16s(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NegInt16s x)
	// result: (ZNEGH x (Select0 <types.TypeMask> (PWHILELTH (MOVDconst [0]) (MOVDconst [16]))))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64ZNEGH)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTH, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(16)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpNegInt32s(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NegInt32s x)
	// result: (ZNEGS x (Select0 <types.TypeMask> (PWHILELTS (MOVDconst [0]) (MOVDconst [8]))))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64ZNEGS)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTS, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(8)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpNegInt64s(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NegInt64s x)
	// result: (ZNEGD x (Select0 <types.TypeMask> (PWHILELTD (MOVDconst [0]) (MOVDconst [4]))))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64ZNEGD)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTD, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(4)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpNegInt8s(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (NegInt8s x)
	// result: (ZNEGB x (Select0 <types.TypeMask> (PWHILELTB (MOVDconst [0]) (MOVDconst [32]))))
	for {
		x := v_0
		v.Reset(ssaop.OpARM64ZNEGB)
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect0, types.TypeMask)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64PWHILELTB, types.NewTuple(typ.Mask, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = ssa.Int64ToAuxInt(0)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(32)
		v1.AddArg2(v2, v3)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpNeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// result: (NotEqual (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32 x y)
	// result: (NotEqual (CMPW x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32F x y)
	// result: (NotEqual (FCMPS x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPS, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeq64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64 x y)
	// result: (NotEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64F x y)
	// result: (NotEqual (FCMPD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64FCMPD, types.TypeFlags)
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
	// result: (NotEqual (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNeqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (NeqPtr x y)
	// result: (NotEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpNot(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Not x)
	// result: (XOR (MOVDconst [1]) x)
	for {
		x := v_0
		v.Reset(ssaop.OpARM64XOR)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg2(v0, x)
		return true
	}
}
func rewriteValue_OpOffPtr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (OffPtr [off] ptr:(SP))
	// cond: ssa.Is32Bit(off)
	// result: (MOVDaddr [int32(off)] ptr)
	for {
		off := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		if ptr.Op != ssaop.OpSP || !(ssa.Is32Bit(off)) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDaddr)
		v.AuxInt = ssa.Int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// result: (ADDconst [off] ptr)
	for {
		off := ssa.AuxIntToInt64(v.AuxInt)
		ptr := v_0
		v.Reset(ssaop.OpARM64ADDconst)
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
	// result: (FMOVDfpgp <t> (VUADDLV <typ.Float64> (VCNT <typ.Float64> (FMOVDgpfp <typ.Float64> (ZeroExt16to64 x)))))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpARM64FMOVDfpgp)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VUADDLV, typ.Float64)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VCNT, typ.Float64)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64FMOVDgpfp, typ.Float64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v3.AddArg(x)
		v2.AddArg(v3)
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
	// result: (FMOVDfpgp <t> (VUADDLV <typ.Float64> (VCNT <typ.Float64> (FMOVDgpfp <typ.Float64> (ZeroExt32to64 x)))))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpARM64FMOVDfpgp)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VUADDLV, typ.Float64)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VCNT, typ.Float64)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64FMOVDgpfp, typ.Float64)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v3.AddArg(x)
		v2.AddArg(v3)
		v1.AddArg(v2)
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
	// result: (FMOVDfpgp <t> (VUADDLV <typ.Float64> (VCNT <typ.Float64> (FMOVDgpfp <typ.Float64> x))))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpARM64FMOVDfpgp)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VUADDLV, typ.Float64)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VCNT, typ.Float64)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64FMOVDgpfp, typ.Float64)
		v2.AddArg(x)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValue_OpPrefetchCache(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PrefetchCache addr mem)
	// result: (PRFM [0] addr mem)
	for {
		addr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64PRFM)
		v.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg2(addr, mem)
		return true
	}
}
func rewriteValue_OpPrefetchCacheStreamed(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (PrefetchCacheStreamed addr mem)
	// result: (PRFM [1] addr mem)
	for {
		addr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64PRFM)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		v.AddArg2(addr, mem)
		return true
	}
}
func rewriteValue_OpPubBarrier(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (PubBarrier mem)
	// result: (DMB [0xe] mem)
	for {
		mem := v_0
		v.Reset(ssaop.OpARM64DMB)
		v.AuxInt = ssa.Int64ToAuxInt(0xe)
		v.AddArg(mem)
		return true
	}
}
func rewriteValue_OpRotateLeft16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVDconst [c]))
	// result: (Or16 (Lsh16x64 <t> x (MOVDconst [c&15])) (Rsh16Ux64 <t> x (MOVDconst [-c&15])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpOr16)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh16x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(c & 15)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh16Ux64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(-c & 15)
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
		v.Reset(ssaop.OpARM64RORW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ORshiftLL, typ.UInt32)
		v0.AuxInt = ssa.Int64ToAuxInt(16)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg2(v1, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, typ.Int64)
		v2.AddArg(y)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValue_OpRotateLeft32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RotateLeft32 x y)
	// result: (RORW x (NEG <y.Type> y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64RORW)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, y.Type)
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
	// result: (ROR x (NEG <y.Type> y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64ROR)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, y.Type)
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
	// match: (RotateLeft8 <t> x (MOVDconst [c]))
	// result: (Or8 (Lsh8x64 <t> x (MOVDconst [c&7])) (Rsh8Ux64 <t> x (MOVDconst [-c&7])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		c := ssa.AuxIntToInt64(v_1.AuxInt)
		v.Reset(ssaop.OpOr8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh8x64, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v1.AuxInt = ssa.Int64ToAuxInt(c & 7)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh8Ux64, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(-c & 7)
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
		v.Reset(ssaop.OpARM64OR)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SLL, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ANDconst, typ.Int64)
		v1.AuxInt = ssa.Int64ToAuxInt(7)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64SRL, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64ANDconst, typ.Int64)
		v4.AuxInt = ssa.Int64ToAuxInt(7)
		v5 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, typ.Int64)
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
	// match: (Rsh16Ux16 <t> [bounded] x y)
	// result: (Rsh64Ux16 <t> [bounded] (ZeroExt16to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux16)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh16Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 <t> [bounded] x y)
	// result: (Rsh64Ux32 <t> [bounded] (ZeroExt16to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux32)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh16Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 <t> [bounded] x y)
	// result: (Rsh64Ux64 <t> [bounded] (ZeroExt16to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh16Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 <t> [bounded] x y)
	// result: (Rsh64Ux8 <t> [bounded] (ZeroExt16to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux8)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 <t> [bounded] x y)
	// result: (Rsh64x16 <t> [bounded] (SignExt16to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x16)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 <t> [bounded] x y)
	// result: (Rsh64x32 <t> [bounded] (SignExt16to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x32)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 <t> [bounded] x y)
	// result: (Rsh64x64 <t> [bounded] (SignExt16to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 <t> [bounded] x y)
	// result: (Rsh64x8 <t> [bounded] (SignExt16to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x8)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh32Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 <t> [bounded] x y)
	// result: (Rsh64Ux16 <t> [bounded] (ZeroExt32to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux16)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh32Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 <t> [bounded] x y)
	// result: (Rsh64Ux32 <t> [bounded] (ZeroExt32to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux32)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh32Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 <t> [bounded] x y)
	// result: (Rsh64Ux64 <t> [bounded] (ZeroExt32to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh32Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 <t> [bounded] x y)
	// result: (Rsh64Ux8 <t> [bounded] (ZeroExt32to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux8)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 <t> [bounded] x y)
	// result: (Rsh64x16 <t> [bounded] (SignExt32to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x16)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 <t> [bounded] x y)
	// result: (Rsh64x32 <t> [bounded] (SignExt32to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x32)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x64 <t> [bounded] x y)
	// result: (Rsh64x64 <t> [bounded] (SignExt32to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 <t> [bounded] x y)
	// result: (Rsh64x8 <t> [bounded] (SignExt32to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x8)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh64Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux16 <t> [bounded] x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (Rsh64Ux32 <t> [bounded] x (ZeroExt16to32 y))
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRsh64Ux32)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh64Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux32 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (CSEL [ssaop.OpARM64LessThanU] (SRL <t> x y) (Const64 <t> [0]) (CMPWconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64CSEL)
		v.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CMPWconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValue_OpRsh64Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64Ux64 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (CSEL [ssaop.OpARM64LessThanU] (SRL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64CSEL)
		v.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SRL, t)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpConst64, t)
		v1.AuxInt = ssa.Int64ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg(y)
		v.AddArg3(v0, v1, v2)
		return true
	}
	return false
}
func rewriteValue_OpRsh64Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRL <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SRL)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64Ux8 <t> [bounded] x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (Rsh64Ux32 <t> [bounded] x (ZeroExt8to32 y))
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRsh64Ux32)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh64x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SRA)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x16 <t> [bounded] x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (Rsh64x32 <t> [bounded] x (ZeroExt16to32 y))
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRsh64x32)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh64x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x32 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SRA)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x32 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA x (CSEL [ssaop.OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPWconst [64] y)))
	for {
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, y.Type)
		v0.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v1 := b.NewValue0(v.Pos, ssaop.OpConst64, y.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(63)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CMPWconst, types.TypeFlags)
		v2.AuxInt = ssa.Int32ToAuxInt(64)
		v2.AddArg(y)
		v0.AddArg3(y, v1, v2)
		v.AddArg2(x, v0)
		return true
	}
	return false
}
func rewriteValue_OpRsh64x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh64x64 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SRA)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x64 <t> x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (SRA x (CSEL [ssaop.OpARM64LessThanU] <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, y.Type)
		v0.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v1 := b.NewValue0(v.Pos, ssaop.OpConst64, y.Type)
		v1.AuxInt = ssa.Int64ToAuxInt(63)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = ssa.Int64ToAuxInt(64)
		v2.AddArg(y)
		v0.AddArg3(y, v1, v2)
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
	// match: (Rsh64x8 <t> x y)
	// cond: ssa.ShiftIsBounded(v)
	// result: (SRA <t> x y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		if !(ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpARM64SRA)
		v.Type = t
		v.AddArg2(x, y)
		return true
	}
	// match: (Rsh64x8 <t> [bounded] x y)
	// cond: !ssa.ShiftIsBounded(v)
	// result: (Rsh64x32 <t> [bounded] x (ZeroExt8to32 y))
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		if !(!ssa.ShiftIsBounded(v)) {
			break
		}
		v.Reset(ssaop.OpRsh64x32)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
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
	// match: (Rsh8Ux16 <t> [bounded] x y)
	// result: (Rsh64Ux16 <t> [bounded] (ZeroExt8to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux16)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh8Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 <t> [bounded] x y)
	// result: (Rsh64Ux32 <t> [bounded] (ZeroExt8to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux32)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh8Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 <t> [bounded] x y)
	// result: (Rsh64Ux64 <t> [bounded] (ZeroExt8to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux64)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh8Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 <t> [bounded] x y)
	// result: (Rsh64Ux8 <t> [bounded] (ZeroExt8to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64Ux8)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 <t> [bounded] x y)
	// result: (Rsh64x16 <t> [bounded] (SignExt8to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x16)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 <t> [bounded] x y)
	// result: (Rsh64x32 <t> [bounded] (SignExt8to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x32)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 <t> [bounded] x y)
	// result: (Rsh64x64 <t> [bounded] (SignExt8to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x64)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpRsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 <t> [bounded] x y)
	// result: (Rsh64x8 <t> [bounded] (SignExt8to64 x) y)
	for {
		t := v.Type
		bounded := ssa.AuxIntToBool(v.AuxInt)
		x := v_0
		y := v_1
		v.Reset(ssaop.OpRsh64x8)
		v.Type = t
		v.AuxInt = ssa.BoolToAuxInt(bounded)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValue_OpScalableVectorLen(v *ssa.Value) bool {
	// match: (ScalableVectorLen)
	// result: (RDVL [1])
	for {
		v.Reset(ssaop.OpARM64RDVL)
		v.AuxInt = ssa.Int64ToAuxInt(1)
		return true
	}
}
func rewriteValue_OpSelect0(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Mul64uhilo x y))
	// result: (UMULH x y)
	for {
		if v_0.Op != ssaop.OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64UMULH)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select0 (Add64carry x y c))
	// result: (Select0 <typ.UInt64> (ADCSflags x y (Select1 <types.TypeFlags> (ADDSconstflags [-1] c))))
	for {
		if v_0.Op != ssaop.OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64ADCSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v1 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64ADDSconstflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v2.AuxInt = ssa.Int64ToAuxInt(-1)
		v2.AddArg(c)
		v1.AddArg(v2)
		v0.AddArg3(x, y, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 (Sub64borrow x y bo))
	// result: (Select0 <typ.UInt64> (SBCSflags x y (Select1 <types.TypeFlags> (NEGSflags bo))))
	for {
		if v_0.Op != ssaop.OpSub64borrow {
			break
		}
		bo := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64SBCSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v1 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64NEGSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v2.AddArg(bo)
		v1.AddArg(v2)
		v0.AddArg3(x, y, v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 (Mul64uover x y))
	// result: (MUL x y)
	for {
		if v_0.Op != ssaop.OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64MUL)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValue_OpSelect1(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Mul64uhilo x y))
	// result: (MUL x y)
	for {
		if v_0.Op != ssaop.OpMul64uhilo {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64MUL)
		v.AddArg2(x, y)
		return true
	}
	// match: (Select1 (Add64carry x y c))
	// result: (ADCzerocarry <typ.UInt64> (Select1 <types.TypeFlags> (ADCSflags x y (Select1 <types.TypeFlags> (ADDSconstflags [-1] c)))))
	for {
		if v_0.Op != ssaop.OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpARM64ADCzerocarry)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64ADCSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v2 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64ADDSconstflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v3.AuxInt = ssa.Int64ToAuxInt(-1)
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
		if v_0.Op != ssaop.OpSub64borrow {
			break
		}
		bo := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.Reset(ssaop.OpARM64NEG)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64NGCzerocarry, typ.UInt64)
		v1 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64SBCSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v3 := b.NewValue0(v.Pos, ssaop.OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64NEGSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
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
		if v_0.Op != ssaop.OpMul64uover {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64UMULH, typ.UInt64)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValue_OpSelectN(v *ssa.Value) bool {
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
		if call.Op != ssaop.OpARM64CALLstatic || len(call.Args) != 1 {
			break
		}
		sym := ssa.AuxToCall(call.Aux)
		s1 := call.Args[0]
		if s1.Op != ssaop.OpARM64MOVDstore {
			break
		}
		_ = s1.Args[2]
		s1_1 := s1.Args[1]
		if s1_1.Op != ssaop.OpARM64MOVDconst {
			break
		}
		sz := ssa.AuxIntToInt64(s1_1.AuxInt)
		s2 := s1.Args[2]
		if s2.Op != ssaop.OpARM64MOVDstore {
			break
		}
		_ = s2.Args[2]
		src := s2.Args[1]
		s3 := s2.Args[2]
		if s3.Op != ssaop.OpARM64MOVDstore {
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
		if call.Op != ssaop.OpARM64CALLstatic || len(call.Args) != 4 {
			break
		}
		sym := ssa.AuxToCall(call.Aux)
		mem := call.Args[3]
		dst := call.Args[0]
		src := call.Args[1]
		call_2 := call.Args[2]
		if call_2.Op != ssaop.OpARM64MOVDconst {
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
func rewriteValue_OpShiftAllLeftInt16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftInt16x8 x y)
	// result: (VSSHL8H x (VDUPHbcast [0] (VMOVHins [0] x (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y)))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VSSHL8H)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPHbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVHins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v2.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(127)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v4.AddArg(y)
		v2.AddArg3(y, v3, v4)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllLeftInt32x4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftInt32x4 x y)
	// result: (VSSHL4S x (VDUPSbcast [0] (VMOVSins [0] x (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y)))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VSSHL4S)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPSbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVSins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v2.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(127)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v4.AddArg(y)
		v2.AddArg3(y, v3, v4)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllLeftInt64x2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftInt64x2 x y)
	// result: (VSSHL2D x (VDUPDbcast [0] (VMOVDins [0] x (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y)))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VSSHL2D)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPDbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVDins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v2.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(127)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v4.AddArg(y)
		v2.AddArg3(y, v3, v4)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllLeftInt8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftInt8x16 x y)
	// result: (VSSHL16B x (VDUPBbcast [0] (VMOVBins [0] x (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y)))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VSSHL16B)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPBbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVBins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v2.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(127)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v4.AddArg(y)
		v2.AddArg3(y, v3, v4)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllLeftUint16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftUint16x8 x y)
	// result: (VUSHL8H x (VDUPHbcast [0] (VMOVHins [0] x (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y)))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VUSHL8H)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPHbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVHins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v2.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(127)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v4.AddArg(y)
		v2.AddArg3(y, v3, v4)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllLeftUint32x4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftUint32x4 x y)
	// result: (VUSHL4S x (VDUPSbcast [0] (VMOVSins [0] x (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y)))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VUSHL4S)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPSbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVSins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v2.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(127)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v4.AddArg(y)
		v2.AddArg3(y, v3, v4)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllLeftUint64x2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftUint64x2 x y)
	// result: (VUSHL2D x (VDUPDbcast [0] (VMOVDins [0] x (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y)))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VUSHL2D)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPDbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVDins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v2.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(127)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v4.AddArg(y)
		v2.AddArg3(y, v3, v4)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllLeftUint8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllLeftUint8x16 x y)
	// result: (VUSHL16B x (VDUPBbcast [0] (VMOVBins [0] x (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y)))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VUSHL16B)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPBbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVBins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v2.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = ssa.Int64ToAuxInt(127)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v4.AddArg(y)
		v2.AddArg3(y, v3, v4)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllRightInt16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightInt16x8 x y)
	// result: (VSSHL8H x (VDUPHbcast [0] (VMOVHins [0] x (NEG <typ.Int64> (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y))))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VSSHL8H)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPHbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVHins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v3.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v5 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v5.AuxInt = ssa.Int64ToAuxInt(127)
		v5.AddArg(y)
		v3.AddArg3(y, v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllRightInt32x4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightInt32x4 x y)
	// result: (VSSHL4S x (VDUPSbcast [0] (VMOVSins [0] x (NEG <typ.Int64> (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y))))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VSSHL4S)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPSbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVSins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v3.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v5 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v5.AuxInt = ssa.Int64ToAuxInt(127)
		v5.AddArg(y)
		v3.AddArg3(y, v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllRightInt64x2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightInt64x2 x y)
	// result: (VSSHL2D x (VDUPDbcast [0] (VMOVDins [0] x (NEG <typ.Int64> (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y))))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VSSHL2D)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPDbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVDins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v3.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v5 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v5.AuxInt = ssa.Int64ToAuxInt(127)
		v5.AddArg(y)
		v3.AddArg3(y, v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllRightInt8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightInt8x16 x y)
	// result: (VSSHL16B x (VDUPBbcast [0] (VMOVBins [0] x (NEG <typ.Int64> (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y))))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VSSHL16B)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPBbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVBins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v3.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v5 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v5.AuxInt = ssa.Int64ToAuxInt(127)
		v5.AddArg(y)
		v3.AddArg3(y, v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllRightUint16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightUint16x8 x y)
	// result: (VUSHL8H x (VDUPHbcast [0] (VMOVHins [0] x (NEG <typ.Int64> (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y))))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VUSHL8H)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPHbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVHins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v3.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v5 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v5.AuxInt = ssa.Int64ToAuxInt(127)
		v5.AddArg(y)
		v3.AddArg3(y, v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllRightUint32x4(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightUint32x4 x y)
	// result: (VUSHL4S x (VDUPSbcast [0] (VMOVSins [0] x (NEG <typ.Int64> (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y))))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VUSHL4S)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPSbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVSins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v3.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v5 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v5.AuxInt = ssa.Int64ToAuxInt(127)
		v5.AddArg(y)
		v3.AddArg3(y, v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllRightUint64x2(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightUint64x2 x y)
	// result: (VUSHL2D x (VDUPDbcast [0] (VMOVDins [0] x (NEG <typ.Int64> (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y))))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VUSHL2D)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPDbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVDins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v3.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v5 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v5.AuxInt = ssa.Int64ToAuxInt(127)
		v5.AddArg(y)
		v3.AddArg3(y, v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpShiftAllRightUint8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ShiftAllRightUint8x16 x y)
	// result: (VUSHL16B x (VDUPBbcast [0] (VMOVBins [0] x (NEG <typ.Int64> (CSEL <typ.UInt64> [ssaop.OpARM64LessThanU] y (MOVDconst [127]) (CMPconst [127] y))))))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARM64VUSHL16B)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64VDUPBbcast, typ.Vec128)
		v0.AuxInt = ssa.Uint8ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64VMOVBins, typ.Vec128)
		v1.AuxInt = ssa.Uint8ToAuxInt(0)
		v2 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, typ.Int64)
		v3 := b.NewValue0(v.Pos, ssaop.OpARM64CSEL, typ.UInt64)
		v3.AuxInt = ssa.OpToAuxInt(ssaop.OpARM64LessThanU)
		v4 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = ssa.Int64ToAuxInt(127)
		v5 := b.NewValue0(v.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
		v5.AuxInt = ssa.Int64ToAuxInt(127)
		v5.AddArg(y)
		v3.AddArg3(y, v4, v5)
		v2.AddArg(v3)
		v1.AddArg2(x, v2)
		v0.AddArg(v1)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValue_OpSlicemask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SRAconst (NEG <t> x) [63])
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpARM64SRAconst)
		v.AuxInt = ssa.Int64ToAuxInt(63)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64NEG, t)
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
		v.Reset(ssaop.OpARM64MOVBstore)
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
		v.Reset(ssaop.OpARM64MOVHstore)
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
		v.Reset(ssaop.OpARM64MOVWstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && !t.IsFloat() && !t.IsSIMD()
	// result: (MOVDstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && !t.IsFloat() && !t.IsSIMD()) {
			break
		}
		v.Reset(ssaop.OpARM64MOVDstore)
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
		v.Reset(ssaop.OpARM64FMOVSstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
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
		v.Reset(ssaop.OpARM64FMOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 16
	// result: (FMOVQstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 16) {
			break
		}
		v.Reset(ssaop.OpARM64FMOVQstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 32 && t.IsSIMD()
	// result: (ZSTRstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 32 && t.IsSIMD()) {
			break
		}
		v.Reset(ssaop.OpARM64ZSTRstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.Size() == 8 && t.IsSIMD()
	// result: (PSTRstore ptr val mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		val := v_1
		mem := v_2
		if !(t.Size() == 8 && t.IsSIMD()) {
			break
		}
		v.Reset(ssaop.OpARM64PSTRstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValue_OpStoreMasked8(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (StoreMasked8 {t} ptr mask val mem)
	// cond: t.Size() == 32
	// result: (ZST1BPredstore ptr val mask mem)
	for {
		t := ssa.AuxToType(v.Aux)
		ptr := v_0
		mask := v_1
		val := v_2
		mem := v_3
		if !(t.Size() == 32) {
			break
		}
		v.Reset(ssaop.OpARM64ZST1BPredstore)
		v.AddArg4(ptr, val, mask, mem)
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
	// result: (MOVBstore ptr (MOVDconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] ptr mem)
	// result: (MOVHstore ptr (MOVDconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [4] ptr mem)
	// result: (MOVWstore ptr (MOVDconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [3] ptr mem)
	// result: (MOVBstore [2] ptr (MOVDconst [0]) (MOVHstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVHstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [5] ptr mem)
	// result: (MOVBstore [4] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 5 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [6] ptr mem)
	// result: (MOVHstore [4] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 6 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(4)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [7] ptr mem)
	// result: (MOVWstore [3] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 7 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVWstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [8] ptr mem)
	// result: (MOVDstore ptr (MOVDconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 8 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVDstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [9] ptr mem)
	// result: (MOVBstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 9 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVBstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [10] ptr mem)
	// result: (MOVHstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 10 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVHstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [11] ptr mem)
	// result: (MOVDstore [3] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 11 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [12] ptr mem)
	// result: (MOVWstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 12 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVWstore)
		v.AuxInt = ssa.Int32ToAuxInt(8)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [13] ptr mem)
	// result: (MOVDstore [5] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 13 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(5)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [14] ptr mem)
	// result: (MOVDstore [6] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 14 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(6)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [15] ptr mem)
	// result: (MOVDstore [7] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 15 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64MOVDstore)
		v.AuxInt = ssa.Int32ToAuxInt(7)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDstore, types.TypeMem)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [16] ptr mem)
	// result: (STP [0] ptr (MOVDconst [0]) (MOVDconst [0]) mem)
	for {
		if ssa.AuxIntToInt64(v.AuxInt) != 16 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARM64STP)
		v.AuxInt = ssa.Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = ssa.Int64ToAuxInt(0)
		v.AddArg4(ptr, v0, v0, mem)
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
		v.Reset(ssaop.OpARM64LoweredZero)
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
		v.Reset(ssaop.OpARM64LoweredZeroLoop)
		v.AuxInt = ssa.Int64ToAuxInt(s)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValue_OpZeroSIMD(v *ssa.Value) bool {
	// match: (ZeroSIMD <t>)
	// cond: t.Size() == 16
	// result: (VMOVI16B [0] <t>)
	for {
		t := v.Type
		if !(t.Size() == 16) {
			break
		}
		v.Reset(ssaop.OpARM64VMOVI16B)
		v.Type = t
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		return true
	}
	// match: (ZeroSIMD <t>)
	// cond: t.Size() == 32 && t.IsSIMD()
	// result: (ZDUPBconst [0])
	for {
		t := v.Type
		if !(t.Size() == 32 && t.IsSIMD()) {
			break
		}
		v.Reset(ssaop.OpARM64ZDUPBconst)
		v.AuxInt = ssa.Int8ToAuxInt(0)
		return true
	}
	// match: (ZeroSIMD <t>)
	// cond: t.Size() == 8 && t.IsSIMD()
	// result: (PPFALSEB)
	for {
		t := v.Type
		if !(t.Size() == 8 && t.IsSIMD()) {
			break
		}
		v.Reset(ssaop.OpARM64PPFALSEB)
		return true
	}
	return false
}
func rewriteValue_Opbroadcast1To16Int8x16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (broadcast1To16Int8x16 x)
	// result: (VDUPBbcast [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARM64VDUPBbcast)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValue_Opbroadcast1To16Uint8x16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (broadcast1To16Uint8x16 x)
	// result: (VDUPBbcast [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARM64VDUPBbcast)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValue_Opbroadcast1To2Float64x2(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (broadcast1To2Float64x2 x)
	// result: (VDUPDbcast [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARM64VDUPDbcast)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValue_Opbroadcast1To2Int64x2(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (broadcast1To2Int64x2 x)
	// result: (VDUPDbcast [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARM64VDUPDbcast)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValue_Opbroadcast1To2Uint64x2(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (broadcast1To2Uint64x2 x)
	// result: (VDUPDbcast [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARM64VDUPDbcast)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValue_Opbroadcast1To4Float32x4(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (broadcast1To4Float32x4 x)
	// result: (VDUPSbcast [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARM64VDUPSbcast)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValue_Opbroadcast1To4Int32x4(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (broadcast1To4Int32x4 x)
	// result: (VDUPSbcast [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARM64VDUPSbcast)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValue_Opbroadcast1To4Uint32x4(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (broadcast1To4Uint32x4 x)
	// result: (VDUPSbcast [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARM64VDUPSbcast)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValue_Opbroadcast1To8Int16x8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (broadcast1To8Int16x8 x)
	// result: (VDUPHbcast [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARM64VDUPHbcast)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValue_Opbroadcast1To8Uint16x8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (broadcast1To8Uint16x8 x)
	// result: (VDUPHbcast [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARM64VDUPHbcast)
		v.AuxInt = ssa.Uint8ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func RewriteBlock(b *ssa.Block) bool {
	typ := &b.Func.Config.Types
	switch b.Kind {
	case block.BlockARM64EQ:
		// match: (EQ (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (TST x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (EQ (TSTconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ANDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (TSTW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (EQ (TSTWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ANDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (EQ (CMNconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ADDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (EQ (CMNWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ADDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (CMNW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64EQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMP {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != ssaop.OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (CMNW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPW {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != ssaop.OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] x) yes no)
		// result: (Z x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64Z, x)
			return true
		}
		// match: (EQ (CMPWconst [0] x) yes no)
		// result: (ZW x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64ZW, x)
			return true
		}
		// match: (EQ (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (EQ (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (EQ (CMNW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (EQ (CMP a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMP, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (EQ (CMPW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (EQ (TSTconst [c] x) yes no)
		// cond: ssa.OneBit(c)
		// result: (TBZ [int64(ssa.Ntz64(c))] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64TSTconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			x := v_0.Args[0]
			if !(ssa.OneBit(c)) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Ntz64(c)))
			return true
		}
		// match: (EQ (TSTWconst [c] x) yes no)
		// cond: ssa.OneBit(int64(uint32(c)))
		// result: (TBZ [int64(ssa.Ntz64(int64(uint32(c))))] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64TSTWconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if !(ssa.OneBit(int64(uint32(c)))) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Ntz64(int64(uint32(c)))))
			return true
		}
		// match: (EQ (FlagConstant [fc]) yes no)
		// cond: fc.Eq()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Eq()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (EQ (FlagConstant [fc]) yes no)
		// cond: !fc.Eq()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Eq()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (EQ (InvertFlags cmp) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64EQ, cmp)
			return true
		}
	case block.BlockARM64FGE:
		// match: (FGE (InvertFlags cmp) yes no)
		// result: (FLE cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64FLE, cmp)
			return true
		}
	case block.BlockARM64FGT:
		// match: (FGT (InvertFlags cmp) yes no)
		// result: (FLT cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64FLT, cmp)
			return true
		}
	case block.BlockARM64FLE:
		// match: (FLE (InvertFlags cmp) yes no)
		// result: (FGE cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64FGE, cmp)
			return true
		}
	case block.BlockARM64FLT:
		// match: (FLT (InvertFlags cmp) yes no)
		// result: (FGT cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64FGT, cmp)
			return true
		}
	case block.BlockARM64GE:
		// match: (GE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (TST x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64GE, v0)
				return true
			}
			break
		}
		// match: (GE (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GE (TSTconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ANDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64GE, v0)
			return true
		}
		// match: (GE (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (TSTW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64GE, v0)
				return true
			}
			break
		}
		// match: (GE (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GE (TSTWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ANDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64GE, v0)
			return true
		}
		// match: (GE (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GEnoov (CMNconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ADDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64GEnoov, v0)
			return true
		}
		// match: (GE (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GEnoov (CMNWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ADDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64GEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (GEnoov (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64GEnoov, v0)
				return true
			}
			break
		}
		// match: (GE (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (GEnoov (CMNW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64GEnoov, v0)
				return true
			}
			break
		}
		// match: (GE (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (GEnoov (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64GEnoov, v0)
			return true
		}
		// match: (GE (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (GEnoov (CMNW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64GEnoov, v0)
			return true
		}
		// match: (GE (CMPWconst [0] x) yes no)
		// result: (TBZ [31] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(31)
			return true
		}
		// match: (GE (CMPconst [0] x) yes no)
		// result: (TBZ [63] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(63)
			return true
		}
		// match: (GE (CMPWconst [128] x) yes no)
		// cond: ssa.ZeroUpper56Bits(x)
		// result: (TBNZ [7] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 128 {
				break
			}
			x := v_0.Args[0]
			if !(ssa.ZeroUpper56Bits(x)) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(7)
			return true
		}
		// match: (GE (CMPconst [128] x) yes no)
		// cond: ssa.ZeroUpper56Bits(x)
		// result: (TBNZ [7] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 128 {
				break
			}
			x := v_0.Args[0]
			if !(ssa.ZeroUpper56Bits(x)) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(7)
			return true
		}
		// match: (GE (FlagConstant [fc]) yes no)
		// cond: fc.Ge()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Ge()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GE (FlagConstant [fc]) yes no)
		// cond: !fc.Ge()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Ge()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (GE (InvertFlags cmp) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64LE, cmp)
			return true
		}
	case block.BlockARM64GEnoov:
		// match: (GEnoov (FlagConstant [fc]) yes no)
		// cond: fc.GeNoov()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.GeNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GEnoov (FlagConstant [fc]) yes no)
		// cond: !fc.GeNoov()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.GeNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockARM64GT:
		// match: (GT (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (TST x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64GT, v0)
				return true
			}
			break
		}
		// match: (GT (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GT (TSTconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ANDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64GT, v0)
			return true
		}
		// match: (GT (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (TSTW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64GT, v0)
				return true
			}
			break
		}
		// match: (GT (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GT (TSTWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ANDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64GT, v0)
			return true
		}
		// match: (GT (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GTnoov (CMNconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ADDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64GTnoov, v0)
			return true
		}
		// match: (GT (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GTnoov (CMNWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ADDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64GTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (GTnoov (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64GTnoov, v0)
				return true
			}
			break
		}
		// match: (GT (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (GTnoov (CMNW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64GTnoov, v0)
				return true
			}
			break
		}
		// match: (GT (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (GTnoov (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64GTnoov, v0)
			return true
		}
		// match: (GT (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (GTnoov (CMNW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64GTnoov, v0)
			return true
		}
		// match: (GT (FlagConstant [fc]) yes no)
		// cond: fc.Gt()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Gt()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GT (FlagConstant [fc]) yes no)
		// cond: !fc.Gt()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Gt()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (GT (InvertFlags cmp) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64LT, cmp)
			return true
		}
	case block.BlockARM64GTnoov:
		// match: (GTnoov (FlagConstant [fc]) yes no)
		// cond: fc.GtNoov()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.GtNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GTnoov (FlagConstant [fc]) yes no)
		// cond: !fc.GtNoov()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.GtNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockIf:
		// match: (If (Equal cc) yes no)
		// result: (EQ cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64Equal {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64EQ, cc)
			return true
		}
		// match: (If (NotEqual cc) yes no)
		// result: (NE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64NotEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64NE, cc)
			return true
		}
		// match: (If (LessThan cc) yes no)
		// result: (LT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64LT, cc)
			return true
		}
		// match: (If (LessThanU cc) yes no)
		// result: (ULT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64ULT, cc)
			return true
		}
		// match: (If (LessEqual cc) yes no)
		// result: (LE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64LE, cc)
			return true
		}
		// match: (If (LessEqualU cc) yes no)
		// result: (ULE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64ULE, cc)
			return true
		}
		// match: (If (GreaterThan cc) yes no)
		// result: (GT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64GT, cc)
			return true
		}
		// match: (If (GreaterThanU cc) yes no)
		// result: (UGT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64UGT, cc)
			return true
		}
		// match: (If (GreaterEqual cc) yes no)
		// result: (GE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64GE, cc)
			return true
		}
		// match: (If (GreaterEqualU cc) yes no)
		// result: (UGE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64UGE, cc)
			return true
		}
		// match: (If (LessThanF cc) yes no)
		// result: (FLT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessThanF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64FLT, cc)
			return true
		}
		// match: (If (LessEqualF cc) yes no)
		// result: (FLE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessEqualF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64FLE, cc)
			return true
		}
		// match: (If (GreaterThanF cc) yes no)
		// result: (FGT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterThanF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64FGT, cc)
			return true
		}
		// match: (If (GreaterEqualF cc) yes no)
		// result: (FGE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterEqualF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64FGE, cc)
			return true
		}
		// match: (If cond yes no)
		// result: (TBNZ [0] cond yes no)
		for {
			cond := b.Controls[0]
			b.ResetWithControl(block.BlockARM64TBNZ, cond)
			b.AuxInt = ssa.Int64ToAuxInt(0)
			return true
		}
	case block.BlockJumpTable:
		// match: (JumpTable idx)
		// result: (JUMPTABLE {ssa.MakeJumpTableSym(b)} idx (MOVDaddr <typ.Uintptr> {ssa.MakeJumpTableSym(b)} (SB)))
		for {
			idx := b.Controls[0]
			v0 := b.NewValue0(b.Pos, ssaop.OpARM64MOVDaddr, typ.Uintptr)
			v0.Aux = ssa.SymToAux(ssa.MakeJumpTableSym(b))
			v1 := b.NewValue0(b.Pos, ssaop.OpSB, typ.Uintptr)
			v0.AddArg(v1)
			b.ResetWithControl2(block.BlockARM64JUMPTABLE, idx, v0)
			b.Aux = ssa.SymToAux(ssa.MakeJumpTableSym(b))
			return true
		}
	case block.BlockARM64LE:
		// match: (LE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (TST x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64LE, v0)
				return true
			}
			break
		}
		// match: (LE (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LE (TSTconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ANDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64LE, v0)
			return true
		}
		// match: (LE (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (TSTW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64LE, v0)
				return true
			}
			break
		}
		// match: (LE (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LE (TSTWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ANDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64LE, v0)
			return true
		}
		// match: (LE (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LEnoov (CMNconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ADDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64LEnoov, v0)
			return true
		}
		// match: (LE (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LEnoov (CMNWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ADDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64LEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (LEnoov (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64LEnoov, v0)
				return true
			}
			break
		}
		// match: (LE (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (LEnoov (CMNW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64LEnoov, v0)
				return true
			}
			break
		}
		// match: (LE (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (LEnoov (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64LEnoov, v0)
			return true
		}
		// match: (LE (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (LEnoov (CMNW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64LEnoov, v0)
			return true
		}
		// match: (LE (FlagConstant [fc]) yes no)
		// cond: fc.Le()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Le()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LE (FlagConstant [fc]) yes no)
		// cond: !fc.Le()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Le()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (LE (InvertFlags cmp) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64GE, cmp)
			return true
		}
	case block.BlockARM64LEnoov:
		// match: (LEnoov (FlagConstant [fc]) yes no)
		// cond: fc.LeNoov()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.LeNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LEnoov (FlagConstant [fc]) yes no)
		// cond: !fc.LeNoov()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.LeNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockARM64LT:
		// match: (LT (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (TST x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64LT, v0)
				return true
			}
			break
		}
		// match: (LT (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LT (TSTconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ANDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64LT, v0)
			return true
		}
		// match: (LT (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (TSTW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64LT, v0)
				return true
			}
			break
		}
		// match: (LT (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LT (TSTWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ANDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64LT, v0)
			return true
		}
		// match: (LT (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LTnoov (CMNconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ADDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64LTnoov, v0)
			return true
		}
		// match: (LT (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LTnoov (CMNWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ADDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64LTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (LTnoov (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64LTnoov, v0)
				return true
			}
			break
		}
		// match: (LT (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (LTnoov (CMNW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64LTnoov, v0)
				return true
			}
			break
		}
		// match: (LT (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (LTnoov (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64LTnoov, v0)
			return true
		}
		// match: (LT (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (LTnoov (CMNW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64LTnoov, v0)
			return true
		}
		// match: (LT (CMPWconst [0] x) yes no)
		// result: (TBNZ [31] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(31)
			return true
		}
		// match: (LT (CMPconst [0] x) yes no)
		// result: (TBNZ [63] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(63)
			return true
		}
		// match: (LT (CMPWconst [128] x) yes no)
		// cond: ssa.ZeroUpper56Bits(x)
		// result: (TBZ [7] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 128 {
				break
			}
			x := v_0.Args[0]
			if !(ssa.ZeroUpper56Bits(x)) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(7)
			return true
		}
		// match: (LT (CMPconst [128] x) yes no)
		// cond: ssa.ZeroUpper56Bits(x)
		// result: (TBZ [7] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 128 {
				break
			}
			x := v_0.Args[0]
			if !(ssa.ZeroUpper56Bits(x)) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(7)
			return true
		}
		// match: (LT (FlagConstant [fc]) yes no)
		// cond: fc.Lt()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Lt()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LT (FlagConstant [fc]) yes no)
		// cond: !fc.Lt()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Lt()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (LT (InvertFlags cmp) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64GT, cmp)
			return true
		}
	case block.BlockARM64LTnoov:
		// match: (LTnoov (FlagConstant [fc]) yes no)
		// cond: fc.LtNoov()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.LtNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LTnoov (FlagConstant [fc]) yes no)
		// cond: !fc.LtNoov()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.LtNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockARM64NE:
		// match: (NE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (TST x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64NE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (NE (TSTconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ANDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (TSTW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64AND {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64NE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (NE (TSTWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ANDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (NE (CMNconst [c] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ADDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (NE (CMNWconst [int32(c)] y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			if x.Op != ssaop.OpARM64ADDconst {
				break
			}
			c := ssa.AuxIntToInt64(x.AuxInt)
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64NE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (CMNW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64ADD {
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
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARM64NE, v0)
				return true
			}
			break
		}
		// match: (NE (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMP {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != ssaop.OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (CMNW x y) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPW {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			z := v_0.Args[1]
			if z.Op != ssaop.OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPconst [0] x) yes no)
		// result: (NZ x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64NZ, x)
			return true
		}
		// match: (NE (CMPWconst [0] x) yes no)
		// result: (NZW x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64NZW, x)
			return true
		}
		// match: (NE (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (NE (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (NE (CMNW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMNW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (NE (CMP a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMP, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NE (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (NE (CMPW a (MULW <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			z := v_0.Args[0]
			if z.Op != ssaop.OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARM64MULW, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NE (TSTconst [c] x) yes no)
		// cond: ssa.OneBit(c)
		// result: (TBNZ [int64(ssa.Ntz64(c))] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64TSTconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			x := v_0.Args[0]
			if !(ssa.OneBit(c)) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Ntz64(c)))
			return true
		}
		// match: (NE (TSTWconst [c] x) yes no)
		// cond: ssa.OneBit(int64(uint32(c)))
		// result: (TBNZ [int64(ssa.Ntz64(int64(uint32(c))))] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64TSTWconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if !(ssa.OneBit(int64(uint32(c)))) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Ntz64(int64(uint32(c)))))
			return true
		}
		// match: (NE (FlagConstant [fc]) yes no)
		// cond: fc.Ne()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Ne()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (NE (FlagConstant [fc]) yes no)
		// cond: !fc.Ne()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Ne()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (NE (InvertFlags cmp) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64NE, cmp)
			return true
		}
	case block.BlockARM64NZ:
		// match: (NZ (Equal cc) yes no)
		// result: (EQ cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64Equal {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64EQ, cc)
			return true
		}
		// match: (NZ (NotEqual cc) yes no)
		// result: (NE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64NotEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64NE, cc)
			return true
		}
		// match: (NZ (LessThan cc) yes no)
		// result: (LT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64LT, cc)
			return true
		}
		// match: (NZ (LessThanU cc) yes no)
		// result: (ULT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64ULT, cc)
			return true
		}
		// match: (NZ (LessEqual cc) yes no)
		// result: (LE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64LE, cc)
			return true
		}
		// match: (NZ (LessEqualU cc) yes no)
		// result: (ULE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64ULE, cc)
			return true
		}
		// match: (NZ (GreaterThan cc) yes no)
		// result: (GT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64GT, cc)
			return true
		}
		// match: (NZ (GreaterThanU cc) yes no)
		// result: (UGT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64UGT, cc)
			return true
		}
		// match: (NZ (GreaterEqual cc) yes no)
		// result: (GE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64GE, cc)
			return true
		}
		// match: (NZ (GreaterEqualU cc) yes no)
		// result: (UGE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64UGE, cc)
			return true
		}
		// match: (NZ (LessThanF cc) yes no)
		// result: (FLT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessThanF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64FLT, cc)
			return true
		}
		// match: (NZ (LessEqualF cc) yes no)
		// result: (FLE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessEqualF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64FLE, cc)
			return true
		}
		// match: (NZ (GreaterThanF cc) yes no)
		// result: (FGT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterThanF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64FGT, cc)
			return true
		}
		// match: (NZ (GreaterEqualF cc) yes no)
		// result: (FGE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterEqualF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64FGE, cc)
			return true
		}
		// match: (NZ sub:(SUB x y))
		// cond: sub.Uses == 1
		// result: (NE (CMP x y))
		for b.Controls[0].Op == ssaop.OpARM64SUB {
			sub := b.Controls[0]
			y := sub.Args[1]
			x := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, ssaop.OpARM64CMP, types.TypeFlags)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NZ sub:(SUBconst [c] y))
		// cond: sub.Uses == 1
		// result: (NE (CMPconst [c] y))
		for b.Controls[0].Op == ssaop.OpARM64SUBconst {
			sub := b.Controls[0]
			c := ssa.AuxIntToInt64(sub.AuxInt)
			y := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NZ (ANDconst [c] x) yes no)
		// cond: ssa.OneBit(c)
		// result: (TBNZ [int64(ssa.Ntz64(c))] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64ANDconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			x := v_0.Args[0]
			if !(ssa.OneBit(c)) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Ntz64(c)))
			return true
		}
		// match: (NZ s:(SRLconst [63] x) yes no)
		// cond: s.Uses == 1
		// result: (TBNZ [63] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64SRLconst {
			s := b.Controls[0]
			if ssa.AuxIntToInt64(s.AuxInt) != 63 {
				break
			}
			x := s.Args[0]
			if !(s.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(63)
			return true
		}
		// match: (NZ s:(SRAconst [63] x) yes no)
		// cond: s.Uses == 1
		// result: (TBNZ [63] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64SRAconst {
			s := b.Controls[0]
			if ssa.AuxIntToInt64(s.AuxInt) != 63 {
				break
			}
			x := s.Args[0]
			if !(s.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(63)
			return true
		}
		// match: (NZ (MOVDconst [0]) yes no)
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64MOVDconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (NZ (MOVDconst [c]) yes no)
		// cond: c != 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64MOVDconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(c != 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
	case block.BlockARM64NZW:
		// match: (NZW sub:(SUB x y))
		// cond: sub.Uses == 1
		// result: (NE (CMPW x y))
		for b.Controls[0].Op == ssaop.OpARM64SUB {
			sub := b.Controls[0]
			y := sub.Args[1]
			x := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NZW sub:(SUBconst [c] y))
		// cond: sub.Uses == 1
		// result: (NE (CMPWconst [int32(c)] y))
		for b.Controls[0].Op == ssaop.OpARM64SUBconst {
			sub := b.Controls[0]
			c := ssa.AuxIntToInt64(sub.AuxInt)
			y := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, ssaop.OpARM64CMPWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (NZW (ANDconst [c] x) yes no)
		// cond: ssa.OneBit(int64(uint32(c)))
		// result: (TBNZ [int64(ssa.Ntz64(int64(uint32(c))))] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64ANDconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			x := v_0.Args[0]
			if !(ssa.OneBit(int64(uint32(c)))) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Ntz64(int64(uint32(c)))))
			return true
		}
		// match: (NZW (MOVDconst [c]) yes no)
		// cond: int32(c) == 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64MOVDconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(int32(c) == 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (NZW (MOVDconst [c]) yes no)
		// cond: int32(c) != 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64MOVDconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(int32(c) != 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
	case block.BlockARM64TBNZ:
		// match: (TBNZ [0] (Equal cc) yes no)
		// result: (EQ cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64Equal {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64EQ, cc)
			return true
		}
		// match: (TBNZ [0] (NotEqual cc) yes no)
		// result: (NE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64NotEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64NE, cc)
			return true
		}
		// match: (TBNZ [0] (LessThan cc) yes no)
		// result: (LT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64LT, cc)
			return true
		}
		// match: (TBNZ [0] (LessThanU cc) yes no)
		// result: (ULT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64ULT, cc)
			return true
		}
		// match: (TBNZ [0] (LessEqual cc) yes no)
		// result: (LE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64LE, cc)
			return true
		}
		// match: (TBNZ [0] (LessEqualU cc) yes no)
		// result: (ULE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64ULE, cc)
			return true
		}
		// match: (TBNZ [0] (GreaterThan cc) yes no)
		// result: (GT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64GT, cc)
			return true
		}
		// match: (TBNZ [0] (GreaterThanU cc) yes no)
		// result: (UGT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64UGT, cc)
			return true
		}
		// match: (TBNZ [0] (GreaterEqual cc) yes no)
		// result: (GE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64GE, cc)
			return true
		}
		// match: (TBNZ [0] (GreaterEqualU cc) yes no)
		// result: (UGE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64UGE, cc)
			return true
		}
		// match: (TBNZ [0] (LessThanF cc) yes no)
		// result: (FLT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessThanF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64FLT, cc)
			return true
		}
		// match: (TBNZ [0] (LessEqualF cc) yes no)
		// result: (FLE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64LessEqualF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64FLE, cc)
			return true
		}
		// match: (TBNZ [0] (GreaterThanF cc) yes no)
		// result: (FGT cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterThanF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64FGT, cc)
			return true
		}
		// match: (TBNZ [0] (GreaterEqualF cc) yes no)
		// result: (FGE cc yes no)
		for b.Controls[0].Op == ssaop.OpARM64GreaterEqualF {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64FGE, cc)
			return true
		}
		// match: (TBNZ [0] (XORconst [1] x) yes no)
		// result: (TBZ [0] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64XORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			x := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(0)
			return true
		}
		// match: (TBNZ [t] sv:(SRLconst [s] x) yes no)
		// cond: t+s < 64 && sv.Uses == 1
		// result: (TBNZ [t+s] x yes no )
		for b.Controls[0].Op == ssaop.OpARM64SRLconst {
			sv := b.Controls[0]
			s := ssa.AuxIntToInt64(sv.AuxInt)
			x := sv.Args[0]
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(t+s < 64 && sv.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(t + s)
			return true
		}
		// match: (TBNZ [t] (SRLconst [s] x) yes no)
		// cond: t+s >= 64
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64SRLconst {
			v_0 := b.Controls[0]
			s := ssa.AuxIntToInt64(v_0.AuxInt)
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(t+s >= 64) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (TBNZ [t] sv:(SLLconst [s] x) yes no)
		// cond: t-s >= 0 && sv.Uses == 1
		// result: (TBNZ [t-s] x yes no )
		for b.Controls[0].Op == ssaop.OpARM64SLLconst {
			sv := b.Controls[0]
			s := ssa.AuxIntToInt64(sv.AuxInt)
			x := sv.Args[0]
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(t-s >= 0 && sv.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(t - s)
			return true
		}
		// match: (TBNZ [t] (SLLconst [s] x) yes no)
		// cond: t-s < 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64SLLconst {
			v_0 := b.Controls[0]
			s := ssa.AuxIntToInt64(v_0.AuxInt)
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(t-s < 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (TBNZ [t] rv:(RORconst [r] x) yes no)
		// cond: rv.Uses == 1
		// result: (TBNZ [int64(uint64(t+r)%64)] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64RORconst {
			rv := b.Controls[0]
			r := ssa.AuxIntToInt64(rv.AuxInt)
			x := rv.Args[0]
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(rv.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(int64(uint64(t+r) % 64))
			return true
		}
		// match: (TBNZ [t] sv:(SRAconst [s] x) yes no)
		// cond: t+s < 64 && sv.Uses == 1
		// result: (TBNZ [t+s] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64SRAconst {
			sv := b.Controls[0]
			s := ssa.AuxIntToInt64(sv.AuxInt)
			x := sv.Args[0]
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(t+s < 64 && sv.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(t + s)
			return true
		}
		// match: (TBNZ [t] sv:(SRAconst [s] x) yes no)
		// cond: t+s >= 64 && sv.Uses == 1
		// result: (TBNZ [63 ] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64SRAconst {
			sv := b.Controls[0]
			s := ssa.AuxIntToInt64(sv.AuxInt)
			x := sv.Args[0]
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(t+s >= 64 && sv.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(63)
			return true
		}
	case block.BlockARM64TBZ:
		// match: (TBZ [0] (XORconst [1] x) yes no)
		// result: (TBNZ [0] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64XORconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 1 {
				break
			}
			x := v_0.Args[0]
			if ssa.AuxIntToInt64(b.AuxInt) != 0 {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(0)
			return true
		}
		// match: (TBZ [t] sv:(SRLconst [s] x) yes no)
		// cond: t+s < 64 && sv.Uses == 1
		// result: (TBZ [t+s] x yes no )
		for b.Controls[0].Op == ssaop.OpARM64SRLconst {
			sv := b.Controls[0]
			s := ssa.AuxIntToInt64(sv.AuxInt)
			x := sv.Args[0]
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(t+s < 64 && sv.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(t + s)
			return true
		}
		// match: (TBZ [t] (SRLconst [s] x) yes no)
		// cond: t+s >= 64
		// result: (First yes no )
		for b.Controls[0].Op == ssaop.OpARM64SRLconst {
			v_0 := b.Controls[0]
			s := ssa.AuxIntToInt64(v_0.AuxInt)
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(t+s >= 64) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (TBZ [t] sv:(SLLconst [s] x) yes no)
		// cond: t-s >= 0 && sv.Uses == 1
		// result: (TBZ [t-s] x yes no )
		for b.Controls[0].Op == ssaop.OpARM64SLLconst {
			sv := b.Controls[0]
			s := ssa.AuxIntToInt64(sv.AuxInt)
			x := sv.Args[0]
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(t-s >= 0 && sv.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(t - s)
			return true
		}
		// match: (TBZ [t] (SLLconst [s] x) yes no)
		// cond: t-s < 0
		// result: (First yes no )
		for b.Controls[0].Op == ssaop.OpARM64SLLconst {
			v_0 := b.Controls[0]
			s := ssa.AuxIntToInt64(v_0.AuxInt)
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(t-s < 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (TBZ [t] rv:(RORconst [r] x) yes no)
		// cond: rv.Uses == 1
		// result: (TBZ [int64(uint64(t+r)%64)] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64RORconst {
			rv := b.Controls[0]
			r := ssa.AuxIntToInt64(rv.AuxInt)
			x := rv.Args[0]
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(rv.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(int64(uint64(t+r) % 64))
			return true
		}
		// match: (TBZ [t] sv:(SRAconst [s] x) yes no)
		// cond: t+s < 64 && sv.Uses == 1
		// result: (TBZ [t+s] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64SRAconst {
			sv := b.Controls[0]
			s := ssa.AuxIntToInt64(sv.AuxInt)
			x := sv.Args[0]
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(t+s < 64 && sv.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(t + s)
			return true
		}
		// match: (TBZ [t] sv:(SRAconst [s] x) yes no)
		// cond: t+s >= 64 && sv.Uses == 1
		// result: (TBZ [63 ] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64SRAconst {
			sv := b.Controls[0]
			s := ssa.AuxIntToInt64(sv.AuxInt)
			x := sv.Args[0]
			t := ssa.AuxIntToInt64(b.AuxInt)
			if !(t+s >= 64 && sv.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(63)
			return true
		}
	case block.BlockARM64UGE:
		// match: (UGE (CMPWconst [128] x) yes no)
		// cond: ssa.ZeroUpper56Bits(x)
		// result: (TBNZ [7] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 128 {
				break
			}
			x := v_0.Args[0]
			if !(ssa.ZeroUpper56Bits(x)) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(7)
			return true
		}
		// match: (UGE (CMPconst [128] x) yes no)
		// cond: ssa.ZeroUpper56Bits(x)
		// result: (TBNZ [7] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 128 {
				break
			}
			x := v_0.Args[0]
			if !(ssa.ZeroUpper56Bits(x)) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBNZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(7)
			return true
		}
		// match: (UGE (FlagConstant [fc]) yes no)
		// cond: fc.Uge()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Uge()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (UGE (FlagConstant [fc]) yes no)
		// cond: !fc.Uge()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Uge()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (UGE (InvertFlags cmp) yes no)
		// result: (ULE cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64ULE, cmp)
			return true
		}
	case block.BlockARM64UGT:
		// match: (UGT (CMPconst [0] x))
		// result: (NE (CMPconst [0] x))
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(0)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (UGT (CMPWconst [0] x))
		// result: (NE (CMPWconst [0] x))
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMPWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(0)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARM64NE, v0)
			return true
		}
		// match: (UGT (FlagConstant [fc]) yes no)
		// cond: fc.Ugt()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Ugt()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (UGT (FlagConstant [fc]) yes no)
		// cond: !fc.Ugt()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Ugt()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (UGT (InvertFlags cmp) yes no)
		// result: (ULT cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64ULT, cmp)
			return true
		}
	case block.BlockARM64ULE:
		// match: (ULE (CMPconst [0] x))
		// result: (EQ (CMPconst [0] x))
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(0)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (ULE (CMPWconst [0] x))
		// result: (EQ (CMPWconst [0] x))
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			x := v_0.Args[0]
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARM64CMPWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(0)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (ULE (FlagConstant [fc]) yes no)
		// cond: fc.Ule()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Ule()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (ULE (FlagConstant [fc]) yes no)
		// cond: !fc.Ule()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Ule()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (ULE (InvertFlags cmp) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64UGE, cmp)
			return true
		}
	case block.BlockARM64ULT:
		// match: (ULT (CMPWconst [128] x) yes no)
		// cond: ssa.ZeroUpper56Bits(x)
		// result: (TBZ [7] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPWconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt32(v_0.AuxInt) != 128 {
				break
			}
			x := v_0.Args[0]
			if !(ssa.ZeroUpper56Bits(x)) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(7)
			return true
		}
		// match: (ULT (CMPconst [128] x) yes no)
		// cond: ssa.ZeroUpper56Bits(x)
		// result: (TBZ [7] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64CMPconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 128 {
				break
			}
			x := v_0.Args[0]
			if !(ssa.ZeroUpper56Bits(x)) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(7)
			return true
		}
		// match: (ULT (FlagConstant [fc]) yes no)
		// cond: fc.Ult()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Ult()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (ULT (FlagConstant [fc]) yes no)
		// cond: !fc.Ult()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64FlagConstant {
			v_0 := b.Controls[0]
			fc := ssa.AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Ult()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (ULT (InvertFlags cmp) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == ssaop.OpARM64InvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARM64UGT, cmp)
			return true
		}
	case block.BlockARM64Z:
		// match: (Z sub:(SUB x y))
		// cond: sub.Uses == 1
		// result: (EQ (CMP x y))
		for b.Controls[0].Op == ssaop.OpARM64SUB {
			sub := b.Controls[0]
			y := sub.Args[1]
			x := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, ssaop.OpARM64CMP, types.TypeFlags)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (Z sub:(SUBconst [c] y))
		// cond: sub.Uses == 1
		// result: (EQ (CMPconst [c] y))
		for b.Controls[0].Op == ssaop.OpARM64SUBconst {
			sub := b.Controls[0]
			c := ssa.AuxIntToInt64(sub.AuxInt)
			y := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, ssaop.OpARM64CMPconst, types.TypeFlags)
			v0.AuxInt = ssa.Int64ToAuxInt(c)
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (Z (ANDconst [c] x) yes no)
		// cond: ssa.OneBit(c)
		// result: (TBZ [int64(ssa.Ntz64(c))] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64ANDconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			x := v_0.Args[0]
			if !(ssa.OneBit(c)) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Ntz64(c)))
			return true
		}
		// match: (Z s:(SRLconst [63] x) yes no)
		// cond: s.Uses == 1
		// result: (TBZ [63] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64SRLconst {
			s := b.Controls[0]
			if ssa.AuxIntToInt64(s.AuxInt) != 63 {
				break
			}
			x := s.Args[0]
			if !(s.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(63)
			return true
		}
		// match: (Z s:(SRAconst [63] x) yes no)
		// cond: s.Uses == 1
		// result: (TBZ [63] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64SRAconst {
			s := b.Controls[0]
			if ssa.AuxIntToInt64(s.AuxInt) != 63 {
				break
			}
			x := s.Args[0]
			if !(s.Uses == 1) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(63)
			return true
		}
		// match: (Z (MOVDconst [0]) yes no)
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64MOVDconst {
			v_0 := b.Controls[0]
			if ssa.AuxIntToInt64(v_0.AuxInt) != 0 {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (Z (MOVDconst [c]) yes no)
		// cond: c != 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64MOVDconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(c != 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockARM64ZW:
		// match: (ZW sub:(SUB x y))
		// cond: sub.Uses == 1
		// result: (EQ (CMPW x y))
		for b.Controls[0].Op == ssaop.OpARM64SUB {
			sub := b.Controls[0]
			y := sub.Args[1]
			x := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, ssaop.OpARM64CMPW, types.TypeFlags)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (ZW sub:(SUBconst [c] y))
		// cond: sub.Uses == 1
		// result: (EQ (CMPWconst [int32(c)] y))
		for b.Controls[0].Op == ssaop.OpARM64SUBconst {
			sub := b.Controls[0]
			c := ssa.AuxIntToInt64(sub.AuxInt)
			y := sub.Args[0]
			if !(sub.Uses == 1) {
				break
			}
			v0 := b.NewValue0(sub.Pos, ssaop.OpARM64CMPWconst, types.TypeFlags)
			v0.AuxInt = ssa.Int32ToAuxInt(int32(c))
			v0.AddArg(y)
			b.ResetWithControl(block.BlockARM64EQ, v0)
			return true
		}
		// match: (ZW (ANDconst [c] x) yes no)
		// cond: ssa.OneBit(int64(uint32(c)))
		// result: (TBZ [int64(ssa.Ntz64(int64(uint32(c))))] x yes no)
		for b.Controls[0].Op == ssaop.OpARM64ANDconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			x := v_0.Args[0]
			if !(ssa.OneBit(int64(uint32(c)))) {
				break
			}
			b.ResetWithControl(block.BlockARM64TBZ, x)
			b.AuxInt = ssa.Int64ToAuxInt(int64(ssa.Ntz64(int64(uint32(c)))))
			return true
		}
		// match: (ZW (MOVDconst [c]) yes no)
		// cond: int32(c) == 0
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARM64MOVDconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(int32(c) == 0) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (ZW (MOVDconst [c]) yes no)
		// cond: int32(c) != 0
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARM64MOVDconst {
			v_0 := b.Controls[0]
			c := ssa.AuxIntToInt64(v_0.AuxInt)
			if !(int32(c) != 0) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	}
	return false
}
