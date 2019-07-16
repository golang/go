// Code generated from gen/ARM64.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

import "fmt"
import "math"
import "cmd/internal/obj"
import "cmd/internal/objabi"
import "cmd/compile/internal/types"

var _ = fmt.Println   // in case not otherwise used
var _ = math.MinInt8  // in case not otherwise used
var _ = obj.ANOP      // in case not otherwise used
var _ = objabi.GOROOT // in case not otherwise used
var _ = types.TypeMem // in case not otherwise used

func rewriteValueARM64(v *Value) bool {
	switch v.Op {
	case OpARM64ADCSflags:
		return rewriteValueARM64_OpARM64ADCSflags_0(v)
	case OpARM64ADD:
		return rewriteValueARM64_OpARM64ADD_0(v) || rewriteValueARM64_OpARM64ADD_10(v) || rewriteValueARM64_OpARM64ADD_20(v)
	case OpARM64ADDconst:
		return rewriteValueARM64_OpARM64ADDconst_0(v)
	case OpARM64ADDshiftLL:
		return rewriteValueARM64_OpARM64ADDshiftLL_0(v)
	case OpARM64ADDshiftRA:
		return rewriteValueARM64_OpARM64ADDshiftRA_0(v)
	case OpARM64ADDshiftRL:
		return rewriteValueARM64_OpARM64ADDshiftRL_0(v)
	case OpARM64AND:
		return rewriteValueARM64_OpARM64AND_0(v) || rewriteValueARM64_OpARM64AND_10(v)
	case OpARM64ANDconst:
		return rewriteValueARM64_OpARM64ANDconst_0(v)
	case OpARM64ANDshiftLL:
		return rewriteValueARM64_OpARM64ANDshiftLL_0(v)
	case OpARM64ANDshiftRA:
		return rewriteValueARM64_OpARM64ANDshiftRA_0(v)
	case OpARM64ANDshiftRL:
		return rewriteValueARM64_OpARM64ANDshiftRL_0(v)
	case OpARM64BIC:
		return rewriteValueARM64_OpARM64BIC_0(v)
	case OpARM64BICshiftLL:
		return rewriteValueARM64_OpARM64BICshiftLL_0(v)
	case OpARM64BICshiftRA:
		return rewriteValueARM64_OpARM64BICshiftRA_0(v)
	case OpARM64BICshiftRL:
		return rewriteValueARM64_OpARM64BICshiftRL_0(v)
	case OpARM64CMN:
		return rewriteValueARM64_OpARM64CMN_0(v)
	case OpARM64CMNW:
		return rewriteValueARM64_OpARM64CMNW_0(v)
	case OpARM64CMNWconst:
		return rewriteValueARM64_OpARM64CMNWconst_0(v)
	case OpARM64CMNconst:
		return rewriteValueARM64_OpARM64CMNconst_0(v)
	case OpARM64CMNshiftLL:
		return rewriteValueARM64_OpARM64CMNshiftLL_0(v)
	case OpARM64CMNshiftRA:
		return rewriteValueARM64_OpARM64CMNshiftRA_0(v)
	case OpARM64CMNshiftRL:
		return rewriteValueARM64_OpARM64CMNshiftRL_0(v)
	case OpARM64CMP:
		return rewriteValueARM64_OpARM64CMP_0(v)
	case OpARM64CMPW:
		return rewriteValueARM64_OpARM64CMPW_0(v)
	case OpARM64CMPWconst:
		return rewriteValueARM64_OpARM64CMPWconst_0(v)
	case OpARM64CMPconst:
		return rewriteValueARM64_OpARM64CMPconst_0(v)
	case OpARM64CMPshiftLL:
		return rewriteValueARM64_OpARM64CMPshiftLL_0(v)
	case OpARM64CMPshiftRA:
		return rewriteValueARM64_OpARM64CMPshiftRA_0(v)
	case OpARM64CMPshiftRL:
		return rewriteValueARM64_OpARM64CMPshiftRL_0(v)
	case OpARM64CSEL:
		return rewriteValueARM64_OpARM64CSEL_0(v)
	case OpARM64CSEL0:
		return rewriteValueARM64_OpARM64CSEL0_0(v)
	case OpARM64DIV:
		return rewriteValueARM64_OpARM64DIV_0(v)
	case OpARM64DIVW:
		return rewriteValueARM64_OpARM64DIVW_0(v)
	case OpARM64EON:
		return rewriteValueARM64_OpARM64EON_0(v)
	case OpARM64EONshiftLL:
		return rewriteValueARM64_OpARM64EONshiftLL_0(v)
	case OpARM64EONshiftRA:
		return rewriteValueARM64_OpARM64EONshiftRA_0(v)
	case OpARM64EONshiftRL:
		return rewriteValueARM64_OpARM64EONshiftRL_0(v)
	case OpARM64Equal:
		return rewriteValueARM64_OpARM64Equal_0(v)
	case OpARM64FADDD:
		return rewriteValueARM64_OpARM64FADDD_0(v)
	case OpARM64FADDS:
		return rewriteValueARM64_OpARM64FADDS_0(v)
	case OpARM64FCMPD:
		return rewriteValueARM64_OpARM64FCMPD_0(v)
	case OpARM64FCMPS:
		return rewriteValueARM64_OpARM64FCMPS_0(v)
	case OpARM64FMOVDfpgp:
		return rewriteValueARM64_OpARM64FMOVDfpgp_0(v)
	case OpARM64FMOVDgpfp:
		return rewriteValueARM64_OpARM64FMOVDgpfp_0(v)
	case OpARM64FMOVDload:
		return rewriteValueARM64_OpARM64FMOVDload_0(v)
	case OpARM64FMOVDloadidx:
		return rewriteValueARM64_OpARM64FMOVDloadidx_0(v)
	case OpARM64FMOVDstore:
		return rewriteValueARM64_OpARM64FMOVDstore_0(v)
	case OpARM64FMOVDstoreidx:
		return rewriteValueARM64_OpARM64FMOVDstoreidx_0(v)
	case OpARM64FMOVSload:
		return rewriteValueARM64_OpARM64FMOVSload_0(v)
	case OpARM64FMOVSloadidx:
		return rewriteValueARM64_OpARM64FMOVSloadidx_0(v)
	case OpARM64FMOVSstore:
		return rewriteValueARM64_OpARM64FMOVSstore_0(v)
	case OpARM64FMOVSstoreidx:
		return rewriteValueARM64_OpARM64FMOVSstoreidx_0(v)
	case OpARM64FMULD:
		return rewriteValueARM64_OpARM64FMULD_0(v)
	case OpARM64FMULS:
		return rewriteValueARM64_OpARM64FMULS_0(v)
	case OpARM64FNEGD:
		return rewriteValueARM64_OpARM64FNEGD_0(v)
	case OpARM64FNEGS:
		return rewriteValueARM64_OpARM64FNEGS_0(v)
	case OpARM64FNMULD:
		return rewriteValueARM64_OpARM64FNMULD_0(v)
	case OpARM64FNMULS:
		return rewriteValueARM64_OpARM64FNMULS_0(v)
	case OpARM64FSUBD:
		return rewriteValueARM64_OpARM64FSUBD_0(v)
	case OpARM64FSUBS:
		return rewriteValueARM64_OpARM64FSUBS_0(v)
	case OpARM64GreaterEqual:
		return rewriteValueARM64_OpARM64GreaterEqual_0(v)
	case OpARM64GreaterEqualF:
		return rewriteValueARM64_OpARM64GreaterEqualF_0(v)
	case OpARM64GreaterEqualU:
		return rewriteValueARM64_OpARM64GreaterEqualU_0(v)
	case OpARM64GreaterThan:
		return rewriteValueARM64_OpARM64GreaterThan_0(v)
	case OpARM64GreaterThanF:
		return rewriteValueARM64_OpARM64GreaterThanF_0(v)
	case OpARM64GreaterThanU:
		return rewriteValueARM64_OpARM64GreaterThanU_0(v)
	case OpARM64LessEqual:
		return rewriteValueARM64_OpARM64LessEqual_0(v)
	case OpARM64LessEqualF:
		return rewriteValueARM64_OpARM64LessEqualF_0(v)
	case OpARM64LessEqualU:
		return rewriteValueARM64_OpARM64LessEqualU_0(v)
	case OpARM64LessThan:
		return rewriteValueARM64_OpARM64LessThan_0(v)
	case OpARM64LessThanF:
		return rewriteValueARM64_OpARM64LessThanF_0(v)
	case OpARM64LessThanU:
		return rewriteValueARM64_OpARM64LessThanU_0(v)
	case OpARM64MADD:
		return rewriteValueARM64_OpARM64MADD_0(v) || rewriteValueARM64_OpARM64MADD_10(v) || rewriteValueARM64_OpARM64MADD_20(v)
	case OpARM64MADDW:
		return rewriteValueARM64_OpARM64MADDW_0(v) || rewriteValueARM64_OpARM64MADDW_10(v) || rewriteValueARM64_OpARM64MADDW_20(v)
	case OpARM64MNEG:
		return rewriteValueARM64_OpARM64MNEG_0(v) || rewriteValueARM64_OpARM64MNEG_10(v) || rewriteValueARM64_OpARM64MNEG_20(v)
	case OpARM64MNEGW:
		return rewriteValueARM64_OpARM64MNEGW_0(v) || rewriteValueARM64_OpARM64MNEGW_10(v) || rewriteValueARM64_OpARM64MNEGW_20(v)
	case OpARM64MOD:
		return rewriteValueARM64_OpARM64MOD_0(v)
	case OpARM64MODW:
		return rewriteValueARM64_OpARM64MODW_0(v)
	case OpARM64MOVBUload:
		return rewriteValueARM64_OpARM64MOVBUload_0(v)
	case OpARM64MOVBUloadidx:
		return rewriteValueARM64_OpARM64MOVBUloadidx_0(v)
	case OpARM64MOVBUreg:
		return rewriteValueARM64_OpARM64MOVBUreg_0(v)
	case OpARM64MOVBload:
		return rewriteValueARM64_OpARM64MOVBload_0(v)
	case OpARM64MOVBloadidx:
		return rewriteValueARM64_OpARM64MOVBloadidx_0(v)
	case OpARM64MOVBreg:
		return rewriteValueARM64_OpARM64MOVBreg_0(v)
	case OpARM64MOVBstore:
		return rewriteValueARM64_OpARM64MOVBstore_0(v) || rewriteValueARM64_OpARM64MOVBstore_10(v) || rewriteValueARM64_OpARM64MOVBstore_20(v) || rewriteValueARM64_OpARM64MOVBstore_30(v) || rewriteValueARM64_OpARM64MOVBstore_40(v)
	case OpARM64MOVBstoreidx:
		return rewriteValueARM64_OpARM64MOVBstoreidx_0(v) || rewriteValueARM64_OpARM64MOVBstoreidx_10(v)
	case OpARM64MOVBstorezero:
		return rewriteValueARM64_OpARM64MOVBstorezero_0(v)
	case OpARM64MOVBstorezeroidx:
		return rewriteValueARM64_OpARM64MOVBstorezeroidx_0(v)
	case OpARM64MOVDload:
		return rewriteValueARM64_OpARM64MOVDload_0(v)
	case OpARM64MOVDloadidx:
		return rewriteValueARM64_OpARM64MOVDloadidx_0(v)
	case OpARM64MOVDloadidx8:
		return rewriteValueARM64_OpARM64MOVDloadidx8_0(v)
	case OpARM64MOVDreg:
		return rewriteValueARM64_OpARM64MOVDreg_0(v)
	case OpARM64MOVDstore:
		return rewriteValueARM64_OpARM64MOVDstore_0(v)
	case OpARM64MOVDstoreidx:
		return rewriteValueARM64_OpARM64MOVDstoreidx_0(v)
	case OpARM64MOVDstoreidx8:
		return rewriteValueARM64_OpARM64MOVDstoreidx8_0(v)
	case OpARM64MOVDstorezero:
		return rewriteValueARM64_OpARM64MOVDstorezero_0(v)
	case OpARM64MOVDstorezeroidx:
		return rewriteValueARM64_OpARM64MOVDstorezeroidx_0(v)
	case OpARM64MOVDstorezeroidx8:
		return rewriteValueARM64_OpARM64MOVDstorezeroidx8_0(v)
	case OpARM64MOVHUload:
		return rewriteValueARM64_OpARM64MOVHUload_0(v)
	case OpARM64MOVHUloadidx:
		return rewriteValueARM64_OpARM64MOVHUloadidx_0(v)
	case OpARM64MOVHUloadidx2:
		return rewriteValueARM64_OpARM64MOVHUloadidx2_0(v)
	case OpARM64MOVHUreg:
		return rewriteValueARM64_OpARM64MOVHUreg_0(v) || rewriteValueARM64_OpARM64MOVHUreg_10(v)
	case OpARM64MOVHload:
		return rewriteValueARM64_OpARM64MOVHload_0(v)
	case OpARM64MOVHloadidx:
		return rewriteValueARM64_OpARM64MOVHloadidx_0(v)
	case OpARM64MOVHloadidx2:
		return rewriteValueARM64_OpARM64MOVHloadidx2_0(v)
	case OpARM64MOVHreg:
		return rewriteValueARM64_OpARM64MOVHreg_0(v) || rewriteValueARM64_OpARM64MOVHreg_10(v)
	case OpARM64MOVHstore:
		return rewriteValueARM64_OpARM64MOVHstore_0(v) || rewriteValueARM64_OpARM64MOVHstore_10(v) || rewriteValueARM64_OpARM64MOVHstore_20(v)
	case OpARM64MOVHstoreidx:
		return rewriteValueARM64_OpARM64MOVHstoreidx_0(v) || rewriteValueARM64_OpARM64MOVHstoreidx_10(v)
	case OpARM64MOVHstoreidx2:
		return rewriteValueARM64_OpARM64MOVHstoreidx2_0(v)
	case OpARM64MOVHstorezero:
		return rewriteValueARM64_OpARM64MOVHstorezero_0(v)
	case OpARM64MOVHstorezeroidx:
		return rewriteValueARM64_OpARM64MOVHstorezeroidx_0(v)
	case OpARM64MOVHstorezeroidx2:
		return rewriteValueARM64_OpARM64MOVHstorezeroidx2_0(v)
	case OpARM64MOVQstorezero:
		return rewriteValueARM64_OpARM64MOVQstorezero_0(v)
	case OpARM64MOVWUload:
		return rewriteValueARM64_OpARM64MOVWUload_0(v)
	case OpARM64MOVWUloadidx:
		return rewriteValueARM64_OpARM64MOVWUloadidx_0(v)
	case OpARM64MOVWUloadidx4:
		return rewriteValueARM64_OpARM64MOVWUloadidx4_0(v)
	case OpARM64MOVWUreg:
		return rewriteValueARM64_OpARM64MOVWUreg_0(v) || rewriteValueARM64_OpARM64MOVWUreg_10(v)
	case OpARM64MOVWload:
		return rewriteValueARM64_OpARM64MOVWload_0(v)
	case OpARM64MOVWloadidx:
		return rewriteValueARM64_OpARM64MOVWloadidx_0(v)
	case OpARM64MOVWloadidx4:
		return rewriteValueARM64_OpARM64MOVWloadidx4_0(v)
	case OpARM64MOVWreg:
		return rewriteValueARM64_OpARM64MOVWreg_0(v) || rewriteValueARM64_OpARM64MOVWreg_10(v)
	case OpARM64MOVWstore:
		return rewriteValueARM64_OpARM64MOVWstore_0(v) || rewriteValueARM64_OpARM64MOVWstore_10(v)
	case OpARM64MOVWstoreidx:
		return rewriteValueARM64_OpARM64MOVWstoreidx_0(v)
	case OpARM64MOVWstoreidx4:
		return rewriteValueARM64_OpARM64MOVWstoreidx4_0(v)
	case OpARM64MOVWstorezero:
		return rewriteValueARM64_OpARM64MOVWstorezero_0(v)
	case OpARM64MOVWstorezeroidx:
		return rewriteValueARM64_OpARM64MOVWstorezeroidx_0(v)
	case OpARM64MOVWstorezeroidx4:
		return rewriteValueARM64_OpARM64MOVWstorezeroidx4_0(v)
	case OpARM64MSUB:
		return rewriteValueARM64_OpARM64MSUB_0(v) || rewriteValueARM64_OpARM64MSUB_10(v) || rewriteValueARM64_OpARM64MSUB_20(v)
	case OpARM64MSUBW:
		return rewriteValueARM64_OpARM64MSUBW_0(v) || rewriteValueARM64_OpARM64MSUBW_10(v) || rewriteValueARM64_OpARM64MSUBW_20(v)
	case OpARM64MUL:
		return rewriteValueARM64_OpARM64MUL_0(v) || rewriteValueARM64_OpARM64MUL_10(v) || rewriteValueARM64_OpARM64MUL_20(v)
	case OpARM64MULW:
		return rewriteValueARM64_OpARM64MULW_0(v) || rewriteValueARM64_OpARM64MULW_10(v) || rewriteValueARM64_OpARM64MULW_20(v)
	case OpARM64MVN:
		return rewriteValueARM64_OpARM64MVN_0(v)
	case OpARM64MVNshiftLL:
		return rewriteValueARM64_OpARM64MVNshiftLL_0(v)
	case OpARM64MVNshiftRA:
		return rewriteValueARM64_OpARM64MVNshiftRA_0(v)
	case OpARM64MVNshiftRL:
		return rewriteValueARM64_OpARM64MVNshiftRL_0(v)
	case OpARM64NEG:
		return rewriteValueARM64_OpARM64NEG_0(v)
	case OpARM64NEGshiftLL:
		return rewriteValueARM64_OpARM64NEGshiftLL_0(v)
	case OpARM64NEGshiftRA:
		return rewriteValueARM64_OpARM64NEGshiftRA_0(v)
	case OpARM64NEGshiftRL:
		return rewriteValueARM64_OpARM64NEGshiftRL_0(v)
	case OpARM64NotEqual:
		return rewriteValueARM64_OpARM64NotEqual_0(v)
	case OpARM64OR:
		return rewriteValueARM64_OpARM64OR_0(v) || rewriteValueARM64_OpARM64OR_10(v) || rewriteValueARM64_OpARM64OR_20(v) || rewriteValueARM64_OpARM64OR_30(v) || rewriteValueARM64_OpARM64OR_40(v)
	case OpARM64ORN:
		return rewriteValueARM64_OpARM64ORN_0(v)
	case OpARM64ORNshiftLL:
		return rewriteValueARM64_OpARM64ORNshiftLL_0(v)
	case OpARM64ORNshiftRA:
		return rewriteValueARM64_OpARM64ORNshiftRA_0(v)
	case OpARM64ORNshiftRL:
		return rewriteValueARM64_OpARM64ORNshiftRL_0(v)
	case OpARM64ORconst:
		return rewriteValueARM64_OpARM64ORconst_0(v)
	case OpARM64ORshiftLL:
		return rewriteValueARM64_OpARM64ORshiftLL_0(v) || rewriteValueARM64_OpARM64ORshiftLL_10(v) || rewriteValueARM64_OpARM64ORshiftLL_20(v)
	case OpARM64ORshiftRA:
		return rewriteValueARM64_OpARM64ORshiftRA_0(v)
	case OpARM64ORshiftRL:
		return rewriteValueARM64_OpARM64ORshiftRL_0(v)
	case OpARM64RORWconst:
		return rewriteValueARM64_OpARM64RORWconst_0(v)
	case OpARM64RORconst:
		return rewriteValueARM64_OpARM64RORconst_0(v)
	case OpARM64SBCSflags:
		return rewriteValueARM64_OpARM64SBCSflags_0(v)
	case OpARM64SLL:
		return rewriteValueARM64_OpARM64SLL_0(v)
	case OpARM64SLLconst:
		return rewriteValueARM64_OpARM64SLLconst_0(v)
	case OpARM64SRA:
		return rewriteValueARM64_OpARM64SRA_0(v)
	case OpARM64SRAconst:
		return rewriteValueARM64_OpARM64SRAconst_0(v)
	case OpARM64SRL:
		return rewriteValueARM64_OpARM64SRL_0(v)
	case OpARM64SRLconst:
		return rewriteValueARM64_OpARM64SRLconst_0(v) || rewriteValueARM64_OpARM64SRLconst_10(v)
	case OpARM64STP:
		return rewriteValueARM64_OpARM64STP_0(v)
	case OpARM64SUB:
		return rewriteValueARM64_OpARM64SUB_0(v) || rewriteValueARM64_OpARM64SUB_10(v)
	case OpARM64SUBconst:
		return rewriteValueARM64_OpARM64SUBconst_0(v)
	case OpARM64SUBshiftLL:
		return rewriteValueARM64_OpARM64SUBshiftLL_0(v)
	case OpARM64SUBshiftRA:
		return rewriteValueARM64_OpARM64SUBshiftRA_0(v)
	case OpARM64SUBshiftRL:
		return rewriteValueARM64_OpARM64SUBshiftRL_0(v)
	case OpARM64TST:
		return rewriteValueARM64_OpARM64TST_0(v)
	case OpARM64TSTW:
		return rewriteValueARM64_OpARM64TSTW_0(v)
	case OpARM64TSTWconst:
		return rewriteValueARM64_OpARM64TSTWconst_0(v)
	case OpARM64TSTconst:
		return rewriteValueARM64_OpARM64TSTconst_0(v)
	case OpARM64TSTshiftLL:
		return rewriteValueARM64_OpARM64TSTshiftLL_0(v)
	case OpARM64TSTshiftRA:
		return rewriteValueARM64_OpARM64TSTshiftRA_0(v)
	case OpARM64TSTshiftRL:
		return rewriteValueARM64_OpARM64TSTshiftRL_0(v)
	case OpARM64UBFIZ:
		return rewriteValueARM64_OpARM64UBFIZ_0(v)
	case OpARM64UBFX:
		return rewriteValueARM64_OpARM64UBFX_0(v)
	case OpARM64UDIV:
		return rewriteValueARM64_OpARM64UDIV_0(v)
	case OpARM64UDIVW:
		return rewriteValueARM64_OpARM64UDIVW_0(v)
	case OpARM64UMOD:
		return rewriteValueARM64_OpARM64UMOD_0(v)
	case OpARM64UMODW:
		return rewriteValueARM64_OpARM64UMODW_0(v)
	case OpARM64XOR:
		return rewriteValueARM64_OpARM64XOR_0(v) || rewriteValueARM64_OpARM64XOR_10(v)
	case OpARM64XORconst:
		return rewriteValueARM64_OpARM64XORconst_0(v)
	case OpARM64XORshiftLL:
		return rewriteValueARM64_OpARM64XORshiftLL_0(v)
	case OpARM64XORshiftRA:
		return rewriteValueARM64_OpARM64XORshiftRA_0(v)
	case OpARM64XORshiftRL:
		return rewriteValueARM64_OpARM64XORshiftRL_0(v)
	case OpAbs:
		return rewriteValueARM64_OpAbs_0(v)
	case OpAdd16:
		return rewriteValueARM64_OpAdd16_0(v)
	case OpAdd32:
		return rewriteValueARM64_OpAdd32_0(v)
	case OpAdd32F:
		return rewriteValueARM64_OpAdd32F_0(v)
	case OpAdd64:
		return rewriteValueARM64_OpAdd64_0(v)
	case OpAdd64F:
		return rewriteValueARM64_OpAdd64F_0(v)
	case OpAdd8:
		return rewriteValueARM64_OpAdd8_0(v)
	case OpAddPtr:
		return rewriteValueARM64_OpAddPtr_0(v)
	case OpAddr:
		return rewriteValueARM64_OpAddr_0(v)
	case OpAnd16:
		return rewriteValueARM64_OpAnd16_0(v)
	case OpAnd32:
		return rewriteValueARM64_OpAnd32_0(v)
	case OpAnd64:
		return rewriteValueARM64_OpAnd64_0(v)
	case OpAnd8:
		return rewriteValueARM64_OpAnd8_0(v)
	case OpAndB:
		return rewriteValueARM64_OpAndB_0(v)
	case OpAtomicAdd32:
		return rewriteValueARM64_OpAtomicAdd32_0(v)
	case OpAtomicAdd32Variant:
		return rewriteValueARM64_OpAtomicAdd32Variant_0(v)
	case OpAtomicAdd64:
		return rewriteValueARM64_OpAtomicAdd64_0(v)
	case OpAtomicAdd64Variant:
		return rewriteValueARM64_OpAtomicAdd64Variant_0(v)
	case OpAtomicAnd8:
		return rewriteValueARM64_OpAtomicAnd8_0(v)
	case OpAtomicCompareAndSwap32:
		return rewriteValueARM64_OpAtomicCompareAndSwap32_0(v)
	case OpAtomicCompareAndSwap64:
		return rewriteValueARM64_OpAtomicCompareAndSwap64_0(v)
	case OpAtomicExchange32:
		return rewriteValueARM64_OpAtomicExchange32_0(v)
	case OpAtomicExchange64:
		return rewriteValueARM64_OpAtomicExchange64_0(v)
	case OpAtomicLoad32:
		return rewriteValueARM64_OpAtomicLoad32_0(v)
	case OpAtomicLoad64:
		return rewriteValueARM64_OpAtomicLoad64_0(v)
	case OpAtomicLoad8:
		return rewriteValueARM64_OpAtomicLoad8_0(v)
	case OpAtomicLoadPtr:
		return rewriteValueARM64_OpAtomicLoadPtr_0(v)
	case OpAtomicOr8:
		return rewriteValueARM64_OpAtomicOr8_0(v)
	case OpAtomicStore32:
		return rewriteValueARM64_OpAtomicStore32_0(v)
	case OpAtomicStore64:
		return rewriteValueARM64_OpAtomicStore64_0(v)
	case OpAtomicStorePtrNoWB:
		return rewriteValueARM64_OpAtomicStorePtrNoWB_0(v)
	case OpAvg64u:
		return rewriteValueARM64_OpAvg64u_0(v)
	case OpBitLen32:
		return rewriteValueARM64_OpBitLen32_0(v)
	case OpBitLen64:
		return rewriteValueARM64_OpBitLen64_0(v)
	case OpBitRev16:
		return rewriteValueARM64_OpBitRev16_0(v)
	case OpBitRev32:
		return rewriteValueARM64_OpBitRev32_0(v)
	case OpBitRev64:
		return rewriteValueARM64_OpBitRev64_0(v)
	case OpBitRev8:
		return rewriteValueARM64_OpBitRev8_0(v)
	case OpBswap32:
		return rewriteValueARM64_OpBswap32_0(v)
	case OpBswap64:
		return rewriteValueARM64_OpBswap64_0(v)
	case OpCeil:
		return rewriteValueARM64_OpCeil_0(v)
	case OpClosureCall:
		return rewriteValueARM64_OpClosureCall_0(v)
	case OpCom16:
		return rewriteValueARM64_OpCom16_0(v)
	case OpCom32:
		return rewriteValueARM64_OpCom32_0(v)
	case OpCom64:
		return rewriteValueARM64_OpCom64_0(v)
	case OpCom8:
		return rewriteValueARM64_OpCom8_0(v)
	case OpCondSelect:
		return rewriteValueARM64_OpCondSelect_0(v)
	case OpConst16:
		return rewriteValueARM64_OpConst16_0(v)
	case OpConst32:
		return rewriteValueARM64_OpConst32_0(v)
	case OpConst32F:
		return rewriteValueARM64_OpConst32F_0(v)
	case OpConst64:
		return rewriteValueARM64_OpConst64_0(v)
	case OpConst64F:
		return rewriteValueARM64_OpConst64F_0(v)
	case OpConst8:
		return rewriteValueARM64_OpConst8_0(v)
	case OpConstBool:
		return rewriteValueARM64_OpConstBool_0(v)
	case OpConstNil:
		return rewriteValueARM64_OpConstNil_0(v)
	case OpCtz16:
		return rewriteValueARM64_OpCtz16_0(v)
	case OpCtz16NonZero:
		return rewriteValueARM64_OpCtz16NonZero_0(v)
	case OpCtz32:
		return rewriteValueARM64_OpCtz32_0(v)
	case OpCtz32NonZero:
		return rewriteValueARM64_OpCtz32NonZero_0(v)
	case OpCtz64:
		return rewriteValueARM64_OpCtz64_0(v)
	case OpCtz64NonZero:
		return rewriteValueARM64_OpCtz64NonZero_0(v)
	case OpCtz8:
		return rewriteValueARM64_OpCtz8_0(v)
	case OpCtz8NonZero:
		return rewriteValueARM64_OpCtz8NonZero_0(v)
	case OpCvt32Fto32:
		return rewriteValueARM64_OpCvt32Fto32_0(v)
	case OpCvt32Fto32U:
		return rewriteValueARM64_OpCvt32Fto32U_0(v)
	case OpCvt32Fto64:
		return rewriteValueARM64_OpCvt32Fto64_0(v)
	case OpCvt32Fto64F:
		return rewriteValueARM64_OpCvt32Fto64F_0(v)
	case OpCvt32Fto64U:
		return rewriteValueARM64_OpCvt32Fto64U_0(v)
	case OpCvt32Uto32F:
		return rewriteValueARM64_OpCvt32Uto32F_0(v)
	case OpCvt32Uto64F:
		return rewriteValueARM64_OpCvt32Uto64F_0(v)
	case OpCvt32to32F:
		return rewriteValueARM64_OpCvt32to32F_0(v)
	case OpCvt32to64F:
		return rewriteValueARM64_OpCvt32to64F_0(v)
	case OpCvt64Fto32:
		return rewriteValueARM64_OpCvt64Fto32_0(v)
	case OpCvt64Fto32F:
		return rewriteValueARM64_OpCvt64Fto32F_0(v)
	case OpCvt64Fto32U:
		return rewriteValueARM64_OpCvt64Fto32U_0(v)
	case OpCvt64Fto64:
		return rewriteValueARM64_OpCvt64Fto64_0(v)
	case OpCvt64Fto64U:
		return rewriteValueARM64_OpCvt64Fto64U_0(v)
	case OpCvt64Uto32F:
		return rewriteValueARM64_OpCvt64Uto32F_0(v)
	case OpCvt64Uto64F:
		return rewriteValueARM64_OpCvt64Uto64F_0(v)
	case OpCvt64to32F:
		return rewriteValueARM64_OpCvt64to32F_0(v)
	case OpCvt64to64F:
		return rewriteValueARM64_OpCvt64to64F_0(v)
	case OpDiv16:
		return rewriteValueARM64_OpDiv16_0(v)
	case OpDiv16u:
		return rewriteValueARM64_OpDiv16u_0(v)
	case OpDiv32:
		return rewriteValueARM64_OpDiv32_0(v)
	case OpDiv32F:
		return rewriteValueARM64_OpDiv32F_0(v)
	case OpDiv32u:
		return rewriteValueARM64_OpDiv32u_0(v)
	case OpDiv64:
		return rewriteValueARM64_OpDiv64_0(v)
	case OpDiv64F:
		return rewriteValueARM64_OpDiv64F_0(v)
	case OpDiv64u:
		return rewriteValueARM64_OpDiv64u_0(v)
	case OpDiv8:
		return rewriteValueARM64_OpDiv8_0(v)
	case OpDiv8u:
		return rewriteValueARM64_OpDiv8u_0(v)
	case OpEq16:
		return rewriteValueARM64_OpEq16_0(v)
	case OpEq32:
		return rewriteValueARM64_OpEq32_0(v)
	case OpEq32F:
		return rewriteValueARM64_OpEq32F_0(v)
	case OpEq64:
		return rewriteValueARM64_OpEq64_0(v)
	case OpEq64F:
		return rewriteValueARM64_OpEq64F_0(v)
	case OpEq8:
		return rewriteValueARM64_OpEq8_0(v)
	case OpEqB:
		return rewriteValueARM64_OpEqB_0(v)
	case OpEqPtr:
		return rewriteValueARM64_OpEqPtr_0(v)
	case OpFloor:
		return rewriteValueARM64_OpFloor_0(v)
	case OpGeq16:
		return rewriteValueARM64_OpGeq16_0(v)
	case OpGeq16U:
		return rewriteValueARM64_OpGeq16U_0(v)
	case OpGeq32:
		return rewriteValueARM64_OpGeq32_0(v)
	case OpGeq32F:
		return rewriteValueARM64_OpGeq32F_0(v)
	case OpGeq32U:
		return rewriteValueARM64_OpGeq32U_0(v)
	case OpGeq64:
		return rewriteValueARM64_OpGeq64_0(v)
	case OpGeq64F:
		return rewriteValueARM64_OpGeq64F_0(v)
	case OpGeq64U:
		return rewriteValueARM64_OpGeq64U_0(v)
	case OpGeq8:
		return rewriteValueARM64_OpGeq8_0(v)
	case OpGeq8U:
		return rewriteValueARM64_OpGeq8U_0(v)
	case OpGetCallerPC:
		return rewriteValueARM64_OpGetCallerPC_0(v)
	case OpGetCallerSP:
		return rewriteValueARM64_OpGetCallerSP_0(v)
	case OpGetClosurePtr:
		return rewriteValueARM64_OpGetClosurePtr_0(v)
	case OpGreater16:
		return rewriteValueARM64_OpGreater16_0(v)
	case OpGreater16U:
		return rewriteValueARM64_OpGreater16U_0(v)
	case OpGreater32:
		return rewriteValueARM64_OpGreater32_0(v)
	case OpGreater32F:
		return rewriteValueARM64_OpGreater32F_0(v)
	case OpGreater32U:
		return rewriteValueARM64_OpGreater32U_0(v)
	case OpGreater64:
		return rewriteValueARM64_OpGreater64_0(v)
	case OpGreater64F:
		return rewriteValueARM64_OpGreater64F_0(v)
	case OpGreater64U:
		return rewriteValueARM64_OpGreater64U_0(v)
	case OpGreater8:
		return rewriteValueARM64_OpGreater8_0(v)
	case OpGreater8U:
		return rewriteValueARM64_OpGreater8U_0(v)
	case OpHmul32:
		return rewriteValueARM64_OpHmul32_0(v)
	case OpHmul32u:
		return rewriteValueARM64_OpHmul32u_0(v)
	case OpHmul64:
		return rewriteValueARM64_OpHmul64_0(v)
	case OpHmul64u:
		return rewriteValueARM64_OpHmul64u_0(v)
	case OpInterCall:
		return rewriteValueARM64_OpInterCall_0(v)
	case OpIsInBounds:
		return rewriteValueARM64_OpIsInBounds_0(v)
	case OpIsNonNil:
		return rewriteValueARM64_OpIsNonNil_0(v)
	case OpIsSliceInBounds:
		return rewriteValueARM64_OpIsSliceInBounds_0(v)
	case OpLeq16:
		return rewriteValueARM64_OpLeq16_0(v)
	case OpLeq16U:
		return rewriteValueARM64_OpLeq16U_0(v)
	case OpLeq32:
		return rewriteValueARM64_OpLeq32_0(v)
	case OpLeq32F:
		return rewriteValueARM64_OpLeq32F_0(v)
	case OpLeq32U:
		return rewriteValueARM64_OpLeq32U_0(v)
	case OpLeq64:
		return rewriteValueARM64_OpLeq64_0(v)
	case OpLeq64F:
		return rewriteValueARM64_OpLeq64F_0(v)
	case OpLeq64U:
		return rewriteValueARM64_OpLeq64U_0(v)
	case OpLeq8:
		return rewriteValueARM64_OpLeq8_0(v)
	case OpLeq8U:
		return rewriteValueARM64_OpLeq8U_0(v)
	case OpLess16:
		return rewriteValueARM64_OpLess16_0(v)
	case OpLess16U:
		return rewriteValueARM64_OpLess16U_0(v)
	case OpLess32:
		return rewriteValueARM64_OpLess32_0(v)
	case OpLess32F:
		return rewriteValueARM64_OpLess32F_0(v)
	case OpLess32U:
		return rewriteValueARM64_OpLess32U_0(v)
	case OpLess64:
		return rewriteValueARM64_OpLess64_0(v)
	case OpLess64F:
		return rewriteValueARM64_OpLess64F_0(v)
	case OpLess64U:
		return rewriteValueARM64_OpLess64U_0(v)
	case OpLess8:
		return rewriteValueARM64_OpLess8_0(v)
	case OpLess8U:
		return rewriteValueARM64_OpLess8U_0(v)
	case OpLoad:
		return rewriteValueARM64_OpLoad_0(v)
	case OpLocalAddr:
		return rewriteValueARM64_OpLocalAddr_0(v)
	case OpLsh16x16:
		return rewriteValueARM64_OpLsh16x16_0(v)
	case OpLsh16x32:
		return rewriteValueARM64_OpLsh16x32_0(v)
	case OpLsh16x64:
		return rewriteValueARM64_OpLsh16x64_0(v)
	case OpLsh16x8:
		return rewriteValueARM64_OpLsh16x8_0(v)
	case OpLsh32x16:
		return rewriteValueARM64_OpLsh32x16_0(v)
	case OpLsh32x32:
		return rewriteValueARM64_OpLsh32x32_0(v)
	case OpLsh32x64:
		return rewriteValueARM64_OpLsh32x64_0(v)
	case OpLsh32x8:
		return rewriteValueARM64_OpLsh32x8_0(v)
	case OpLsh64x16:
		return rewriteValueARM64_OpLsh64x16_0(v)
	case OpLsh64x32:
		return rewriteValueARM64_OpLsh64x32_0(v)
	case OpLsh64x64:
		return rewriteValueARM64_OpLsh64x64_0(v)
	case OpLsh64x8:
		return rewriteValueARM64_OpLsh64x8_0(v)
	case OpLsh8x16:
		return rewriteValueARM64_OpLsh8x16_0(v)
	case OpLsh8x32:
		return rewriteValueARM64_OpLsh8x32_0(v)
	case OpLsh8x64:
		return rewriteValueARM64_OpLsh8x64_0(v)
	case OpLsh8x8:
		return rewriteValueARM64_OpLsh8x8_0(v)
	case OpMod16:
		return rewriteValueARM64_OpMod16_0(v)
	case OpMod16u:
		return rewriteValueARM64_OpMod16u_0(v)
	case OpMod32:
		return rewriteValueARM64_OpMod32_0(v)
	case OpMod32u:
		return rewriteValueARM64_OpMod32u_0(v)
	case OpMod64:
		return rewriteValueARM64_OpMod64_0(v)
	case OpMod64u:
		return rewriteValueARM64_OpMod64u_0(v)
	case OpMod8:
		return rewriteValueARM64_OpMod8_0(v)
	case OpMod8u:
		return rewriteValueARM64_OpMod8u_0(v)
	case OpMove:
		return rewriteValueARM64_OpMove_0(v) || rewriteValueARM64_OpMove_10(v)
	case OpMul16:
		return rewriteValueARM64_OpMul16_0(v)
	case OpMul32:
		return rewriteValueARM64_OpMul32_0(v)
	case OpMul32F:
		return rewriteValueARM64_OpMul32F_0(v)
	case OpMul64:
		return rewriteValueARM64_OpMul64_0(v)
	case OpMul64F:
		return rewriteValueARM64_OpMul64F_0(v)
	case OpMul64uhilo:
		return rewriteValueARM64_OpMul64uhilo_0(v)
	case OpMul8:
		return rewriteValueARM64_OpMul8_0(v)
	case OpNeg16:
		return rewriteValueARM64_OpNeg16_0(v)
	case OpNeg32:
		return rewriteValueARM64_OpNeg32_0(v)
	case OpNeg32F:
		return rewriteValueARM64_OpNeg32F_0(v)
	case OpNeg64:
		return rewriteValueARM64_OpNeg64_0(v)
	case OpNeg64F:
		return rewriteValueARM64_OpNeg64F_0(v)
	case OpNeg8:
		return rewriteValueARM64_OpNeg8_0(v)
	case OpNeq16:
		return rewriteValueARM64_OpNeq16_0(v)
	case OpNeq32:
		return rewriteValueARM64_OpNeq32_0(v)
	case OpNeq32F:
		return rewriteValueARM64_OpNeq32F_0(v)
	case OpNeq64:
		return rewriteValueARM64_OpNeq64_0(v)
	case OpNeq64F:
		return rewriteValueARM64_OpNeq64F_0(v)
	case OpNeq8:
		return rewriteValueARM64_OpNeq8_0(v)
	case OpNeqB:
		return rewriteValueARM64_OpNeqB_0(v)
	case OpNeqPtr:
		return rewriteValueARM64_OpNeqPtr_0(v)
	case OpNilCheck:
		return rewriteValueARM64_OpNilCheck_0(v)
	case OpNot:
		return rewriteValueARM64_OpNot_0(v)
	case OpOffPtr:
		return rewriteValueARM64_OpOffPtr_0(v)
	case OpOr16:
		return rewriteValueARM64_OpOr16_0(v)
	case OpOr32:
		return rewriteValueARM64_OpOr32_0(v)
	case OpOr64:
		return rewriteValueARM64_OpOr64_0(v)
	case OpOr8:
		return rewriteValueARM64_OpOr8_0(v)
	case OpOrB:
		return rewriteValueARM64_OpOrB_0(v)
	case OpPanicBounds:
		return rewriteValueARM64_OpPanicBounds_0(v)
	case OpPopCount16:
		return rewriteValueARM64_OpPopCount16_0(v)
	case OpPopCount32:
		return rewriteValueARM64_OpPopCount32_0(v)
	case OpPopCount64:
		return rewriteValueARM64_OpPopCount64_0(v)
	case OpRotateLeft16:
		return rewriteValueARM64_OpRotateLeft16_0(v)
	case OpRotateLeft32:
		return rewriteValueARM64_OpRotateLeft32_0(v)
	case OpRotateLeft64:
		return rewriteValueARM64_OpRotateLeft64_0(v)
	case OpRotateLeft8:
		return rewriteValueARM64_OpRotateLeft8_0(v)
	case OpRound:
		return rewriteValueARM64_OpRound_0(v)
	case OpRound32F:
		return rewriteValueARM64_OpRound32F_0(v)
	case OpRound64F:
		return rewriteValueARM64_OpRound64F_0(v)
	case OpRoundToEven:
		return rewriteValueARM64_OpRoundToEven_0(v)
	case OpRsh16Ux16:
		return rewriteValueARM64_OpRsh16Ux16_0(v)
	case OpRsh16Ux32:
		return rewriteValueARM64_OpRsh16Ux32_0(v)
	case OpRsh16Ux64:
		return rewriteValueARM64_OpRsh16Ux64_0(v)
	case OpRsh16Ux8:
		return rewriteValueARM64_OpRsh16Ux8_0(v)
	case OpRsh16x16:
		return rewriteValueARM64_OpRsh16x16_0(v)
	case OpRsh16x32:
		return rewriteValueARM64_OpRsh16x32_0(v)
	case OpRsh16x64:
		return rewriteValueARM64_OpRsh16x64_0(v)
	case OpRsh16x8:
		return rewriteValueARM64_OpRsh16x8_0(v)
	case OpRsh32Ux16:
		return rewriteValueARM64_OpRsh32Ux16_0(v)
	case OpRsh32Ux32:
		return rewriteValueARM64_OpRsh32Ux32_0(v)
	case OpRsh32Ux64:
		return rewriteValueARM64_OpRsh32Ux64_0(v)
	case OpRsh32Ux8:
		return rewriteValueARM64_OpRsh32Ux8_0(v)
	case OpRsh32x16:
		return rewriteValueARM64_OpRsh32x16_0(v)
	case OpRsh32x32:
		return rewriteValueARM64_OpRsh32x32_0(v)
	case OpRsh32x64:
		return rewriteValueARM64_OpRsh32x64_0(v)
	case OpRsh32x8:
		return rewriteValueARM64_OpRsh32x8_0(v)
	case OpRsh64Ux16:
		return rewriteValueARM64_OpRsh64Ux16_0(v)
	case OpRsh64Ux32:
		return rewriteValueARM64_OpRsh64Ux32_0(v)
	case OpRsh64Ux64:
		return rewriteValueARM64_OpRsh64Ux64_0(v)
	case OpRsh64Ux8:
		return rewriteValueARM64_OpRsh64Ux8_0(v)
	case OpRsh64x16:
		return rewriteValueARM64_OpRsh64x16_0(v)
	case OpRsh64x32:
		return rewriteValueARM64_OpRsh64x32_0(v)
	case OpRsh64x64:
		return rewriteValueARM64_OpRsh64x64_0(v)
	case OpRsh64x8:
		return rewriteValueARM64_OpRsh64x8_0(v)
	case OpRsh8Ux16:
		return rewriteValueARM64_OpRsh8Ux16_0(v)
	case OpRsh8Ux32:
		return rewriteValueARM64_OpRsh8Ux32_0(v)
	case OpRsh8Ux64:
		return rewriteValueARM64_OpRsh8Ux64_0(v)
	case OpRsh8Ux8:
		return rewriteValueARM64_OpRsh8Ux8_0(v)
	case OpRsh8x16:
		return rewriteValueARM64_OpRsh8x16_0(v)
	case OpRsh8x32:
		return rewriteValueARM64_OpRsh8x32_0(v)
	case OpRsh8x64:
		return rewriteValueARM64_OpRsh8x64_0(v)
	case OpRsh8x8:
		return rewriteValueARM64_OpRsh8x8_0(v)
	case OpSelect0:
		return rewriteValueARM64_OpSelect0_0(v)
	case OpSelect1:
		return rewriteValueARM64_OpSelect1_0(v)
	case OpSignExt16to32:
		return rewriteValueARM64_OpSignExt16to32_0(v)
	case OpSignExt16to64:
		return rewriteValueARM64_OpSignExt16to64_0(v)
	case OpSignExt32to64:
		return rewriteValueARM64_OpSignExt32to64_0(v)
	case OpSignExt8to16:
		return rewriteValueARM64_OpSignExt8to16_0(v)
	case OpSignExt8to32:
		return rewriteValueARM64_OpSignExt8to32_0(v)
	case OpSignExt8to64:
		return rewriteValueARM64_OpSignExt8to64_0(v)
	case OpSlicemask:
		return rewriteValueARM64_OpSlicemask_0(v)
	case OpSqrt:
		return rewriteValueARM64_OpSqrt_0(v)
	case OpStaticCall:
		return rewriteValueARM64_OpStaticCall_0(v)
	case OpStore:
		return rewriteValueARM64_OpStore_0(v)
	case OpSub16:
		return rewriteValueARM64_OpSub16_0(v)
	case OpSub32:
		return rewriteValueARM64_OpSub32_0(v)
	case OpSub32F:
		return rewriteValueARM64_OpSub32F_0(v)
	case OpSub64:
		return rewriteValueARM64_OpSub64_0(v)
	case OpSub64F:
		return rewriteValueARM64_OpSub64F_0(v)
	case OpSub8:
		return rewriteValueARM64_OpSub8_0(v)
	case OpSubPtr:
		return rewriteValueARM64_OpSubPtr_0(v)
	case OpTrunc:
		return rewriteValueARM64_OpTrunc_0(v)
	case OpTrunc16to8:
		return rewriteValueARM64_OpTrunc16to8_0(v)
	case OpTrunc32to16:
		return rewriteValueARM64_OpTrunc32to16_0(v)
	case OpTrunc32to8:
		return rewriteValueARM64_OpTrunc32to8_0(v)
	case OpTrunc64to16:
		return rewriteValueARM64_OpTrunc64to16_0(v)
	case OpTrunc64to32:
		return rewriteValueARM64_OpTrunc64to32_0(v)
	case OpTrunc64to8:
		return rewriteValueARM64_OpTrunc64to8_0(v)
	case OpWB:
		return rewriteValueARM64_OpWB_0(v)
	case OpXor16:
		return rewriteValueARM64_OpXor16_0(v)
	case OpXor32:
		return rewriteValueARM64_OpXor32_0(v)
	case OpXor64:
		return rewriteValueARM64_OpXor64_0(v)
	case OpXor8:
		return rewriteValueARM64_OpXor8_0(v)
	case OpZero:
		return rewriteValueARM64_OpZero_0(v) || rewriteValueARM64_OpZero_10(v) || rewriteValueARM64_OpZero_20(v)
	case OpZeroExt16to32:
		return rewriteValueARM64_OpZeroExt16to32_0(v)
	case OpZeroExt16to64:
		return rewriteValueARM64_OpZeroExt16to64_0(v)
	case OpZeroExt32to64:
		return rewriteValueARM64_OpZeroExt32to64_0(v)
	case OpZeroExt8to16:
		return rewriteValueARM64_OpZeroExt8to16_0(v)
	case OpZeroExt8to32:
		return rewriteValueARM64_OpZeroExt8to32_0(v)
	case OpZeroExt8to64:
		return rewriteValueARM64_OpZeroExt8to64_0(v)
	}
	return false
}
func rewriteValueARM64_OpARM64ADCSflags_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADCSflags x y (Select1 <types.TypeFlags> (ADDSconstflags [-1] (ADCzerocarry <typ.UInt64> c))))
	// cond:
	// result: (ADCSflags x y c)
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpSelect1 {
			break
		}
		if v_2.Type != types.TypeFlags {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpARM64ADDSconstflags {
			break
		}
		if v_2_0.AuxInt != -1 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpARM64ADCzerocarry {
			break
		}
		if v_2_0_0.Type != typ.UInt64 {
			break
		}
		c := v_2_0_0.Args[0]
		v.reset(OpARM64ADCSflags)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(c)
		return true
	}
	// match: (ADCSflags x y (Select1 <types.TypeFlags> (ADDSconstflags [-1] (MOVDconst [0]))))
	// cond:
	// result: (ADDSflags x y)
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpSelect1 {
			break
		}
		if v_2.Type != types.TypeFlags {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpARM64ADDSconstflags {
			break
		}
		if v_2_0.AuxInt != -1 {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_2_0_0.AuxInt != 0 {
			break
		}
		v.reset(OpARM64ADDSflags)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADD_0(v *Value) bool {
	// match: (ADD x (MOVDconst [c]))
	// cond:
	// result: (ADDconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ADDconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADD (MOVDconst [c]) x)
	// cond:
	// result: (ADDconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64ADDconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADD a l:(MUL x y))
	// cond: l.Uses==1 && clobber(l)
	// result: (MADD a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		l := v.Args[1]
		if l.Op != OpARM64MUL {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MADD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD l:(MUL x y) a)
	// cond: l.Uses==1 && clobber(l)
	// result: (MADD a x y)
	for {
		a := v.Args[1]
		l := v.Args[0]
		if l.Op != OpARM64MUL {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MADD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD a l:(MNEG x y))
	// cond: l.Uses==1 && clobber(l)
	// result: (MSUB a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		l := v.Args[1]
		if l.Op != OpARM64MNEG {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MSUB)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD l:(MNEG x y) a)
	// cond: l.Uses==1 && clobber(l)
	// result: (MSUB a x y)
	for {
		a := v.Args[1]
		l := v.Args[0]
		if l.Op != OpARM64MNEG {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MSUB)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD a l:(MULW x y))
	// cond: a.Type.Size() != 8 && l.Uses==1 && clobber(l)
	// result: (MADDW a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		l := v.Args[1]
		if l.Op != OpARM64MULW {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(a.Type.Size() != 8 && l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MADDW)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD l:(MULW x y) a)
	// cond: a.Type.Size() != 8 && l.Uses==1 && clobber(l)
	// result: (MADDW a x y)
	for {
		a := v.Args[1]
		l := v.Args[0]
		if l.Op != OpARM64MULW {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(a.Type.Size() != 8 && l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MADDW)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD a l:(MNEGW x y))
	// cond: a.Type.Size() != 8 && l.Uses==1 && clobber(l)
	// result: (MSUBW a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		l := v.Args[1]
		if l.Op != OpARM64MNEGW {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(a.Type.Size() != 8 && l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MSUBW)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD l:(MNEGW x y) a)
	// cond: a.Type.Size() != 8 && l.Uses==1 && clobber(l)
	// result: (MSUBW a x y)
	for {
		a := v.Args[1]
		l := v.Args[0]
		if l.Op != OpARM64MNEGW {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(a.Type.Size() != 8 && l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MSUBW)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADD_10(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADD x (NEG y))
	// cond:
	// result: (SUB x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64NEG {
			break
		}
		y := v_1.Args[0]
		v.reset(OpARM64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD (NEG y) x)
	// cond:
	// result: (SUB x y)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64NEG {
			break
		}
		y := v_0.Args[0]
		v.reset(OpARM64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ADDshiftLL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (ADD x1:(SLLconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (ADDshiftLL x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (ADD x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ADDshiftRL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ADDshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (ADD x1:(SRLconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (ADDshiftRL x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ADDshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (ADD x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ADDshiftRA x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ADDshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (ADD x1:(SRAconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (ADDshiftRA x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ADDshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (ADD (SLL x (ANDconst <t> [63] y)) (CSEL0 <typ.UInt64> {cc} (SRL <typ.UInt64> x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (ROR x (NEG <t> y))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64ANDconst {
			break
		}
		t := v_0_1.Type
		if v_0_1.AuxInt != 63 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CSEL0 {
			break
		}
		if v_1.Type != typ.UInt64 {
			break
		}
		cc := v_1.Aux
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64SRL {
			break
		}
		if v_1_0.Type != typ.UInt64 {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpARM64SUB {
			break
		}
		if v_1_0_1.Type != t {
			break
		}
		_ = v_1_0_1.Args[1]
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_0_1_0.AuxInt != 64 {
			break
		}
		v_1_0_1_1 := v_1_0_1.Args[1]
		if v_1_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_0_1_1.Type != t {
			break
		}
		if v_1_0_1_1.AuxInt != 63 {
			break
		}
		if y != v_1_0_1_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64CMPconst {
			break
		}
		if v_1_1.AuxInt != 64 {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpARM64SUB {
			break
		}
		if v_1_1_0.Type != t {
			break
		}
		_ = v_1_1_0.Args[1]
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_1_0_0.AuxInt != 64 {
			break
		}
		v_1_1_0_1 := v_1_1_0.Args[1]
		if v_1_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1_0_1.Type != t {
			break
		}
		if v_1_1_0_1.AuxInt != 63 {
			break
		}
		if y != v_1_1_0_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ADD (CSEL0 <typ.UInt64> {cc} (SRL <typ.UInt64> x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))) (SLL x (ANDconst <t> [63] y)))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (ROR x (NEG <t> y))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64CSEL0 {
			break
		}
		if v_0.Type != typ.UInt64 {
			break
		}
		cc := v_0.Aux
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64SRL {
			break
		}
		if v_0_0.Type != typ.UInt64 {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpARM64SUB {
			break
		}
		t := v_0_0_1.Type
		_ = v_0_0_1.Args[1]
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_0_1_0.AuxInt != 64 {
			break
		}
		v_0_0_1_1 := v_0_0_1.Args[1]
		if v_0_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_0_1_1.Type != t {
			break
		}
		if v_0_0_1_1.AuxInt != 63 {
			break
		}
		y := v_0_0_1_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64CMPconst {
			break
		}
		if v_0_1.AuxInt != 64 {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpARM64SUB {
			break
		}
		if v_0_1_0.Type != t {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_1_0_0.AuxInt != 64 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_1_0_1.Type != t {
			break
		}
		if v_0_1_0_1.AuxInt != 63 {
			break
		}
		if y != v_0_1_0_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1.Type != t {
			break
		}
		if v_1_1.AuxInt != 63 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADD_20(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADD (SRL <typ.UInt64> x (ANDconst <t> [63] y)) (CSEL0 <typ.UInt64> {cc} (SLL x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (ROR x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRL {
			break
		}
		if v_0.Type != typ.UInt64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64ANDconst {
			break
		}
		t := v_0_1.Type
		if v_0_1.AuxInt != 63 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CSEL0 {
			break
		}
		if v_1.Type != typ.UInt64 {
			break
		}
		cc := v_1.Aux
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64SLL {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpARM64SUB {
			break
		}
		if v_1_0_1.Type != t {
			break
		}
		_ = v_1_0_1.Args[1]
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_0_1_0.AuxInt != 64 {
			break
		}
		v_1_0_1_1 := v_1_0_1.Args[1]
		if v_1_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_0_1_1.Type != t {
			break
		}
		if v_1_0_1_1.AuxInt != 63 {
			break
		}
		if y != v_1_0_1_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64CMPconst {
			break
		}
		if v_1_1.AuxInt != 64 {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpARM64SUB {
			break
		}
		if v_1_1_0.Type != t {
			break
		}
		_ = v_1_1_0.Args[1]
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_1_0_0.AuxInt != 64 {
			break
		}
		v_1_1_0_1 := v_1_1_0.Args[1]
		if v_1_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1_0_1.Type != t {
			break
		}
		if v_1_1_0_1.AuxInt != 63 {
			break
		}
		if y != v_1_1_0_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD (CSEL0 <typ.UInt64> {cc} (SLL x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))) (SRL <typ.UInt64> x (ANDconst <t> [63] y)))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (ROR x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64CSEL0 {
			break
		}
		if v_0.Type != typ.UInt64 {
			break
		}
		cc := v_0.Aux
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64SLL {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpARM64SUB {
			break
		}
		t := v_0_0_1.Type
		_ = v_0_0_1.Args[1]
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_0_1_0.AuxInt != 64 {
			break
		}
		v_0_0_1_1 := v_0_0_1.Args[1]
		if v_0_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_0_1_1.Type != t {
			break
		}
		if v_0_0_1_1.AuxInt != 63 {
			break
		}
		y := v_0_0_1_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64CMPconst {
			break
		}
		if v_0_1.AuxInt != 64 {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpARM64SUB {
			break
		}
		if v_0_1_0.Type != t {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_1_0_0.AuxInt != 64 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_1_0_1.Type != t {
			break
		}
		if v_0_1_0_1.AuxInt != 63 {
			break
		}
		if y != v_0_1_0_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRL {
			break
		}
		if v_1.Type != typ.UInt64 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1.Type != t {
			break
		}
		if v_1_1.AuxInt != 63 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD (SLL x (ANDconst <t> [31] y)) (CSEL0 <typ.UInt32> {cc} (SRL <typ.UInt32> (MOVWUreg x) (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (RORW x (NEG <t> y))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64ANDconst {
			break
		}
		t := v_0_1.Type
		if v_0_1.AuxInt != 31 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CSEL0 {
			break
		}
		if v_1.Type != typ.UInt32 {
			break
		}
		cc := v_1.Aux
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64SRL {
			break
		}
		if v_1_0.Type != typ.UInt32 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpARM64MOVWUreg {
			break
		}
		if x != v_1_0_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpARM64SUB {
			break
		}
		if v_1_0_1.Type != t {
			break
		}
		_ = v_1_0_1.Args[1]
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_0_1_0.AuxInt != 32 {
			break
		}
		v_1_0_1_1 := v_1_0_1.Args[1]
		if v_1_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_0_1_1.Type != t {
			break
		}
		if v_1_0_1_1.AuxInt != 31 {
			break
		}
		if y != v_1_0_1_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64CMPconst {
			break
		}
		if v_1_1.AuxInt != 64 {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpARM64SUB {
			break
		}
		if v_1_1_0.Type != t {
			break
		}
		_ = v_1_1_0.Args[1]
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_1_0_0.AuxInt != 32 {
			break
		}
		v_1_1_0_1 := v_1_1_0.Args[1]
		if v_1_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1_0_1.Type != t {
			break
		}
		if v_1_1_0_1.AuxInt != 31 {
			break
		}
		if y != v_1_1_0_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ADD (CSEL0 <typ.UInt32> {cc} (SRL <typ.UInt32> (MOVWUreg x) (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))) (SLL x (ANDconst <t> [31] y)))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (RORW x (NEG <t> y))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64CSEL0 {
			break
		}
		if v_0.Type != typ.UInt32 {
			break
		}
		cc := v_0.Aux
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64SRL {
			break
		}
		if v_0_0.Type != typ.UInt32 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpARM64SUB {
			break
		}
		t := v_0_0_1.Type
		_ = v_0_0_1.Args[1]
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_0_1_0.AuxInt != 32 {
			break
		}
		v_0_0_1_1 := v_0_0_1.Args[1]
		if v_0_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_0_1_1.Type != t {
			break
		}
		if v_0_0_1_1.AuxInt != 31 {
			break
		}
		y := v_0_0_1_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64CMPconst {
			break
		}
		if v_0_1.AuxInt != 64 {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpARM64SUB {
			break
		}
		if v_0_1_0.Type != t {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_1_0_0.AuxInt != 32 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_1_0_1.Type != t {
			break
		}
		if v_0_1_0_1.AuxInt != 31 {
			break
		}
		if y != v_0_1_0_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1.Type != t {
			break
		}
		if v_1_1.AuxInt != 31 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ADD (SRL <typ.UInt32> (MOVWUreg x) (ANDconst <t> [31] y)) (CSEL0 <typ.UInt32> {cc} (SLL x (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (RORW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRL {
			break
		}
		if v_0.Type != typ.UInt32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64ANDconst {
			break
		}
		t := v_0_1.Type
		if v_0_1.AuxInt != 31 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CSEL0 {
			break
		}
		if v_1.Type != typ.UInt32 {
			break
		}
		cc := v_1.Aux
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64SLL {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpARM64SUB {
			break
		}
		if v_1_0_1.Type != t {
			break
		}
		_ = v_1_0_1.Args[1]
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_0_1_0.AuxInt != 32 {
			break
		}
		v_1_0_1_1 := v_1_0_1.Args[1]
		if v_1_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_0_1_1.Type != t {
			break
		}
		if v_1_0_1_1.AuxInt != 31 {
			break
		}
		if y != v_1_0_1_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64CMPconst {
			break
		}
		if v_1_1.AuxInt != 64 {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpARM64SUB {
			break
		}
		if v_1_1_0.Type != t {
			break
		}
		_ = v_1_1_0.Args[1]
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_1_0_0.AuxInt != 32 {
			break
		}
		v_1_1_0_1 := v_1_1_0.Args[1]
		if v_1_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1_0_1.Type != t {
			break
		}
		if v_1_1_0_1.AuxInt != 31 {
			break
		}
		if y != v_1_1_0_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD (CSEL0 <typ.UInt32> {cc} (SLL x (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))) (SRL <typ.UInt32> (MOVWUreg x) (ANDconst <t> [31] y)))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (RORW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64CSEL0 {
			break
		}
		if v_0.Type != typ.UInt32 {
			break
		}
		cc := v_0.Aux
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64SLL {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpARM64SUB {
			break
		}
		t := v_0_0_1.Type
		_ = v_0_0_1.Args[1]
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_0_1_0.AuxInt != 32 {
			break
		}
		v_0_0_1_1 := v_0_0_1.Args[1]
		if v_0_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_0_1_1.Type != t {
			break
		}
		if v_0_0_1_1.AuxInt != 31 {
			break
		}
		y := v_0_0_1_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64CMPconst {
			break
		}
		if v_0_1.AuxInt != 64 {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpARM64SUB {
			break
		}
		if v_0_1_0.Type != t {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_1_0_0.AuxInt != 32 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_1_0_1.Type != t {
			break
		}
		if v_0_1_0_1.AuxInt != 31 {
			break
		}
		if y != v_0_1_0_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRL {
			break
		}
		if v_1.Type != typ.UInt32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVWUreg {
			break
		}
		if x != v_1_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1.Type != t {
			break
		}
		if v_1_1.AuxInt != 31 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADDconst_0(v *Value) bool {
	// match: (ADDconst [off1] (MOVDaddr [off2] {sym} ptr))
	// cond:
	// result: (MOVDaddr [off1+off2] {sym} ptr)
	for {
		off1 := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym := v_0.Aux
		ptr := v_0.Args[0]
		v.reset(OpARM64MOVDaddr)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		return true
	}
	// match: (ADDconst [0] x)
	// cond:
	// result: x
	for {
		if v.AuxInt != 0 {
			break
		}
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [c+d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = c + d
		return true
	}
	// match: (ADDconst [c] (ADDconst [d] x))
	// cond:
	// result: (ADDconst [c+d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARM64ADDconst)
		v.AuxInt = c + d
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (SUBconst [d] x))
	// cond:
	// result: (ADDconst [c-d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SUBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARM64ADDconst)
		v.AuxInt = c - d
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADDshiftLL_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDshiftLL (MOVDconst [c]) x [d])
	// cond:
	// result: (ADDconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64ADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftLL x (MOVDconst [c]) [d])
	// cond:
	// result: (ADDconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64(uint64(c) << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL [c] (SRLconst x [64-c]) x)
	// cond:
	// result: (RORconst [64-c] x)
	for {
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = 64 - c
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL <t> [c] (UBFX [bfc] x) x)
	// cond: c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)
	// result: (RORWconst [32-c] x)
	for {
		t := v.Type
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := v_0.AuxInt
		if x != v_0.Args[0] {
			break
		}
		if !(c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = 32 - c
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL <typ.UInt16> [8] (UBFX <typ.UInt16> [armBFAuxInt(8, 8)] x) x)
	// cond:
	// result: (REV16W x)
	for {
		if v.Type != typ.UInt16 {
			break
		}
		if v.AuxInt != 8 {
			break
		}
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFX {
			break
		}
		if v_0.Type != typ.UInt16 {
			break
		}
		if v_0.AuxInt != armBFAuxInt(8, 8) {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL [c] (SRLconst x [64-c]) x2)
	// cond:
	// result: (EXTRconst [64-c] x2 x)
	for {
		c := v.AuxInt
		x2 := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64EXTRconst)
		v.AuxInt = 64 - c
		v.AddArg(x2)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL <t> [c] (UBFX [bfc] x) x2)
	// cond: c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)
	// result: (EXTRWconst [32-c] x2 x)
	for {
		t := v.Type
		c := v.AuxInt
		x2 := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		if !(c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)) {
			break
		}
		v.reset(OpARM64EXTRWconst)
		v.AuxInt = 32 - c
		v.AddArg(x2)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADDshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (ADDshiftRA (MOVDconst [c]) x [d])
	// cond:
	// result: (ADDconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64ADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRA x (MOVDconst [c]) [d])
	// cond:
	// result: (ADDconst x [c>>uint64(d)])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ADDconst)
		v.AuxInt = c >> uint64(d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADDshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (ADDshiftRL (MOVDconst [c]) x [d])
	// cond:
	// result: (ADDconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64ADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRL x (MOVDconst [c]) [d])
	// cond:
	// result: (ADDconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64(uint64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftRL [c] (SLLconst x [64-c]) x)
	// cond:
	// result: (RORconst [ c] x)
	for {
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftRL <t> [c] (SLLconst x [32-c]) (MOVWUreg x))
	// cond: c < 32 && t.Size() == 4
	// result: (RORWconst [c] x)
	for {
		t := v.Type
		c := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 32-c {
			break
		}
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWUreg {
			break
		}
		if x != v_1.Args[0] {
			break
		}
		if !(c < 32 && t.Size() == 4) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64AND_0(v *Value) bool {
	// match: (AND x (MOVDconst [c]))
	// cond:
	// result: (ANDconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ANDconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (AND (MOVDconst [c]) x)
	// cond:
	// result: (ANDconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64ANDconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (AND x x)
	// cond:
	// result: x
	for {
		x := v.Args[1]
		if x != v.Args[0] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (AND x (MVN y))
	// cond:
	// result: (BIC x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MVN {
			break
		}
		y := v_1.Args[0]
		v.reset(OpARM64BIC)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND (MVN y) x)
	// cond:
	// result: (BIC x y)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MVN {
			break
		}
		y := v_0.Args[0]
		v.reset(OpARM64BIC)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ANDshiftLL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ANDshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (AND x1:(SLLconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (ANDshiftLL x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ANDshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (AND x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ANDshiftRL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ANDshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (AND x1:(SRLconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (ANDshiftRL x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ANDshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (AND x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ANDshiftRA x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ANDshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64AND_10(v *Value) bool {
	// match: (AND x1:(SRAconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (ANDshiftRA x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ANDshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ANDconst_0(v *Value) bool {
	// match: (ANDconst [0] _)
	// cond:
	// result: (MOVDconst [0])
	for {
		if v.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (ANDconst [-1] x)
	// cond:
	// result: x
	for {
		if v.AuxInt != -1 {
			break
		}
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [c&d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = c & d
		return true
	}
	// match: (ANDconst [c] (ANDconst [d] x))
	// cond:
	// result: (ANDconst [c&d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ANDconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = c & d
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVWUreg x))
	// cond:
	// result: (ANDconst [c&(1<<32-1)] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = c & (1<<32 - 1)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVHUreg x))
	// cond:
	// result: (ANDconst [c&(1<<16-1)] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = c & (1<<16 - 1)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVBUreg x))
	// cond:
	// result: (ANDconst [c&(1<<8-1)] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = c & (1<<8 - 1)
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [ac] (SLLconst [sc] x))
	// cond: isARM64BFMask(sc, ac, sc)
	// result: (UBFIZ [armBFAuxInt(sc, arm64BFWidth(ac, sc))] x)
	for {
		ac := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, ac, sc)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(sc, arm64BFWidth(ac, sc))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [ac] (SRLconst [sc] x))
	// cond: isARM64BFMask(sc, ac, 0)
	// result: (UBFX [armBFAuxInt(sc, arm64BFWidth(ac, 0))] x)
	for {
		ac := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, ac, 0)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(sc, arm64BFWidth(ac, 0))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ANDshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (ANDshiftLL (MOVDconst [c]) x [d])
	// cond:
	// result: (ANDconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64ANDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftLL x (MOVDconst [c]) [d])
	// cond:
	// result: (ANDconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64(uint64(c) << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftLL x y:(SLLconst x [c]) [d])
	// cond: c==d
	// result: y
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if y.Op != OpARM64SLLconst {
			break
		}
		c := y.AuxInt
		if x != y.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ANDshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (ANDshiftRA (MOVDconst [c]) x [d])
	// cond:
	// result: (ANDconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64ANDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRA x (MOVDconst [c]) [d])
	// cond:
	// result: (ANDconst x [c>>uint64(d)])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ANDconst)
		v.AuxInt = c >> uint64(d)
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftRA x y:(SRAconst x [c]) [d])
	// cond: c==d
	// result: y
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if y.Op != OpARM64SRAconst {
			break
		}
		c := y.AuxInt
		if x != y.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ANDshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (ANDshiftRL (MOVDconst [c]) x [d])
	// cond:
	// result: (ANDconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64ANDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRL x (MOVDconst [c]) [d])
	// cond:
	// result: (ANDconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ANDconst)
		v.AuxInt = int64(uint64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftRL x y:(SRLconst x [c]) [d])
	// cond: c==d
	// result: y
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if y.Op != OpARM64SRLconst {
			break
		}
		c := y.AuxInt
		if x != y.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64BIC_0(v *Value) bool {
	// match: (BIC x (MOVDconst [c]))
	// cond:
	// result: (ANDconst [^c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ANDconst)
		v.AuxInt = ^c
		v.AddArg(x)
		return true
	}
	// match: (BIC x x)
	// cond:
	// result: (MOVDconst [0])
	for {
		x := v.Args[1]
		if x != v.Args[0] {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (BIC x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (BICshiftLL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64BICshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (BIC x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (BICshiftRL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64BICshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (BIC x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (BICshiftRA x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64BICshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64BICshiftLL_0(v *Value) bool {
	// match: (BICshiftLL x (MOVDconst [c]) [d])
	// cond:
	// result: (ANDconst x [^int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ANDconst)
		v.AuxInt = ^int64(uint64(c) << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64BICshiftRA_0(v *Value) bool {
	// match: (BICshiftRA x (MOVDconst [c]) [d])
	// cond:
	// result: (ANDconst x [^(c>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ANDconst)
		v.AuxInt = ^(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64BICshiftRL_0(v *Value) bool {
	// match: (BICshiftRL x (MOVDconst [c]) [d])
	// cond:
	// result: (ANDconst x [^int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ANDconst)
		v.AuxInt = ^int64(uint64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMN_0(v *Value) bool {
	// match: (CMN x (MOVDconst [c]))
	// cond:
	// result: (CMNconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64CMNconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (CMN (MOVDconst [c]) x)
	// cond:
	// result: (CMNconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64CMNconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (CMN x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (CMNshiftLL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64CMNshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (CMN x1:(SLLconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (CMNshiftLL x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64CMNshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (CMN x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (CMNshiftRL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64CMNshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (CMN x1:(SRLconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (CMNshiftRL x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64CMNshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (CMN x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (CMNshiftRA x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64CMNshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (CMN x1:(SRAconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (CMNshiftRA x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64CMNshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMNW_0(v *Value) bool {
	// match: (CMNW x (MOVDconst [c]))
	// cond:
	// result: (CMNWconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64CMNWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (CMNW (MOVDconst [c]) x)
	// cond:
	// result: (CMNWconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64CMNWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMNWconst_0(v *Value) bool {
	// match: (CMNWconst (MOVDconst [x]) [y])
	// cond: int32(x)==int32(-y)
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) == int32(-y)) {
			break
		}
		v.reset(OpARM64FlagEQ)
		return true
	}
	// match: (CMNWconst (MOVDconst [x]) [y])
	// cond: int32(x)<int32(-y) && uint32(x)<uint32(-y)
	// result: (FlagLT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) < int32(-y) && uint32(x) < uint32(-y)) {
			break
		}
		v.reset(OpARM64FlagLT_ULT)
		return true
	}
	// match: (CMNWconst (MOVDconst [x]) [y])
	// cond: int32(x)<int32(-y) && uint32(x)>uint32(-y)
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) < int32(-y) && uint32(x) > uint32(-y)) {
			break
		}
		v.reset(OpARM64FlagLT_UGT)
		return true
	}
	// match: (CMNWconst (MOVDconst [x]) [y])
	// cond: int32(x)>int32(-y) && uint32(x)<uint32(-y)
	// result: (FlagGT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) > int32(-y) && uint32(x) < uint32(-y)) {
			break
		}
		v.reset(OpARM64FlagGT_ULT)
		return true
	}
	// match: (CMNWconst (MOVDconst [x]) [y])
	// cond: int32(x)>int32(-y) && uint32(x)>uint32(-y)
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) > int32(-y) && uint32(x) > uint32(-y)) {
			break
		}
		v.reset(OpARM64FlagGT_UGT)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMNconst_0(v *Value) bool {
	// match: (CMNconst (MOVDconst [x]) [y])
	// cond: int64(x)==int64(-y)
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int64(x) == int64(-y)) {
			break
		}
		v.reset(OpARM64FlagEQ)
		return true
	}
	// match: (CMNconst (MOVDconst [x]) [y])
	// cond: int64(x)<int64(-y) && uint64(x)<uint64(-y)
	// result: (FlagLT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int64(x) < int64(-y) && uint64(x) < uint64(-y)) {
			break
		}
		v.reset(OpARM64FlagLT_ULT)
		return true
	}
	// match: (CMNconst (MOVDconst [x]) [y])
	// cond: int64(x)<int64(-y) && uint64(x)>uint64(-y)
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int64(x) < int64(-y) && uint64(x) > uint64(-y)) {
			break
		}
		v.reset(OpARM64FlagLT_UGT)
		return true
	}
	// match: (CMNconst (MOVDconst [x]) [y])
	// cond: int64(x)>int64(-y) && uint64(x)<uint64(-y)
	// result: (FlagGT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int64(x) > int64(-y) && uint64(x) < uint64(-y)) {
			break
		}
		v.reset(OpARM64FlagGT_ULT)
		return true
	}
	// match: (CMNconst (MOVDconst [x]) [y])
	// cond: int64(x)>int64(-y) && uint64(x)>uint64(-y)
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int64(x) > int64(-y) && uint64(x) > uint64(-y)) {
			break
		}
		v.reset(OpARM64FlagGT_UGT)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMNshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (CMNshiftLL (MOVDconst [c]) x [d])
	// cond:
	// result: (CMNconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64CMNconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftLL x (MOVDconst [c]) [d])
	// cond:
	// result: (CMNconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64CMNconst)
		v.AuxInt = int64(uint64(c) << uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMNshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (CMNshiftRA (MOVDconst [c]) x [d])
	// cond:
	// result: (CMNconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64CMNconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRA x (MOVDconst [c]) [d])
	// cond:
	// result: (CMNconst x [c>>uint64(d)])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64CMNconst)
		v.AuxInt = c >> uint64(d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMNshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (CMNshiftRL (MOVDconst [c]) x [d])
	// cond:
	// result: (CMNconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64CMNconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRL x (MOVDconst [c]) [d])
	// cond:
	// result: (CMNconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64CMNconst)
		v.AuxInt = int64(uint64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMP_0(v *Value) bool {
	b := v.Block
	// match: (CMP x (MOVDconst [c]))
	// cond:
	// result: (CMPconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64CMPconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (CMP (MOVDconst [c]) x)
	// cond:
	// result: (InvertFlags (CMPconst [c] x))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = c
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (CMPshiftLL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64CMPshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (CMP x0:(SLLconst [c] y) x1)
	// cond: clobberIfDead(x0)
	// result: (InvertFlags (CMPshiftLL x1 y [c]))
	for {
		x1 := v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpARM64SLLconst {
			break
		}
		c := x0.AuxInt
		y := x0.Args[0]
		if !(clobberIfDead(x0)) {
			break
		}
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPshiftLL, types.TypeFlags)
		v0.AuxInt = c
		v0.AddArg(x1)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (CMPshiftRL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64CMPshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (CMP x0:(SRLconst [c] y) x1)
	// cond: clobberIfDead(x0)
	// result: (InvertFlags (CMPshiftRL x1 y [c]))
	for {
		x1 := v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpARM64SRLconst {
			break
		}
		c := x0.AuxInt
		y := x0.Args[0]
		if !(clobberIfDead(x0)) {
			break
		}
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPshiftRL, types.TypeFlags)
		v0.AuxInt = c
		v0.AddArg(x1)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (CMPshiftRA x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64CMPshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (CMP x0:(SRAconst [c] y) x1)
	// cond: clobberIfDead(x0)
	// result: (InvertFlags (CMPshiftRA x1 y [c]))
	for {
		x1 := v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpARM64SRAconst {
			break
		}
		c := x0.AuxInt
		y := x0.Args[0]
		if !(clobberIfDead(x0)) {
			break
		}
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPshiftRA, types.TypeFlags)
		v0.AuxInt = c
		v0.AddArg(x1)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPW_0(v *Value) bool {
	b := v.Block
	// match: (CMPW x (MOVDconst [c]))
	// cond:
	// result: (CMPWconst [int64(int32(c))] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64CMPWconst)
		v.AuxInt = int64(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPW (MOVDconst [c]) x)
	// cond:
	// result: (InvertFlags (CMPWconst [int64(int32(c))] x))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPWconst, types.TypeFlags)
		v0.AuxInt = int64(int32(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPWconst_0(v *Value) bool {
	// match: (CMPWconst (MOVDconst [x]) [y])
	// cond: int32(x)==int32(y)
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) == int32(y)) {
			break
		}
		v.reset(OpARM64FlagEQ)
		return true
	}
	// match: (CMPWconst (MOVDconst [x]) [y])
	// cond: int32(x)<int32(y) && uint32(x)<uint32(y)
	// result: (FlagLT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) < int32(y) && uint32(x) < uint32(y)) {
			break
		}
		v.reset(OpARM64FlagLT_ULT)
		return true
	}
	// match: (CMPWconst (MOVDconst [x]) [y])
	// cond: int32(x)<int32(y) && uint32(x)>uint32(y)
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) < int32(y) && uint32(x) > uint32(y)) {
			break
		}
		v.reset(OpARM64FlagLT_UGT)
		return true
	}
	// match: (CMPWconst (MOVDconst [x]) [y])
	// cond: int32(x)>int32(y) && uint32(x)<uint32(y)
	// result: (FlagGT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) > int32(y) && uint32(x) < uint32(y)) {
			break
		}
		v.reset(OpARM64FlagGT_ULT)
		return true
	}
	// match: (CMPWconst (MOVDconst [x]) [y])
	// cond: int32(x)>int32(y) && uint32(x)>uint32(y)
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) > int32(y) && uint32(x) > uint32(y)) {
			break
		}
		v.reset(OpARM64FlagGT_UGT)
		return true
	}
	// match: (CMPWconst (MOVBUreg _) [c])
	// cond: 0xff < int32(c)
	// result: (FlagLT_ULT)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVBUreg {
			break
		}
		if !(0xff < int32(c)) {
			break
		}
		v.reset(OpARM64FlagLT_ULT)
		return true
	}
	// match: (CMPWconst (MOVHUreg _) [c])
	// cond: 0xffff < int32(c)
	// result: (FlagLT_ULT)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVHUreg {
			break
		}
		if !(0xffff < int32(c)) {
			break
		}
		v.reset(OpARM64FlagLT_ULT)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPconst_0(v *Value) bool {
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: x==y
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(x == y) {
			break
		}
		v.reset(OpARM64FlagEQ)
		return true
	}
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: x<y && uint64(x)<uint64(y)
	// result: (FlagLT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(x < y && uint64(x) < uint64(y)) {
			break
		}
		v.reset(OpARM64FlagLT_ULT)
		return true
	}
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: x<y && uint64(x)>uint64(y)
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(x < y && uint64(x) > uint64(y)) {
			break
		}
		v.reset(OpARM64FlagLT_UGT)
		return true
	}
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: x>y && uint64(x)<uint64(y)
	// result: (FlagGT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(x > y && uint64(x) < uint64(y)) {
			break
		}
		v.reset(OpARM64FlagGT_ULT)
		return true
	}
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: x>y && uint64(x)>uint64(y)
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(x > y && uint64(x) > uint64(y)) {
			break
		}
		v.reset(OpARM64FlagGT_UGT)
		return true
	}
	// match: (CMPconst (MOVBUreg _) [c])
	// cond: 0xff < c
	// result: (FlagLT_ULT)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVBUreg {
			break
		}
		if !(0xff < c) {
			break
		}
		v.reset(OpARM64FlagLT_ULT)
		return true
	}
	// match: (CMPconst (MOVHUreg _) [c])
	// cond: 0xffff < c
	// result: (FlagLT_ULT)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVHUreg {
			break
		}
		if !(0xffff < c) {
			break
		}
		v.reset(OpARM64FlagLT_ULT)
		return true
	}
	// match: (CMPconst (MOVWUreg _) [c])
	// cond: 0xffffffff < c
	// result: (FlagLT_ULT)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVWUreg {
			break
		}
		if !(0xffffffff < c) {
			break
		}
		v.reset(OpARM64FlagLT_ULT)
		return true
	}
	// match: (CMPconst (ANDconst _ [m]) [n])
	// cond: 0 <= m && m < n
	// result: (FlagLT_ULT)
	for {
		n := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ANDconst {
			break
		}
		m := v_0.AuxInt
		if !(0 <= m && m < n) {
			break
		}
		v.reset(OpARM64FlagLT_ULT)
		return true
	}
	// match: (CMPconst (SRLconst _ [c]) [n])
	// cond: 0 <= n && 0 < c && c <= 63 && (1<<uint64(64-c)) <= uint64(n)
	// result: (FlagLT_ULT)
	for {
		n := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		c := v_0.AuxInt
		if !(0 <= n && 0 < c && c <= 63 && (1<<uint64(64-c)) <= uint64(n)) {
			break
		}
		v.reset(OpARM64FlagLT_ULT)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (CMPshiftLL (MOVDconst [c]) x [d])
	// cond:
	// result: (InvertFlags (CMPconst [c] (SLLconst <x.Type> x [d])))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = c
		v1 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v1.AuxInt = d
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftLL x (MOVDconst [c]) [d])
	// cond:
	// result: (CMPconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64CMPconst)
		v.AuxInt = int64(uint64(c) << uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (CMPshiftRA (MOVDconst [c]) x [d])
	// cond:
	// result: (InvertFlags (CMPconst [c] (SRAconst <x.Type> x [d])))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = c
		v1 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v1.AuxInt = d
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRA x (MOVDconst [c]) [d])
	// cond:
	// result: (CMPconst x [c>>uint64(d)])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64CMPconst)
		v.AuxInt = c >> uint64(d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (CMPshiftRL (MOVDconst [c]) x [d])
	// cond:
	// result: (InvertFlags (CMPconst [c] (SRLconst <x.Type> x [d])))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = c
		v1 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v1.AuxInt = d
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRL x (MOVDconst [c]) [d])
	// cond:
	// result: (CMPconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64CMPconst)
		v.AuxInt = int64(uint64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CSEL_0(v *Value) bool {
	// match: (CSEL {cc} x (MOVDconst [0]) flag)
	// cond:
	// result: (CSEL0 {cc} x flag)
	for {
		cc := v.Aux
		flag := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpARM64CSEL0)
		v.Aux = cc
		v.AddArg(x)
		v.AddArg(flag)
		return true
	}
	// match: (CSEL {cc} (MOVDconst [0]) y flag)
	// cond:
	// result: (CSEL0 {arm64Negate(cc.(Op))} y flag)
	for {
		cc := v.Aux
		flag := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		y := v.Args[1]
		v.reset(OpARM64CSEL0)
		v.Aux = arm64Negate(cc.(Op))
		v.AddArg(y)
		v.AddArg(flag)
		return true
	}
	// match: (CSEL {cc} x y (InvertFlags cmp))
	// cond:
	// result: (CSEL {arm64Invert(cc.(Op))} x y cmp)
	for {
		cc := v.Aux
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64InvertFlags {
			break
		}
		cmp := v_2.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = arm64Invert(cc.(Op))
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(cmp)
		return true
	}
	// match: (CSEL {cc} x _ flag)
	// cond: ccARM64Eval(cc, flag) > 0
	// result: x
	for {
		cc := v.Aux
		flag := v.Args[2]
		x := v.Args[0]
		if !(ccARM64Eval(cc, flag) > 0) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (CSEL {cc} _ y flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: y
	for {
		cc := v.Aux
		flag := v.Args[2]
		y := v.Args[1]
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (CSEL {cc} x y (CMPWconst [0] boolval))
	// cond: cc.(Op) == OpARM64NotEqual && flagArg(boolval) != nil
	// result: (CSEL {boolval.Op} x y flagArg(boolval))
	for {
		cc := v.Aux
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64CMPWconst {
			break
		}
		if v_2.AuxInt != 0 {
			break
		}
		boolval := v_2.Args[0]
		if !(cc.(Op) == OpARM64NotEqual && flagArg(boolval) != nil) {
			break
		}
		v.reset(OpARM64CSEL)
		v.Aux = boolval.Op
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flagArg(boolval))
		return true
	}
	// match: (CSEL {cc} x y (CMPWconst [0] boolval))
	// cond: cc.(Op) == OpARM64Equal && flagArg(boolval) != nil
	// result: (CSEL {arm64Negate(boolval.Op)} x y flagArg(boolval))
	for {
		cc := v.Aux
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64CMPWconst {
			break
		}
		if v_2.AuxInt != 0 {
			break
		}
		boolval := v_2.Args[0]
		if !(cc.(Op) == OpARM64Equal && flagArg(boolval) != nil) {
			break
		}
		v.reset(OpARM64CSEL)
		v.Aux = arm64Negate(boolval.Op)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flagArg(boolval))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CSEL0_0(v *Value) bool {
	// match: (CSEL0 {cc} x (InvertFlags cmp))
	// cond:
	// result: (CSEL0 {arm64Invert(cc.(Op))} x cmp)
	for {
		cc := v.Aux
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64InvertFlags {
			break
		}
		cmp := v_1.Args[0]
		v.reset(OpARM64CSEL0)
		v.Aux = arm64Invert(cc.(Op))
		v.AddArg(x)
		v.AddArg(cmp)
		return true
	}
	// match: (CSEL0 {cc} x flag)
	// cond: ccARM64Eval(cc, flag) > 0
	// result: x
	for {
		cc := v.Aux
		flag := v.Args[1]
		x := v.Args[0]
		if !(ccARM64Eval(cc, flag) > 0) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (CSEL0 {cc} _ flag)
	// cond: ccARM64Eval(cc, flag) < 0
	// result: (MOVDconst [0])
	for {
		cc := v.Aux
		flag := v.Args[1]
		if !(ccARM64Eval(cc, flag) < 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (CSEL0 {cc} x (CMPWconst [0] boolval))
	// cond: cc.(Op) == OpARM64NotEqual && flagArg(boolval) != nil
	// result: (CSEL0 {boolval.Op} x flagArg(boolval))
	for {
		cc := v.Aux
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CMPWconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		boolval := v_1.Args[0]
		if !(cc.(Op) == OpARM64NotEqual && flagArg(boolval) != nil) {
			break
		}
		v.reset(OpARM64CSEL0)
		v.Aux = boolval.Op
		v.AddArg(x)
		v.AddArg(flagArg(boolval))
		return true
	}
	// match: (CSEL0 {cc} x (CMPWconst [0] boolval))
	// cond: cc.(Op) == OpARM64Equal && flagArg(boolval) != nil
	// result: (CSEL0 {arm64Negate(boolval.Op)} x flagArg(boolval))
	for {
		cc := v.Aux
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CMPWconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		boolval := v_1.Args[0]
		if !(cc.(Op) == OpARM64Equal && flagArg(boolval) != nil) {
			break
		}
		v.reset(OpARM64CSEL0)
		v.Aux = arm64Negate(boolval.Op)
		v.AddArg(x)
		v.AddArg(flagArg(boolval))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64DIV_0(v *Value) bool {
	// match: (DIV (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [c/d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = c / d
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64DIVW_0(v *Value) bool {
	// match: (DIVW (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [int64(int32(c)/int32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(int32(c) / int32(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64EON_0(v *Value) bool {
	// match: (EON x (MOVDconst [c]))
	// cond:
	// result: (XORconst [^c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64XORconst)
		v.AuxInt = ^c
		v.AddArg(x)
		return true
	}
	// match: (EON x x)
	// cond:
	// result: (MOVDconst [-1])
	for {
		x := v.Args[1]
		if x != v.Args[0] {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -1
		return true
	}
	// match: (EON x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (EONshiftLL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64EONshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (EON x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (EONshiftRL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64EONshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (EON x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (EONshiftRA x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64EONshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64EONshiftLL_0(v *Value) bool {
	// match: (EONshiftLL x (MOVDconst [c]) [d])
	// cond:
	// result: (XORconst x [^int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64XORconst)
		v.AuxInt = ^int64(uint64(c) << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (EONshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [-1])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -1
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64EONshiftRA_0(v *Value) bool {
	// match: (EONshiftRA x (MOVDconst [c]) [d])
	// cond:
	// result: (XORconst x [^(c>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64XORconst)
		v.AuxInt = ^(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (EONshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [-1])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -1
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64EONshiftRL_0(v *Value) bool {
	// match: (EONshiftRL x (MOVDconst [c]) [d])
	// cond:
	// result: (XORconst x [^int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64XORconst)
		v.AuxInt = ^int64(uint64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (EONshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [-1])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -1
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64Equal_0(v *Value) bool {
	// match: (Equal (FlagEQ))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagEQ {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (Equal (FlagLT_ULT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (Equal (FlagLT_UGT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (Equal (FlagGT_ULT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (Equal (FlagGT_UGT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (Equal (InvertFlags x))
	// cond:
	// result: (Equal x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64FADDD_0(v *Value) bool {
	// match: (FADDD a (FMULD x y))
	// cond:
	// result: (FMADDD a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARM64FMADDD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FADDD (FMULD x y) a)
	// cond:
	// result: (FMADDD a x y)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FMADDD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FADDD a (FNMULD x y))
	// cond:
	// result: (FMSUBD a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FNMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARM64FMSUBD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FADDD (FNMULD x y) a)
	// cond:
	// result: (FMSUBD a x y)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FNMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FMSUBD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FADDS_0(v *Value) bool {
	// match: (FADDS a (FMULS x y))
	// cond:
	// result: (FMADDS a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FMULS {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARM64FMADDS)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FADDS (FMULS x y) a)
	// cond:
	// result: (FMADDS a x y)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FMADDS)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FADDS a (FNMULS x y))
	// cond:
	// result: (FMSUBS a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FNMULS {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARM64FMSUBS)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FADDS (FNMULS x y) a)
	// cond:
	// result: (FMSUBS a x y)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FNMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FMSUBS)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FCMPD_0(v *Value) bool {
	b := v.Block
	// match: (FCMPD x (FMOVDconst [0]))
	// cond:
	// result: (FCMPD0 x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FMOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpARM64FCMPD0)
		v.AddArg(x)
		return true
	}
	// match: (FCMPD (FMOVDconst [0]) x)
	// cond:
	// result: (InvertFlags (FCMPD0 x))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FMOVDconst {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD0, types.TypeFlags)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FCMPS_0(v *Value) bool {
	b := v.Block
	// match: (FCMPS x (FMOVSconst [0]))
	// cond:
	// result: (FCMPS0 x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FMOVSconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpARM64FCMPS0)
		v.AddArg(x)
		return true
	}
	// match: (FCMPS (FMOVSconst [0]) x)
	// cond:
	// result: (InvertFlags (FCMPS0 x))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FMOVSconst {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS0, types.TypeFlags)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDfpgp_0(v *Value) bool {
	b := v.Block
	// match: (FMOVDfpgp <t> (Arg [off] {sym}))
	// cond:
	// result: @b.Func.Entry (Arg <t> [off] {sym})
	for {
		t := v.Type
		v_0 := v.Args[0]
		if v_0.Op != OpArg {
			break
		}
		off := v_0.AuxInt
		sym := v_0.Aux
		b = b.Func.Entry
		v0 := b.NewValue0(v.Pos, OpArg, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDgpfp_0(v *Value) bool {
	b := v.Block
	// match: (FMOVDgpfp <t> (Arg [off] {sym}))
	// cond:
	// result: @b.Func.Entry (Arg <t> [off] {sym})
	for {
		t := v.Type
		v_0 := v.Args[0]
		if v_0.Op != OpArg {
			break
		}
		off := v_0.AuxInt
		sym := v_0.Aux
		b = b.Func.Entry
		v0 := b.NewValue0(v.Pos, OpArg, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (FMOVDload [off] {sym} ptr (MOVDstore [off] {sym} ptr val _))
	// cond:
	// result: (FMOVDgpfp val)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDstore {
			break
		}
		if v_1.AuxInt != off {
			break
		}
		if v_1.Aux != sym {
			break
		}
		_ = v_1.Args[2]
		if ptr != v_1.Args[0] {
			break
		}
		val := v_1.Args[1]
		v.reset(OpARM64FMOVDgpfp)
		v.AddArg(val)
		return true
	}
	// match: (FMOVDload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVDload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVDload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVDload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVDloadidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64FMOVDloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVDload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVDload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVDload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDloadidx_0(v *Value) bool {
	// match: (FMOVDloadidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (FMOVDload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64FMOVDload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVDloadidx (MOVDconst [c]) ptr mem)
	// cond:
	// result: (FMOVDload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARM64FMOVDload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDstore_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (FMOVDstore [off] {sym} ptr (FMOVDgpfp val) mem)
	// cond:
	// result: (MOVDstore [off] {sym} ptr val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FMOVDgpfp {
			break
		}
		val := v_1.Args[0]
		v.reset(OpARM64MOVDstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVDstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVDstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVDstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVDstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVDstoreidx ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64FMOVDstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVDstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVDstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVDstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVDstoreidx_0(v *Value) bool {
	// match: (FMOVDstoreidx ptr (MOVDconst [c]) val mem)
	// cond:
	// result: (FMOVDstore [c] ptr val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARM64FMOVDstore)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVDstoreidx (MOVDconst [c]) idx val mem)
	// cond:
	// result: (FMOVDstore [c] idx val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		idx := v.Args[1]
		val := v.Args[2]
		v.reset(OpARM64FMOVDstore)
		v.AuxInt = c
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVSload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (FMOVSload [off] {sym} ptr (MOVWstore [off] {sym} ptr val _))
	// cond:
	// result: (FMOVSgpfp val)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWstore {
			break
		}
		if v_1.AuxInt != off {
			break
		}
		if v_1.Aux != sym {
			break
		}
		_ = v_1.Args[2]
		if ptr != v_1.Args[0] {
			break
		}
		val := v_1.Args[1]
		v.reset(OpARM64FMOVSgpfp)
		v.AddArg(val)
		return true
	}
	// match: (FMOVSload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVSload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVSload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVSload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVSloadidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64FMOVSloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVSload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVSload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVSload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVSloadidx_0(v *Value) bool {
	// match: (FMOVSloadidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (FMOVSload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64FMOVSload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVSloadidx (MOVDconst [c]) ptr mem)
	// cond:
	// result: (FMOVSload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARM64FMOVSload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVSstore_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (FMOVSstore [off] {sym} ptr (FMOVSgpfp val) mem)
	// cond:
	// result: (MOVWstore [off] {sym} ptr val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FMOVSgpfp {
			break
		}
		val := v_1.Args[0]
		v.reset(OpARM64MOVWstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVSstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVSstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVSstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVSstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (FMOVSstoreidx ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64FMOVSstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVSstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVSstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64FMOVSstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMOVSstoreidx_0(v *Value) bool {
	// match: (FMOVSstoreidx ptr (MOVDconst [c]) val mem)
	// cond:
	// result: (FMOVSstore [c] ptr val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARM64FMOVSstore)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (FMOVSstoreidx (MOVDconst [c]) idx val mem)
	// cond:
	// result: (FMOVSstore [c] idx val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		idx := v.Args[1]
		val := v.Args[2]
		v.reset(OpARM64FMOVSstore)
		v.AuxInt = c
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMULD_0(v *Value) bool {
	// match: (FMULD (FNEGD x) y)
	// cond:
	// result: (FNMULD x y)
	for {
		y := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FNEGD {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64FNMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FMULD y (FNEGD x))
	// cond:
	// result: (FNMULD x y)
	for {
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FNEGD {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64FNMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FMULS_0(v *Value) bool {
	// match: (FMULS (FNEGS x) y)
	// cond:
	// result: (FNMULS x y)
	for {
		y := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FNEGS {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64FNMULS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FMULS y (FNEGS x))
	// cond:
	// result: (FNMULS x y)
	for {
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FNEGS {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64FNMULS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FNEGD_0(v *Value) bool {
	// match: (FNEGD (FMULD x y))
	// cond:
	// result: (FNMULD x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FNMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FNEGD (FNMULD x y))
	// cond:
	// result: (FMULD x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FNMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FNEGS_0(v *Value) bool {
	// match: (FNEGS (FMULS x y))
	// cond:
	// result: (FNMULS x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FNMULS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FNEGS (FNMULS x y))
	// cond:
	// result: (FMULS x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FNMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FMULS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FNMULD_0(v *Value) bool {
	// match: (FNMULD (FNEGD x) y)
	// cond:
	// result: (FMULD x y)
	for {
		y := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FNEGD {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64FMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FNMULD y (FNEGD x))
	// cond:
	// result: (FMULD x y)
	for {
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FNEGD {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64FMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FNMULS_0(v *Value) bool {
	// match: (FNMULS (FNEGS x) y)
	// cond:
	// result: (FMULS x y)
	for {
		y := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FNEGS {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64FMULS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FNMULS y (FNEGS x))
	// cond:
	// result: (FMULS x y)
	for {
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FNEGS {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64FMULS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FSUBD_0(v *Value) bool {
	// match: (FSUBD a (FMULD x y))
	// cond:
	// result: (FMSUBD a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARM64FMSUBD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FSUBD (FMULD x y) a)
	// cond:
	// result: (FNMSUBD a x y)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FNMSUBD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FSUBD a (FNMULD x y))
	// cond:
	// result: (FMADDD a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FNMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARM64FMADDD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FSUBD (FNMULD x y) a)
	// cond:
	// result: (FNMADDD a x y)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FNMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FNMADDD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64FSUBS_0(v *Value) bool {
	// match: (FSUBS a (FMULS x y))
	// cond:
	// result: (FMSUBS a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FMULS {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARM64FMSUBS)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FSUBS (FMULS x y) a)
	// cond:
	// result: (FNMSUBS a x y)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FNMSUBS)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FSUBS a (FNMULS x y))
	// cond:
	// result: (FMADDS a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FNMULS {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARM64FMADDS)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (FSUBS (FNMULS x y) a)
	// cond:
	// result: (FNMADDS a x y)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FNMULS {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64FNMADDS)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64GreaterEqual_0(v *Value) bool {
	// match: (GreaterEqual (FlagEQ))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagEQ {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterEqual (FlagLT_ULT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterEqual (FlagLT_UGT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterEqual (FlagGT_ULT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterEqual (FlagGT_UGT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterEqual (InvertFlags x))
	// cond:
	// result: (LessEqual x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64GreaterEqualF_0(v *Value) bool {
	// match: (GreaterEqualF (InvertFlags x))
	// cond:
	// result: (LessEqualF x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64GreaterEqualU_0(v *Value) bool {
	// match: (GreaterEqualU (FlagEQ))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagEQ {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterEqualU (FlagLT_ULT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterEqualU (FlagLT_UGT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterEqualU (FlagGT_ULT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterEqualU (FlagGT_UGT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterEqualU (InvertFlags x))
	// cond:
	// result: (LessEqualU x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64GreaterThan_0(v *Value) bool {
	// match: (GreaterThan (FlagEQ))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagEQ {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterThan (FlagLT_ULT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterThan (FlagLT_UGT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterThan (FlagGT_ULT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterThan (FlagGT_UGT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterThan (InvertFlags x))
	// cond:
	// result: (LessThan x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64GreaterThanF_0(v *Value) bool {
	// match: (GreaterThanF (InvertFlags x))
	// cond:
	// result: (LessThanF x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64GreaterThanU_0(v *Value) bool {
	// match: (GreaterThanU (FlagEQ))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagEQ {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterThanU (FlagLT_ULT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterThanU (FlagLT_UGT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterThanU (FlagGT_ULT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterThanU (FlagGT_UGT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterThanU (InvertFlags x))
	// cond:
	// result: (LessThanU x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64LessEqual_0(v *Value) bool {
	// match: (LessEqual (FlagEQ))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagEQ {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessEqual (FlagLT_ULT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessEqual (FlagLT_UGT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessEqual (FlagGT_ULT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessEqual (FlagGT_UGT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessEqual (InvertFlags x))
	// cond:
	// result: (GreaterEqual x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64LessEqualF_0(v *Value) bool {
	// match: (LessEqualF (InvertFlags x))
	// cond:
	// result: (GreaterEqualF x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64LessEqualU_0(v *Value) bool {
	// match: (LessEqualU (FlagEQ))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagEQ {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessEqualU (FlagLT_ULT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessEqualU (FlagLT_UGT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessEqualU (FlagGT_ULT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessEqualU (FlagGT_UGT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessEqualU (InvertFlags x))
	// cond:
	// result: (GreaterEqualU x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64LessThan_0(v *Value) bool {
	// match: (LessThan (FlagEQ))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagEQ {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessThan (FlagLT_ULT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessThan (FlagLT_UGT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessThan (FlagGT_ULT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessThan (FlagGT_UGT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessThan (InvertFlags x))
	// cond:
	// result: (GreaterThan x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64LessThanF_0(v *Value) bool {
	// match: (LessThanF (InvertFlags x))
	// cond:
	// result: (GreaterThanF x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64LessThanU_0(v *Value) bool {
	// match: (LessThanU (FlagEQ))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagEQ {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessThanU (FlagLT_ULT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessThanU (FlagLT_UGT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessThanU (FlagGT_ULT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessThanU (FlagGT_UGT))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessThanU (InvertFlags x))
	// cond:
	// result: (GreaterThanU x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64MADD_0(v *Value) bool {
	b := v.Block
	// match: (MADD a x (MOVDconst [-1]))
	// cond:
	// result: (SUB a x)
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != -1 {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MADD a _ (MOVDconst [0]))
	// cond:
	// result: a
	for {
		_ = v.Args[2]
		a := v.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = a.Type
		v.AddArg(a)
		return true
	}
	// match: (MADD a x (MOVDconst [1]))
	// cond:
	// result: (ADD a x)
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != 1 {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (ADDshiftLL a x [log2(c)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c-1) && c>=3
	// result: (ADD a (ADDshiftLL <x.Type> x x [log2(c-1)]))
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(isPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c+1) && c>=7
	// result: (SUB a (SUBshiftLL <x.Type> x x [log2(c+1)]))
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(isPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [2]) [log2(c/3)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 3)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [2]) [log2(c/5)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 5)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [3]) [log2(c/7)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 7)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [3]) [log2(c/9)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 9)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MADD_10(v *Value) bool {
	b := v.Block
	// match: (MADD a (MOVDconst [-1]) x)
	// cond:
	// result: (SUB a x)
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != -1 {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MADD a (MOVDconst [0]) _)
	// cond:
	// result: a
	for {
		_ = v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = a.Type
		v.AddArg(a)
		return true
	}
	// match: (MADD a (MOVDconst [1]) x)
	// cond:
	// result: (ADD a x)
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c)
	// result: (ADDshiftLL a x [log2(c)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c-1) && c>=3
	// result: (ADD a (ADDshiftLL <x.Type> x x [log2(c-1)]))
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c+1) && c>=7
	// result: (SUB a (SUBshiftLL <x.Type> x x [log2(c+1)]))
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [2]) [log2(c/3)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 3)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [2]) [log2(c/5)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 5)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [3]) [log2(c/7)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 7)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [3]) [log2(c/9)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 9)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MADD_20(v *Value) bool {
	b := v.Block
	// match: (MADD (MOVDconst [c]) x y)
	// cond:
	// result: (ADDconst [c] (MUL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARM64ADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (MADD a (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (ADDconst [c*d] a)
	for {
		_ = v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		d := v_2.AuxInt
		v.reset(OpARM64ADDconst)
		v.AuxInt = c * d
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MADDW_0(v *Value) bool {
	b := v.Block
	// match: (MADDW a x (MOVDconst [c]))
	// cond: int32(c)==-1
	// result: (SUB a x)
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MADDW a _ (MOVDconst [c]))
	// cond: int32(c)==0
	// result: a
	for {
		_ = v.Args[2]
		a := v.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpCopy)
		v.Type = a.Type
		v.AddArg(a)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: int32(c)==1
	// result: (ADD a x)
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(int32(c) == 1) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (ADDshiftLL a x [log2(c)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c-1) && int32(c)>=3
	// result: (ADD a (ADDshiftLL <x.Type> x x [log2(c-1)]))
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c+1) && int32(c)>=7
	// result: (SUB a (SUBshiftLL <x.Type> x x [log2(c+1)]))
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [2]) [log2(c/3)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 3)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [2]) [log2(c/5)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 5)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [3]) [log2(c/7)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 7)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [3]) [log2(c/9)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 9)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MADDW_10(v *Value) bool {
	b := v.Block
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: int32(c)==-1
	// result: (SUB a x)
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) _)
	// cond: int32(c)==0
	// result: a
	for {
		_ = v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpCopy)
		v.Type = a.Type
		v.AddArg(a)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: int32(c)==1
	// result: (ADD a x)
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == 1) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c)
	// result: (ADDshiftLL a x [log2(c)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c-1) && int32(c)>=3
	// result: (ADD a (ADDshiftLL <x.Type> x x [log2(c-1)]))
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c+1) && int32(c)>=7
	// result: (SUB a (SUBshiftLL <x.Type> x x [log2(c+1)]))
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [2]) [log2(c/3)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 3)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [2]) [log2(c/5)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 5)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (SUBshiftLL a (SUBshiftLL <x.Type> x x [3]) [log2(c/7)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 7)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (ADDshiftLL a (ADDshiftLL <x.Type> x x [3]) [log2(c/9)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 9)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MADDW_20(v *Value) bool {
	b := v.Block
	// match: (MADDW (MOVDconst [c]) x y)
	// cond:
	// result: (ADDconst [c] (MULW <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARM64ADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (MADDW a (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (ADDconst [int64(int32(c)*int32(d))] a)
	for {
		_ = v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		d := v_2.AuxInt
		v.reset(OpARM64ADDconst)
		v.AuxInt = int64(int32(c) * int32(d))
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MNEG_0(v *Value) bool {
	b := v.Block
	// match: (MNEG x (MOVDconst [-1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != -1 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MNEG (MOVDconst [-1]) x)
	// cond:
	// result: x
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MNEG _ (MOVDconst [0]))
	// cond:
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (MNEG (MOVDconst [0]) _)
	// cond:
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (MNEG x (MOVDconst [1]))
	// cond:
	// result: (NEG x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpARM64NEG)
		v.AddArg(x)
		return true
	}
	// match: (MNEG (MOVDconst [1]) x)
	// cond:
	// result: (NEG x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		v.reset(OpARM64NEG)
		v.AddArg(x)
		return true
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (NEG (SLLconst <x.Type> [log2(c)] x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEG (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c)
	// result: (NEG (SLLconst <x.Type> [log2(c)] x))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: isPowerOfTwo(c-1) && c >= 3
	// result: (NEG (ADDshiftLL <x.Type> x x [log2(c-1)]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEG (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c-1) && c >= 3
	// result: (NEG (ADDshiftLL <x.Type> x x [log2(c-1)]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MNEG_10(v *Value) bool {
	b := v.Block
	// match: (MNEG x (MOVDconst [c]))
	// cond: isPowerOfTwo(c+1) && c >= 7
	// result: (NEG (ADDshiftLL <x.Type> (NEG <x.Type> x) x [log2(c+1)]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v1 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEG (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c+1) && c >= 7
	// result: (NEG (ADDshiftLL <x.Type> (NEG <x.Type> x) x [log2(c+1)]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v1 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (SLLconst <x.Type> [log2(c/3)] (SUBshiftLL <x.Type> x x [2]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.Type = x.Type
		v.AuxInt = log2(c / 3)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEG (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (SLLconst <x.Type> [log2(c/3)] (SUBshiftLL <x.Type> x x [2]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.Type = x.Type
		v.AuxInt = log2(c / 3)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (NEG (SLLconst <x.Type> [log2(c/5)] (ADDshiftLL <x.Type> x x [2])))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = log2(c / 5)
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = 2
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (MNEG (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (NEG (SLLconst <x.Type> [log2(c/5)] (ADDshiftLL <x.Type> x x [2])))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = log2(c / 5)
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = 2
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (SLLconst <x.Type> [log2(c/7)] (SUBshiftLL <x.Type> x x [3]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.Type = x.Type
		v.AuxInt = log2(c / 7)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEG (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (SLLconst <x.Type> [log2(c/7)] (SUBshiftLL <x.Type> x x [3]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.Type = x.Type
		v.AuxInt = log2(c / 7)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEG x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (NEG (SLLconst <x.Type> [log2(c/9)] (ADDshiftLL <x.Type> x x [3])))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = log2(c / 9)
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = 3
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (MNEG (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (NEG (SLLconst <x.Type> [log2(c/9)] (ADDshiftLL <x.Type> x x [3])))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = log2(c / 9)
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = 3
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MNEG_20(v *Value) bool {
	// match: (MNEG (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [-c*d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -c * d
		return true
	}
	// match: (MNEG (MOVDconst [d]) (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [-c*d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -c * d
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MNEGW_0(v *Value) bool {
	b := v.Block
	// match: (MNEGW x (MOVDconst [c]))
	// cond: int32(c)==-1
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MNEGW (MOVDconst [c]) x)
	// cond: int32(c)==-1
	// result: x
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MNEGW _ (MOVDconst [c]))
	// cond: int32(c)==0
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (MNEGW (MOVDconst [c]) _)
	// cond: int32(c)==0
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: int32(c)==1
	// result: (NEG x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == 1) {
			break
		}
		v.reset(OpARM64NEG)
		v.AddArg(x)
		return true
	}
	// match: (MNEGW (MOVDconst [c]) x)
	// cond: int32(c)==1
	// result: (NEG x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(int32(c) == 1) {
			break
		}
		v.reset(OpARM64NEG)
		v.AddArg(x)
		return true
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (NEG (SLLconst <x.Type> [log2(c)] x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEGW (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c)
	// result: (NEG (SLLconst <x.Type> [log2(c)] x))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: isPowerOfTwo(c-1) && int32(c) >= 3
	// result: (NEG (ADDshiftLL <x.Type> x x [log2(c-1)]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEGW (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c-1) && int32(c) >= 3
	// result: (NEG (ADDshiftLL <x.Type> x x [log2(c-1)]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MNEGW_10(v *Value) bool {
	b := v.Block
	// match: (MNEGW x (MOVDconst [c]))
	// cond: isPowerOfTwo(c+1) && int32(c) >= 7
	// result: (NEG (ADDshiftLL <x.Type> (NEG <x.Type> x) x [log2(c+1)]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v1 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEGW (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c+1) && int32(c) >= 7
	// result: (NEG (ADDshiftLL <x.Type> (NEG <x.Type> x) x [log2(c+1)]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v1 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (SLLconst <x.Type> [log2(c/3)] (SUBshiftLL <x.Type> x x [2]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.Type = x.Type
		v.AuxInt = log2(c / 3)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEGW (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (SLLconst <x.Type> [log2(c/3)] (SUBshiftLL <x.Type> x x [2]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.Type = x.Type
		v.AuxInt = log2(c / 3)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (NEG (SLLconst <x.Type> [log2(c/5)] (ADDshiftLL <x.Type> x x [2])))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = log2(c / 5)
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = 2
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (MNEGW (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (NEG (SLLconst <x.Type> [log2(c/5)] (ADDshiftLL <x.Type> x x [2])))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = log2(c / 5)
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = 2
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (SLLconst <x.Type> [log2(c/7)] (SUBshiftLL <x.Type> x x [3]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.Type = x.Type
		v.AuxInt = log2(c / 7)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEGW (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (SLLconst <x.Type> [log2(c/7)] (SUBshiftLL <x.Type> x x [3]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.Type = x.Type
		v.AuxInt = log2(c / 7)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MNEGW x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (NEG (SLLconst <x.Type> [log2(c/9)] (ADDshiftLL <x.Type> x x [3])))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = log2(c / 9)
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = 3
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (MNEGW (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (NEG (SLLconst <x.Type> [log2(c/9)] (ADDshiftLL <x.Type> x x [3])))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64NEG)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = log2(c / 9)
		v1 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v1.AuxInt = 3
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MNEGW_20(v *Value) bool {
	// match: (MNEGW (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [-int64(int32(c)*int32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -int64(int32(c) * int32(d))
		return true
	}
	// match: (MNEGW (MOVDconst [d]) (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [-int64(int32(c)*int32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -int64(int32(c) * int32(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOD_0(v *Value) bool {
	// match: (MOD (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [c%d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = c % d
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MODW_0(v *Value) bool {
	// match: (MODW (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [int64(int32(c)%int32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(int32(c) % int32(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBUload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVBUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBUload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBUload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBUload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBUloadidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVBUloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBUload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBUload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBUload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBUload [off] {sym} ptr (MOVBstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVBstorezero {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (MOVBUload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVDconst [int64(read8(sym, off))])
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSB {
			break
		}
		if !(symIsRO(sym)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(read8(sym, off))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBUloadidx_0(v *Value) bool {
	// match: (MOVBUloadidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVBUload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVBUload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBUloadidx (MOVDconst [c]) ptr mem)
	// cond:
	// result: (MOVBUload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARM64MOVBUload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBUloadidx ptr idx (MOVBstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVBstorezeroidx {
			break
		}
		_ = v_2.Args[2]
		ptr2 := v_2.Args[0]
		idx2 := v_2.Args[1]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBUreg_0(v *Value) bool {
	// match: (MOVBUreg x:(MOVBUload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (ANDconst [c] x))
	// cond:
	// result: (ANDconst [c&(1<<8-1)] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ANDconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = c & (1<<8 - 1)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [int64(uint8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(uint8(c))
		return true
	}
	// match: (MOVBUreg x)
	// cond: x.Type.IsBoolean()
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
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
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<8-1, sc)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(sc, arm64BFWidth(1<<8-1, sc))
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (SRLconst [sc] x))
	// cond: isARM64BFMask(sc, 1<<8-1, 0)
	// result: (UBFX [armBFAuxInt(sc, 8)] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<8-1, 0)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(sc, 8)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVBload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBloadidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVBloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBload [off] {sym} ptr (MOVBstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVBstorezero {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBloadidx_0(v *Value) bool {
	// match: (MOVBloadidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVBload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVBload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBloadidx (MOVDconst [c]) ptr mem)
	// cond:
	// result: (MOVBload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARM64MOVBload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBloadidx ptr idx (MOVBstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVBstorezeroidx {
			break
		}
		_ = v_2.Args[2]
		ptr2 := v_2.Args[0]
		idx2 := v_2.Args[1]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBreg_0(v *Value) bool {
	// match: (MOVBreg x:(MOVBload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [int64(int8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(int8(c))
		return true
	}
	// match: (MOVBreg (SLLconst [lc] x))
	// cond: lc < 8
	// result: (SBFIZ [armBFAuxInt(lc, 8-lc)] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := v_0.AuxInt
		x := v_0.Args[0]
		if !(lc < 8) {
			break
		}
		v.reset(OpARM64SBFIZ)
		v.AuxInt = armBFAuxInt(lc, 8-lc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstore_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBstoreidx ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVDconst [0]) mem)
	// cond:
	// result: (MOVBstorezero [off] {sym} ptr mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVBstorezero)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBreg x) mem)
	// cond:
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVBreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBUreg x) mem)
	// cond:
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVBUreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHreg x) mem)
	// cond:
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVHreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVHUreg x) mem)
	// cond:
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWreg x) mem)
	// cond:
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVWUreg x) mem)
	// cond:
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstore_10(v *Value) bool {
	// match: (MOVBstore [i] {s} ptr0 (SRLconst [8] w) x:(MOVBstore [i-1] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 8 {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (SRLconst [8] w) x:(MOVBstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 8 {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr0 (UBFX [armBFAuxInt(8, 8)] w) x:(MOVBstore [i-1] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64UBFX {
			break
		}
		if v_1.AuxInt != armBFAuxInt(8, 8) {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (UBFX [armBFAuxInt(8, 8)] w) x:(MOVBstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64UBFX {
			break
		}
		if v_1.AuxInt != armBFAuxInt(8, 8) {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr0 (UBFX [armBFAuxInt(8, 24)] w) x:(MOVBstore [i-1] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64UBFX {
			break
		}
		if v_1.AuxInt != armBFAuxInt(8, 24) {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (UBFX [armBFAuxInt(8, 24)] w) x:(MOVBstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64UBFX {
			break
		}
		if v_1.AuxInt != armBFAuxInt(8, 24) {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr0 (SRLconst [8] (MOVDreg w)) x:(MOVBstore [i-1] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 8 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVDreg {
			break
		}
		w := v_1_0.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (SRLconst [8] (MOVDreg w)) x:(MOVBstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 8 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVDreg {
			break
		}
		w := v_1_0.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr0 (SRLconst [j] w) x:(MOVBstore [i-1] {s} ptr1 w0:(SRLconst [j-8] w) mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpARM64SRLconst {
			break
		}
		if w0.AuxInt != j-8 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (SRLconst [j] w) x:(MOVBstoreidx ptr1 idx1 w0:(SRLconst [j-8] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w0 mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		w0 := x.Args[2]
		if w0.Op != OpARM64SRLconst {
			break
		}
		if w0.AuxInt != j-8 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstore_20(v *Value) bool {
	b := v.Block
	// match: (MOVBstore [i] {s} ptr0 (UBFX [bfc] w) x:(MOVBstore [i-1] {s} ptr1 w0:(UBFX [bfc2] w) mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && getARM64BFwidth(bfc) == 32 - getARM64BFlsb(bfc) && getARM64BFwidth(bfc2) == 32 - getARM64BFlsb(bfc2) && getARM64BFlsb(bfc2) == getARM64BFlsb(bfc) - 8 && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64UBFX {
			break
		}
		bfc := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpARM64UBFX {
			break
		}
		bfc2 := w0.AuxInt
		if w != w0.Args[0] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && getARM64BFwidth(bfc) == 32-getARM64BFlsb(bfc) && getARM64BFwidth(bfc2) == 32-getARM64BFlsb(bfc2) && getARM64BFlsb(bfc2) == getARM64BFlsb(bfc)-8 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (UBFX [bfc] w) x:(MOVBstoreidx ptr1 idx1 w0:(UBFX [bfc2] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && getARM64BFwidth(bfc) == 32 - getARM64BFlsb(bfc) && getARM64BFwidth(bfc2) == 32 - getARM64BFlsb(bfc2) && getARM64BFlsb(bfc2) == getARM64BFlsb(bfc) - 8 && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w0 mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64UBFX {
			break
		}
		bfc := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		w0 := x.Args[2]
		if w0.Op != OpARM64UBFX {
			break
		}
		bfc2 := w0.AuxInt
		if w != w0.Args[0] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && getARM64BFwidth(bfc) == 32-getARM64BFlsb(bfc) && getARM64BFwidth(bfc2) == 32-getARM64BFlsb(bfc2) && getARM64BFlsb(bfc2) == getARM64BFlsb(bfc)-8 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr0 (SRLconst [j] (MOVDreg w)) x:(MOVBstore [i-1] {s} ptr1 w0:(SRLconst [j-8] (MOVDreg w)) mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr0 w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := v_1.AuxInt
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVDreg {
			break
		}
		w := v_1_0.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpARM64SRLconst {
			break
		}
		if w0.AuxInt != j-8 {
			break
		}
		w0_0 := w0.Args[0]
		if w0_0.Op != OpARM64MOVDreg {
			break
		}
		if w != w0_0.Args[0] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr0 idx0) (SRLconst [j] (MOVDreg w)) x:(MOVBstoreidx ptr1 idx1 w0:(SRLconst [j-8] (MOVDreg w)) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr1 idx1 w0 mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := v_1.AuxInt
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVDreg {
			break
		}
		w := v_1_0.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		w0 := x.Args[2]
		if w0.Op != OpARM64SRLconst {
			break
		}
		if w0.AuxInt != j-8 {
			break
		}
		w0_0 := w0.Args[0]
		if w0_0.Op != OpARM64MOVDreg {
			break
		}
		if w != w0_0.Args[0] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr w x0:(MOVBstore [i-1] {s} ptr (SRLconst [8] w) x1:(MOVBstore [i-2] {s} ptr (SRLconst [16] w) x2:(MOVBstore [i-3] {s} ptr (SRLconst [24] w) x3:(MOVBstore [i-4] {s} ptr (SRLconst [32] w) x4:(MOVBstore [i-5] {s} ptr (SRLconst [40] w) x5:(MOVBstore [i-6] {s} ptr (SRLconst [48] w) x6:(MOVBstore [i-7] {s} ptr (SRLconst [56] w) mem))))))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6)
	// result: (MOVDstore [i-7] {s} ptr (REV <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		w := v.Args[1]
		x0 := v.Args[2]
		if x0.Op != OpARM64MOVBstore {
			break
		}
		if x0.AuxInt != i-1 {
			break
		}
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64SRLconst {
			break
		}
		if x0_1.AuxInt != 8 {
			break
		}
		if w != x0_1.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore {
			break
		}
		if x1.AuxInt != i-2 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64SRLconst {
			break
		}
		if x1_1.AuxInt != 16 {
			break
		}
		if w != x1_1.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstore {
			break
		}
		if x2.AuxInt != i-3 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[2]
		if ptr != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64SRLconst {
			break
		}
		if x2_1.AuxInt != 24 {
			break
		}
		if w != x2_1.Args[0] {
			break
		}
		x3 := x2.Args[2]
		if x3.Op != OpARM64MOVBstore {
			break
		}
		if x3.AuxInt != i-4 {
			break
		}
		if x3.Aux != s {
			break
		}
		_ = x3.Args[2]
		if ptr != x3.Args[0] {
			break
		}
		x3_1 := x3.Args[1]
		if x3_1.Op != OpARM64SRLconst {
			break
		}
		if x3_1.AuxInt != 32 {
			break
		}
		if w != x3_1.Args[0] {
			break
		}
		x4 := x3.Args[2]
		if x4.Op != OpARM64MOVBstore {
			break
		}
		if x4.AuxInt != i-5 {
			break
		}
		if x4.Aux != s {
			break
		}
		_ = x4.Args[2]
		if ptr != x4.Args[0] {
			break
		}
		x4_1 := x4.Args[1]
		if x4_1.Op != OpARM64SRLconst {
			break
		}
		if x4_1.AuxInt != 40 {
			break
		}
		if w != x4_1.Args[0] {
			break
		}
		x5 := x4.Args[2]
		if x5.Op != OpARM64MOVBstore {
			break
		}
		if x5.AuxInt != i-6 {
			break
		}
		if x5.Aux != s {
			break
		}
		_ = x5.Args[2]
		if ptr != x5.Args[0] {
			break
		}
		x5_1 := x5.Args[1]
		if x5_1.Op != OpARM64SRLconst {
			break
		}
		if x5_1.AuxInt != 48 {
			break
		}
		if w != x5_1.Args[0] {
			break
		}
		x6 := x5.Args[2]
		if x6.Op != OpARM64MOVBstore {
			break
		}
		if x6.AuxInt != i-7 {
			break
		}
		if x6.Aux != s {
			break
		}
		mem := x6.Args[2]
		if ptr != x6.Args[0] {
			break
		}
		x6_1 := x6.Args[1]
		if x6_1.Op != OpARM64SRLconst {
			break
		}
		if x6_1.AuxInt != 56 {
			break
		}
		if w != x6_1.Args[0] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = i - 7
		v.Aux = s
		v.AddArg(ptr)
		v0 := b.NewValue0(x6.Pos, OpARM64REV, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [7] {s} p w x0:(MOVBstore [6] {s} p (SRLconst [8] w) x1:(MOVBstore [5] {s} p (SRLconst [16] w) x2:(MOVBstore [4] {s} p (SRLconst [24] w) x3:(MOVBstore [3] {s} p (SRLconst [32] w) x4:(MOVBstore [2] {s} p (SRLconst [40] w) x5:(MOVBstore [1] {s} p1:(ADD ptr1 idx1) (SRLconst [48] w) x6:(MOVBstoreidx ptr0 idx0 (SRLconst [56] w) mem))))))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6)
	// result: (MOVDstoreidx ptr0 idx0 (REV <w.Type> w) mem)
	for {
		if v.AuxInt != 7 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		w := v.Args[1]
		x0 := v.Args[2]
		if x0.Op != OpARM64MOVBstore {
			break
		}
		if x0.AuxInt != 6 {
			break
		}
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64SRLconst {
			break
		}
		if x0_1.AuxInt != 8 {
			break
		}
		if w != x0_1.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore {
			break
		}
		if x1.AuxInt != 5 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64SRLconst {
			break
		}
		if x1_1.AuxInt != 16 {
			break
		}
		if w != x1_1.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstore {
			break
		}
		if x2.AuxInt != 4 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[2]
		if p != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64SRLconst {
			break
		}
		if x2_1.AuxInt != 24 {
			break
		}
		if w != x2_1.Args[0] {
			break
		}
		x3 := x2.Args[2]
		if x3.Op != OpARM64MOVBstore {
			break
		}
		if x3.AuxInt != 3 {
			break
		}
		if x3.Aux != s {
			break
		}
		_ = x3.Args[2]
		if p != x3.Args[0] {
			break
		}
		x3_1 := x3.Args[1]
		if x3_1.Op != OpARM64SRLconst {
			break
		}
		if x3_1.AuxInt != 32 {
			break
		}
		if w != x3_1.Args[0] {
			break
		}
		x4 := x3.Args[2]
		if x4.Op != OpARM64MOVBstore {
			break
		}
		if x4.AuxInt != 2 {
			break
		}
		if x4.Aux != s {
			break
		}
		_ = x4.Args[2]
		if p != x4.Args[0] {
			break
		}
		x4_1 := x4.Args[1]
		if x4_1.Op != OpARM64SRLconst {
			break
		}
		if x4_1.AuxInt != 40 {
			break
		}
		if w != x4_1.Args[0] {
			break
		}
		x5 := x4.Args[2]
		if x5.Op != OpARM64MOVBstore {
			break
		}
		if x5.AuxInt != 1 {
			break
		}
		if x5.Aux != s {
			break
		}
		_ = x5.Args[2]
		p1 := x5.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		x5_1 := x5.Args[1]
		if x5_1.Op != OpARM64SRLconst {
			break
		}
		if x5_1.AuxInt != 48 {
			break
		}
		if w != x5_1.Args[0] {
			break
		}
		x6 := x5.Args[2]
		if x6.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x6.Args[3]
		ptr0 := x6.Args[0]
		idx0 := x6.Args[1]
		x6_2 := x6.Args[2]
		if x6_2.Op != OpARM64SRLconst {
			break
		}
		if x6_2.AuxInt != 56 {
			break
		}
		if w != x6_2.Args[0] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6)) {
			break
		}
		v.reset(OpARM64MOVDstoreidx)
		v.AddArg(ptr0)
		v.AddArg(idx0)
		v0 := b.NewValue0(x5.Pos, OpARM64REV, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr w x0:(MOVBstore [i-1] {s} ptr (UBFX [armBFAuxInt(8, 24)] w) x1:(MOVBstore [i-2] {s} ptr (UBFX [armBFAuxInt(16, 16)] w) x2:(MOVBstore [i-3] {s} ptr (UBFX [armBFAuxInt(24, 8)] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2)
	// result: (MOVWstore [i-3] {s} ptr (REVW <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		w := v.Args[1]
		x0 := v.Args[2]
		if x0.Op != OpARM64MOVBstore {
			break
		}
		if x0.AuxInt != i-1 {
			break
		}
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64UBFX {
			break
		}
		if x0_1.AuxInt != armBFAuxInt(8, 24) {
			break
		}
		if w != x0_1.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore {
			break
		}
		if x1.AuxInt != i-2 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64UBFX {
			break
		}
		if x1_1.AuxInt != armBFAuxInt(16, 16) {
			break
		}
		if w != x1_1.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstore {
			break
		}
		if x2.AuxInt != i-3 {
			break
		}
		if x2.Aux != s {
			break
		}
		mem := x2.Args[2]
		if ptr != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64UBFX {
			break
		}
		if x2_1.AuxInt != armBFAuxInt(24, 8) {
			break
		}
		if w != x2_1.Args[0] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = i - 3
		v.Aux = s
		v.AddArg(ptr)
		v0 := b.NewValue0(x2.Pos, OpARM64REVW, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [3] {s} p w x0:(MOVBstore [2] {s} p (UBFX [armBFAuxInt(8, 24)] w) x1:(MOVBstore [1] {s} p1:(ADD ptr1 idx1) (UBFX [armBFAuxInt(16, 16)] w) x2:(MOVBstoreidx ptr0 idx0 (UBFX [armBFAuxInt(24, 8)] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2)
	// result: (MOVWstoreidx ptr0 idx0 (REVW <w.Type> w) mem)
	for {
		if v.AuxInt != 3 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		w := v.Args[1]
		x0 := v.Args[2]
		if x0.Op != OpARM64MOVBstore {
			break
		}
		if x0.AuxInt != 2 {
			break
		}
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64UBFX {
			break
		}
		if x0_1.AuxInt != armBFAuxInt(8, 24) {
			break
		}
		if w != x0_1.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore {
			break
		}
		if x1.AuxInt != 1 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64UBFX {
			break
		}
		if x1_1.AuxInt != armBFAuxInt(16, 16) {
			break
		}
		if w != x1_1.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x2.Args[3]
		ptr0 := x2.Args[0]
		idx0 := x2.Args[1]
		x2_2 := x2.Args[2]
		if x2_2.Op != OpARM64UBFX {
			break
		}
		if x2_2.AuxInt != armBFAuxInt(24, 8) {
			break
		}
		if w != x2_2.Args[0] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr0)
		v.AddArg(idx0)
		v0 := b.NewValue0(x1.Pos, OpARM64REVW, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr w x0:(MOVBstore [i-1] {s} ptr (SRLconst [8] (MOVDreg w)) x1:(MOVBstore [i-2] {s} ptr (SRLconst [16] (MOVDreg w)) x2:(MOVBstore [i-3] {s} ptr (SRLconst [24] (MOVDreg w)) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2)
	// result: (MOVWstore [i-3] {s} ptr (REVW <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		w := v.Args[1]
		x0 := v.Args[2]
		if x0.Op != OpARM64MOVBstore {
			break
		}
		if x0.AuxInt != i-1 {
			break
		}
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64SRLconst {
			break
		}
		if x0_1.AuxInt != 8 {
			break
		}
		x0_1_0 := x0_1.Args[0]
		if x0_1_0.Op != OpARM64MOVDreg {
			break
		}
		if w != x0_1_0.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore {
			break
		}
		if x1.AuxInt != i-2 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64SRLconst {
			break
		}
		if x1_1.AuxInt != 16 {
			break
		}
		x1_1_0 := x1_1.Args[0]
		if x1_1_0.Op != OpARM64MOVDreg {
			break
		}
		if w != x1_1_0.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstore {
			break
		}
		if x2.AuxInt != i-3 {
			break
		}
		if x2.Aux != s {
			break
		}
		mem := x2.Args[2]
		if ptr != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64SRLconst {
			break
		}
		if x2_1.AuxInt != 24 {
			break
		}
		x2_1_0 := x2_1.Args[0]
		if x2_1_0.Op != OpARM64MOVDreg {
			break
		}
		if w != x2_1_0.Args[0] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = i - 3
		v.Aux = s
		v.AddArg(ptr)
		v0 := b.NewValue0(x2.Pos, OpARM64REVW, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [3] {s} p w x0:(MOVBstore [2] {s} p (SRLconst [8] (MOVDreg w)) x1:(MOVBstore [1] {s} p1:(ADD ptr1 idx1) (SRLconst [16] (MOVDreg w)) x2:(MOVBstoreidx ptr0 idx0 (SRLconst [24] (MOVDreg w)) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2)
	// result: (MOVWstoreidx ptr0 idx0 (REVW <w.Type> w) mem)
	for {
		if v.AuxInt != 3 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		w := v.Args[1]
		x0 := v.Args[2]
		if x0.Op != OpARM64MOVBstore {
			break
		}
		if x0.AuxInt != 2 {
			break
		}
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64SRLconst {
			break
		}
		if x0_1.AuxInt != 8 {
			break
		}
		x0_1_0 := x0_1.Args[0]
		if x0_1_0.Op != OpARM64MOVDreg {
			break
		}
		if w != x0_1_0.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore {
			break
		}
		if x1.AuxInt != 1 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64SRLconst {
			break
		}
		if x1_1.AuxInt != 16 {
			break
		}
		x1_1_0 := x1_1.Args[0]
		if x1_1_0.Op != OpARM64MOVDreg {
			break
		}
		if w != x1_1_0.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x2.Args[3]
		ptr0 := x2.Args[0]
		idx0 := x2.Args[1]
		x2_2 := x2.Args[2]
		if x2_2.Op != OpARM64SRLconst {
			break
		}
		if x2_2.AuxInt != 24 {
			break
		}
		x2_2_0 := x2_2.Args[0]
		if x2_2_0.Op != OpARM64MOVDreg {
			break
		}
		if w != x2_2_0.Args[0] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr0)
		v.AddArg(idx0)
		v0 := b.NewValue0(x1.Pos, OpARM64REVW, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstore_30(v *Value) bool {
	b := v.Block
	// match: (MOVBstore [i] {s} ptr w x0:(MOVBstore [i-1] {s} ptr (SRLconst [8] w) x1:(MOVBstore [i-2] {s} ptr (SRLconst [16] w) x2:(MOVBstore [i-3] {s} ptr (SRLconst [24] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2)
	// result: (MOVWstore [i-3] {s} ptr (REVW <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		w := v.Args[1]
		x0 := v.Args[2]
		if x0.Op != OpARM64MOVBstore {
			break
		}
		if x0.AuxInt != i-1 {
			break
		}
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64SRLconst {
			break
		}
		if x0_1.AuxInt != 8 {
			break
		}
		if w != x0_1.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore {
			break
		}
		if x1.AuxInt != i-2 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if ptr != x1.Args[0] {
			break
		}
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64SRLconst {
			break
		}
		if x1_1.AuxInt != 16 {
			break
		}
		if w != x1_1.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstore {
			break
		}
		if x2.AuxInt != i-3 {
			break
		}
		if x2.Aux != s {
			break
		}
		mem := x2.Args[2]
		if ptr != x2.Args[0] {
			break
		}
		x2_1 := x2.Args[1]
		if x2_1.Op != OpARM64SRLconst {
			break
		}
		if x2_1.AuxInt != 24 {
			break
		}
		if w != x2_1.Args[0] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = i - 3
		v.Aux = s
		v.AddArg(ptr)
		v0 := b.NewValue0(x2.Pos, OpARM64REVW, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [3] {s} p w x0:(MOVBstore [2] {s} p (SRLconst [8] w) x1:(MOVBstore [1] {s} p1:(ADD ptr1 idx1) (SRLconst [16] w) x2:(MOVBstoreidx ptr0 idx0 (SRLconst [24] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2)
	// result: (MOVWstoreidx ptr0 idx0 (REVW <w.Type> w) mem)
	for {
		if v.AuxInt != 3 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		w := v.Args[1]
		x0 := v.Args[2]
		if x0.Op != OpARM64MOVBstore {
			break
		}
		if x0.AuxInt != 2 {
			break
		}
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64SRLconst {
			break
		}
		if x0_1.AuxInt != 8 {
			break
		}
		if w != x0_1.Args[0] {
			break
		}
		x1 := x0.Args[2]
		if x1.Op != OpARM64MOVBstore {
			break
		}
		if x1.AuxInt != 1 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		x1_1 := x1.Args[1]
		if x1_1.Op != OpARM64SRLconst {
			break
		}
		if x1_1.AuxInt != 16 {
			break
		}
		if w != x1_1.Args[0] {
			break
		}
		x2 := x1.Args[2]
		if x2.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x2.Args[3]
		ptr0 := x2.Args[0]
		idx0 := x2.Args[1]
		x2_2 := x2.Args[2]
		if x2_2.Op != OpARM64SRLconst {
			break
		}
		if x2_2.AuxInt != 24 {
			break
		}
		if w != x2_2.Args[0] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr0)
		v.AddArg(idx0)
		v0 := b.NewValue0(x1.Pos, OpARM64REVW, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr w x:(MOVBstore [i-1] {s} ptr (SRLconst [8] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr (REV16W <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		w := v.Args[1]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpARM64SRLconst {
			break
		}
		if x_1.AuxInt != 8 {
			break
		}
		if w != x_1.Args[0] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(ptr)
		v0 := b.NewValue0(x.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr1 idx1) w x:(MOVBstoreidx ptr0 idx0 (SRLconst [8] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr0 idx0 (REV16W <w.Type> w) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx1 := v_0.Args[1]
		ptr1 := v_0.Args[0]
		w := v.Args[1]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		ptr0 := x.Args[0]
		idx0 := x.Args[1]
		x_2 := x.Args[2]
		if x_2.Op != OpARM64SRLconst {
			break
		}
		if x_2.AuxInt != 8 {
			break
		}
		if w != x_2.Args[0] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr0)
		v.AddArg(idx0)
		v0 := b.NewValue0(v.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr w x:(MOVBstore [i-1] {s} ptr (UBFX [armBFAuxInt(8, 8)] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr (REV16W <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		w := v.Args[1]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpARM64UBFX {
			break
		}
		if x_1.AuxInt != armBFAuxInt(8, 8) {
			break
		}
		if w != x_1.Args[0] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(ptr)
		v0 := b.NewValue0(x.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr1 idx1) w x:(MOVBstoreidx ptr0 idx0 (UBFX [armBFAuxInt(8, 8)] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr0 idx0 (REV16W <w.Type> w) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx1 := v_0.Args[1]
		ptr1 := v_0.Args[0]
		w := v.Args[1]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		ptr0 := x.Args[0]
		idx0 := x.Args[1]
		x_2 := x.Args[2]
		if x_2.Op != OpARM64UBFX {
			break
		}
		if x_2.AuxInt != armBFAuxInt(8, 8) {
			break
		}
		if w != x_2.Args[0] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr0)
		v.AddArg(idx0)
		v0 := b.NewValue0(v.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr w x:(MOVBstore [i-1] {s} ptr (SRLconst [8] (MOVDreg w)) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr (REV16W <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		w := v.Args[1]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpARM64SRLconst {
			break
		}
		if x_1.AuxInt != 8 {
			break
		}
		x_1_0 := x_1.Args[0]
		if x_1_0.Op != OpARM64MOVDreg {
			break
		}
		if w != x_1_0.Args[0] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(ptr)
		v0 := b.NewValue0(x.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr1 idx1) w x:(MOVBstoreidx ptr0 idx0 (SRLconst [8] (MOVDreg w)) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr0 idx0 (REV16W <w.Type> w) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx1 := v_0.Args[1]
		ptr1 := v_0.Args[0]
		w := v.Args[1]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		ptr0 := x.Args[0]
		idx0 := x.Args[1]
		x_2 := x.Args[2]
		if x_2.Op != OpARM64SRLconst {
			break
		}
		if x_2.AuxInt != 8 {
			break
		}
		x_2_0 := x_2.Args[0]
		if x_2_0.Op != OpARM64MOVDreg {
			break
		}
		if w != x_2_0.Args[0] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr0)
		v.AddArg(idx0)
		v0 := b.NewValue0(v.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} ptr w x:(MOVBstore [i-1] {s} ptr (UBFX [armBFAuxInt(8, 24)] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr (REV16W <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		w := v.Args[1]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpARM64UBFX {
			break
		}
		if x_1.AuxInt != armBFAuxInt(8, 24) {
			break
		}
		if w != x_1.Args[0] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(ptr)
		v0 := b.NewValue0(x.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr1 idx1) w x:(MOVBstoreidx ptr0 idx0 (UBFX [armBFAuxInt(8, 24)] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr0 idx0 (REV16W <w.Type> w) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx1 := v_0.Args[1]
		ptr1 := v_0.Args[0]
		w := v.Args[1]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		ptr0 := x.Args[0]
		idx0 := x.Args[1]
		x_2 := x.Args[2]
		if x_2.Op != OpARM64UBFX {
			break
		}
		if x_2.AuxInt != armBFAuxInt(8, 24) {
			break
		}
		if w != x_2.Args[0] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr0)
		v.AddArg(idx0)
		v0 := b.NewValue0(v.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstore_40(v *Value) bool {
	b := v.Block
	// match: (MOVBstore [i] {s} ptr w x:(MOVBstore [i-1] {s} ptr (SRLconst [8] (MOVDreg w)) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstore [i-1] {s} ptr (REV16W <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		w := v.Args[1]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpARM64SRLconst {
			break
		}
		if x_1.AuxInt != 8 {
			break
		}
		x_1_0 := x_1.Args[0]
		if x_1_0.Op != OpARM64MOVDreg {
			break
		}
		if w != x_1_0.Args[0] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(ptr)
		v0 := b.NewValue0(x.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [1] {s} (ADD ptr1 idx1) w x:(MOVBstoreidx ptr0 idx0 (SRLconst [8] (MOVDreg w)) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstoreidx ptr0 idx0 (REV16W <w.Type> w) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx1 := v_0.Args[1]
		ptr1 := v_0.Args[0]
		w := v.Args[1]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		ptr0 := x.Args[0]
		idx0 := x.Args[1]
		x_2 := x.Args[2]
		if x_2.Op != OpARM64SRLconst {
			break
		}
		if x_2.AuxInt != 8 {
			break
		}
		x_2_0 := x_2.Args[0]
		if x_2_0.Op != OpARM64MOVDreg {
			break
		}
		if w != x_2_0.Args[0] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr0)
		v.AddArg(idx0)
		v0 := b.NewValue0(v.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstoreidx_0(v *Value) bool {
	// match: (MOVBstoreidx ptr (MOVDconst [c]) val mem)
	// cond:
	// result: (MOVBstore [c] ptr val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx (MOVDconst [c]) idx val mem)
	// cond:
	// result: (MOVBstore [c] idx val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		idx := v.Args[1]
		val := v.Args[2]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = c
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVDconst [0]) mem)
	// cond:
	// result: (MOVBstorezeroidx ptr idx mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVBstorezeroidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVBreg x) mem)
	// cond:
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVBreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVBUreg x) mem)
	// cond:
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVBUreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVHreg x) mem)
	// cond:
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVHreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVHUreg x) mem)
	// cond:
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVHUreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVWreg x) mem)
	// cond:
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx (MOVWUreg x) mem)
	// cond:
	// result: (MOVBstoreidx ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVBstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx ptr (ADDconst [1] idx) (SRLconst [8] w) x:(MOVBstoreidx ptr idx w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstoreidx ptr idx w mem)
	for {
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ADDconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		idx := v_1.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64SRLconst {
			break
		}
		if v_2.AuxInt != 8 {
			break
		}
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		if ptr != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstoreidx_10(v *Value) bool {
	b := v.Block
	// match: (MOVBstoreidx ptr (ADDconst [3] idx) w x0:(MOVBstoreidx ptr (ADDconst [2] idx) (UBFX [armBFAuxInt(8, 24)] w) x1:(MOVBstoreidx ptr (ADDconst [1] idx) (UBFX [armBFAuxInt(16, 16)] w) x2:(MOVBstoreidx ptr idx (UBFX [armBFAuxInt(24, 8)] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2)
	// result: (MOVWstoreidx ptr idx (REVW <w.Type> w) mem)
	for {
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ADDconst {
			break
		}
		if v_1.AuxInt != 3 {
			break
		}
		idx := v_1.Args[0]
		w := v.Args[2]
		x0 := v.Args[3]
		if x0.Op != OpARM64MOVBstoreidx {
			break
		}
		_ = x0.Args[3]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64ADDconst {
			break
		}
		if x0_1.AuxInt != 2 {
			break
		}
		if idx != x0_1.Args[0] {
			break
		}
		x0_2 := x0.Args[2]
		if x0_2.Op != OpARM64UBFX {
			break
		}
		if x0_2.AuxInt != armBFAuxInt(8, 24) {
			break
		}
		if w != x0_2.Args[0] {
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 1 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		x1_2 := x1.Args[2]
		if x1_2.Op != OpARM64UBFX {
			break
		}
		if x1_2.AuxInt != armBFAuxInt(16, 16) {
			break
		}
		if w != x1_2.Args[0] {
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
		if idx != x2.Args[1] {
			break
		}
		x2_2 := x2.Args[2]
		if x2_2.Op != OpARM64UBFX {
			break
		}
		if x2_2.AuxInt != armBFAuxInt(24, 8) {
			break
		}
		if w != x2_2.Args[0] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v0 := b.NewValue0(v.Pos, OpARM64REVW, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx w x0:(MOVBstoreidx ptr (ADDconst [1] idx) (UBFX [armBFAuxInt(8, 24)] w) x1:(MOVBstoreidx ptr (ADDconst [2] idx) (UBFX [armBFAuxInt(16, 16)] w) x2:(MOVBstoreidx ptr (ADDconst [3] idx) (UBFX [armBFAuxInt(24, 8)] w) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2)
	// result: (MOVWstoreidx ptr idx w mem)
	for {
		_ = v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		w := v.Args[2]
		x0 := v.Args[3]
		if x0.Op != OpARM64MOVBstoreidx {
			break
		}
		_ = x0.Args[3]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64ADDconst {
			break
		}
		if x0_1.AuxInt != 1 {
			break
		}
		if idx != x0_1.Args[0] {
			break
		}
		x0_2 := x0.Args[2]
		if x0_2.Op != OpARM64UBFX {
			break
		}
		if x0_2.AuxInt != armBFAuxInt(8, 24) {
			break
		}
		if w != x0_2.Args[0] {
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 2 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		x1_2 := x1.Args[2]
		if x1_2.Op != OpARM64UBFX {
			break
		}
		if x1_2.AuxInt != armBFAuxInt(16, 16) {
			break
		}
		if w != x1_2.Args[0] {
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
		if x2_1.Op != OpARM64ADDconst {
			break
		}
		if x2_1.AuxInt != 3 {
			break
		}
		if idx != x2_1.Args[0] {
			break
		}
		x2_2 := x2.Args[2]
		if x2_2.Op != OpARM64UBFX {
			break
		}
		if x2_2.AuxInt != armBFAuxInt(24, 8) {
			break
		}
		if w != x2_2.Args[0] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx ptr (ADDconst [1] idx) w x:(MOVBstoreidx ptr idx (UBFX [armBFAuxInt(8, 8)] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstoreidx ptr idx (REV16W <w.Type> w) mem)
	for {
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ADDconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		idx := v_1.Args[0]
		w := v.Args[2]
		x := v.Args[3]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		if ptr != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		x_2 := x.Args[2]
		if x_2.Op != OpARM64UBFX {
			break
		}
		if x_2.AuxInt != armBFAuxInt(8, 8) {
			break
		}
		if w != x_2.Args[0] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v0 := b.NewValue0(v.Pos, OpARM64REV16W, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx ptr idx w x:(MOVBstoreidx ptr (ADDconst [1] idx) (UBFX [armBFAuxInt(8, 8)] w) mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstoreidx ptr idx w mem)
	for {
		_ = v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		w := v.Args[2]
		x := v.Args[3]
		if x.Op != OpARM64MOVBstoreidx {
			break
		}
		mem := x.Args[3]
		if ptr != x.Args[0] {
			break
		}
		x_1 := x.Args[1]
		if x_1.Op != OpARM64ADDconst {
			break
		}
		if x_1.AuxInt != 1 {
			break
		}
		if idx != x_1.Args[0] {
			break
		}
		x_2 := x.Args[2]
		if x_2.Op != OpARM64UBFX {
			break
		}
		if x_2.AuxInt != armBFAuxInt(8, 8) {
			break
		}
		if w != x_2.Args[0] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstorezero_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstorezero [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBstorezero)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstorezero [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVBstorezero)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstorezero [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVBstorezeroidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVBstorezeroidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstorezero [i] {s} ptr0 x:(MOVBstorezero [j] {s} ptr1 mem))
	// cond: x.Uses == 1 && areAdjacentOffsets(i,j,1) && is32Bit(min(i,j)) && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVHstorezero [min(i,j)] {s} ptr0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		ptr0 := v.Args[0]
		x := v.Args[1]
		if x.Op != OpARM64MOVBstorezero {
			break
		}
		j := x.AuxInt
		if x.Aux != s {
			break
		}
		mem := x.Args[1]
		ptr1 := x.Args[0]
		if !(x.Uses == 1 && areAdjacentOffsets(i, j, 1) && is32Bit(min(i, j)) && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = min(i, j)
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstorezero [1] {s} (ADD ptr0 idx0) x:(MOVBstorezeroidx ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVHstorezeroidx ptr1 idx1 mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		x := v.Args[1]
		if x.Op != OpARM64MOVBstorezeroidx {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstorezeroidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVBstorezeroidx_0(v *Value) bool {
	// match: (MOVBstorezeroidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVBstorezero [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVBstorezero)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstorezeroidx (MOVDconst [c]) idx mem)
	// cond:
	// result: (MOVBstorezero [c] idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		idx := v.Args[1]
		v.reset(OpARM64MOVBstorezero)
		v.AuxInt = c
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstorezeroidx ptr (ADDconst [1] idx) x:(MOVBstorezeroidx ptr idx mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVHstorezeroidx ptr idx mem)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ADDconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		idx := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVBstorezeroidx {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVHstorezeroidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVDload [off] {sym} ptr (FMOVDstore [off] {sym} ptr val _))
	// cond:
	// result: (FMOVDfpgp val)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FMOVDstore {
			break
		}
		if v_1.AuxInt != off {
			break
		}
		if v_1.Aux != sym {
			break
		}
		_ = v_1.Args[2]
		if ptr != v_1.Args[0] {
			break
		}
		val := v_1.Args[1]
		v.reset(OpARM64FMOVDfpgp)
		v.AddArg(val)
		return true
	}
	// match: (MOVDload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVDload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDloadidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVDloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDload [off] {sym} (ADDshiftLL [3] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDloadidx8 ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 3 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVDloadidx8)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVDload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDload [off] {sym} ptr (MOVDstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDstorezero {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (MOVDload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVDconst [int64(read64(sym, off, config.BigEndian))])
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSB {
			break
		}
		if !(symIsRO(sym)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(read64(sym, off, config.BigEndian))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDloadidx_0(v *Value) bool {
	// match: (MOVDloadidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVDload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVDload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDloadidx (MOVDconst [c]) ptr mem)
	// cond:
	// result: (MOVDload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARM64MOVDload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDloadidx ptr (SLLconst [3] idx) mem)
	// cond:
	// result: (MOVDloadidx8 ptr idx mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		if v_1.AuxInt != 3 {
			break
		}
		idx := v_1.Args[0]
		v.reset(OpARM64MOVDloadidx8)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDloadidx (SLLconst [3] idx) ptr mem)
	// cond:
	// result: (MOVDloadidx8 ptr idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v.Args[1]
		v.reset(OpARM64MOVDloadidx8)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDloadidx ptr idx (MOVDstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDstorezeroidx {
			break
		}
		_ = v_2.Args[2]
		ptr2 := v_2.Args[0]
		idx2 := v_2.Args[1]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDloadidx8_0(v *Value) bool {
	// match: (MOVDloadidx8 ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVDload [c<<3] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVDload)
		v.AuxInt = c << 3
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDloadidx8 ptr idx (MOVDstorezeroidx8 ptr2 idx2 _))
	// cond: isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)
	// result: (MOVDconst [0])
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDstorezeroidx8 {
			break
		}
		_ = v_2.Args[2]
		ptr2 := v_2.Args[0]
		idx2 := v_2.Args[1]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDreg_0(v *Value) bool {
	// match: (MOVDreg x)
	// cond: x.Uses == 1
	// result: (MOVDnop x)
	for {
		x := v.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.reset(OpARM64MOVDnop)
		v.AddArg(x)
		return true
	}
	// match: (MOVDreg (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = c
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstore_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVDstore [off] {sym} ptr (FMOVDfpgp val) mem)
	// cond:
	// result: (FMOVDstore [off] {sym} ptr val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FMOVDfpgp {
			break
		}
		val := v_1.Args[0]
		v.reset(OpARM64FMOVDstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstoreidx ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVDstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstore [off] {sym} (ADDshiftLL [3] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstoreidx8 ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 3 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVDstoreidx8)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstore [off] {sym} ptr (MOVDconst [0]) mem)
	// cond:
	// result: (MOVDstorezero [off] {sym} ptr mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstoreidx_0(v *Value) bool {
	// match: (MOVDstoreidx ptr (MOVDconst [c]) val mem)
	// cond:
	// result: (MOVDstore [c] ptr val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARM64MOVDstore)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstoreidx (MOVDconst [c]) idx val mem)
	// cond:
	// result: (MOVDstore [c] idx val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		idx := v.Args[1]
		val := v.Args[2]
		v.reset(OpARM64MOVDstore)
		v.AuxInt = c
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstoreidx ptr (SLLconst [3] idx) val mem)
	// cond:
	// result: (MOVDstoreidx8 ptr idx val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		if v_1.AuxInt != 3 {
			break
		}
		idx := v_1.Args[0]
		val := v.Args[2]
		v.reset(OpARM64MOVDstoreidx8)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstoreidx (SLLconst [3] idx) ptr val mem)
	// cond:
	// result: (MOVDstoreidx8 ptr idx val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v.Args[1]
		val := v.Args[2]
		v.reset(OpARM64MOVDstoreidx8)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstoreidx ptr idx (MOVDconst [0]) mem)
	// cond:
	// result: (MOVDstorezeroidx ptr idx mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVDstorezeroidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstoreidx8_0(v *Value) bool {
	// match: (MOVDstoreidx8 ptr (MOVDconst [c]) val mem)
	// cond:
	// result: (MOVDstore [c<<3] ptr val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARM64MOVDstore)
		v.AuxInt = c << 3
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstoreidx8 ptr idx (MOVDconst [0]) mem)
	// cond:
	// result: (MOVDstorezeroidx8 ptr idx mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVDstorezeroidx8)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstorezero_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVDstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDstorezero [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstorezero [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstorezero [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstorezeroidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVDstorezeroidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstorezero [off] {sym} (ADDshiftLL [3] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVDstorezeroidx8 ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 3 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVDstorezeroidx8)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstorezero [i] {s} ptr0 x:(MOVDstorezero [j] {s} ptr1 mem))
	// cond: x.Uses == 1 && areAdjacentOffsets(i,j,8) && is32Bit(min(i,j)) && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVQstorezero [min(i,j)] {s} ptr0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		ptr0 := v.Args[0]
		x := v.Args[1]
		if x.Op != OpARM64MOVDstorezero {
			break
		}
		j := x.AuxInt
		if x.Aux != s {
			break
		}
		mem := x.Args[1]
		ptr1 := x.Args[0]
		if !(x.Uses == 1 && areAdjacentOffsets(i, j, 8) && is32Bit(min(i, j)) && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVQstorezero)
		v.AuxInt = min(i, j)
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstorezero [8] {s} p0:(ADD ptr0 idx0) x:(MOVDstorezeroidx ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVQstorezero [0] {s} p0 mem)
	for {
		if v.AuxInt != 8 {
			break
		}
		s := v.Aux
		_ = v.Args[1]
		p0 := v.Args[0]
		if p0.Op != OpARM64ADD {
			break
		}
		idx0 := p0.Args[1]
		ptr0 := p0.Args[0]
		x := v.Args[1]
		if x.Op != OpARM64MOVDstorezeroidx {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVQstorezero)
		v.AuxInt = 0
		v.Aux = s
		v.AddArg(p0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstorezero [8] {s} p0:(ADDshiftLL [3] ptr0 idx0) x:(MOVDstorezeroidx8 ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVQstorezero [0] {s} p0 mem)
	for {
		if v.AuxInt != 8 {
			break
		}
		s := v.Aux
		_ = v.Args[1]
		p0 := v.Args[0]
		if p0.Op != OpARM64ADDshiftLL {
			break
		}
		if p0.AuxInt != 3 {
			break
		}
		idx0 := p0.Args[1]
		ptr0 := p0.Args[0]
		x := v.Args[1]
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
		v.AuxInt = 0
		v.Aux = s
		v.AddArg(p0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstorezeroidx_0(v *Value) bool {
	// match: (MOVDstorezeroidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVDstorezero [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstorezeroidx (MOVDconst [c]) idx mem)
	// cond:
	// result: (MOVDstorezero [c] idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		idx := v.Args[1]
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = c
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstorezeroidx ptr (SLLconst [3] idx) mem)
	// cond:
	// result: (MOVDstorezeroidx8 ptr idx mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		if v_1.AuxInt != 3 {
			break
		}
		idx := v_1.Args[0]
		v.reset(OpARM64MOVDstorezeroidx8)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstorezeroidx (SLLconst [3] idx) ptr mem)
	// cond:
	// result: (MOVDstorezeroidx8 ptr idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v.Args[1]
		v.reset(OpARM64MOVDstorezeroidx8)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstorezeroidx8_0(v *Value) bool {
	// match: (MOVDstorezeroidx8 ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVDstorezero [c<<3] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = c << 3
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHUload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVHUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHUload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHUload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHUloadidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHUloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (ADDshiftLL [1] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHUloadidx2 ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHUloadidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHUload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHUload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUload [off] {sym} ptr (MOVHstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVHstorezero {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (MOVHUload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVDconst [int64(read16(sym, off, config.BigEndian))])
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSB {
			break
		}
		if !(symIsRO(sym)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(read16(sym, off, config.BigEndian))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHUloadidx_0(v *Value) bool {
	// match: (MOVHUloadidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVHUload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVHUload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUloadidx (MOVDconst [c]) ptr mem)
	// cond:
	// result: (MOVHUload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARM64MOVHUload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUloadidx ptr (SLLconst [1] idx) mem)
	// cond:
	// result: (MOVHUloadidx2 ptr idx mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		idx := v_1.Args[0]
		v.reset(OpARM64MOVHUloadidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUloadidx ptr (ADD idx idx) mem)
	// cond:
	// result: (MOVHUloadidx2 ptr idx mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ADD {
			break
		}
		idx := v_1.Args[1]
		if idx != v_1.Args[0] {
			break
		}
		v.reset(OpARM64MOVHUloadidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUloadidx (ADD idx idx) ptr mem)
	// cond:
	// result: (MOVHUloadidx2 ptr idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		if idx != v_0.Args[0] {
			break
		}
		ptr := v.Args[1]
		v.reset(OpARM64MOVHUloadidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUloadidx ptr idx (MOVHstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVHstorezeroidx {
			break
		}
		_ = v_2.Args[2]
		ptr2 := v_2.Args[0]
		idx2 := v_2.Args[1]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHUloadidx2_0(v *Value) bool {
	// match: (MOVHUloadidx2 ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVHUload [c<<1] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVHUload)
		v.AuxInt = c << 1
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUloadidx2 ptr idx (MOVHstorezeroidx2 ptr2 idx2 _))
	// cond: isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)
	// result: (MOVDconst [0])
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVHstorezeroidx2 {
			break
		}
		_ = v_2.Args[2]
		ptr2 := v_2.Args[0]
		idx2 := v_2.Args[1]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHUreg_0(v *Value) bool {
	// match: (MOVHUreg x:(MOVBUload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHUloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUloadidx2 _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHUloadidx2 {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (ANDconst [c] x))
	// cond:
	// result: (ANDconst [c&(1<<16-1)] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ANDconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = c & (1<<16 - 1)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [int64(uint16(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(uint16(c))
		return true
	}
	// match: (MOVHUreg (SLLconst [sc] x))
	// cond: isARM64BFMask(sc, 1<<16-1, sc)
	// result: (UBFIZ [armBFAuxInt(sc, arm64BFWidth(1<<16-1, sc))] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<16-1, sc)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(sc, arm64BFWidth(1<<16-1, sc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHUreg_10(v *Value) bool {
	// match: (MOVHUreg (SRLconst [sc] x))
	// cond: isARM64BFMask(sc, 1<<16-1, 0)
	// result: (UBFX [armBFAuxInt(sc, 16)] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<16-1, 0)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(sc, 16)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVHload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHloadidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHload [off] {sym} (ADDshiftLL [1] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHloadidx2 ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHloadidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHload [off] {sym} ptr (MOVHstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVHstorezero {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHloadidx_0(v *Value) bool {
	// match: (MOVHloadidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVHload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVHload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHloadidx (MOVDconst [c]) ptr mem)
	// cond:
	// result: (MOVHload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARM64MOVHload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHloadidx ptr (SLLconst [1] idx) mem)
	// cond:
	// result: (MOVHloadidx2 ptr idx mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		idx := v_1.Args[0]
		v.reset(OpARM64MOVHloadidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHloadidx ptr (ADD idx idx) mem)
	// cond:
	// result: (MOVHloadidx2 ptr idx mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ADD {
			break
		}
		idx := v_1.Args[1]
		if idx != v_1.Args[0] {
			break
		}
		v.reset(OpARM64MOVHloadidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHloadidx (ADD idx idx) ptr mem)
	// cond:
	// result: (MOVHloadidx2 ptr idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		if idx != v_0.Args[0] {
			break
		}
		ptr := v.Args[1]
		v.reset(OpARM64MOVHloadidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHloadidx ptr idx (MOVHstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVHstorezeroidx {
			break
		}
		_ = v_2.Args[2]
		ptr2 := v_2.Args[0]
		idx2 := v_2.Args[1]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHloadidx2_0(v *Value) bool {
	// match: (MOVHloadidx2 ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVHload [c<<1] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVHload)
		v.AuxInt = c << 1
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHloadidx2 ptr idx (MOVHstorezeroidx2 ptr2 idx2 _))
	// cond: isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)
	// result: (MOVDconst [0])
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVHstorezeroidx2 {
			break
		}
		_ = v_2.Args[2]
		ptr2 := v_2.Args[0]
		idx2 := v_2.Args[1]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHreg_0(v *Value) bool {
	// match: (MOVHreg x:(MOVBload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHloadidx2 _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHloadidx2 {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHreg_10(v *Value) bool {
	// match: (MOVHreg (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [int64(int16(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(int16(c))
		return true
	}
	// match: (MOVHreg (SLLconst [lc] x))
	// cond: lc < 16
	// result: (SBFIZ [armBFAuxInt(lc, 16-lc)] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := v_0.AuxInt
		x := v_0.Args[0]
		if !(lc < 16) {
			break
		}
		v.reset(OpARM64SBFIZ)
		v.AuxInt = armBFAuxInt(lc, 16-lc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstore_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVHstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstoreidx ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [off] {sym} (ADDshiftLL [1] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVDconst [0]) mem)
	// cond:
	// result: (MOVHstorezero [off] {sym} ptr mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHreg x) mem)
	// cond:
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVHreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MOVHstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVHUreg x) mem)
	// cond:
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MOVHstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWreg x) mem)
	// cond:
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MOVHstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [off] {sym} ptr (MOVWUreg x) mem)
	// cond:
	// result: (MOVHstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MOVHstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [i] {s} ptr0 (SRLconst [16] w) x:(MOVHstore [i-2] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVWstore [i-2] {s} ptr0 w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 16 {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstore {
			break
		}
		if x.AuxInt != i-2 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = i - 2
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstore_10(v *Value) bool {
	b := v.Block
	// match: (MOVHstore [2] {s} (ADD ptr0 idx0) (SRLconst [16] w) x:(MOVHstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVWstoreidx ptr1 idx1 w mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 16 {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [2] {s} (ADDshiftLL [1] ptr0 idx0) (SRLconst [16] w) x:(MOVHstoreidx2 ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVWstoreidx ptr1 (SLLconst <idx1.Type> [1] idx1) w mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 16 {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstoreidx2 {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr1)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = 1
		v0.AddArg(idx1)
		v.AddArg(v0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [i] {s} ptr0 (UBFX [armBFAuxInt(16, 16)] w) x:(MOVHstore [i-2] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVWstore [i-2] {s} ptr0 w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64UBFX {
			break
		}
		if v_1.AuxInt != armBFAuxInt(16, 16) {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstore {
			break
		}
		if x.AuxInt != i-2 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = i - 2
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [2] {s} (ADD ptr0 idx0) (UBFX [armBFAuxInt(16, 16)] w) x:(MOVHstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVWstoreidx ptr1 idx1 w mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64UBFX {
			break
		}
		if v_1.AuxInt != armBFAuxInt(16, 16) {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [2] {s} (ADDshiftLL [1] ptr0 idx0) (UBFX [armBFAuxInt(16, 16)] w) x:(MOVHstoreidx2 ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVWstoreidx ptr1 (SLLconst <idx1.Type> [1] idx1) w mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64UBFX {
			break
		}
		if v_1.AuxInt != armBFAuxInt(16, 16) {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstoreidx2 {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr1)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = 1
		v0.AddArg(idx1)
		v.AddArg(v0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [i] {s} ptr0 (SRLconst [16] (MOVDreg w)) x:(MOVHstore [i-2] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVWstore [i-2] {s} ptr0 w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 16 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVDreg {
			break
		}
		w := v_1_0.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstore {
			break
		}
		if x.AuxInt != i-2 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = i - 2
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [2] {s} (ADD ptr0 idx0) (SRLconst [16] (MOVDreg w)) x:(MOVHstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVWstoreidx ptr1 idx1 w mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 16 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVDreg {
			break
		}
		w := v_1_0.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [2] {s} (ADDshiftLL [1] ptr0 idx0) (SRLconst [16] (MOVDreg w)) x:(MOVHstoreidx2 ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVWstoreidx ptr1 (SLLconst <idx1.Type> [1] idx1) w mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 16 {
			break
		}
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVDreg {
			break
		}
		w := v_1_0.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstoreidx2 {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr1)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = 1
		v0.AddArg(idx1)
		v.AddArg(v0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [i] {s} ptr0 (SRLconst [j] w) x:(MOVHstore [i-2] {s} ptr1 w0:(SRLconst [j-16] w) mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVWstore [i-2] {s} ptr0 w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstore {
			break
		}
		if x.AuxInt != i-2 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpARM64SRLconst {
			break
		}
		if w0.AuxInt != j-16 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = i - 2
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [2] {s} (ADD ptr0 idx0) (SRLconst [j] w) x:(MOVHstoreidx ptr1 idx1 w0:(SRLconst [j-16] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVWstoreidx ptr1 idx1 w0 mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		w0 := x.Args[2]
		if w0.Op != OpARM64SRLconst {
			break
		}
		if w0.AuxInt != j-16 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstore_20(v *Value) bool {
	b := v.Block
	// match: (MOVHstore [2] {s} (ADDshiftLL [1] ptr0 idx0) (SRLconst [j] w) x:(MOVHstoreidx2 ptr1 idx1 w0:(SRLconst [j-16] w) mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVWstoreidx ptr1 (SLLconst <idx1.Type> [1] idx1) w0 mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstoreidx2 {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		w0 := x.Args[2]
		if w0.Op != OpARM64SRLconst {
			break
		}
		if w0.AuxInt != j-16 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		if !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr1)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = 1
		v0.AddArg(idx1)
		v.AddArg(v0)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstoreidx_0(v *Value) bool {
	// match: (MOVHstoreidx ptr (MOVDconst [c]) val mem)
	// cond:
	// result: (MOVHstore [c] ptr val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARM64MOVHstore)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx (MOVDconst [c]) idx val mem)
	// cond:
	// result: (MOVHstore [c] idx val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		idx := v.Args[1]
		val := v.Args[2]
		v.reset(OpARM64MOVHstore)
		v.AuxInt = c
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx ptr (SLLconst [1] idx) val mem)
	// cond:
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		idx := v_1.Args[0]
		val := v.Args[2]
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx ptr (ADD idx idx) val mem)
	// cond:
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ADD {
			break
		}
		idx := v_1.Args[1]
		if idx != v_1.Args[0] {
			break
		}
		val := v.Args[2]
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx (SLLconst [1] idx) ptr val mem)
	// cond:
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		idx := v_0.Args[0]
		ptr := v.Args[1]
		val := v.Args[2]
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx (ADD idx idx) ptr val mem)
	// cond:
	// result: (MOVHstoreidx2 ptr idx val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		if idx != v_0.Args[0] {
			break
		}
		ptr := v.Args[1]
		val := v.Args[2]
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVDconst [0]) mem)
	// cond:
	// result: (MOVHstorezeroidx ptr idx mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVHstorezeroidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVHreg x) mem)
	// cond:
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVHreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVHUreg x) mem)
	// cond:
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVHUreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx ptr idx (MOVWreg x) mem)
	// cond:
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstoreidx_10(v *Value) bool {
	// match: (MOVHstoreidx ptr idx (MOVWUreg x) mem)
	// cond:
	// result: (MOVHstoreidx ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVHstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx ptr (ADDconst [2] idx) (SRLconst [16] w) x:(MOVHstoreidx ptr idx w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstoreidx ptr idx w mem)
	for {
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ADDconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		idx := v_1.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64SRLconst {
			break
		}
		if v_2.AuxInt != 16 {
			break
		}
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpARM64MOVHstoreidx {
			break
		}
		mem := x.Args[3]
		if ptr != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstoreidx2_0(v *Value) bool {
	// match: (MOVHstoreidx2 ptr (MOVDconst [c]) val mem)
	// cond:
	// result: (MOVHstore [c<<1] ptr val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARM64MOVHstore)
		v.AuxInt = c << 1
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVDconst [0]) mem)
	// cond:
	// result: (MOVHstorezeroidx2 ptr idx mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVHstorezeroidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVHreg x) mem)
	// cond:
	// result: (MOVHstoreidx2 ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVHreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVHUreg x) mem)
	// cond:
	// result: (MOVHstoreidx2 ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVHUreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVWreg x) mem)
	// cond:
	// result: (MOVHstoreidx2 ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx2 ptr idx (MOVWUreg x) mem)
	// cond:
	// result: (MOVHstoreidx2 ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVHstoreidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstorezero_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVHstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHstorezero [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstorezero [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstorezero [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstorezeroidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHstorezeroidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstorezero [off] {sym} (ADDshiftLL [1] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVHstorezeroidx2 ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVHstorezeroidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstorezero [i] {s} ptr0 x:(MOVHstorezero [j] {s} ptr1 mem))
	// cond: x.Uses == 1 && areAdjacentOffsets(i,j,2) && is32Bit(min(i,j)) && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVWstorezero [min(i,j)] {s} ptr0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		ptr0 := v.Args[0]
		x := v.Args[1]
		if x.Op != OpARM64MOVHstorezero {
			break
		}
		j := x.AuxInt
		if x.Aux != s {
			break
		}
		mem := x.Args[1]
		ptr1 := x.Args[0]
		if !(x.Uses == 1 && areAdjacentOffsets(i, j, 2) && is32Bit(min(i, j)) && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = min(i, j)
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstorezero [2] {s} (ADD ptr0 idx0) x:(MOVHstorezeroidx ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVWstorezeroidx ptr1 idx1 mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		x := v.Args[1]
		if x.Op != OpARM64MOVHstorezeroidx {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstorezeroidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstorezero [2] {s} (ADDshiftLL [1] ptr0 idx0) x:(MOVHstorezeroidx2 ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVWstorezeroidx ptr1 (SLLconst <idx1.Type> [1] idx1) mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		x := v.Args[1]
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
		v.AddArg(ptr1)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = 1
		v0.AddArg(idx1)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstorezeroidx_0(v *Value) bool {
	// match: (MOVHstorezeroidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVHstorezero [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstorezeroidx (MOVDconst [c]) idx mem)
	// cond:
	// result: (MOVHstorezero [c] idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		idx := v.Args[1]
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = c
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstorezeroidx ptr (SLLconst [1] idx) mem)
	// cond:
	// result: (MOVHstorezeroidx2 ptr idx mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		idx := v_1.Args[0]
		v.reset(OpARM64MOVHstorezeroidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstorezeroidx ptr (ADD idx idx) mem)
	// cond:
	// result: (MOVHstorezeroidx2 ptr idx mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ADD {
			break
		}
		idx := v_1.Args[1]
		if idx != v_1.Args[0] {
			break
		}
		v.reset(OpARM64MOVHstorezeroidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstorezeroidx (SLLconst [1] idx) ptr mem)
	// cond:
	// result: (MOVHstorezeroidx2 ptr idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		idx := v_0.Args[0]
		ptr := v.Args[1]
		v.reset(OpARM64MOVHstorezeroidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstorezeroidx (ADD idx idx) ptr mem)
	// cond:
	// result: (MOVHstorezeroidx2 ptr idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		if idx != v_0.Args[0] {
			break
		}
		ptr := v.Args[1]
		v.reset(OpARM64MOVHstorezeroidx2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstorezeroidx ptr (ADDconst [2] idx) x:(MOVHstorezeroidx ptr idx mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVWstorezeroidx ptr idx mem)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ADDconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		idx := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVHstorezeroidx {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVWstorezeroidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstorezeroidx2_0(v *Value) bool {
	// match: (MOVHstorezeroidx2 ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVHstorezero [c<<1] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVHstorezero)
		v.AuxInt = c << 1
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVQstorezero_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVQstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVQstorezero [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVQstorezero)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstorezero [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVQstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVQstorezero)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWUload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVWUload [off] {sym} ptr (FMOVSstore [off] {sym} ptr val _))
	// cond:
	// result: (FMOVSfpgp val)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FMOVSstore {
			break
		}
		if v_1.AuxInt != off {
			break
		}
		if v_1.Aux != sym {
			break
		}
		_ = v_1.Args[2]
		if ptr != v_1.Args[0] {
			break
		}
		val := v_1.Args[1]
		v.reset(OpARM64FMOVSfpgp)
		v.AddArg(val)
		return true
	}
	// match: (MOVWUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWUload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWUload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWUload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWUloadidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWUloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWUload [off] {sym} (ADDshiftLL [2] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWUloadidx4 ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWUloadidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWUload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWUload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWUload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWUload [off] {sym} ptr (MOVWstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWstorezero {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (MOVWUload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVDconst [int64(read32(sym, off, config.BigEndian))])
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSB {
			break
		}
		if !(symIsRO(sym)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(read32(sym, off, config.BigEndian))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWUloadidx_0(v *Value) bool {
	// match: (MOVWUloadidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVWUload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVWUload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWUloadidx (MOVDconst [c]) ptr mem)
	// cond:
	// result: (MOVWUload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARM64MOVWUload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWUloadidx ptr (SLLconst [2] idx) mem)
	// cond:
	// result: (MOVWUloadidx4 ptr idx mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		idx := v_1.Args[0]
		v.reset(OpARM64MOVWUloadidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWUloadidx (SLLconst [2] idx) ptr mem)
	// cond:
	// result: (MOVWUloadidx4 ptr idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v.Args[1]
		v.reset(OpARM64MOVWUloadidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWUloadidx ptr idx (MOVWstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWstorezeroidx {
			break
		}
		_ = v_2.Args[2]
		ptr2 := v_2.Args[0]
		idx2 := v_2.Args[1]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWUloadidx4_0(v *Value) bool {
	// match: (MOVWUloadidx4 ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVWUload [c<<2] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVWUload)
		v.AuxInt = c << 2
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWUloadidx4 ptr idx (MOVWstorezeroidx4 ptr2 idx2 _))
	// cond: isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)
	// result: (MOVDconst [0])
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWstorezeroidx4 {
			break
		}
		_ = v_2.Args[2]
		ptr2 := v_2.Args[0]
		idx2 := v_2.Args[1]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWUreg_0(v *Value) bool {
	// match: (MOVWUreg x:(MOVBUload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVWUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHUloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVWUloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUloadidx2 _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHUloadidx2 {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVWUloadidx4 _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVWUloadidx4 {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVBUreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg x:(MOVHUreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWUreg_10(v *Value) bool {
	// match: (MOVWUreg x:(MOVWUreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVWUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (ANDconst [c] x))
	// cond:
	// result: (ANDconst [c&(1<<32-1)] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ANDconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARM64ANDconst)
		v.AuxInt = c & (1<<32 - 1)
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [int64(uint32(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(uint32(c))
		return true
	}
	// match: (MOVWUreg (SLLconst [sc] x))
	// cond: isARM64BFMask(sc, 1<<32-1, sc)
	// result: (UBFIZ [armBFAuxInt(sc, arm64BFWidth(1<<32-1, sc))] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<32-1, sc)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(sc, arm64BFWidth(1<<32-1, sc))
		v.AddArg(x)
		return true
	}
	// match: (MOVWUreg (SRLconst [sc] x))
	// cond: isARM64BFMask(sc, 1<<32-1, 0)
	// result: (UBFX [armBFAuxInt(sc, 32)] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<32-1, 0)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(sc, 32)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVWload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWloadidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off] {sym} (ADDshiftLL [2] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWloadidx4 ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWloadidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off] {sym} ptr (MOVWstorezero [off2] {sym2} ptr2 _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVDconst [0])
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWstorezero {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWloadidx_0(v *Value) bool {
	// match: (MOVWloadidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVWload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVWload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx (MOVDconst [c]) ptr mem)
	// cond:
	// result: (MOVWload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARM64MOVWload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx ptr (SLLconst [2] idx) mem)
	// cond:
	// result: (MOVWloadidx4 ptr idx mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		idx := v_1.Args[0]
		v.reset(OpARM64MOVWloadidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx (SLLconst [2] idx) ptr mem)
	// cond:
	// result: (MOVWloadidx4 ptr idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v.Args[1]
		v.reset(OpARM64MOVWloadidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx ptr idx (MOVWstorezeroidx ptr2 idx2 _))
	// cond: (isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2))
	// result: (MOVDconst [0])
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWstorezeroidx {
			break
		}
		_ = v_2.Args[2]
		ptr2 := v_2.Args[0]
		idx2 := v_2.Args[1]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2) || isSamePtr(ptr, idx2) && isSamePtr(idx, ptr2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWloadidx4_0(v *Value) bool {
	// match: (MOVWloadidx4 ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVWload [c<<2] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVWload)
		v.AuxInt = c << 2
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx4 ptr idx (MOVWstorezeroidx4 ptr2 idx2 _))
	// cond: isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)
	// result: (MOVDconst [0])
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWstorezeroidx4 {
			break
		}
		_ = v_2.Args[2]
		ptr2 := v_2.Args[0]
		idx2 := v_2.Args[1]
		if !(isSamePtr(ptr, ptr2) && isSamePtr(idx, idx2)) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWreg_0(v *Value) bool {
	// match: (MOVWreg x:(MOVBload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWload _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVWload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHUloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWloadidx _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVWloadidx {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWreg_10(v *Value) bool {
	// match: (MOVWreg x:(MOVHloadidx2 _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHloadidx2 {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHUloadidx2 _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHUloadidx2 {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWloadidx4 _ _ _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVWloadidx4 {
			break
		}
		_ = x.Args[2]
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVBUreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVBUreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVHreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVHreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg x:(MOVWreg _))
	// cond:
	// result: (MOVDreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARM64MOVWreg {
			break
		}
		v.reset(OpARM64MOVDreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [int64(int32(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(int32(c))
		return true
	}
	// match: (MOVWreg (SLLconst [lc] x))
	// cond: lc < 32
	// result: (SBFIZ [armBFAuxInt(lc, 32-lc)] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := v_0.AuxInt
		x := v_0.Args[0]
		if !(lc < 32) {
			break
		}
		v.reset(OpARM64SBFIZ)
		v.AuxInt = armBFAuxInt(lc, 32-lc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstore_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstore [off] {sym} ptr (FMOVSfpgp val) mem)
	// cond:
	// result: (FMOVSstore [off] {sym} ptr val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FMOVSfpgp {
			break
		}
		val := v_1.Args[0]
		v.reset(OpARM64FMOVSstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off] {sym} (ADD ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstoreidx ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off] {sym} (ADDshiftLL [2] ptr idx) val mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstoreidx4 ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWstoreidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVDconst [0]) mem)
	// cond:
	// result: (MOVWstorezero [off] {sym} ptr mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWreg x) mem)
	// cond:
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MOVWstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWUreg x) mem)
	// cond:
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MOVWstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [i] {s} ptr0 (SRLconst [32] w) x:(MOVWstore [i-4] {s} ptr1 w mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVDstore [i-4] {s} ptr0 w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 32 {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVWstore {
			break
		}
		if x.AuxInt != i-4 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		if w != x.Args[1] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = i - 4
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [4] {s} (ADD ptr0 idx0) (SRLconst [32] w) x:(MOVWstoreidx ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVDstoreidx ptr1 idx1 w mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 32 {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVWstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstore_10(v *Value) bool {
	b := v.Block
	// match: (MOVWstore [4] {s} (ADDshiftLL [2] ptr0 idx0) (SRLconst [32] w) x:(MOVWstoreidx4 ptr1 idx1 w mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVDstoreidx ptr1 (SLLconst <idx1.Type> [2] idx1) w mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != 32 {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVWstoreidx4 {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstoreidx)
		v.AddArg(ptr1)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = 2
		v0.AddArg(idx1)
		v.AddArg(v0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [i] {s} ptr0 (SRLconst [j] w) x:(MOVWstore [i-4] {s} ptr1 w0:(SRLconst [j-32] w) mem))
	// cond: x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVDstore [i-4] {s} ptr0 w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		ptr0 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVWstore {
			break
		}
		if x.AuxInt != i-4 {
			break
		}
		if x.Aux != s {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		w0 := x.Args[1]
		if w0.Op != OpARM64SRLconst {
			break
		}
		if w0.AuxInt != j-32 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		if !(x.Uses == 1 && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = i - 4
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [4] {s} (ADD ptr0 idx0) (SRLconst [j] w) x:(MOVWstoreidx ptr1 idx1 w0:(SRLconst [j-32] w) mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVDstoreidx ptr1 idx1 w0 mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVWstoreidx {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		w0 := x.Args[2]
		if w0.Op != OpARM64SRLconst {
			break
		}
		if w0.AuxInt != j-32 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstoreidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [4] {s} (ADDshiftLL [2] ptr0 idx0) (SRLconst [j] w) x:(MOVWstoreidx4 ptr1 idx1 w0:(SRLconst [j-32] w) mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVDstoreidx ptr1 (SLLconst <idx1.Type> [2] idx1) w0 mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		s := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		j := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVWstoreidx4 {
			break
		}
		mem := x.Args[3]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		w0 := x.Args[2]
		if w0.Op != OpARM64SRLconst {
			break
		}
		if w0.AuxInt != j-32 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		if !(x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstoreidx)
		v.AddArg(ptr1)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = 2
		v0.AddArg(idx1)
		v.AddArg(v0)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstoreidx_0(v *Value) bool {
	// match: (MOVWstoreidx ptr (MOVDconst [c]) val mem)
	// cond:
	// result: (MOVWstore [c] ptr val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARM64MOVWstore)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx (MOVDconst [c]) idx val mem)
	// cond:
	// result: (MOVWstore [c] idx val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		idx := v.Args[1]
		val := v.Args[2]
		v.reset(OpARM64MOVWstore)
		v.AuxInt = c
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx ptr (SLLconst [2] idx) val mem)
	// cond:
	// result: (MOVWstoreidx4 ptr idx val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		idx := v_1.Args[0]
		val := v.Args[2]
		v.reset(OpARM64MOVWstoreidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx (SLLconst [2] idx) ptr val mem)
	// cond:
	// result: (MOVWstoreidx4 ptr idx val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v.Args[1]
		val := v.Args[2]
		v.reset(OpARM64MOVWstoreidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVDconst [0]) mem)
	// cond:
	// result: (MOVWstorezeroidx ptr idx mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVWstorezeroidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVWreg x) mem)
	// cond:
	// result: (MOVWstoreidx ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx ptr idx (MOVWUreg x) mem)
	// cond:
	// result: (MOVWstoreidx ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVWstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx ptr (ADDconst [4] idx) (SRLconst [32] w) x:(MOVWstoreidx ptr idx w mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVDstoreidx ptr idx w mem)
	for {
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ADDconst {
			break
		}
		if v_1.AuxInt != 4 {
			break
		}
		idx := v_1.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64SRLconst {
			break
		}
		if v_2.AuxInt != 32 {
			break
		}
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpARM64MOVWstoreidx {
			break
		}
		mem := x.Args[3]
		if ptr != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		if w != x.Args[2] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstoreidx4_0(v *Value) bool {
	// match: (MOVWstoreidx4 ptr (MOVDconst [c]) val mem)
	// cond:
	// result: (MOVWstore [c<<2] ptr val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARM64MOVWstore)
		v.AuxInt = c << 2
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx4 ptr idx (MOVDconst [0]) mem)
	// cond:
	// result: (MOVWstorezeroidx4 ptr idx mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVWstorezeroidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx4 ptr idx (MOVWreg x) mem)
	// cond:
	// result: (MOVWstoreidx4 ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVWstoreidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx4 ptr idx (MOVWUreg x) mem)
	// cond:
	// result: (MOVWstoreidx4 ptr idx x mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVWUreg {
			break
		}
		x := v_2.Args[0]
		v.reset(OpARM64MOVWstoreidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstorezero_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstorezero [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstorezero [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstorezero [off] {sym} (ADD ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstorezeroidx ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWstorezeroidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstorezero [off] {sym} (ADDshiftLL [2] ptr idx) mem)
	// cond: off == 0 && sym == nil
	// result: (MOVWstorezeroidx4 ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(off == 0 && sym == nil) {
			break
		}
		v.reset(OpARM64MOVWstorezeroidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstorezero [i] {s} ptr0 x:(MOVWstorezero [j] {s} ptr1 mem))
	// cond: x.Uses == 1 && areAdjacentOffsets(i,j,4) && is32Bit(min(i,j)) && isSamePtr(ptr0, ptr1) && clobber(x)
	// result: (MOVDstorezero [min(i,j)] {s} ptr0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		ptr0 := v.Args[0]
		x := v.Args[1]
		if x.Op != OpARM64MOVWstorezero {
			break
		}
		j := x.AuxInt
		if x.Aux != s {
			break
		}
		mem := x.Args[1]
		ptr1 := x.Args[0]
		if !(x.Uses == 1 && areAdjacentOffsets(i, j, 4) && is32Bit(min(i, j)) && isSamePtr(ptr0, ptr1) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = min(i, j)
		v.Aux = s
		v.AddArg(ptr0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstorezero [4] {s} (ADD ptr0 idx0) x:(MOVWstorezeroidx ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)
	// result: (MOVDstorezeroidx ptr1 idx1 mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADD {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		x := v.Args[1]
		if x.Op != OpARM64MOVWstorezeroidx {
			break
		}
		mem := x.Args[2]
		ptr1 := x.Args[0]
		idx1 := x.Args[1]
		if !(x.Uses == 1 && s == nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstorezeroidx)
		v.AddArg(ptr1)
		v.AddArg(idx1)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstorezero [4] {s} (ADDshiftLL [2] ptr0 idx0) x:(MOVWstorezeroidx4 ptr1 idx1 mem))
	// cond: x.Uses == 1 && s == nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && clobber(x)
	// result: (MOVDstorezeroidx ptr1 (SLLconst <idx1.Type> [2] idx1) mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDshiftLL {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		idx0 := v_0.Args[1]
		ptr0 := v_0.Args[0]
		x := v.Args[1]
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
		v.AddArg(ptr1)
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, idx1.Type)
		v0.AuxInt = 2
		v0.AddArg(idx1)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstorezeroidx_0(v *Value) bool {
	// match: (MOVWstorezeroidx ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVWstorezero [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstorezeroidx (MOVDconst [c]) idx mem)
	// cond:
	// result: (MOVWstorezero [c] idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		idx := v.Args[1]
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = c
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstorezeroidx ptr (SLLconst [2] idx) mem)
	// cond:
	// result: (MOVWstorezeroidx4 ptr idx mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		idx := v_1.Args[0]
		v.reset(OpARM64MOVWstorezeroidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstorezeroidx (SLLconst [2] idx) ptr mem)
	// cond:
	// result: (MOVWstorezeroidx4 ptr idx mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v.Args[1]
		v.reset(OpARM64MOVWstorezeroidx4)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstorezeroidx ptr (ADDconst [4] idx) x:(MOVWstorezeroidx ptr idx mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: (MOVDstorezeroidx ptr idx mem)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ADDconst {
			break
		}
		if v_1.AuxInt != 4 {
			break
		}
		idx := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpARM64MOVWstorezeroidx {
			break
		}
		mem := x.Args[2]
		if ptr != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpARM64MOVDstorezeroidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstorezeroidx4_0(v *Value) bool {
	// match: (MOVWstorezeroidx4 ptr (MOVDconst [c]) mem)
	// cond:
	// result: (MOVWstorezero [c<<2] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVWstorezero)
		v.AuxInt = c << 2
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MSUB_0(v *Value) bool {
	b := v.Block
	// match: (MSUB a x (MOVDconst [-1]))
	// cond:
	// result: (ADD a x)
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != -1 {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MSUB a _ (MOVDconst [0]))
	// cond:
	// result: a
	for {
		_ = v.Args[2]
		a := v.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = a.Type
		v.AddArg(a)
		return true
	}
	// match: (MSUB a x (MOVDconst [1]))
	// cond:
	// result: (SUB a x)
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != 1 {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (SUBshiftLL a x [log2(c)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c-1) && c>=3
	// result: (SUB a (ADDshiftLL <x.Type> x x [log2(c-1)]))
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(isPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c+1) && c>=7
	// result: (ADD a (SUBshiftLL <x.Type> x x [log2(c+1)]))
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(isPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [2]) [log2(c/3)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 3)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [2]) [log2(c/5)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 5)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [3]) [log2(c/7)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 7)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [3]) [log2(c/9)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 9)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MSUB_10(v *Value) bool {
	b := v.Block
	// match: (MSUB a (MOVDconst [-1]) x)
	// cond:
	// result: (ADD a x)
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != -1 {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MSUB a (MOVDconst [0]) _)
	// cond:
	// result: a
	for {
		_ = v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = a.Type
		v.AddArg(a)
		return true
	}
	// match: (MSUB a (MOVDconst [1]) x)
	// cond:
	// result: (SUB a x)
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c)
	// result: (SUBshiftLL a x [log2(c)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c-1) && c>=3
	// result: (SUB a (ADDshiftLL <x.Type> x x [log2(c-1)]))
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c+1) && c>=7
	// result: (ADD a (SUBshiftLL <x.Type> x x [log2(c+1)]))
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [2]) [log2(c/3)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 3)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [2]) [log2(c/5)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 5)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [3]) [log2(c/7)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 7)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [3]) [log2(c/9)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 9)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MSUB_20(v *Value) bool {
	b := v.Block
	// match: (MSUB (MOVDconst [c]) x y)
	// cond:
	// result: (ADDconst [c] (MNEG <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARM64ADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64MNEG, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (MSUB a (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (SUBconst [c*d] a)
	for {
		_ = v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		d := v_2.AuxInt
		v.reset(OpARM64SUBconst)
		v.AuxInt = c * d
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MSUBW_0(v *Value) bool {
	b := v.Block
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: int32(c)==-1
	// result: (ADD a x)
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MSUBW a _ (MOVDconst [c]))
	// cond: int32(c)==0
	// result: a
	for {
		_ = v.Args[2]
		a := v.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpCopy)
		v.Type = a.Type
		v.AddArg(a)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: int32(c)==1
	// result: (SUB a x)
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(int32(c) == 1) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (SUBshiftLL a x [log2(c)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c-1) && int32(c)>=3
	// result: (SUB a (ADDshiftLL <x.Type> x x [log2(c-1)]))
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: isPowerOfTwo(c+1) && int32(c)>=7
	// result: (ADD a (SUBshiftLL <x.Type> x x [log2(c+1)]))
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [2]) [log2(c/3)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 3)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [2]) [log2(c/5)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 5)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [3]) [log2(c/7)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 7)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [3]) [log2(c/9)])
	for {
		_ = v.Args[2]
		a := v.Args[0]
		x := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		c := v_2.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 9)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MSUBW_10(v *Value) bool {
	b := v.Block
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: int32(c)==-1
	// result: (ADD a x)
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) _)
	// cond: int32(c)==0
	// result: a
	for {
		_ = v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpCopy)
		v.Type = a.Type
		v.AddArg(a)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: int32(c)==1
	// result: (SUB a x)
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == 1) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c)
	// result: (SUBshiftLL a x [log2(c)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c-1) && int32(c)>=3
	// result: (SUB a (ADDshiftLL <x.Type> x x [log2(c-1)]))
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64SUB)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c+1) && int32(c)>=7
	// result: (ADD a (SUBshiftLL <x.Type> x x [log2(c+1)]))
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64ADD)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [2]) [log2(c/3)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 3)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [2]) [log2(c/5)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 5)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (ADDshiftLL a (SUBshiftLL <x.Type> x x [3]) [log2(c/7)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c / 7)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64SUBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (SUBshiftLL a (ADDshiftLL <x.Type> x x [3]) [log2(c/9)])
	for {
		x := v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = log2(c / 9)
		v.AddArg(a)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MSUBW_20(v *Value) bool {
	b := v.Block
	// match: (MSUBW (MOVDconst [c]) x y)
	// cond:
	// result: (ADDconst [c] (MNEGW <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARM64ADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64MNEGW, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (MSUBW a (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (SUBconst [int64(int32(c)*int32(d))] a)
	for {
		_ = v.Args[2]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		d := v_2.AuxInt
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64(int32(c) * int32(d))
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MUL_0(v *Value) bool {
	// match: (MUL (NEG x) y)
	// cond:
	// result: (MNEG x y)
	for {
		y := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64NEG {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64MNEG)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (MUL y (NEG x))
	// cond:
	// result: (MNEG x y)
	for {
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64NEG {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MNEG)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (MUL x (MOVDconst [-1]))
	// cond:
	// result: (NEG x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != -1 {
			break
		}
		v.reset(OpARM64NEG)
		v.AddArg(x)
		return true
	}
	// match: (MUL (MOVDconst [-1]) x)
	// cond:
	// result: (NEG x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		v.reset(OpARM64NEG)
		v.AddArg(x)
		return true
	}
	// match: (MUL _ (MOVDconst [0]))
	// cond:
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (MUL (MOVDconst [0]) _)
	// cond:
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (MUL x (MOVDconst [1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MUL (MOVDconst [1]) x)
	// cond:
	// result: x
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (SLLconst [log2(c)] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c)
		v.AddArg(x)
		return true
	}
	// match: (MUL (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c)
	// result: (SLLconst [log2(c)] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MUL_10(v *Value) bool {
	b := v.Block
	// match: (MUL x (MOVDconst [c]))
	// cond: isPowerOfTwo(c-1) && c >= 3
	// result: (ADDshiftLL x x [log2(c-1)])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c - 1)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	// match: (MUL (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c-1) && c >= 3
	// result: (ADDshiftLL x x [log2(c-1)])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c - 1)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: isPowerOfTwo(c+1) && c >= 7
	// result: (ADDshiftLL (NEG <x.Type> x) x [log2(c+1)])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c + 1)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (MUL (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c+1) && c >= 7
	// result: (ADDshiftLL (NEG <x.Type> x) x [log2(c+1)])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c + 1)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (SLLconst [log2(c/3)] (ADDshiftLL <x.Type> x x [1]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 3)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 1
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (SLLconst [log2(c/3)] (ADDshiftLL <x.Type> x x [1]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 3)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 1
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (SLLconst [log2(c/5)] (ADDshiftLL <x.Type> x x [2]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 5)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (SLLconst [log2(c/5)] (ADDshiftLL <x.Type> x x [2]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 5)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (SLLconst [log2(c/7)] (ADDshiftLL <x.Type> (NEG <x.Type> x) x [3]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 7)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v1 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo(c/7)
	// result: (SLLconst [log2(c/7)] (ADDshiftLL <x.Type> (NEG <x.Type> x) x [3]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 7)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v1 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MUL_20(v *Value) bool {
	b := v.Block
	// match: (MUL x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (SLLconst [log2(c/9)] (ADDshiftLL <x.Type> x x [3]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 9)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (SLLconst [log2(c/9)] (ADDshiftLL <x.Type> x x [3]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 9)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [c*d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = c * d
		return true
	}
	// match: (MUL (MOVDconst [d]) (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [c*d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = c * d
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MULW_0(v *Value) bool {
	// match: (MULW (NEG x) y)
	// cond:
	// result: (MNEGW x y)
	for {
		y := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64NEG {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64MNEGW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (MULW y (NEG x))
	// cond:
	// result: (MNEGW x y)
	for {
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64NEG {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARM64MNEGW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: int32(c)==-1
	// result: (NEG x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARM64NEG)
		v.AddArg(x)
		return true
	}
	// match: (MULW (MOVDconst [c]) x)
	// cond: int32(c)==-1
	// result: (NEG x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARM64NEG)
		v.AddArg(x)
		return true
	}
	// match: (MULW _ (MOVDconst [c]))
	// cond: int32(c)==0
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (MULW (MOVDconst [c]) _)
	// cond: int32(c)==0
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: int32(c)==1
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == 1) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MULW (MOVDconst [c]) x)
	// cond: int32(c)==1
	// result: x
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(int32(c) == 1) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (SLLconst [log2(c)] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c)
		v.AddArg(x)
		return true
	}
	// match: (MULW (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c)
	// result: (SLLconst [log2(c)] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MULW_10(v *Value) bool {
	b := v.Block
	// match: (MULW x (MOVDconst [c]))
	// cond: isPowerOfTwo(c-1) && int32(c) >= 3
	// result: (ADDshiftLL x x [log2(c-1)])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c - 1)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	// match: (MULW (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c-1) && int32(c) >= 3
	// result: (ADDshiftLL x x [log2(c-1)])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c - 1)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: isPowerOfTwo(c+1) && int32(c) >= 7
	// result: (ADDshiftLL (NEG <x.Type> x) x [log2(c+1)])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c + 1)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (MULW (MOVDconst [c]) x)
	// cond: isPowerOfTwo(c+1) && int32(c) >= 7
	// result: (ADDshiftLL (NEG <x.Type> x) x [log2(c+1)])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c + 1)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (SLLconst [log2(c/3)] (ADDshiftLL <x.Type> x x [1]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 3)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 1
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULW (MOVDconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (SLLconst [log2(c/3)] (ADDshiftLL <x.Type> x x [1]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 3)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 1
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (SLLconst [log2(c/5)] (ADDshiftLL <x.Type> x x [2]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 5)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULW (MOVDconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (SLLconst [log2(c/5)] (ADDshiftLL <x.Type> x x [2]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 5)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULW x (MOVDconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (SLLconst [log2(c/7)] (ADDshiftLL <x.Type> (NEG <x.Type> x) x [3]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 7)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v1 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULW (MOVDconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (SLLconst [log2(c/7)] (ADDshiftLL <x.Type> (NEG <x.Type> x) x [3]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 7)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v1 := b.NewValue0(v.Pos, OpARM64NEG, x.Type)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MULW_20(v *Value) bool {
	b := v.Block
	// match: (MULW x (MOVDconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (SLLconst [log2(c/9)] (ADDshiftLL <x.Type> x x [3]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 9)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULW (MOVDconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (SLLconst [log2(c/9)] (ADDshiftLL <x.Type> x x [3]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c / 9)
		v0 := b.NewValue0(v.Pos, OpARM64ADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULW (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [int64(int32(c)*int32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(int32(c) * int32(d))
		return true
	}
	// match: (MULW (MOVDconst [d]) (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [int64(int32(c)*int32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(int32(c) * int32(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MVN_0(v *Value) bool {
	// match: (MVN (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [^c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = ^c
		return true
	}
	// match: (MVN x:(SLLconst [c] y))
	// cond: clobberIfDead(x)
	// result: (MVNshiftLL [c] y)
	for {
		x := v.Args[0]
		if x.Op != OpARM64SLLconst {
			break
		}
		c := x.AuxInt
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64MVNshiftLL)
		v.AuxInt = c
		v.AddArg(y)
		return true
	}
	// match: (MVN x:(SRLconst [c] y))
	// cond: clobberIfDead(x)
	// result: (MVNshiftRL [c] y)
	for {
		x := v.Args[0]
		if x.Op != OpARM64SRLconst {
			break
		}
		c := x.AuxInt
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64MVNshiftRL)
		v.AuxInt = c
		v.AddArg(y)
		return true
	}
	// match: (MVN x:(SRAconst [c] y))
	// cond: clobberIfDead(x)
	// result: (MVNshiftRA [c] y)
	for {
		x := v.Args[0]
		if x.Op != OpARM64SRAconst {
			break
		}
		c := x.AuxInt
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64MVNshiftRA)
		v.AuxInt = c
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MVNshiftLL_0(v *Value) bool {
	// match: (MVNshiftLL (MOVDconst [c]) [d])
	// cond:
	// result: (MOVDconst [^int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = ^int64(uint64(c) << uint64(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MVNshiftRA_0(v *Value) bool {
	// match: (MVNshiftRA (MOVDconst [c]) [d])
	// cond:
	// result: (MOVDconst [^(c>>uint64(d))])
	for {
		d := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = ^(c >> uint64(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MVNshiftRL_0(v *Value) bool {
	// match: (MVNshiftRL (MOVDconst [c]) [d])
	// cond:
	// result: (MOVDconst [^int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = ^int64(uint64(c) >> uint64(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64NEG_0(v *Value) bool {
	// match: (NEG (MUL x y))
	// cond:
	// result: (MNEG x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MUL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64MNEG)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (NEG (MULW x y))
	// cond:
	// result: (MNEGW x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MULW {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64MNEGW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (NEG (MOVDconst [c]))
	// cond:
	// result: (MOVDconst [-c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -c
		return true
	}
	// match: (NEG x:(SLLconst [c] y))
	// cond: clobberIfDead(x)
	// result: (NEGshiftLL [c] y)
	for {
		x := v.Args[0]
		if x.Op != OpARM64SLLconst {
			break
		}
		c := x.AuxInt
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64NEGshiftLL)
		v.AuxInt = c
		v.AddArg(y)
		return true
	}
	// match: (NEG x:(SRLconst [c] y))
	// cond: clobberIfDead(x)
	// result: (NEGshiftRL [c] y)
	for {
		x := v.Args[0]
		if x.Op != OpARM64SRLconst {
			break
		}
		c := x.AuxInt
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64NEGshiftRL)
		v.AuxInt = c
		v.AddArg(y)
		return true
	}
	// match: (NEG x:(SRAconst [c] y))
	// cond: clobberIfDead(x)
	// result: (NEGshiftRA [c] y)
	for {
		x := v.Args[0]
		if x.Op != OpARM64SRAconst {
			break
		}
		c := x.AuxInt
		y := x.Args[0]
		if !(clobberIfDead(x)) {
			break
		}
		v.reset(OpARM64NEGshiftRA)
		v.AuxInt = c
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64NEGshiftLL_0(v *Value) bool {
	// match: (NEGshiftLL (MOVDconst [c]) [d])
	// cond:
	// result: (MOVDconst [-int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -int64(uint64(c) << uint64(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64NEGshiftRA_0(v *Value) bool {
	// match: (NEGshiftRA (MOVDconst [c]) [d])
	// cond:
	// result: (MOVDconst [-(c>>uint64(d))])
	for {
		d := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -(c >> uint64(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64NEGshiftRL_0(v *Value) bool {
	// match: (NEGshiftRL (MOVDconst [c]) [d])
	// cond:
	// result: (MOVDconst [-int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -int64(uint64(c) >> uint64(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64NotEqual_0(v *Value) bool {
	// match: (NotEqual (FlagEQ))
	// cond:
	// result: (MOVDconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagEQ {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (NotEqual (FlagLT_ULT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (NotEqual (FlagLT_UGT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagLT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (NotEqual (FlagGT_ULT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_ULT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (NotEqual (FlagGT_UGT))
	// cond:
	// result: (MOVDconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARM64FlagGT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 1
		return true
	}
	// match: (NotEqual (InvertFlags x))
	// cond:
	// result: (NotEqual x)
	for {
		v_0 := v.Args[0]
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
func rewriteValueARM64_OpARM64OR_0(v *Value) bool {
	// match: (OR x (MOVDconst [c]))
	// cond:
	// result: (ORconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (OR (MOVDconst [c]) x)
	// cond:
	// result: (ORconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64ORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (OR x x)
	// cond:
	// result: x
	for {
		x := v.Args[1]
		if x != v.Args[0] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (OR x (MVN y))
	// cond:
	// result: (ORN x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MVN {
			break
		}
		y := v_1.Args[0]
		v.reset(OpARM64ORN)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR (MVN y) x)
	// cond:
	// result: (ORN x y)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MVN {
			break
		}
		y := v_0.Args[0]
		v.reset(OpARM64ORN)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORshiftLL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (OR x1:(SLLconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (ORshiftLL x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (OR x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORshiftRL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (OR x1:(SRLconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (ORshiftRL x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (OR x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORshiftRA x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64OR_10(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OR x1:(SRAconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (ORshiftRA x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (OR (SLL x (ANDconst <t> [63] y)) (CSEL0 <typ.UInt64> {cc} (SRL <typ.UInt64> x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (ROR x (NEG <t> y))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64ANDconst {
			break
		}
		t := v_0_1.Type
		if v_0_1.AuxInt != 63 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CSEL0 {
			break
		}
		if v_1.Type != typ.UInt64 {
			break
		}
		cc := v_1.Aux
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64SRL {
			break
		}
		if v_1_0.Type != typ.UInt64 {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpARM64SUB {
			break
		}
		if v_1_0_1.Type != t {
			break
		}
		_ = v_1_0_1.Args[1]
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_0_1_0.AuxInt != 64 {
			break
		}
		v_1_0_1_1 := v_1_0_1.Args[1]
		if v_1_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_0_1_1.Type != t {
			break
		}
		if v_1_0_1_1.AuxInt != 63 {
			break
		}
		if y != v_1_0_1_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64CMPconst {
			break
		}
		if v_1_1.AuxInt != 64 {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpARM64SUB {
			break
		}
		if v_1_1_0.Type != t {
			break
		}
		_ = v_1_1_0.Args[1]
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_1_0_0.AuxInt != 64 {
			break
		}
		v_1_1_0_1 := v_1_1_0.Args[1]
		if v_1_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1_0_1.Type != t {
			break
		}
		if v_1_1_0_1.AuxInt != 63 {
			break
		}
		if y != v_1_1_0_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (OR (CSEL0 <typ.UInt64> {cc} (SRL <typ.UInt64> x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))) (SLL x (ANDconst <t> [63] y)))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (ROR x (NEG <t> y))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64CSEL0 {
			break
		}
		if v_0.Type != typ.UInt64 {
			break
		}
		cc := v_0.Aux
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64SRL {
			break
		}
		if v_0_0.Type != typ.UInt64 {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpARM64SUB {
			break
		}
		t := v_0_0_1.Type
		_ = v_0_0_1.Args[1]
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_0_1_0.AuxInt != 64 {
			break
		}
		v_0_0_1_1 := v_0_0_1.Args[1]
		if v_0_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_0_1_1.Type != t {
			break
		}
		if v_0_0_1_1.AuxInt != 63 {
			break
		}
		y := v_0_0_1_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64CMPconst {
			break
		}
		if v_0_1.AuxInt != 64 {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpARM64SUB {
			break
		}
		if v_0_1_0.Type != t {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_1_0_0.AuxInt != 64 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_1_0_1.Type != t {
			break
		}
		if v_0_1_0_1.AuxInt != 63 {
			break
		}
		if y != v_0_1_0_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1.Type != t {
			break
		}
		if v_1_1.AuxInt != 63 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (OR (SRL <typ.UInt64> x (ANDconst <t> [63] y)) (CSEL0 <typ.UInt64> {cc} (SLL x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (ROR x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRL {
			break
		}
		if v_0.Type != typ.UInt64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64ANDconst {
			break
		}
		t := v_0_1.Type
		if v_0_1.AuxInt != 63 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CSEL0 {
			break
		}
		if v_1.Type != typ.UInt64 {
			break
		}
		cc := v_1.Aux
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64SLL {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpARM64SUB {
			break
		}
		if v_1_0_1.Type != t {
			break
		}
		_ = v_1_0_1.Args[1]
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_0_1_0.AuxInt != 64 {
			break
		}
		v_1_0_1_1 := v_1_0_1.Args[1]
		if v_1_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_0_1_1.Type != t {
			break
		}
		if v_1_0_1_1.AuxInt != 63 {
			break
		}
		if y != v_1_0_1_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64CMPconst {
			break
		}
		if v_1_1.AuxInt != 64 {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpARM64SUB {
			break
		}
		if v_1_1_0.Type != t {
			break
		}
		_ = v_1_1_0.Args[1]
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_1_0_0.AuxInt != 64 {
			break
		}
		v_1_1_0_1 := v_1_1_0.Args[1]
		if v_1_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1_0_1.Type != t {
			break
		}
		if v_1_1_0_1.AuxInt != 63 {
			break
		}
		if y != v_1_1_0_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR (CSEL0 <typ.UInt64> {cc} (SLL x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))) (SRL <typ.UInt64> x (ANDconst <t> [63] y)))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (ROR x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64CSEL0 {
			break
		}
		if v_0.Type != typ.UInt64 {
			break
		}
		cc := v_0.Aux
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64SLL {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpARM64SUB {
			break
		}
		t := v_0_0_1.Type
		_ = v_0_0_1.Args[1]
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_0_1_0.AuxInt != 64 {
			break
		}
		v_0_0_1_1 := v_0_0_1.Args[1]
		if v_0_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_0_1_1.Type != t {
			break
		}
		if v_0_0_1_1.AuxInt != 63 {
			break
		}
		y := v_0_0_1_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64CMPconst {
			break
		}
		if v_0_1.AuxInt != 64 {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpARM64SUB {
			break
		}
		if v_0_1_0.Type != t {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_1_0_0.AuxInt != 64 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_1_0_1.Type != t {
			break
		}
		if v_0_1_0_1.AuxInt != 63 {
			break
		}
		if y != v_0_1_0_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRL {
			break
		}
		if v_1.Type != typ.UInt64 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1.Type != t {
			break
		}
		if v_1_1.AuxInt != 63 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR (SLL x (ANDconst <t> [31] y)) (CSEL0 <typ.UInt32> {cc} (SRL <typ.UInt32> (MOVWUreg x) (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (RORW x (NEG <t> y))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64ANDconst {
			break
		}
		t := v_0_1.Type
		if v_0_1.AuxInt != 31 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CSEL0 {
			break
		}
		if v_1.Type != typ.UInt32 {
			break
		}
		cc := v_1.Aux
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64SRL {
			break
		}
		if v_1_0.Type != typ.UInt32 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpARM64MOVWUreg {
			break
		}
		if x != v_1_0_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpARM64SUB {
			break
		}
		if v_1_0_1.Type != t {
			break
		}
		_ = v_1_0_1.Args[1]
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_0_1_0.AuxInt != 32 {
			break
		}
		v_1_0_1_1 := v_1_0_1.Args[1]
		if v_1_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_0_1_1.Type != t {
			break
		}
		if v_1_0_1_1.AuxInt != 31 {
			break
		}
		if y != v_1_0_1_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64CMPconst {
			break
		}
		if v_1_1.AuxInt != 64 {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpARM64SUB {
			break
		}
		if v_1_1_0.Type != t {
			break
		}
		_ = v_1_1_0.Args[1]
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_1_0_0.AuxInt != 32 {
			break
		}
		v_1_1_0_1 := v_1_1_0.Args[1]
		if v_1_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1_0_1.Type != t {
			break
		}
		if v_1_1_0_1.AuxInt != 31 {
			break
		}
		if y != v_1_1_0_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (OR (CSEL0 <typ.UInt32> {cc} (SRL <typ.UInt32> (MOVWUreg x) (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))) (SLL x (ANDconst <t> [31] y)))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (RORW x (NEG <t> y))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64CSEL0 {
			break
		}
		if v_0.Type != typ.UInt32 {
			break
		}
		cc := v_0.Aux
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64SRL {
			break
		}
		if v_0_0.Type != typ.UInt32 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpARM64SUB {
			break
		}
		t := v_0_0_1.Type
		_ = v_0_0_1.Args[1]
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_0_1_0.AuxInt != 32 {
			break
		}
		v_0_0_1_1 := v_0_0_1.Args[1]
		if v_0_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_0_1_1.Type != t {
			break
		}
		if v_0_0_1_1.AuxInt != 31 {
			break
		}
		y := v_0_0_1_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64CMPconst {
			break
		}
		if v_0_1.AuxInt != 64 {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpARM64SUB {
			break
		}
		if v_0_1_0.Type != t {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_1_0_0.AuxInt != 32 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_1_0_1.Type != t {
			break
		}
		if v_0_1_0_1.AuxInt != 31 {
			break
		}
		if y != v_0_1_0_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1.Type != t {
			break
		}
		if v_1_1.AuxInt != 31 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (OR (SRL <typ.UInt32> (MOVWUreg x) (ANDconst <t> [31] y)) (CSEL0 <typ.UInt32> {cc} (SLL x (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (RORW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRL {
			break
		}
		if v_0.Type != typ.UInt32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64ANDconst {
			break
		}
		t := v_0_1.Type
		if v_0_1.AuxInt != 31 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CSEL0 {
			break
		}
		if v_1.Type != typ.UInt32 {
			break
		}
		cc := v_1.Aux
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64SLL {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpARM64SUB {
			break
		}
		if v_1_0_1.Type != t {
			break
		}
		_ = v_1_0_1.Args[1]
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_0_1_0.AuxInt != 32 {
			break
		}
		v_1_0_1_1 := v_1_0_1.Args[1]
		if v_1_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_0_1_1.Type != t {
			break
		}
		if v_1_0_1_1.AuxInt != 31 {
			break
		}
		if y != v_1_0_1_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64CMPconst {
			break
		}
		if v_1_1.AuxInt != 64 {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpARM64SUB {
			break
		}
		if v_1_1_0.Type != t {
			break
		}
		_ = v_1_1_0.Args[1]
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_1_0_0.AuxInt != 32 {
			break
		}
		v_1_1_0_1 := v_1_1_0.Args[1]
		if v_1_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1_0_1.Type != t {
			break
		}
		if v_1_1_0_1.AuxInt != 31 {
			break
		}
		if y != v_1_1_0_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR (CSEL0 <typ.UInt32> {cc} (SLL x (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))) (SRL <typ.UInt32> (MOVWUreg x) (ANDconst <t> [31] y)))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (RORW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64CSEL0 {
			break
		}
		if v_0.Type != typ.UInt32 {
			break
		}
		cc := v_0.Aux
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64SLL {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpARM64SUB {
			break
		}
		t := v_0_0_1.Type
		_ = v_0_0_1.Args[1]
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_0_1_0.AuxInt != 32 {
			break
		}
		v_0_0_1_1 := v_0_0_1.Args[1]
		if v_0_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_0_1_1.Type != t {
			break
		}
		if v_0_0_1_1.AuxInt != 31 {
			break
		}
		y := v_0_0_1_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64CMPconst {
			break
		}
		if v_0_1.AuxInt != 64 {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpARM64SUB {
			break
		}
		if v_0_1_0.Type != t {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_1_0_0.AuxInt != 32 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_1_0_1.Type != t {
			break
		}
		if v_0_1_0_1.AuxInt != 31 {
			break
		}
		if y != v_0_1_0_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRL {
			break
		}
		if v_1.Type != typ.UInt32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVWUreg {
			break
		}
		if x != v_1_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1.Type != t {
			break
		}
		if v_1_1.AuxInt != 31 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR (UBFIZ [bfc] x) (ANDconst [ac] y))
	// cond: ac == ^((1<<uint(getARM64BFwidth(bfc))-1) << uint(getARM64BFlsb(bfc)))
	// result: (BFI [bfc] y x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ANDconst {
			break
		}
		ac := v_1.AuxInt
		y := v_1.Args[0]
		if !(ac == ^((1<<uint(getARM64BFwidth(bfc)) - 1) << uint(getARM64BFlsb(bfc)))) {
			break
		}
		v.reset(OpARM64BFI)
		v.AuxInt = bfc
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64OR_20(v *Value) bool {
	b := v.Block
	// match: (OR (ANDconst [ac] y) (UBFIZ [bfc] x))
	// cond: ac == ^((1<<uint(getARM64BFwidth(bfc))-1) << uint(getARM64BFlsb(bfc)))
	// result: (BFI [bfc] y x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ANDconst {
			break
		}
		ac := v_0.AuxInt
		y := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64UBFIZ {
			break
		}
		bfc := v_1.AuxInt
		x := v_1.Args[0]
		if !(ac == ^((1<<uint(getARM64BFwidth(bfc)) - 1) << uint(getARM64BFlsb(bfc)))) {
			break
		}
		v.reset(OpARM64BFI)
		v.AuxInt = bfc
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	// match: (OR (UBFX [bfc] x) (ANDconst [ac] y))
	// cond: ac == ^(1<<uint(getARM64BFwidth(bfc))-1)
	// result: (BFXIL [bfc] y x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64ANDconst {
			break
		}
		ac := v_1.AuxInt
		y := v_1.Args[0]
		if !(ac == ^(1<<uint(getARM64BFwidth(bfc)) - 1)) {
			break
		}
		v.reset(OpARM64BFXIL)
		v.AuxInt = bfc
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	// match: (OR (ANDconst [ac] y) (UBFX [bfc] x))
	// cond: ac == ^(1<<uint(getARM64BFwidth(bfc))-1)
	// result: (BFXIL [bfc] y x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ANDconst {
			break
		}
		ac := v_0.AuxInt
		y := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64UBFX {
			break
		}
		bfc := v_1.AuxInt
		x := v_1.Args[0]
		if !(ac == ^(1<<uint(getARM64BFwidth(bfc)) - 1)) {
			break
		}
		v.reset(OpARM64BFXIL)
		v.AuxInt = bfc
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [i3] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i2] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i1] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (MOVWUload <t> {s} (OffPtr <p.Type> [i0] p) mem)
	for {
		t := v.Type
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		s0 := o1.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 24 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		i3 := x0.AuxInt
		s := x0.Aux
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := o1.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i2 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := o0.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i1 := x2.AuxInt
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		y3 := v.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		i0 := x3.AuxInt
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		if !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3)
		v0 := b.NewValue0(x3.Pos, OpARM64MOVWUload, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.Aux = s
		v1 := b.NewValue0(x3.Pos, OpOffPtr, p.Type)
		v1.AuxInt = i0
		v1.AddArg(p)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	// match: (OR <t> y3:(MOVDnop x3:(MOVBUload [i0] {s} p mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [i3] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i2] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i1] {s} p mem))))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (MOVWUload <t> {s} (OffPtr <p.Type> [i0] p) mem)
	for {
		t := v.Type
		_ = v.Args[1]
		y3 := v.Args[0]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		i0 := x3.AuxInt
		s := x3.Aux
		mem := x3.Args[1]
		p := x3.Args[0]
		o0 := v.Args[1]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		s0 := o1.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 24 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		i3 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[1]
		if p != x0.Args[0] {
			break
		}
		if mem != x0.Args[1] {
			break
		}
		y1 := o1.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i2 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := o0.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i1 := x2.AuxInt
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		if !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3)
		v0 := b.NewValue0(x2.Pos, OpARM64MOVWUload, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.Aux = s
		v1 := b.NewValue0(x2.Pos, OpOffPtr, p.Type)
		v1.AuxInt = i0
		v1.AddArg(p)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [3] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [2] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr0 idx0 mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (MOVWUloadidx <t> ptr0 idx0 mem)
	for {
		t := v.Type
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		s0 := o1.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 24 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		if x0.AuxInt != 3 {
			break
		}
		s := x0.Aux
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := o1.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 2 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := o0.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		if x2.AuxInt != 1 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		p1 := x2.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x2.Args[1] {
			break
		}
		y3 := v.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x3.Args[2]
		ptr0 := x3.Args[0]
		idx0 := x3.Args[1]
		if mem != x3.Args[2] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3)
		v0 := b.NewValue0(x2.Pos, OpARM64MOVWUloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr0)
		v0.AddArg(idx0)
		v0.AddArg(mem)
		return true
	}
	// match: (OR <t> y3:(MOVDnop x3:(MOVBUloadidx ptr0 idx0 mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [3] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [2] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (MOVWUloadidx <t> ptr0 idx0 mem)
	for {
		t := v.Type
		_ = v.Args[1]
		y3 := v.Args[0]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUloadidx {
			break
		}
		mem := x3.Args[2]
		ptr0 := x3.Args[0]
		idx0 := x3.Args[1]
		o0 := v.Args[1]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		s0 := o1.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 24 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		if x0.AuxInt != 3 {
			break
		}
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		if mem != x0.Args[1] {
			break
		}
		y1 := o1.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 2 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := o0.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		if x2.AuxInt != 1 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		p1 := x2.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x2.Args[1] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3)
		v0 := b.NewValue0(x2.Pos, OpARM64MOVWUloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr0)
		v0.AddArg(idx0)
		v0.AddArg(mem)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUloadidx ptr (ADDconst [3] idx) mem))) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr idx mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (MOVWUloadidx <t> ptr idx mem)
	for {
		t := v.Type
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		s0 := o1.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 24 {
			break
		}
		y0 := s0.Args[0]
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
		if x0_1.Op != OpARM64ADDconst {
			break
		}
		if x0_1.AuxInt != 3 {
			break
		}
		idx := x0_1.Args[0]
		y1 := o1.Args[1]
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 2 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y2 := o0.Args[1]
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
		if x2_1.Op != OpARM64ADDconst {
			break
		}
		if x2_1.AuxInt != 1 {
			break
		}
		if idx != x2_1.Args[0] {
			break
		}
		if mem != x2.Args[2] {
			break
		}
		y3 := v.Args[1]
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
		if idx != x3.Args[1] {
			break
		}
		if mem != x3.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3)
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (OR <t> y3:(MOVDnop x3:(MOVBUloadidx ptr idx mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUloadidx ptr (ADDconst [3] idx) mem))) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [1] idx) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (MOVWUloadidx <t> ptr idx mem)
	for {
		t := v.Type
		_ = v.Args[1]
		y3 := v.Args[0]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUloadidx {
			break
		}
		mem := x3.Args[2]
		ptr := x3.Args[0]
		idx := x3.Args[1]
		o0 := v.Args[1]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		s0 := o1.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 24 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x0.Args[2]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64ADDconst {
			break
		}
		if x0_1.AuxInt != 3 {
			break
		}
		if idx != x0_1.Args[0] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y1 := o1.Args[1]
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 2 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y2 := o0.Args[1]
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
		if x2_1.Op != OpARM64ADDconst {
			break
		}
		if x2_1.AuxInt != 1 {
			break
		}
		if idx != x2_1.Args[0] {
			break
		}
		if mem != x2.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3)
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [i7] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i6] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i4] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i3] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [i2] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [i1] {s} p mem))) y7:(MOVDnop x7:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (MOVDload <t> {s} (OffPtr <p.Type> [i0] p) mem)
	for {
		t := v.Type
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 24 {
			break
		}
		_ = o2.Args[1]
		o3 := o2.Args[0]
		if o3.Op != OpARM64ORshiftLL {
			break
		}
		if o3.AuxInt != 32 {
			break
		}
		_ = o3.Args[1]
		o4 := o3.Args[0]
		if o4.Op != OpARM64ORshiftLL {
			break
		}
		if o4.AuxInt != 40 {
			break
		}
		_ = o4.Args[1]
		o5 := o4.Args[0]
		if o5.Op != OpARM64ORshiftLL {
			break
		}
		if o5.AuxInt != 48 {
			break
		}
		_ = o5.Args[1]
		s0 := o5.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 56 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		i7 := x0.AuxInt
		s := x0.Aux
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := o5.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i6 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := o4.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i5 := x2.AuxInt
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		y3 := o3.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		i4 := x3.AuxInt
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		y4 := o2.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		i3 := x4.AuxInt
		if x4.Aux != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] {
			break
		}
		if mem != x4.Args[1] {
			break
		}
		y5 := o1.Args[1]
		if y5.Op != OpARM64MOVDnop {
			break
		}
		x5 := y5.Args[0]
		if x5.Op != OpARM64MOVBUload {
			break
		}
		i2 := x5.AuxInt
		if x5.Aux != s {
			break
		}
		_ = x5.Args[1]
		if p != x5.Args[0] {
			break
		}
		if mem != x5.Args[1] {
			break
		}
		y6 := o0.Args[1]
		if y6.Op != OpARM64MOVDnop {
			break
		}
		x6 := y6.Args[0]
		if x6.Op != OpARM64MOVBUload {
			break
		}
		i1 := x6.AuxInt
		if x6.Aux != s {
			break
		}
		_ = x6.Args[1]
		if p != x6.Args[0] {
			break
		}
		if mem != x6.Args[1] {
			break
		}
		y7 := v.Args[1]
		if y7.Op != OpARM64MOVDnop {
			break
		}
		x7 := y7.Args[0]
		if x7.Op != OpARM64MOVBUload {
			break
		}
		i0 := x7.AuxInt
		if x7.Aux != s {
			break
		}
		_ = x7.Args[1]
		if p != x7.Args[0] {
			break
		}
		if mem != x7.Args[1] {
			break
		}
		if !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
		v0 := b.NewValue0(x7.Pos, OpARM64MOVDload, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.Aux = s
		v1 := b.NewValue0(x7.Pos, OpOffPtr, p.Type)
		v1.AuxInt = i0
		v1.AddArg(p)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64OR_30(v *Value) bool {
	b := v.Block
	// match: (OR <t> y7:(MOVDnop x7:(MOVBUload [i0] {s} p mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [i7] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i6] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i4] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i3] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [i2] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [i1] {s} p mem))))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (MOVDload <t> {s} (OffPtr <p.Type> [i0] p) mem)
	for {
		t := v.Type
		_ = v.Args[1]
		y7 := v.Args[0]
		if y7.Op != OpARM64MOVDnop {
			break
		}
		x7 := y7.Args[0]
		if x7.Op != OpARM64MOVBUload {
			break
		}
		i0 := x7.AuxInt
		s := x7.Aux
		mem := x7.Args[1]
		p := x7.Args[0]
		o0 := v.Args[1]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 24 {
			break
		}
		_ = o2.Args[1]
		o3 := o2.Args[0]
		if o3.Op != OpARM64ORshiftLL {
			break
		}
		if o3.AuxInt != 32 {
			break
		}
		_ = o3.Args[1]
		o4 := o3.Args[0]
		if o4.Op != OpARM64ORshiftLL {
			break
		}
		if o4.AuxInt != 40 {
			break
		}
		_ = o4.Args[1]
		o5 := o4.Args[0]
		if o5.Op != OpARM64ORshiftLL {
			break
		}
		if o5.AuxInt != 48 {
			break
		}
		_ = o5.Args[1]
		s0 := o5.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 56 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		i7 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[1]
		if p != x0.Args[0] {
			break
		}
		if mem != x0.Args[1] {
			break
		}
		y1 := o5.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i6 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := o4.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i5 := x2.AuxInt
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		y3 := o3.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		i4 := x3.AuxInt
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		y4 := o2.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		i3 := x4.AuxInt
		if x4.Aux != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] {
			break
		}
		if mem != x4.Args[1] {
			break
		}
		y5 := o1.Args[1]
		if y5.Op != OpARM64MOVDnop {
			break
		}
		x5 := y5.Args[0]
		if x5.Op != OpARM64MOVBUload {
			break
		}
		i2 := x5.AuxInt
		if x5.Aux != s {
			break
		}
		_ = x5.Args[1]
		if p != x5.Args[0] {
			break
		}
		if mem != x5.Args[1] {
			break
		}
		y6 := o0.Args[1]
		if y6.Op != OpARM64MOVDnop {
			break
		}
		x6 := y6.Args[0]
		if x6.Op != OpARM64MOVBUload {
			break
		}
		i1 := x6.AuxInt
		if x6.Aux != s {
			break
		}
		_ = x6.Args[1]
		if p != x6.Args[0] {
			break
		}
		if mem != x6.Args[1] {
			break
		}
		if !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
		v0 := b.NewValue0(x6.Pos, OpARM64MOVDload, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.Aux = s
		v1 := b.NewValue0(x6.Pos, OpOffPtr, p.Type)
		v1.AuxInt = i0
		v1.AddArg(p)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [7] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [6] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [4] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [3] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [2] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y7:(MOVDnop x7:(MOVBUloadidx ptr0 idx0 mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (MOVDloadidx <t> ptr0 idx0 mem)
	for {
		t := v.Type
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 24 {
			break
		}
		_ = o2.Args[1]
		o3 := o2.Args[0]
		if o3.Op != OpARM64ORshiftLL {
			break
		}
		if o3.AuxInt != 32 {
			break
		}
		_ = o3.Args[1]
		o4 := o3.Args[0]
		if o4.Op != OpARM64ORshiftLL {
			break
		}
		if o4.AuxInt != 40 {
			break
		}
		_ = o4.Args[1]
		o5 := o4.Args[0]
		if o5.Op != OpARM64ORshiftLL {
			break
		}
		if o5.AuxInt != 48 {
			break
		}
		_ = o5.Args[1]
		s0 := o5.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 56 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		if x0.AuxInt != 7 {
			break
		}
		s := x0.Aux
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := o5.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 6 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := o4.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		if x2.AuxInt != 5 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		y3 := o3.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		if x3.AuxInt != 4 {
			break
		}
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		y4 := o2.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		if x4.AuxInt != 3 {
			break
		}
		if x4.Aux != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] {
			break
		}
		if mem != x4.Args[1] {
			break
		}
		y5 := o1.Args[1]
		if y5.Op != OpARM64MOVDnop {
			break
		}
		x5 := y5.Args[0]
		if x5.Op != OpARM64MOVBUload {
			break
		}
		if x5.AuxInt != 2 {
			break
		}
		if x5.Aux != s {
			break
		}
		_ = x5.Args[1]
		if p != x5.Args[0] {
			break
		}
		if mem != x5.Args[1] {
			break
		}
		y6 := o0.Args[1]
		if y6.Op != OpARM64MOVDnop {
			break
		}
		x6 := y6.Args[0]
		if x6.Op != OpARM64MOVBUload {
			break
		}
		if x6.AuxInt != 1 {
			break
		}
		if x6.Aux != s {
			break
		}
		_ = x6.Args[1]
		p1 := x6.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x6.Args[1] {
			break
		}
		y7 := v.Args[1]
		if y7.Op != OpARM64MOVDnop {
			break
		}
		x7 := y7.Args[0]
		if x7.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x7.Args[2]
		ptr0 := x7.Args[0]
		idx0 := x7.Args[1]
		if mem != x7.Args[2] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
		v0 := b.NewValue0(x6.Pos, OpARM64MOVDloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr0)
		v0.AddArg(idx0)
		v0.AddArg(mem)
		return true
	}
	// match: (OR <t> y7:(MOVDnop x7:(MOVBUloadidx ptr0 idx0 mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [7] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [6] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [4] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [3] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [2] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (MOVDloadidx <t> ptr0 idx0 mem)
	for {
		t := v.Type
		_ = v.Args[1]
		y7 := v.Args[0]
		if y7.Op != OpARM64MOVDnop {
			break
		}
		x7 := y7.Args[0]
		if x7.Op != OpARM64MOVBUloadidx {
			break
		}
		mem := x7.Args[2]
		ptr0 := x7.Args[0]
		idx0 := x7.Args[1]
		o0 := v.Args[1]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 24 {
			break
		}
		_ = o2.Args[1]
		o3 := o2.Args[0]
		if o3.Op != OpARM64ORshiftLL {
			break
		}
		if o3.AuxInt != 32 {
			break
		}
		_ = o3.Args[1]
		o4 := o3.Args[0]
		if o4.Op != OpARM64ORshiftLL {
			break
		}
		if o4.AuxInt != 40 {
			break
		}
		_ = o4.Args[1]
		o5 := o4.Args[0]
		if o5.Op != OpARM64ORshiftLL {
			break
		}
		if o5.AuxInt != 48 {
			break
		}
		_ = o5.Args[1]
		s0 := o5.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 56 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		if x0.AuxInt != 7 {
			break
		}
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		if mem != x0.Args[1] {
			break
		}
		y1 := o5.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 6 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := o4.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		if x2.AuxInt != 5 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		y3 := o3.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		if x3.AuxInt != 4 {
			break
		}
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		y4 := o2.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		if x4.AuxInt != 3 {
			break
		}
		if x4.Aux != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] {
			break
		}
		if mem != x4.Args[1] {
			break
		}
		y5 := o1.Args[1]
		if y5.Op != OpARM64MOVDnop {
			break
		}
		x5 := y5.Args[0]
		if x5.Op != OpARM64MOVBUload {
			break
		}
		if x5.AuxInt != 2 {
			break
		}
		if x5.Aux != s {
			break
		}
		_ = x5.Args[1]
		if p != x5.Args[0] {
			break
		}
		if mem != x5.Args[1] {
			break
		}
		y6 := o0.Args[1]
		if y6.Op != OpARM64MOVDnop {
			break
		}
		x6 := y6.Args[0]
		if x6.Op != OpARM64MOVBUload {
			break
		}
		if x6.AuxInt != 1 {
			break
		}
		if x6.Aux != s {
			break
		}
		_ = x6.Args[1]
		p1 := x6.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x6.Args[1] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
		v0 := b.NewValue0(x6.Pos, OpARM64MOVDloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr0)
		v0.AddArg(idx0)
		v0.AddArg(mem)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUloadidx ptr (ADDconst [7] idx) mem))) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [6] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [5] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [4] idx) mem))) y4:(MOVDnop x4:(MOVBUloadidx ptr (ADDconst [3] idx) mem))) y5:(MOVDnop x5:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y6:(MOVDnop x6:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y7:(MOVDnop x7:(MOVBUloadidx ptr idx mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (MOVDloadidx <t> ptr idx mem)
	for {
		t := v.Type
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 24 {
			break
		}
		_ = o2.Args[1]
		o3 := o2.Args[0]
		if o3.Op != OpARM64ORshiftLL {
			break
		}
		if o3.AuxInt != 32 {
			break
		}
		_ = o3.Args[1]
		o4 := o3.Args[0]
		if o4.Op != OpARM64ORshiftLL {
			break
		}
		if o4.AuxInt != 40 {
			break
		}
		_ = o4.Args[1]
		o5 := o4.Args[0]
		if o5.Op != OpARM64ORshiftLL {
			break
		}
		if o5.AuxInt != 48 {
			break
		}
		_ = o5.Args[1]
		s0 := o5.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 56 {
			break
		}
		y0 := s0.Args[0]
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
		if x0_1.Op != OpARM64ADDconst {
			break
		}
		if x0_1.AuxInt != 7 {
			break
		}
		idx := x0_1.Args[0]
		y1 := o5.Args[1]
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 6 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y2 := o4.Args[1]
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
		if x2_1.Op != OpARM64ADDconst {
			break
		}
		if x2_1.AuxInt != 5 {
			break
		}
		if idx != x2_1.Args[0] {
			break
		}
		if mem != x2.Args[2] {
			break
		}
		y3 := o3.Args[1]
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
		if x3_1.Op != OpARM64ADDconst {
			break
		}
		if x3_1.AuxInt != 4 {
			break
		}
		if idx != x3_1.Args[0] {
			break
		}
		if mem != x3.Args[2] {
			break
		}
		y4 := o2.Args[1]
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
		if x4_1.Op != OpARM64ADDconst {
			break
		}
		if x4_1.AuxInt != 3 {
			break
		}
		if idx != x4_1.Args[0] {
			break
		}
		if mem != x4.Args[2] {
			break
		}
		y5 := o1.Args[1]
		if y5.Op != OpARM64MOVDnop {
			break
		}
		x5 := y5.Args[0]
		if x5.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x5.Args[2]
		if ptr != x5.Args[0] {
			break
		}
		x5_1 := x5.Args[1]
		if x5_1.Op != OpARM64ADDconst {
			break
		}
		if x5_1.AuxInt != 2 {
			break
		}
		if idx != x5_1.Args[0] {
			break
		}
		if mem != x5.Args[2] {
			break
		}
		y6 := o0.Args[1]
		if y6.Op != OpARM64MOVDnop {
			break
		}
		x6 := y6.Args[0]
		if x6.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x6.Args[2]
		if ptr != x6.Args[0] {
			break
		}
		x6_1 := x6.Args[1]
		if x6_1.Op != OpARM64ADDconst {
			break
		}
		if x6_1.AuxInt != 1 {
			break
		}
		if idx != x6_1.Args[0] {
			break
		}
		if mem != x6.Args[2] {
			break
		}
		y7 := v.Args[1]
		if y7.Op != OpARM64MOVDnop {
			break
		}
		x7 := y7.Args[0]
		if x7.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x7.Args[2]
		if ptr != x7.Args[0] {
			break
		}
		if idx != x7.Args[1] {
			break
		}
		if mem != x7.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (OR <t> y7:(MOVDnop x7:(MOVBUloadidx ptr idx mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUloadidx ptr (ADDconst [7] idx) mem))) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [6] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [5] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [4] idx) mem))) y4:(MOVDnop x4:(MOVBUloadidx ptr (ADDconst [3] idx) mem))) y5:(MOVDnop x5:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y6:(MOVDnop x6:(MOVBUloadidx ptr (ADDconst [1] idx) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (MOVDloadidx <t> ptr idx mem)
	for {
		t := v.Type
		_ = v.Args[1]
		y7 := v.Args[0]
		if y7.Op != OpARM64MOVDnop {
			break
		}
		x7 := y7.Args[0]
		if x7.Op != OpARM64MOVBUloadidx {
			break
		}
		mem := x7.Args[2]
		ptr := x7.Args[0]
		idx := x7.Args[1]
		o0 := v.Args[1]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 24 {
			break
		}
		_ = o2.Args[1]
		o3 := o2.Args[0]
		if o3.Op != OpARM64ORshiftLL {
			break
		}
		if o3.AuxInt != 32 {
			break
		}
		_ = o3.Args[1]
		o4 := o3.Args[0]
		if o4.Op != OpARM64ORshiftLL {
			break
		}
		if o4.AuxInt != 40 {
			break
		}
		_ = o4.Args[1]
		o5 := o4.Args[0]
		if o5.Op != OpARM64ORshiftLL {
			break
		}
		if o5.AuxInt != 48 {
			break
		}
		_ = o5.Args[1]
		s0 := o5.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 56 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x0.Args[2]
		if ptr != x0.Args[0] {
			break
		}
		x0_1 := x0.Args[1]
		if x0_1.Op != OpARM64ADDconst {
			break
		}
		if x0_1.AuxInt != 7 {
			break
		}
		if idx != x0_1.Args[0] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y1 := o5.Args[1]
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 6 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y2 := o4.Args[1]
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
		if x2_1.Op != OpARM64ADDconst {
			break
		}
		if x2_1.AuxInt != 5 {
			break
		}
		if idx != x2_1.Args[0] {
			break
		}
		if mem != x2.Args[2] {
			break
		}
		y3 := o3.Args[1]
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
		if x3_1.Op != OpARM64ADDconst {
			break
		}
		if x3_1.AuxInt != 4 {
			break
		}
		if idx != x3_1.Args[0] {
			break
		}
		if mem != x3.Args[2] {
			break
		}
		y4 := o2.Args[1]
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
		if x4_1.Op != OpARM64ADDconst {
			break
		}
		if x4_1.AuxInt != 3 {
			break
		}
		if idx != x4_1.Args[0] {
			break
		}
		if mem != x4.Args[2] {
			break
		}
		y5 := o1.Args[1]
		if y5.Op != OpARM64MOVDnop {
			break
		}
		x5 := y5.Args[0]
		if x5.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x5.Args[2]
		if ptr != x5.Args[0] {
			break
		}
		x5_1 := x5.Args[1]
		if x5_1.Op != OpARM64ADDconst {
			break
		}
		if x5_1.AuxInt != 2 {
			break
		}
		if idx != x5_1.Args[0] {
			break
		}
		if mem != x5.Args[2] {
			break
		}
		y6 := o0.Args[1]
		if y6.Op != OpARM64MOVDnop {
			break
		}
		x6 := y6.Args[0]
		if x6.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x6.Args[2]
		if ptr != x6.Args[0] {
			break
		}
		x6_1 := x6.Args[1]
		if x6_1.Op != OpARM64ADDconst {
			break
		}
		if x6_1.AuxInt != 1 {
			break
		}
		if idx != x6_1.Args[0] {
			break
		}
		if mem != x6.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i3] {s} p mem)))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (REVW <t> (MOVWUload <t> {s} (OffPtr <p.Type> [i0] p) mem))
	for {
		t := v.Type
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		s0 := o1.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 24 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := o1.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := o0.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i2 := x2.AuxInt
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		y3 := v.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		i3 := x3.AuxInt
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		if !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3)
		v0 := b.NewValue0(x3.Pos, OpARM64REVW, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x3.Pos, OpARM64MOVWUload, t)
		v1.Aux = s
		v2 := b.NewValue0(x3.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> y3:(MOVDnop x3:(MOVBUload [i3] {s} p mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (REVW <t> (MOVWUload <t> {s} (OffPtr <p.Type> [i0] p) mem))
	for {
		t := v.Type
		_ = v.Args[1]
		y3 := v.Args[0]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		i3 := x3.AuxInt
		s := x3.Aux
		mem := x3.Args[1]
		p := x3.Args[0]
		o0 := v.Args[1]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		s0 := o1.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 24 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[1]
		if p != x0.Args[0] {
			break
		}
		if mem != x0.Args[1] {
			break
		}
		y1 := o1.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := o0.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i2 := x2.AuxInt
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		if !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3)
		v0 := b.NewValue0(x2.Pos, OpARM64REVW, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x2.Pos, OpARM64MOVWUload, t)
		v1.Aux = s
		v2 := b.NewValue0(x2.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUloadidx ptr0 idx0 mem))) y1:(MOVDnop x1:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [3] {s} p mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (REVW <t> (MOVWUloadidx <t> ptr0 idx0 mem))
	for {
		t := v.Type
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		s0 := o1.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 24 {
			break
		}
		y0 := s0.Args[0]
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
		y1 := o1.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 1 {
			break
		}
		s := x1.Aux
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x1.Args[1] {
			break
		}
		y2 := o0.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		if x2.AuxInt != 2 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		p := x2.Args[0]
		if mem != x2.Args[1] {
			break
		}
		y3 := v.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		if x3.AuxInt != 3 {
			break
		}
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3)
		v0 := b.NewValue0(x3.Pos, OpARM64REVW, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x3.Pos, OpARM64MOVWUloadidx, t)
		v1.AddArg(ptr0)
		v1.AddArg(idx0)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> y3:(MOVDnop x3:(MOVBUload [3] {s} p mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUloadidx ptr0 idx0 mem))) y1:(MOVDnop x1:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [2] {s} p mem))))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (REVW <t> (MOVWUloadidx <t> ptr0 idx0 mem))
	for {
		t := v.Type
		_ = v.Args[1]
		y3 := v.Args[0]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		if x3.AuxInt != 3 {
			break
		}
		s := x3.Aux
		mem := x3.Args[1]
		p := x3.Args[0]
		o0 := v.Args[1]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		s0 := o1.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 24 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x0.Args[2]
		ptr0 := x0.Args[0]
		idx0 := x0.Args[1]
		if mem != x0.Args[2] {
			break
		}
		y1 := o1.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 1 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x1.Args[1] {
			break
		}
		y2 := o0.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		if x2.AuxInt != 2 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3)
		v0 := b.NewValue0(x2.Pos, OpARM64REVW, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x2.Pos, OpARM64MOVWUloadidx, t)
		v1.AddArg(ptr0)
		v1.AddArg(idx0)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUloadidx ptr idx mem))) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [3] idx) mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (REVW <t> (MOVWUloadidx <t> ptr idx mem))
	for {
		t := v.Type
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		s0 := o1.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 24 {
			break
		}
		y0 := s0.Args[0]
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
		y1 := o1.Args[1]
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 1 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y2 := o0.Args[1]
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
		if x2_1.Op != OpARM64ADDconst {
			break
		}
		if x2_1.AuxInt != 2 {
			break
		}
		if idx != x2_1.Args[0] {
			break
		}
		if mem != x2.Args[2] {
			break
		}
		y3 := v.Args[1]
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
		if x3_1.Op != OpARM64ADDconst {
			break
		}
		if x3_1.AuxInt != 3 {
			break
		}
		if idx != x3_1.Args[0] {
			break
		}
		if mem != x3.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3)
		v0 := b.NewValue0(v.Pos, OpARM64REVW, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWUloadidx, t)
		v1.AddArg(ptr)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64OR_40(v *Value) bool {
	b := v.Block
	// match: (OR <t> y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [3] idx) mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUloadidx ptr idx mem))) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [2] idx) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3) (REVW <t> (MOVWUloadidx <t> ptr idx mem))
	for {
		t := v.Type
		_ = v.Args[1]
		y3 := v.Args[0]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUloadidx {
			break
		}
		mem := x3.Args[2]
		ptr := x3.Args[0]
		x3_1 := x3.Args[1]
		if x3_1.Op != OpARM64ADDconst {
			break
		}
		if x3_1.AuxInt != 3 {
			break
		}
		idx := x3_1.Args[0]
		o0 := v.Args[1]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		s0 := o1.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 24 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x0.Args[2]
		if ptr != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y1 := o1.Args[1]
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 1 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y2 := o0.Args[1]
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
		if x2_1.Op != OpARM64ADDconst {
			break
		}
		if x2_1.AuxInt != 2 {
			break
		}
		if idx != x2_1.Args[0] {
			break
		}
		if mem != x2.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(o0) && clobber(o1) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3)
		v0 := b.NewValue0(v.Pos, OpARM64REVW, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWUloadidx, t)
		v1.AddArg(ptr)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i3] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i4] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [i5] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [i6] {s} p mem))) y7:(MOVDnop x7:(MOVBUload [i7] {s} p mem)))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (REV <t> (MOVDload <t> {s} (OffPtr <p.Type> [i0] p) mem))
	for {
		t := v.Type
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 24 {
			break
		}
		_ = o2.Args[1]
		o3 := o2.Args[0]
		if o3.Op != OpARM64ORshiftLL {
			break
		}
		if o3.AuxInt != 32 {
			break
		}
		_ = o3.Args[1]
		o4 := o3.Args[0]
		if o4.Op != OpARM64ORshiftLL {
			break
		}
		if o4.AuxInt != 40 {
			break
		}
		_ = o4.Args[1]
		o5 := o4.Args[0]
		if o5.Op != OpARM64ORshiftLL {
			break
		}
		if o5.AuxInt != 48 {
			break
		}
		_ = o5.Args[1]
		s0 := o5.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 56 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := o5.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := o4.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i2 := x2.AuxInt
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		y3 := o3.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		i3 := x3.AuxInt
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		y4 := o2.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		i4 := x4.AuxInt
		if x4.Aux != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] {
			break
		}
		if mem != x4.Args[1] {
			break
		}
		y5 := o1.Args[1]
		if y5.Op != OpARM64MOVDnop {
			break
		}
		x5 := y5.Args[0]
		if x5.Op != OpARM64MOVBUload {
			break
		}
		i5 := x5.AuxInt
		if x5.Aux != s {
			break
		}
		_ = x5.Args[1]
		if p != x5.Args[0] {
			break
		}
		if mem != x5.Args[1] {
			break
		}
		y6 := o0.Args[1]
		if y6.Op != OpARM64MOVDnop {
			break
		}
		x6 := y6.Args[0]
		if x6.Op != OpARM64MOVBUload {
			break
		}
		i6 := x6.AuxInt
		if x6.Aux != s {
			break
		}
		_ = x6.Args[1]
		if p != x6.Args[0] {
			break
		}
		if mem != x6.Args[1] {
			break
		}
		y7 := v.Args[1]
		if y7.Op != OpARM64MOVDnop {
			break
		}
		x7 := y7.Args[0]
		if x7.Op != OpARM64MOVBUload {
			break
		}
		i7 := x7.AuxInt
		if x7.Aux != s {
			break
		}
		_ = x7.Args[1]
		if p != x7.Args[0] {
			break
		}
		if mem != x7.Args[1] {
			break
		}
		if !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
		v0 := b.NewValue0(x7.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x7.Pos, OpARM64MOVDload, t)
		v1.Aux = s
		v2 := b.NewValue0(x7.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> y7:(MOVDnop x7:(MOVBUload [i7] {s} p mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i3] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i4] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [i5] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [i6] {s} p mem))))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (REV <t> (MOVDload <t> {s} (OffPtr <p.Type> [i0] p) mem))
	for {
		t := v.Type
		_ = v.Args[1]
		y7 := v.Args[0]
		if y7.Op != OpARM64MOVDnop {
			break
		}
		x7 := y7.Args[0]
		if x7.Op != OpARM64MOVBUload {
			break
		}
		i7 := x7.AuxInt
		s := x7.Aux
		mem := x7.Args[1]
		p := x7.Args[0]
		o0 := v.Args[1]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 24 {
			break
		}
		_ = o2.Args[1]
		o3 := o2.Args[0]
		if o3.Op != OpARM64ORshiftLL {
			break
		}
		if o3.AuxInt != 32 {
			break
		}
		_ = o3.Args[1]
		o4 := o3.Args[0]
		if o4.Op != OpARM64ORshiftLL {
			break
		}
		if o4.AuxInt != 40 {
			break
		}
		_ = o4.Args[1]
		o5 := o4.Args[0]
		if o5.Op != OpARM64ORshiftLL {
			break
		}
		if o5.AuxInt != 48 {
			break
		}
		_ = o5.Args[1]
		s0 := o5.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 56 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[1]
		if p != x0.Args[0] {
			break
		}
		if mem != x0.Args[1] {
			break
		}
		y1 := o5.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := o4.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i2 := x2.AuxInt
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		y3 := o3.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		i3 := x3.AuxInt
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		y4 := o2.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		i4 := x4.AuxInt
		if x4.Aux != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] {
			break
		}
		if mem != x4.Args[1] {
			break
		}
		y5 := o1.Args[1]
		if y5.Op != OpARM64MOVDnop {
			break
		}
		x5 := y5.Args[0]
		if x5.Op != OpARM64MOVBUload {
			break
		}
		i5 := x5.AuxInt
		if x5.Aux != s {
			break
		}
		_ = x5.Args[1]
		if p != x5.Args[0] {
			break
		}
		if mem != x5.Args[1] {
			break
		}
		y6 := o0.Args[1]
		if y6.Op != OpARM64MOVDnop {
			break
		}
		x6 := y6.Args[0]
		if x6.Op != OpARM64MOVBUload {
			break
		}
		i6 := x6.AuxInt
		if x6.Aux != s {
			break
		}
		_ = x6.Args[1]
		if p != x6.Args[0] {
			break
		}
		if mem != x6.Args[1] {
			break
		}
		if !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
		v0 := b.NewValue0(x6.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x6.Pos, OpARM64MOVDload, t)
		v1.Aux = s
		v2 := b.NewValue0(x6.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUloadidx ptr0 idx0 mem))) y1:(MOVDnop x1:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [3] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [4] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [5] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [6] {s} p mem))) y7:(MOVDnop x7:(MOVBUload [7] {s} p mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (REV <t> (MOVDloadidx <t> ptr0 idx0 mem))
	for {
		t := v.Type
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 24 {
			break
		}
		_ = o2.Args[1]
		o3 := o2.Args[0]
		if o3.Op != OpARM64ORshiftLL {
			break
		}
		if o3.AuxInt != 32 {
			break
		}
		_ = o3.Args[1]
		o4 := o3.Args[0]
		if o4.Op != OpARM64ORshiftLL {
			break
		}
		if o4.AuxInt != 40 {
			break
		}
		_ = o4.Args[1]
		o5 := o4.Args[0]
		if o5.Op != OpARM64ORshiftLL {
			break
		}
		if o5.AuxInt != 48 {
			break
		}
		_ = o5.Args[1]
		s0 := o5.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 56 {
			break
		}
		y0 := s0.Args[0]
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
		y1 := o5.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 1 {
			break
		}
		s := x1.Aux
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x1.Args[1] {
			break
		}
		y2 := o4.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		if x2.AuxInt != 2 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		p := x2.Args[0]
		if mem != x2.Args[1] {
			break
		}
		y3 := o3.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		if x3.AuxInt != 3 {
			break
		}
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		y4 := o2.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		if x4.AuxInt != 4 {
			break
		}
		if x4.Aux != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] {
			break
		}
		if mem != x4.Args[1] {
			break
		}
		y5 := o1.Args[1]
		if y5.Op != OpARM64MOVDnop {
			break
		}
		x5 := y5.Args[0]
		if x5.Op != OpARM64MOVBUload {
			break
		}
		if x5.AuxInt != 5 {
			break
		}
		if x5.Aux != s {
			break
		}
		_ = x5.Args[1]
		if p != x5.Args[0] {
			break
		}
		if mem != x5.Args[1] {
			break
		}
		y6 := o0.Args[1]
		if y6.Op != OpARM64MOVDnop {
			break
		}
		x6 := y6.Args[0]
		if x6.Op != OpARM64MOVBUload {
			break
		}
		if x6.AuxInt != 6 {
			break
		}
		if x6.Aux != s {
			break
		}
		_ = x6.Args[1]
		if p != x6.Args[0] {
			break
		}
		if mem != x6.Args[1] {
			break
		}
		y7 := v.Args[1]
		if y7.Op != OpARM64MOVDnop {
			break
		}
		x7 := y7.Args[0]
		if x7.Op != OpARM64MOVBUload {
			break
		}
		if x7.AuxInt != 7 {
			break
		}
		if x7.Aux != s {
			break
		}
		_ = x7.Args[1]
		if p != x7.Args[0] {
			break
		}
		if mem != x7.Args[1] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
		v0 := b.NewValue0(x7.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x7.Pos, OpARM64MOVDloadidx, t)
		v1.AddArg(ptr0)
		v1.AddArg(idx0)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> y7:(MOVDnop x7:(MOVBUload [7] {s} p mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUloadidx ptr0 idx0 mem))) y1:(MOVDnop x1:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [3] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [4] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [5] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [6] {s} p mem))))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (REV <t> (MOVDloadidx <t> ptr0 idx0 mem))
	for {
		t := v.Type
		_ = v.Args[1]
		y7 := v.Args[0]
		if y7.Op != OpARM64MOVDnop {
			break
		}
		x7 := y7.Args[0]
		if x7.Op != OpARM64MOVBUload {
			break
		}
		if x7.AuxInt != 7 {
			break
		}
		s := x7.Aux
		mem := x7.Args[1]
		p := x7.Args[0]
		o0 := v.Args[1]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 24 {
			break
		}
		_ = o2.Args[1]
		o3 := o2.Args[0]
		if o3.Op != OpARM64ORshiftLL {
			break
		}
		if o3.AuxInt != 32 {
			break
		}
		_ = o3.Args[1]
		o4 := o3.Args[0]
		if o4.Op != OpARM64ORshiftLL {
			break
		}
		if o4.AuxInt != 40 {
			break
		}
		_ = o4.Args[1]
		o5 := o4.Args[0]
		if o5.Op != OpARM64ORshiftLL {
			break
		}
		if o5.AuxInt != 48 {
			break
		}
		_ = o5.Args[1]
		s0 := o5.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 56 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x0.Args[2]
		ptr0 := x0.Args[0]
		idx0 := x0.Args[1]
		if mem != x0.Args[2] {
			break
		}
		y1 := o5.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 1 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x1.Args[1] {
			break
		}
		y2 := o4.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		if x2.AuxInt != 2 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		y3 := o3.Args[1]
		if y3.Op != OpARM64MOVDnop {
			break
		}
		x3 := y3.Args[0]
		if x3.Op != OpARM64MOVBUload {
			break
		}
		if x3.AuxInt != 3 {
			break
		}
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		y4 := o2.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		if x4.AuxInt != 4 {
			break
		}
		if x4.Aux != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] {
			break
		}
		if mem != x4.Args[1] {
			break
		}
		y5 := o1.Args[1]
		if y5.Op != OpARM64MOVDnop {
			break
		}
		x5 := y5.Args[0]
		if x5.Op != OpARM64MOVBUload {
			break
		}
		if x5.AuxInt != 5 {
			break
		}
		if x5.Aux != s {
			break
		}
		_ = x5.Args[1]
		if p != x5.Args[0] {
			break
		}
		if mem != x5.Args[1] {
			break
		}
		y6 := o0.Args[1]
		if y6.Op != OpARM64MOVDnop {
			break
		}
		x6 := y6.Args[0]
		if x6.Op != OpARM64MOVBUload {
			break
		}
		if x6.AuxInt != 6 {
			break
		}
		if x6.Aux != s {
			break
		}
		_ = x6.Args[1]
		if p != x6.Args[0] {
			break
		}
		if mem != x6.Args[1] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
		v0 := b.NewValue0(x6.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x6.Pos, OpARM64MOVDloadidx, t)
		v1.AddArg(ptr0)
		v1.AddArg(idx0)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUloadidx ptr idx mem))) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [3] idx) mem))) y4:(MOVDnop x4:(MOVBUloadidx ptr (ADDconst [4] idx) mem))) y5:(MOVDnop x5:(MOVBUloadidx ptr (ADDconst [5] idx) mem))) y6:(MOVDnop x6:(MOVBUloadidx ptr (ADDconst [6] idx) mem))) y7:(MOVDnop x7:(MOVBUloadidx ptr (ADDconst [7] idx) mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (REV <t> (MOVDloadidx <t> ptr idx mem))
	for {
		t := v.Type
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 24 {
			break
		}
		_ = o2.Args[1]
		o3 := o2.Args[0]
		if o3.Op != OpARM64ORshiftLL {
			break
		}
		if o3.AuxInt != 32 {
			break
		}
		_ = o3.Args[1]
		o4 := o3.Args[0]
		if o4.Op != OpARM64ORshiftLL {
			break
		}
		if o4.AuxInt != 40 {
			break
		}
		_ = o4.Args[1]
		o5 := o4.Args[0]
		if o5.Op != OpARM64ORshiftLL {
			break
		}
		if o5.AuxInt != 48 {
			break
		}
		_ = o5.Args[1]
		s0 := o5.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 56 {
			break
		}
		y0 := s0.Args[0]
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
		y1 := o5.Args[1]
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 1 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y2 := o4.Args[1]
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
		if x2_1.Op != OpARM64ADDconst {
			break
		}
		if x2_1.AuxInt != 2 {
			break
		}
		if idx != x2_1.Args[0] {
			break
		}
		if mem != x2.Args[2] {
			break
		}
		y3 := o3.Args[1]
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
		if x3_1.Op != OpARM64ADDconst {
			break
		}
		if x3_1.AuxInt != 3 {
			break
		}
		if idx != x3_1.Args[0] {
			break
		}
		if mem != x3.Args[2] {
			break
		}
		y4 := o2.Args[1]
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
		if x4_1.Op != OpARM64ADDconst {
			break
		}
		if x4_1.AuxInt != 4 {
			break
		}
		if idx != x4_1.Args[0] {
			break
		}
		if mem != x4.Args[2] {
			break
		}
		y5 := o1.Args[1]
		if y5.Op != OpARM64MOVDnop {
			break
		}
		x5 := y5.Args[0]
		if x5.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x5.Args[2]
		if ptr != x5.Args[0] {
			break
		}
		x5_1 := x5.Args[1]
		if x5_1.Op != OpARM64ADDconst {
			break
		}
		if x5_1.AuxInt != 5 {
			break
		}
		if idx != x5_1.Args[0] {
			break
		}
		if mem != x5.Args[2] {
			break
		}
		y6 := o0.Args[1]
		if y6.Op != OpARM64MOVDnop {
			break
		}
		x6 := y6.Args[0]
		if x6.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x6.Args[2]
		if ptr != x6.Args[0] {
			break
		}
		x6_1 := x6.Args[1]
		if x6_1.Op != OpARM64ADDconst {
			break
		}
		if x6_1.AuxInt != 6 {
			break
		}
		if idx != x6_1.Args[0] {
			break
		}
		if mem != x6.Args[2] {
			break
		}
		y7 := v.Args[1]
		if y7.Op != OpARM64MOVDnop {
			break
		}
		x7 := y7.Args[0]
		if x7.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x7.Args[2]
		if ptr != x7.Args[0] {
			break
		}
		x7_1 := x7.Args[1]
		if x7_1.Op != OpARM64ADDconst {
			break
		}
		if x7_1.AuxInt != 7 {
			break
		}
		if idx != x7_1.Args[0] {
			break
		}
		if mem != x7.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
		v0 := b.NewValue0(v.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDloadidx, t)
		v1.AddArg(ptr)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> y7:(MOVDnop x7:(MOVBUloadidx ptr (ADDconst [7] idx) mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUloadidx ptr idx mem))) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [3] idx) mem))) y4:(MOVDnop x4:(MOVBUloadidx ptr (ADDconst [4] idx) mem))) y5:(MOVDnop x5:(MOVBUloadidx ptr (ADDconst [5] idx) mem))) y6:(MOVDnop x6:(MOVBUloadidx ptr (ADDconst [6] idx) mem))))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)
	// result: @mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) (REV <t> (MOVDloadidx <t> ptr idx mem))
	for {
		t := v.Type
		_ = v.Args[1]
		y7 := v.Args[0]
		if y7.Op != OpARM64MOVDnop {
			break
		}
		x7 := y7.Args[0]
		if x7.Op != OpARM64MOVBUloadidx {
			break
		}
		mem := x7.Args[2]
		ptr := x7.Args[0]
		x7_1 := x7.Args[1]
		if x7_1.Op != OpARM64ADDconst {
			break
		}
		if x7_1.AuxInt != 7 {
			break
		}
		idx := x7_1.Args[0]
		o0 := v.Args[1]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 8 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 16 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 24 {
			break
		}
		_ = o2.Args[1]
		o3 := o2.Args[0]
		if o3.Op != OpARM64ORshiftLL {
			break
		}
		if o3.AuxInt != 32 {
			break
		}
		_ = o3.Args[1]
		o4 := o3.Args[0]
		if o4.Op != OpARM64ORshiftLL {
			break
		}
		if o4.AuxInt != 40 {
			break
		}
		_ = o4.Args[1]
		o5 := o4.Args[0]
		if o5.Op != OpARM64ORshiftLL {
			break
		}
		if o5.AuxInt != 48 {
			break
		}
		_ = o5.Args[1]
		s0 := o5.Args[0]
		if s0.Op != OpARM64SLLconst {
			break
		}
		if s0.AuxInt != 56 {
			break
		}
		y0 := s0.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x0.Args[2]
		if ptr != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y1 := o5.Args[1]
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 1 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y2 := o4.Args[1]
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
		if x2_1.Op != OpARM64ADDconst {
			break
		}
		if x2_1.AuxInt != 2 {
			break
		}
		if idx != x2_1.Args[0] {
			break
		}
		if mem != x2.Args[2] {
			break
		}
		y3 := o3.Args[1]
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
		if x3_1.Op != OpARM64ADDconst {
			break
		}
		if x3_1.AuxInt != 3 {
			break
		}
		if idx != x3_1.Args[0] {
			break
		}
		if mem != x3.Args[2] {
			break
		}
		y4 := o2.Args[1]
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
		if x4_1.Op != OpARM64ADDconst {
			break
		}
		if x4_1.AuxInt != 4 {
			break
		}
		if idx != x4_1.Args[0] {
			break
		}
		if mem != x4.Args[2] {
			break
		}
		y5 := o1.Args[1]
		if y5.Op != OpARM64MOVDnop {
			break
		}
		x5 := y5.Args[0]
		if x5.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x5.Args[2]
		if ptr != x5.Args[0] {
			break
		}
		x5_1 := x5.Args[1]
		if x5_1.Op != OpARM64ADDconst {
			break
		}
		if x5_1.AuxInt != 5 {
			break
		}
		if idx != x5_1.Args[0] {
			break
		}
		if mem != x5.Args[2] {
			break
		}
		y6 := o0.Args[1]
		if y6.Op != OpARM64MOVDnop {
			break
		}
		x6 := y6.Args[0]
		if x6.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x6.Args[2]
		if ptr != x6.Args[0] {
			break
		}
		x6_1 := x6.Args[1]
		if x6_1.Op != OpARM64ADDconst {
			break
		}
		if x6_1.AuxInt != 6 {
			break
		}
		if idx != x6_1.Args[0] {
			break
		}
		if mem != x6.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 && o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) && clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) && clobber(o4) && clobber(o5) && clobber(s0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4, x5, x6, x7)
		v0 := b.NewValue0(v.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDloadidx, t)
		v1.AddArg(ptr)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORN_0(v *Value) bool {
	// match: (ORN x (MOVDconst [c]))
	// cond:
	// result: (ORconst [^c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ORconst)
		v.AuxInt = ^c
		v.AddArg(x)
		return true
	}
	// match: (ORN x x)
	// cond:
	// result: (MOVDconst [-1])
	for {
		x := v.Args[1]
		if x != v.Args[0] {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -1
		return true
	}
	// match: (ORN x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORNshiftLL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORNshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (ORN x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORNshiftRL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORNshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (ORN x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (ORNshiftRA x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64ORNshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORNshiftLL_0(v *Value) bool {
	// match: (ORNshiftLL x (MOVDconst [c]) [d])
	// cond:
	// result: (ORconst x [^int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ORconst)
		v.AuxInt = ^int64(uint64(c) << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ORNshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [-1])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -1
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORNshiftRA_0(v *Value) bool {
	// match: (ORNshiftRA x (MOVDconst [c]) [d])
	// cond:
	// result: (ORconst x [^(c>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ORconst)
		v.AuxInt = ^(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ORNshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [-1])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -1
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORNshiftRL_0(v *Value) bool {
	// match: (ORNshiftRL x (MOVDconst [c]) [d])
	// cond:
	// result: (ORconst x [^int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ORconst)
		v.AuxInt = ^int64(uint64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ORNshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [-1])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -1
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORconst_0(v *Value) bool {
	// match: (ORconst [0] x)
	// cond:
	// result: x
	for {
		if v.AuxInt != 0 {
			break
		}
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ORconst [-1] _)
	// cond:
	// result: (MOVDconst [-1])
	for {
		if v.AuxInt != -1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = -1
		return true
	}
	// match: (ORconst [c] (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [c|d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = c | d
		return true
	}
	// match: (ORconst [c] (ORconst [d] x))
	// cond:
	// result: (ORconst [c|d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ORconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARM64ORconst)
		v.AuxInt = c | d
		v.AddArg(x)
		return true
	}
	// match: (ORconst [c1] (ANDconst [c2] x))
	// cond: c2|c1 == ^0
	// result: (ORconst [c1] x)
	for {
		c1 := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ANDconst {
			break
		}
		c2 := v_0.AuxInt
		x := v_0.Args[0]
		if !(c2|c1 == ^0) {
			break
		}
		v.reset(OpARM64ORconst)
		v.AuxInt = c1
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORshiftLL_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ORshiftLL (MOVDconst [c]) x [d])
	// cond:
	// result: (ORconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64ORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftLL x (MOVDconst [c]) [d])
	// cond:
	// result: (ORconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ORconst)
		v.AuxInt = int64(uint64(c) << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL x y:(SLLconst x [c]) [d])
	// cond: c==d
	// result: y
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if y.Op != OpARM64SLLconst {
			break
		}
		c := y.AuxInt
		if x != y.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (ORshiftLL [c] (SRLconst x [64-c]) x)
	// cond:
	// result: (RORconst [64-c] x)
	for {
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = 64 - c
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL <t> [c] (UBFX [bfc] x) x)
	// cond: c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)
	// result: (RORWconst [32-c] x)
	for {
		t := v.Type
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := v_0.AuxInt
		if x != v_0.Args[0] {
			break
		}
		if !(c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = 32 - c
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL <typ.UInt16> [8] (UBFX <typ.UInt16> [armBFAuxInt(8, 8)] x) x)
	// cond:
	// result: (REV16W x)
	for {
		if v.Type != typ.UInt16 {
			break
		}
		if v.AuxInt != 8 {
			break
		}
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFX {
			break
		}
		if v_0.Type != typ.UInt16 {
			break
		}
		if v_0.AuxInt != armBFAuxInt(8, 8) {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL [c] (SRLconst x [64-c]) x2)
	// cond:
	// result: (EXTRconst [64-c] x2 x)
	for {
		c := v.AuxInt
		x2 := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64EXTRconst)
		v.AuxInt = 64 - c
		v.AddArg(x2)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL <t> [c] (UBFX [bfc] x) x2)
	// cond: c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)
	// result: (EXTRWconst [32-c] x2 x)
	for {
		t := v.Type
		c := v.AuxInt
		x2 := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		if !(c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)) {
			break
		}
		v.reset(OpARM64EXTRWconst)
		v.AuxInt = 32 - c
		v.AddArg(x2)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL [sc] (UBFX [bfc] x) (SRLconst [sc] y))
	// cond: sc == getARM64BFwidth(bfc)
	// result: (BFXIL [bfc] y x)
	for {
		sc := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		if v_1.AuxInt != sc {
			break
		}
		y := v_1.Args[0]
		if !(sc == getARM64BFwidth(bfc)) {
			break
		}
		v.reset(OpARM64BFXIL)
		v.AuxInt = bfc
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem)))
	// cond: i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0) && clobber(x1) && clobber(y0) && clobber(y1)
	// result: @mergePoint(b,x0,x1) (MOVHUload <t> {s} (OffPtr <p.Type> [i0] p) mem)
	for {
		t := v.Type
		if v.AuxInt != 8 {
			break
		}
		_ = v.Args[1]
		y0 := v.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := v.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(y0) && clobber(y1)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(x1.Pos, OpARM64MOVHUload, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.Aux = s
		v1 := b.NewValue0(x1.Pos, OpOffPtr, p.Type)
		v1.AuxInt = i0
		v1.AddArg(p)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORshiftLL_10(v *Value) bool {
	b := v.Block
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUloadidx ptr0 idx0 mem)) y1:(MOVDnop x1:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b,x0,x1) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x0) && clobber(x1) && clobber(y0) && clobber(y1)
	// result: @mergePoint(b,x0,x1) (MOVHUloadidx <t> ptr0 idx0 mem)
	for {
		t := v.Type
		if v.AuxInt != 8 {
			break
		}
		_ = v.Args[1]
		y0 := v.Args[0]
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
		y1 := v.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 1 {
			break
		}
		s := x1.Aux
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x1.Args[1] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b, x0, x1) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x0) && clobber(x1) && clobber(y0) && clobber(y1)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(x1.Pos, OpARM64MOVHUloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr0)
		v0.AddArg(idx0)
		v0.AddArg(mem)
		return true
	}
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUloadidx ptr idx mem)) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [1] idx) mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0) && clobber(x1) && clobber(y0) && clobber(y1)
	// result: @mergePoint(b,x0,x1) (MOVHUloadidx <t> ptr idx mem)
	for {
		t := v.Type
		if v.AuxInt != 8 {
			break
		}
		_ = v.Args[1]
		y0 := v.Args[0]
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
		y1 := v.Args[1]
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 1 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(y0) && clobber(y1)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpARM64MOVHUloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] x0:(MOVHUload [i0] {s} p mem) y1:(MOVDnop x1:(MOVBUload [i2] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i3] {s} p mem)))
	// cond: i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y1) && clobber(y2) && clobber(o0)
	// result: @mergePoint(b,x0,x1,x2) (MOVWUload <t> {s} (OffPtr <p.Type> [i0] p) mem)
	for {
		t := v.Type
		if v.AuxInt != 24 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 16 {
			break
		}
		_ = o0.Args[1]
		x0 := o0.Args[0]
		if x0.Op != OpARM64MOVHUload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
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
		i2 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := v.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i3 := x2.AuxInt
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		if !(i2 == i0+2 && i3 == i0+3 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y1) && clobber(y2) && clobber(o0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2)
		v0 := b.NewValue0(x2.Pos, OpARM64MOVWUload, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.Aux = s
		v1 := b.NewValue0(x2.Pos, OpOffPtr, p.Type)
		v1.AuxInt = i0
		v1.AddArg(p)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] x0:(MOVHUloadidx ptr0 idx0 mem) y1:(MOVDnop x1:(MOVBUload [2] {s} p1:(ADD ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [3] {s} p mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y1) && clobber(y2) && clobber(o0)
	// result: @mergePoint(b,x0,x1,x2) (MOVWUloadidx <t> ptr0 idx0 mem)
	for {
		t := v.Type
		if v.AuxInt != 24 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 16 {
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
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 2 {
			break
		}
		s := x1.Aux
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x1.Args[1] {
			break
		}
		y2 := v.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		if x2.AuxInt != 3 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		p := x2.Args[0]
		if mem != x2.Args[1] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y1) && clobber(y2) && clobber(o0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2)
		v0 := b.NewValue0(x2.Pos, OpARM64MOVWUloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr0)
		v0.AddArg(idx0)
		v0.AddArg(mem)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] x0:(MOVHUloadidx ptr idx mem) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [3] idx) mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y1) && clobber(y2) && clobber(o0)
	// result: @mergePoint(b,x0,x1,x2) (MOVWUloadidx <t> ptr idx mem)
	for {
		t := v.Type
		if v.AuxInt != 24 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 16 {
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 2 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y2 := v.Args[1]
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
		if x2_1.Op != OpARM64ADDconst {
			break
		}
		if x2_1.AuxInt != 3 {
			break
		}
		if idx != x2_1.Args[0] {
			break
		}
		if mem != x2.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y1) && clobber(y2) && clobber(o0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2)
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] x0:(MOVHUloadidx2 ptr0 idx0 mem) y1:(MOVDnop x1:(MOVBUload [2] {s} p1:(ADDshiftLL [1] ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [3] {s} p mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y1) && clobber(y2) && clobber(o0)
	// result: @mergePoint(b,x0,x1,x2) (MOVWUloadidx <t> ptr0 (SLLconst <idx0.Type> [1] idx0) mem)
	for {
		t := v.Type
		if v.AuxInt != 24 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 16 {
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
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 2 {
			break
		}
		s := x1.Aux
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADDshiftLL {
			break
		}
		if p1.AuxInt != 1 {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x1.Args[1] {
			break
		}
		y2 := v.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		if x2.AuxInt != 3 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		p := x2.Args[0]
		if mem != x2.Args[1] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y1) && clobber(y2) && clobber(o0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2)
		v0 := b.NewValue0(x2.Pos, OpARM64MOVWUloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr0)
		v1 := b.NewValue0(x2.Pos, OpARM64SLLconst, idx0.Type)
		v1.AuxInt = 1
		v1.AddArg(idx0)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] x0:(MOVWUload [i0] {s} p mem) y1:(MOVDnop x1:(MOVBUload [i4] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i6] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i7] {s} p mem)))
	// cond: i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (MOVDload <t> {s} (OffPtr <p.Type> [i0] p) mem)
	for {
		t := v.Type
		if v.AuxInt != 56 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 32 {
			break
		}
		_ = o2.Args[1]
		x0 := o2.Args[0]
		if x0.Op != OpARM64MOVWUload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
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
		i4 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
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
		i5 := x2.AuxInt
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
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
		i6 := x3.AuxInt
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		y4 := v.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		i7 := x4.AuxInt
		if x4.Aux != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] {
			break
		}
		if mem != x4.Args[1] {
			break
		}
		if !(i4 == i0+4 && i5 == i0+5 && i6 == i0+6 && i7 == i0+7 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4)
		v0 := b.NewValue0(x4.Pos, OpARM64MOVDload, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.Aux = s
		v1 := b.NewValue0(x4.Pos, OpOffPtr, p.Type)
		v1.AuxInt = i0
		v1.AddArg(p)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] x0:(MOVWUloadidx ptr0 idx0 mem) y1:(MOVDnop x1:(MOVBUload [4] {s} p1:(ADD ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [6] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [7] {s} p mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (MOVDloadidx <t> ptr0 idx0 mem)
	for {
		t := v.Type
		if v.AuxInt != 56 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 32 {
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
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 4 {
			break
		}
		s := x1.Aux
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
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
		if x2.Op != OpARM64MOVBUload {
			break
		}
		if x2.AuxInt != 5 {
			break
		}
		if x2.Aux != s {
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
		if x3.Op != OpARM64MOVBUload {
			break
		}
		if x3.AuxInt != 6 {
			break
		}
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		y4 := v.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		if x4.AuxInt != 7 {
			break
		}
		if x4.Aux != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] {
			break
		}
		if mem != x4.Args[1] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4)
		v0 := b.NewValue0(x4.Pos, OpARM64MOVDloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr0)
		v0.AddArg(idx0)
		v0.AddArg(mem)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] x0:(MOVWUloadidx4 ptr0 idx0 mem) y1:(MOVDnop x1:(MOVBUload [4] {s} p1:(ADDshiftLL [2] ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUload [5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [6] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [7] {s} p mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (MOVDloadidx <t> ptr0 (SLLconst <idx0.Type> [2] idx0) mem)
	for {
		t := v.Type
		if v.AuxInt != 56 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 32 {
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
		if x1.Op != OpARM64MOVBUload {
			break
		}
		if x1.AuxInt != 4 {
			break
		}
		s := x1.Aux
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADDshiftLL {
			break
		}
		if p1.AuxInt != 2 {
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
		if x2.Op != OpARM64MOVBUload {
			break
		}
		if x2.AuxInt != 5 {
			break
		}
		if x2.Aux != s {
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
		if x3.Op != OpARM64MOVBUload {
			break
		}
		if x3.AuxInt != 6 {
			break
		}
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		y4 := v.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		if x4.AuxInt != 7 {
			break
		}
		if x4.Aux != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] {
			break
		}
		if mem != x4.Args[1] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4)
		v0 := b.NewValue0(x4.Pos, OpARM64MOVDloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr0)
		v1 := b.NewValue0(x4.Pos, OpARM64SLLconst, idx0.Type)
		v1.AuxInt = 2
		v1.AddArg(idx0)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] x0:(MOVWUloadidx ptr idx mem) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [4] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [5] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [6] idx) mem))) y4:(MOVDnop x4:(MOVBUloadidx ptr (ADDconst [7] idx) mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (MOVDloadidx <t> ptr idx mem)
	for {
		t := v.Type
		if v.AuxInt != 56 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 32 {
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 4 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
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
		if x2_1.Op != OpARM64ADDconst {
			break
		}
		if x2_1.AuxInt != 5 {
			break
		}
		if idx != x2_1.Args[0] {
			break
		}
		if mem != x2.Args[2] {
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
		if x3_1.Op != OpARM64ADDconst {
			break
		}
		if x3_1.AuxInt != 6 {
			break
		}
		if idx != x3_1.Args[0] {
			break
		}
		if mem != x3.Args[2] {
			break
		}
		y4 := v.Args[1]
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
		if x4_1.Op != OpARM64ADDconst {
			break
		}
		if x4_1.AuxInt != 7 {
			break
		}
		if idx != x4_1.Args[0] {
			break
		}
		if mem != x4.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDloadidx, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AddArg(ptr)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORshiftLL_20(v *Value) bool {
	b := v.Block
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUload [i1] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0) && clobber(x1) && clobber(y0) && clobber(y1)
	// result: @mergePoint(b,x0,x1) (REV16W <t> (MOVHUload <t> [i0] {s} p mem))
	for {
		t := v.Type
		if v.AuxInt != 8 {
			break
		}
		_ = v.Args[1]
		y0 := v.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		i1 := x0.AuxInt
		s := x0.Aux
		mem := x0.Args[1]
		p := x0.Args[0]
		y1 := v.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUload {
			break
		}
		i0 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(y0) && clobber(y1)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(x1.Pos, OpARM64REV16W, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x1.Pos, OpARM64MOVHUload, t)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem)) y1:(MOVDnop x1:(MOVBUloadidx ptr0 idx0 mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b,x0,x1) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x0) && clobber(x1) && clobber(y0) && clobber(y1)
	// result: @mergePoint(b,x0,x1) (REV16W <t> (MOVHUloadidx <t> ptr0 idx0 mem))
	for {
		t := v.Type
		if v.AuxInt != 8 {
			break
		}
		_ = v.Args[1]
		y0 := v.Args[0]
		if y0.Op != OpARM64MOVDnop {
			break
		}
		x0 := y0.Args[0]
		if x0.Op != OpARM64MOVBUload {
			break
		}
		if x0.AuxInt != 1 {
			break
		}
		s := x0.Aux
		mem := x0.Args[1]
		p1 := x0.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		y1 := v.Args[1]
		if y1.Op != OpARM64MOVDnop {
			break
		}
		x1 := y1.Args[0]
		if x1.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x1.Args[2]
		ptr0 := x1.Args[0]
		idx0 := x1.Args[1]
		if mem != x1.Args[2] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b, x0, x1) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && clobber(x0) && clobber(x1) && clobber(y0) && clobber(y1)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(x0.Pos, OpARM64REV16W, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x0.Pos, OpARM64MOVHUloadidx, t)
		v1.AddArg(ptr0)
		v1.AddArg(idx0)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUloadidx ptr (ADDconst [1] idx) mem)) y1:(MOVDnop x1:(MOVBUloadidx ptr idx mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b,x0,x1) != nil && clobber(x0) && clobber(x1) && clobber(y0) && clobber(y1)
	// result: @mergePoint(b,x0,x1) (REV16W <t> (MOVHUloadidx <t> ptr idx mem))
	for {
		t := v.Type
		if v.AuxInt != 8 {
			break
		}
		_ = v.Args[1]
		y0 := v.Args[0]
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
		if x0_1.Op != OpARM64ADDconst {
			break
		}
		if x0_1.AuxInt != 1 {
			break
		}
		idx := x0_1.Args[0]
		y1 := v.Args[1]
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
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(y0) && clobber(y1)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpARM64REV16W, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHUloadidx, t)
		v1.AddArg(ptr)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] y0:(REV16W x0:(MOVHUload [i2] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 && i2 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(o0)
	// result: @mergePoint(b,x0,x1,x2) (REVW <t> (MOVWUload <t> {s} (OffPtr <p.Type> [i0] p) mem))
	for {
		t := v.Type
		if v.AuxInt != 24 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 16 {
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
		i2 := x0.AuxInt
		s := x0.Aux
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
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
			break
		}
		y2 := v.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUload {
			break
		}
		i0 := x2.AuxInt
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
			break
		}
		if !(i1 == i0+1 && i2 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(o0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2)
		v0 := b.NewValue0(x2.Pos, OpARM64REVW, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x2.Pos, OpARM64MOVWUload, t)
		v1.Aux = s
		v2 := b.NewValue0(x2.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] y0:(REV16W x0:(MOVHUload [2] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr0 idx0 mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(o0)
	// result: @mergePoint(b,x0,x1,x2) (REVW <t> (MOVWUloadidx <t> ptr0 idx0 mem))
	for {
		t := v.Type
		if v.AuxInt != 24 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 16 {
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
		if x0.AuxInt != 2 {
			break
		}
		s := x0.Aux
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
		if x1.AuxInt != 1 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		p1 := x1.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x1.Args[1] {
			break
		}
		y2 := v.Args[1]
		if y2.Op != OpARM64MOVDnop {
			break
		}
		x2 := y2.Args[0]
		if x2.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x2.Args[2]
		ptr0 := x2.Args[0]
		idx0 := x2.Args[1]
		if mem != x2.Args[2] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(o0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2)
		v0 := b.NewValue0(x1.Pos, OpARM64REVW, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x1.Pos, OpARM64MOVWUloadidx, t)
		v1.AddArg(ptr0)
		v1.AddArg(idx0)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] y0:(REV16W x0:(MOVHUloadidx ptr (ADDconst [2] idx) mem)) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr idx mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b,x0,x1,x2) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(o0)
	// result: @mergePoint(b,x0,x1,x2) (REVW <t> (MOVWUloadidx <t> ptr idx mem))
	for {
		t := v.Type
		if v.AuxInt != 24 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 16 {
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
		if x0_1.Op != OpARM64ADDconst {
			break
		}
		if x0_1.AuxInt != 2 {
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 1 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y2 := v.Args[1]
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
		if idx != x2.Args[1] {
			break
		}
		if mem != x2.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && o0.Uses == 1 && mergePoint(b, x0, x1, x2) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(o0)) {
			break
		}
		b = mergePoint(b, x0, x1, x2)
		v0 := b.NewValue0(v.Pos, OpARM64REVW, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWUloadidx, t)
		v1.AddArg(ptr)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] y0:(REVW x0:(MOVWUload [i4] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [i3] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i1] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (REV <t> (MOVDload <t> {s} (OffPtr <p.Type> [i0] p) mem))
	for {
		t := v.Type
		if v.AuxInt != 56 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 32 {
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
		i4 := x0.AuxInt
		s := x0.Aux
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
		i3 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
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
		i2 := x2.AuxInt
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
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
		i1 := x3.AuxInt
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		if p != x3.Args[0] {
			break
		}
		if mem != x3.Args[1] {
			break
		}
		y4 := v.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUload {
			break
		}
		i0 := x4.AuxInt
		if x4.Aux != s {
			break
		}
		_ = x4.Args[1]
		if p != x4.Args[0] {
			break
		}
		if mem != x4.Args[1] {
			break
		}
		if !(i1 == i0+1 && i2 == i0+2 && i3 == i0+3 && i4 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4)
		v0 := b.NewValue0(x4.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x4.Pos, OpARM64MOVDload, t)
		v1.Aux = s
		v2 := b.NewValue0(x4.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] y0:(REVW x0:(MOVWUload [4] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [3] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [1] {s} p1:(ADD ptr1 idx1) mem))) y4:(MOVDnop x4:(MOVBUloadidx ptr0 idx0 mem)))
	// cond: s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (REV <t> (MOVDloadidx <t> ptr0 idx0 mem))
	for {
		t := v.Type
		if v.AuxInt != 56 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 32 {
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
		if x0.AuxInt != 4 {
			break
		}
		s := x0.Aux
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
		if x1.AuxInt != 3 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[1]
		if p != x1.Args[0] {
			break
		}
		if mem != x1.Args[1] {
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
		if x2.AuxInt != 2 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[1]
		if p != x2.Args[0] {
			break
		}
		if mem != x2.Args[1] {
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
		if x3.AuxInt != 1 {
			break
		}
		if x3.Aux != s {
			break
		}
		_ = x3.Args[1]
		p1 := x3.Args[0]
		if p1.Op != OpARM64ADD {
			break
		}
		idx1 := p1.Args[1]
		ptr1 := p1.Args[0]
		if mem != x3.Args[1] {
			break
		}
		y4 := v.Args[1]
		if y4.Op != OpARM64MOVDnop {
			break
		}
		x4 := y4.Args[0]
		if x4.Op != OpARM64MOVBUloadidx {
			break
		}
		_ = x4.Args[2]
		ptr0 := x4.Args[0]
		idx0 := x4.Args[1]
		if mem != x4.Args[2] {
			break
		}
		if !(s == nil && x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && (isSamePtr(ptr0, ptr1) && isSamePtr(idx0, idx1) || isSamePtr(ptr0, idx1) && isSamePtr(idx0, ptr1)) && isSamePtr(p1, p) && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4)
		v0 := b.NewValue0(x3.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(x3.Pos, OpARM64MOVDloadidx, t)
		v1.AddArg(ptr0)
		v1.AddArg(idx0)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] y0:(REVW x0:(MOVWUloadidx ptr (ADDconst [4] idx) mem)) y1:(MOVDnop x1:(MOVBUloadidx ptr (ADDconst [3] idx) mem))) y2:(MOVDnop x2:(MOVBUloadidx ptr (ADDconst [2] idx) mem))) y3:(MOVDnop x3:(MOVBUloadidx ptr (ADDconst [1] idx) mem))) y4:(MOVDnop x4:(MOVBUloadidx ptr idx mem)))
	// cond: x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b,x0,x1,x2,x3,x4) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)
	// result: @mergePoint(b,x0,x1,x2,x3,x4) (REV <t> (MOVDloadidx <t> ptr idx mem))
	for {
		t := v.Type
		if v.AuxInt != 56 {
			break
		}
		_ = v.Args[1]
		o0 := v.Args[0]
		if o0.Op != OpARM64ORshiftLL {
			break
		}
		if o0.AuxInt != 48 {
			break
		}
		_ = o0.Args[1]
		o1 := o0.Args[0]
		if o1.Op != OpARM64ORshiftLL {
			break
		}
		if o1.AuxInt != 40 {
			break
		}
		_ = o1.Args[1]
		o2 := o1.Args[0]
		if o2.Op != OpARM64ORshiftLL {
			break
		}
		if o2.AuxInt != 32 {
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
		if x0_1.Op != OpARM64ADDconst {
			break
		}
		if x0_1.AuxInt != 4 {
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
		if x1_1.Op != OpARM64ADDconst {
			break
		}
		if x1_1.AuxInt != 3 {
			break
		}
		if idx != x1_1.Args[0] {
			break
		}
		if mem != x1.Args[2] {
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
		if x2_1.Op != OpARM64ADDconst {
			break
		}
		if x2_1.AuxInt != 2 {
			break
		}
		if idx != x2_1.Args[0] {
			break
		}
		if mem != x2.Args[2] {
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
		if x3_1.Op != OpARM64ADDconst {
			break
		}
		if x3_1.AuxInt != 1 {
			break
		}
		if idx != x3_1.Args[0] {
			break
		}
		if mem != x3.Args[2] {
			break
		}
		y4 := v.Args[1]
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
		if idx != x4.Args[1] {
			break
		}
		if mem != x4.Args[2] {
			break
		}
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 && o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && mergePoint(b, x0, x1, x2, x3, x4) != nil && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) && clobber(o0) && clobber(o1) && clobber(o2)) {
			break
		}
		b = mergePoint(b, x0, x1, x2, x3, x4)
		v0 := b.NewValue0(v.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDloadidx, t)
		v1.AddArg(ptr)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (ORshiftRA (MOVDconst [c]) x [d])
	// cond:
	// result: (ORconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64ORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRA x (MOVDconst [c]) [d])
	// cond:
	// result: (ORconst x [c>>uint64(d)])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ORconst)
		v.AuxInt = c >> uint64(d)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRA x y:(SRAconst x [c]) [d])
	// cond: c==d
	// result: y
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if y.Op != OpARM64SRAconst {
			break
		}
		c := y.AuxInt
		if x != y.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (ORshiftRL (MOVDconst [c]) x [d])
	// cond:
	// result: (ORconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64ORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRL x (MOVDconst [c]) [d])
	// cond:
	// result: (ORconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64ORconst)
		v.AuxInt = int64(uint64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRL x y:(SRLconst x [c]) [d])
	// cond: c==d
	// result: y
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if y.Op != OpARM64SRLconst {
			break
		}
		c := y.AuxInt
		if x != y.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (ORshiftRL [c] (SLLconst x [64-c]) x)
	// cond:
	// result: (RORconst [ c] x)
	for {
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRL <t> [c] (SLLconst x [32-c]) (MOVWUreg x))
	// cond: c < 32 && t.Size() == 4
	// result: (RORWconst [c] x)
	for {
		t := v.Type
		c := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 32-c {
			break
		}
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWUreg {
			break
		}
		if x != v_1.Args[0] {
			break
		}
		if !(c < 32 && t.Size() == 4) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRL [rc] (ANDconst [ac] x) (SLLconst [lc] y))
	// cond: lc > rc && ac == ^((1<<uint(64-lc)-1) << uint64(lc-rc))
	// result: (BFI [armBFAuxInt(lc-rc, 64-lc)] x y)
	for {
		rc := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ANDconst {
			break
		}
		ac := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		lc := v_1.AuxInt
		y := v_1.Args[0]
		if !(lc > rc && ac == ^((1<<uint(64-lc)-1)<<uint64(lc-rc))) {
			break
		}
		v.reset(OpARM64BFI)
		v.AuxInt = armBFAuxInt(lc-rc, 64-lc)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64RORWconst_0(v *Value) bool {
	// match: (RORWconst [c] (RORWconst [d] x))
	// cond:
	// result: (RORWconst [(c+d)&31] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64RORWconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARM64RORWconst)
		v.AuxInt = (c + d) & 31
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64RORconst_0(v *Value) bool {
	// match: (RORconst [c] (RORconst [d] x))
	// cond:
	// result: (RORconst [(c+d)&63] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64RORconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARM64RORconst)
		v.AuxInt = (c + d) & 63
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SBCSflags_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (SBCSflags x y (Select1 <types.TypeFlags> (NEGSflags (NEG <typ.UInt64> (NGCzerocarry <typ.UInt64> bo)))))
	// cond:
	// result: (SBCSflags x y bo)
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpSelect1 {
			break
		}
		if v_2.Type != types.TypeFlags {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpARM64NEGSflags {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpARM64NEG {
			break
		}
		if v_2_0_0.Type != typ.UInt64 {
			break
		}
		v_2_0_0_0 := v_2_0_0.Args[0]
		if v_2_0_0_0.Op != OpARM64NGCzerocarry {
			break
		}
		if v_2_0_0_0.Type != typ.UInt64 {
			break
		}
		bo := v_2_0_0_0.Args[0]
		v.reset(OpARM64SBCSflags)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(bo)
		return true
	}
	// match: (SBCSflags x y (Select1 <types.TypeFlags> (NEGSflags (MOVDconst [0]))))
	// cond:
	// result: (SUBSflags x y)
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpSelect1 {
			break
		}
		if v_2.Type != types.TypeFlags {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpARM64NEGSflags {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_2_0_0.AuxInt != 0 {
			break
		}
		v.reset(OpARM64SUBSflags)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SLL_0(v *Value) bool {
	// match: (SLL x (MOVDconst [c]))
	// cond:
	// result: (SLLconst x [c&63])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64SLLconst)
		v.AuxInt = c & 63
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SLLconst_0(v *Value) bool {
	// match: (SLLconst [c] (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [d<<uint64(c)])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = d << uint64(c)
		return true
	}
	// match: (SLLconst [c] (SRLconst [c] x))
	// cond: 0 < c && c < 64
	// result: (ANDconst [^(1<<uint(c)-1)] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != c {
			break
		}
		x := v_0.Args[0]
		if !(0 < c && c < 64) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = ^(1<<uint(c) - 1)
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [sc] (ANDconst [ac] x))
	// cond: isARM64BFMask(sc, ac, 0)
	// result: (UBFIZ [armBFAuxInt(sc, arm64BFWidth(ac, 0))] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ANDconst {
			break
		}
		ac := v_0.AuxInt
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, ac, 0)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(sc, arm64BFWidth(ac, 0))
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [sc] (MOVWUreg x))
	// cond: isARM64BFMask(sc, 1<<32-1, 0)
	// result: (UBFIZ [armBFAuxInt(sc, 32)] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<32-1, 0)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(sc, 32)
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [sc] (MOVHUreg x))
	// cond: isARM64BFMask(sc, 1<<16-1, 0)
	// result: (UBFIZ [armBFAuxInt(sc, 16)] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<16-1, 0)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(sc, 16)
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [sc] (MOVBUreg x))
	// cond: isARM64BFMask(sc, 1<<8-1, 0)
	// result: (UBFIZ [armBFAuxInt(sc, 8)] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<8-1, 0)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(sc, 8)
		v.AddArg(x)
		return true
	}
	// match: (SLLconst [sc] (UBFIZ [bfc] x))
	// cond: sc+getARM64BFwidth(bfc)+getARM64BFlsb(bfc) < 64
	// result: (UBFIZ [armBFAuxInt(getARM64BFlsb(bfc)+sc, getARM64BFwidth(bfc))] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		if !(sc+getARM64BFwidth(bfc)+getARM64BFlsb(bfc) < 64) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(getARM64BFlsb(bfc)+sc, getARM64BFwidth(bfc))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SRA_0(v *Value) bool {
	// match: (SRA x (MOVDconst [c]))
	// cond:
	// result: (SRAconst x [c&63])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64SRAconst)
		v.AuxInt = c & 63
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SRAconst_0(v *Value) bool {
	// match: (SRAconst [c] (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [d>>uint64(c)])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = d >> uint64(c)
		return true
	}
	// match: (SRAconst [rc] (SLLconst [lc] x))
	// cond: lc > rc
	// result: (SBFIZ [armBFAuxInt(lc-rc, 64-lc)] x)
	for {
		rc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := v_0.AuxInt
		x := v_0.Args[0]
		if !(lc > rc) {
			break
		}
		v.reset(OpARM64SBFIZ)
		v.AuxInt = armBFAuxInt(lc-rc, 64-lc)
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [rc] (SLLconst [lc] x))
	// cond: lc <= rc
	// result: (SBFX [armBFAuxInt(rc-lc, 64-rc)] x)
	for {
		rc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := v_0.AuxInt
		x := v_0.Args[0]
		if !(lc <= rc) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = armBFAuxInt(rc-lc, 64-rc)
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [rc] (MOVWreg x))
	// cond: rc < 32
	// result: (SBFX [armBFAuxInt(rc, 32-rc)] x)
	for {
		rc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVWreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 32) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = armBFAuxInt(rc, 32-rc)
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [rc] (MOVHreg x))
	// cond: rc < 16
	// result: (SBFX [armBFAuxInt(rc, 16-rc)] x)
	for {
		rc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVHreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 16) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = armBFAuxInt(rc, 16-rc)
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [rc] (MOVBreg x))
	// cond: rc < 8
	// result: (SBFX [armBFAuxInt(rc, 8-rc)] x)
	for {
		rc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVBreg {
			break
		}
		x := v_0.Args[0]
		if !(rc < 8) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = armBFAuxInt(rc, 8-rc)
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [sc] (SBFIZ [bfc] x))
	// cond: sc < getARM64BFlsb(bfc)
	// result: (SBFIZ [armBFAuxInt(getARM64BFlsb(bfc)-sc, getARM64BFwidth(bfc))] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SBFIZ {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		if !(sc < getARM64BFlsb(bfc)) {
			break
		}
		v.reset(OpARM64SBFIZ)
		v.AuxInt = armBFAuxInt(getARM64BFlsb(bfc)-sc, getARM64BFwidth(bfc))
		v.AddArg(x)
		return true
	}
	// match: (SRAconst [sc] (SBFIZ [bfc] x))
	// cond: sc >= getARM64BFlsb(bfc) && sc < getARM64BFlsb(bfc)+getARM64BFwidth(bfc)
	// result: (SBFX [armBFAuxInt(sc-getARM64BFlsb(bfc), getARM64BFlsb(bfc)+getARM64BFwidth(bfc)-sc)] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SBFIZ {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		if !(sc >= getARM64BFlsb(bfc) && sc < getARM64BFlsb(bfc)+getARM64BFwidth(bfc)) {
			break
		}
		v.reset(OpARM64SBFX)
		v.AuxInt = armBFAuxInt(sc-getARM64BFlsb(bfc), getARM64BFlsb(bfc)+getARM64BFwidth(bfc)-sc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SRL_0(v *Value) bool {
	// match: (SRL x (MOVDconst [c]))
	// cond:
	// result: (SRLconst x [c&63])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64SRLconst)
		v.AuxInt = c & 63
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SRLconst_0(v *Value) bool {
	// match: (SRLconst [c] (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [int64(uint64(d)>>uint64(c))])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(uint64(d) >> uint64(c))
		return true
	}
	// match: (SRLconst [c] (SLLconst [c] x))
	// cond: 0 < c && c < 64
	// result: (ANDconst [1<<uint(64-c)-1] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != c {
			break
		}
		x := v_0.Args[0]
		if !(0 < c && c < 64) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = 1<<uint(64-c) - 1
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [rc] (SLLconst [lc] x))
	// cond: lc > rc
	// result: (UBFIZ [armBFAuxInt(lc-rc, 64-lc)] x)
	for {
		rc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := v_0.AuxInt
		x := v_0.Args[0]
		if !(lc > rc) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(lc-rc, 64-lc)
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (ANDconst [ac] x))
	// cond: isARM64BFMask(sc, ac, sc)
	// result: (UBFX [armBFAuxInt(sc, arm64BFWidth(ac, sc))] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ANDconst {
			break
		}
		ac := v_0.AuxInt
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, ac, sc)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(sc, arm64BFWidth(ac, sc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (MOVWUreg x))
	// cond: isARM64BFMask(sc, 1<<32-1, sc)
	// result: (UBFX [armBFAuxInt(sc, arm64BFWidth(1<<32-1, sc))] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<32-1, sc)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(sc, arm64BFWidth(1<<32-1, sc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (MOVHUreg x))
	// cond: isARM64BFMask(sc, 1<<16-1, sc)
	// result: (UBFX [armBFAuxInt(sc, arm64BFWidth(1<<16-1, sc))] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVHUreg {
			break
		}
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<16-1, sc)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(sc, arm64BFWidth(1<<16-1, sc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (MOVBUreg x))
	// cond: isARM64BFMask(sc, 1<<8-1, sc)
	// result: (UBFX [armBFAuxInt(sc, arm64BFWidth(1<<8-1, sc))] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVBUreg {
			break
		}
		x := v_0.Args[0]
		if !(isARM64BFMask(sc, 1<<8-1, sc)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(sc, arm64BFWidth(1<<8-1, sc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [rc] (SLLconst [lc] x))
	// cond: lc < rc
	// result: (UBFX [armBFAuxInt(rc-lc, 64-rc)] x)
	for {
		rc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		lc := v_0.AuxInt
		x := v_0.Args[0]
		if !(lc < rc) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(rc-lc, 64-rc)
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFX [bfc] x))
	// cond: sc < getARM64BFwidth(bfc)
	// result: (UBFX [armBFAuxInt(getARM64BFlsb(bfc)+sc, getARM64BFwidth(bfc)-sc)] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		if !(sc < getARM64BFwidth(bfc)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(getARM64BFlsb(bfc)+sc, getARM64BFwidth(bfc)-sc)
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFIZ [bfc] x))
	// cond: sc == getARM64BFlsb(bfc)
	// result: (ANDconst [1<<uint(getARM64BFwidth(bfc))-1] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		if !(sc == getARM64BFlsb(bfc)) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = 1<<uint(getARM64BFwidth(bfc)) - 1
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SRLconst_10(v *Value) bool {
	// match: (SRLconst [sc] (UBFIZ [bfc] x))
	// cond: sc < getARM64BFlsb(bfc)
	// result: (UBFIZ [armBFAuxInt(getARM64BFlsb(bfc)-sc, getARM64BFwidth(bfc))] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		if !(sc < getARM64BFlsb(bfc)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(getARM64BFlsb(bfc)-sc, getARM64BFwidth(bfc))
		v.AddArg(x)
		return true
	}
	// match: (SRLconst [sc] (UBFIZ [bfc] x))
	// cond: sc > getARM64BFlsb(bfc) && sc < getARM64BFlsb(bfc)+getARM64BFwidth(bfc)
	// result: (UBFX [armBFAuxInt(sc-getARM64BFlsb(bfc), getARM64BFlsb(bfc)+getARM64BFwidth(bfc)-sc)] x)
	for {
		sc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFIZ {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		if !(sc > getARM64BFlsb(bfc) && sc < getARM64BFlsb(bfc)+getARM64BFwidth(bfc)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(sc-getARM64BFlsb(bfc), getARM64BFlsb(bfc)+getARM64BFwidth(bfc)-sc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64STP_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (STP [off1] {sym} (ADDconst [off2] ptr) val1 val2 mem)
	// cond: is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (STP [off1+off2] {sym} ptr val1 val2 mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val1 := v.Args[1]
		val2 := v.Args[2]
		if !(is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64STP)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val1)
		v.AddArg(val2)
		v.AddArg(mem)
		return true
	}
	// match: (STP [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val1 val2 mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (STP [off1+off2] {mergeSym(sym1,sym2)} ptr val1 val2 mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val1 := v.Args[1]
		val2 := v.Args[2]
		if !(canMergeSym(sym1, sym2) && is32Bit(off1+off2) && (ptr.Op != OpSB || !config.ctxt.Flag_shared)) {
			break
		}
		v.reset(OpARM64STP)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(val1)
		v.AddArg(val2)
		v.AddArg(mem)
		return true
	}
	// match: (STP [off] {sym} ptr (MOVDconst [0]) (MOVDconst [0]) mem)
	// cond:
	// result: (MOVQstorezero [off] {sym} ptr mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v_2 := v.Args[2]
		if v_2.Op != OpARM64MOVDconst {
			break
		}
		if v_2.AuxInt != 0 {
			break
		}
		v.reset(OpARM64MOVQstorezero)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SUB_0(v *Value) bool {
	b := v.Block
	// match: (SUB x (MOVDconst [c]))
	// cond:
	// result: (SUBconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64SUBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (SUB a l:(MUL x y))
	// cond: l.Uses==1 && clobber(l)
	// result: (MSUB a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		l := v.Args[1]
		if l.Op != OpARM64MUL {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MSUB)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUB a l:(MNEG x y))
	// cond: l.Uses==1 && clobber(l)
	// result: (MADD a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		l := v.Args[1]
		if l.Op != OpARM64MNEG {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MADD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUB a l:(MULW x y))
	// cond: a.Type.Size() != 8 && l.Uses==1 && clobber(l)
	// result: (MSUBW a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		l := v.Args[1]
		if l.Op != OpARM64MULW {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(a.Type.Size() != 8 && l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MSUBW)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUB a l:(MNEGW x y))
	// cond: a.Type.Size() != 8 && l.Uses==1 && clobber(l)
	// result: (MADDW a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		l := v.Args[1]
		if l.Op != OpARM64MNEGW {
			break
		}
		y := l.Args[1]
		x := l.Args[0]
		if !(a.Type.Size() != 8 && l.Uses == 1 && clobber(l)) {
			break
		}
		v.reset(OpARM64MADDW)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUB x x)
	// cond:
	// result: (MOVDconst [0])
	for {
		x := v.Args[1]
		if x != v.Args[0] {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (SUB x (SUB y z))
	// cond:
	// result: (SUB (ADD <v.Type> x z) y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SUB {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, v.Type)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(y)
		return true
	}
	// match: (SUB (SUB x y) z)
	// cond:
	// result: (SUB x (ADD <y.Type> y z))
	for {
		z := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SUB {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARM64SUB)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, y.Type)
		v0.AddArg(y)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (SUB x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (SUBshiftLL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64SUBshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (SUB x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (SUBshiftRL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64SUBshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SUB_10(v *Value) bool {
	// match: (SUB x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (SUBshiftRA x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64SUBshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SUBconst_0(v *Value) bool {
	// match: (SUBconst [0] x)
	// cond:
	// result: x
	for {
		if v.AuxInt != 0 {
			break
		}
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [d-c])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = d - c
		return true
	}
	// match: (SUBconst [c] (SUBconst [d] x))
	// cond:
	// result: (ADDconst [-c-d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SUBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARM64ADDconst)
		v.AuxInt = -c - d
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] (ADDconst [d] x))
	// cond:
	// result: (ADDconst [-c+d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARM64ADDconst)
		v.AuxInt = -c + d
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SUBshiftLL_0(v *Value) bool {
	// match: (SUBshiftLL x (MOVDconst [c]) [d])
	// cond:
	// result: (SUBconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64(uint64(c) << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SUBshiftRA_0(v *Value) bool {
	// match: (SUBshiftRA x (MOVDconst [c]) [d])
	// cond:
	// result: (SUBconst x [c>>uint64(d)])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64SUBconst)
		v.AuxInt = c >> uint64(d)
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64SUBshiftRL_0(v *Value) bool {
	// match: (SUBshiftRL x (MOVDconst [c]) [d])
	// cond:
	// result: (SUBconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64SUBconst)
		v.AuxInt = int64(uint64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64TST_0(v *Value) bool {
	// match: (TST x (MOVDconst [c]))
	// cond:
	// result: (TSTconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64TSTconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (TST (MOVDconst [c]) x)
	// cond:
	// result: (TSTconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64TSTconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (TST x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (TSTshiftLL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64TSTshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (TST x1:(SLLconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (TSTshiftLL x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64TSTshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (TST x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (TSTshiftRL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64TSTshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (TST x1:(SRLconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (TSTshiftRL x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64TSTshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (TST x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (TSTshiftRA x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64TSTshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (TST x1:(SRAconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (TSTshiftRA x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64TSTshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64TSTW_0(v *Value) bool {
	// match: (TSTW x (MOVDconst [c]))
	// cond:
	// result: (TSTWconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64TSTWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (TSTW (MOVDconst [c]) x)
	// cond:
	// result: (TSTWconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64TSTWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64TSTWconst_0(v *Value) bool {
	// match: (TSTWconst (MOVDconst [x]) [y])
	// cond: int32(x&y)==0
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x&y) == 0) {
			break
		}
		v.reset(OpARM64FlagEQ)
		return true
	}
	// match: (TSTWconst (MOVDconst [x]) [y])
	// cond: int32(x&y)<0
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x&y) < 0) {
			break
		}
		v.reset(OpARM64FlagLT_UGT)
		return true
	}
	// match: (TSTWconst (MOVDconst [x]) [y])
	// cond: int32(x&y)>0
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x&y) > 0) {
			break
		}
		v.reset(OpARM64FlagGT_UGT)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64TSTconst_0(v *Value) bool {
	// match: (TSTconst (MOVDconst [x]) [y])
	// cond: int64(x&y)==0
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int64(x&y) == 0) {
			break
		}
		v.reset(OpARM64FlagEQ)
		return true
	}
	// match: (TSTconst (MOVDconst [x]) [y])
	// cond: int64(x&y)<0
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int64(x&y) < 0) {
			break
		}
		v.reset(OpARM64FlagLT_UGT)
		return true
	}
	// match: (TSTconst (MOVDconst [x]) [y])
	// cond: int64(x&y)>0
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int64(x&y) > 0) {
			break
		}
		v.reset(OpARM64FlagGT_UGT)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64TSTshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (TSTshiftLL (MOVDconst [c]) x [d])
	// cond:
	// result: (TSTconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64TSTconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftLL x (MOVDconst [c]) [d])
	// cond:
	// result: (TSTconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64TSTconst)
		v.AuxInt = int64(uint64(c) << uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64TSTshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (TSTshiftRA (MOVDconst [c]) x [d])
	// cond:
	// result: (TSTconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64TSTconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRA x (MOVDconst [c]) [d])
	// cond:
	// result: (TSTconst x [c>>uint64(d)])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64TSTconst)
		v.AuxInt = c >> uint64(d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64TSTshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (TSTshiftRL (MOVDconst [c]) x [d])
	// cond:
	// result: (TSTconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64TSTconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRL x (MOVDconst [c]) [d])
	// cond:
	// result: (TSTconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64TSTconst)
		v.AuxInt = int64(uint64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UBFIZ_0(v *Value) bool {
	// match: (UBFIZ [bfc] (SLLconst [sc] x))
	// cond: sc < getARM64BFwidth(bfc)
	// result: (UBFIZ [armBFAuxInt(getARM64BFlsb(bfc)+sc, getARM64BFwidth(bfc)-sc)] x)
	for {
		bfc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(sc < getARM64BFwidth(bfc)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(getARM64BFlsb(bfc)+sc, getARM64BFwidth(bfc)-sc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UBFX_0(v *Value) bool {
	// match: (UBFX [bfc] (SRLconst [sc] x))
	// cond: sc+getARM64BFwidth(bfc)+getARM64BFlsb(bfc) < 64
	// result: (UBFX [armBFAuxInt(getARM64BFlsb(bfc)+sc, getARM64BFwidth(bfc))] x)
	for {
		bfc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(sc+getARM64BFwidth(bfc)+getARM64BFlsb(bfc) < 64) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(getARM64BFlsb(bfc)+sc, getARM64BFwidth(bfc))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SLLconst [sc] x))
	// cond: sc == getARM64BFlsb(bfc)
	// result: (ANDconst [1<<uint(getARM64BFwidth(bfc))-1] x)
	for {
		bfc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(sc == getARM64BFlsb(bfc)) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = 1<<uint(getARM64BFwidth(bfc)) - 1
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SLLconst [sc] x))
	// cond: sc < getARM64BFlsb(bfc)
	// result: (UBFX [armBFAuxInt(getARM64BFlsb(bfc)-sc, getARM64BFwidth(bfc))] x)
	for {
		bfc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(sc < getARM64BFlsb(bfc)) {
			break
		}
		v.reset(OpARM64UBFX)
		v.AuxInt = armBFAuxInt(getARM64BFlsb(bfc)-sc, getARM64BFwidth(bfc))
		v.AddArg(x)
		return true
	}
	// match: (UBFX [bfc] (SLLconst [sc] x))
	// cond: sc > getARM64BFlsb(bfc) && sc < getARM64BFlsb(bfc)+getARM64BFwidth(bfc)
	// result: (UBFIZ [armBFAuxInt(sc-getARM64BFlsb(bfc), getARM64BFlsb(bfc)+getARM64BFwidth(bfc)-sc)] x)
	for {
		bfc := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		sc := v_0.AuxInt
		x := v_0.Args[0]
		if !(sc > getARM64BFlsb(bfc) && sc < getARM64BFlsb(bfc)+getARM64BFwidth(bfc)) {
			break
		}
		v.reset(OpARM64UBFIZ)
		v.AuxInt = armBFAuxInt(sc-getARM64BFlsb(bfc), getARM64BFlsb(bfc)+getARM64BFwidth(bfc)-sc)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UDIV_0(v *Value) bool {
	// match: (UDIV x (MOVDconst [1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (UDIV x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (SRLconst [log2(c)] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SRLconst)
		v.AuxInt = log2(c)
		v.AddArg(x)
		return true
	}
	// match: (UDIV (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [int64(uint64(c)/uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(uint64(c) / uint64(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UDIVW_0(v *Value) bool {
	// match: (UDIVW x (MOVDconst [c]))
	// cond: uint32(c)==1
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint32(c) == 1) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (UDIVW x (MOVDconst [c]))
	// cond: isPowerOfTwo(c) && is32Bit(c)
	// result: (SRLconst [log2(c)] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64SRLconst)
		v.AuxInt = log2(c)
		v.AddArg(x)
		return true
	}
	// match: (UDIVW (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [int64(uint32(c)/uint32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(uint32(c) / uint32(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UMOD_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (UMOD <typ.UInt64> x y)
	// cond:
	// result: (MSUB <typ.UInt64> x y (UDIV <typ.UInt64> x y))
	for {
		if v.Type != typ.UInt64 {
			break
		}
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64MSUB)
		v.Type = typ.UInt64
		v.AddArg(x)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpARM64UDIV, typ.UInt64)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (UMOD _ (MOVDconst [1]))
	// cond:
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (UMOD x (MOVDconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (ANDconst [c-1] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = c - 1
		v.AddArg(x)
		return true
	}
	// match: (UMOD (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [int64(uint64(c)%uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(uint64(c) % uint64(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64UMODW_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (UMODW <typ.UInt32> x y)
	// cond:
	// result: (MSUBW <typ.UInt32> x y (UDIVW <typ.UInt32> x y))
	for {
		if v.Type != typ.UInt32 {
			break
		}
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64MSUBW)
		v.Type = typ.UInt32
		v.AddArg(x)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpARM64UDIVW, typ.UInt32)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (UMODW _ (MOVDconst [c]))
	// cond: uint32(c)==1
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint32(c) == 1) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (UMODW x (MOVDconst [c]))
	// cond: isPowerOfTwo(c) && is32Bit(c)
	// result: (ANDconst [c-1] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c) && is32Bit(c)) {
			break
		}
		v.reset(OpARM64ANDconst)
		v.AuxInt = c - 1
		v.AddArg(x)
		return true
	}
	// match: (UMODW (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [int64(uint32(c)%uint32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(uint32(c) % uint32(d))
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64XOR_0(v *Value) bool {
	// match: (XOR x (MOVDconst [c]))
	// cond:
	// result: (XORconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64XORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XOR (MOVDconst [c]) x)
	// cond:
	// result: (XORconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64XORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XOR x x)
	// cond:
	// result: (MOVDconst [0])
	for {
		x := v.Args[1]
		if x != v.Args[0] {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (XOR x (MVN y))
	// cond:
	// result: (EON x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MVN {
			break
		}
		y := v_1.Args[0]
		v.reset(OpARM64EON)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR (MVN y) x)
	// cond:
	// result: (EON x y)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MVN {
			break
		}
		y := v_0.Args[0]
		v.reset(OpARM64EON)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR x0 x1:(SLLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (XORshiftLL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64XORshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (XOR x1:(SLLconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (XORshiftLL x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SLLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64XORshiftLL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (XOR x0 x1:(SRLconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (XORshiftRL x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64XORshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (XOR x1:(SRLconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (XORshiftRL x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SRLconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64XORshiftRL)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (XOR x0 x1:(SRAconst [c] y))
	// cond: clobberIfDead(x1)
	// result: (XORshiftRA x0 y [c])
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		x1 := v.Args[1]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64XORshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64XOR_10(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (XOR x1:(SRAconst [c] y) x0)
	// cond: clobberIfDead(x1)
	// result: (XORshiftRA x0 y [c])
	for {
		x0 := v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpARM64SRAconst {
			break
		}
		c := x1.AuxInt
		y := x1.Args[0]
		if !(clobberIfDead(x1)) {
			break
		}
		v.reset(OpARM64XORshiftRA)
		v.AuxInt = c
		v.AddArg(x0)
		v.AddArg(y)
		return true
	}
	// match: (XOR (SLL x (ANDconst <t> [63] y)) (CSEL0 <typ.UInt64> {cc} (SRL <typ.UInt64> x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (ROR x (NEG <t> y))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64ANDconst {
			break
		}
		t := v_0_1.Type
		if v_0_1.AuxInt != 63 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CSEL0 {
			break
		}
		if v_1.Type != typ.UInt64 {
			break
		}
		cc := v_1.Aux
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64SRL {
			break
		}
		if v_1_0.Type != typ.UInt64 {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpARM64SUB {
			break
		}
		if v_1_0_1.Type != t {
			break
		}
		_ = v_1_0_1.Args[1]
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_0_1_0.AuxInt != 64 {
			break
		}
		v_1_0_1_1 := v_1_0_1.Args[1]
		if v_1_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_0_1_1.Type != t {
			break
		}
		if v_1_0_1_1.AuxInt != 63 {
			break
		}
		if y != v_1_0_1_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64CMPconst {
			break
		}
		if v_1_1.AuxInt != 64 {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpARM64SUB {
			break
		}
		if v_1_1_0.Type != t {
			break
		}
		_ = v_1_1_0.Args[1]
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_1_0_0.AuxInt != 64 {
			break
		}
		v_1_1_0_1 := v_1_1_0.Args[1]
		if v_1_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1_0_1.Type != t {
			break
		}
		if v_1_1_0_1.AuxInt != 63 {
			break
		}
		if y != v_1_1_0_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (XOR (CSEL0 <typ.UInt64> {cc} (SRL <typ.UInt64> x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))) (SLL x (ANDconst <t> [63] y)))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (ROR x (NEG <t> y))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64CSEL0 {
			break
		}
		if v_0.Type != typ.UInt64 {
			break
		}
		cc := v_0.Aux
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64SRL {
			break
		}
		if v_0_0.Type != typ.UInt64 {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpARM64SUB {
			break
		}
		t := v_0_0_1.Type
		_ = v_0_0_1.Args[1]
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_0_1_0.AuxInt != 64 {
			break
		}
		v_0_0_1_1 := v_0_0_1.Args[1]
		if v_0_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_0_1_1.Type != t {
			break
		}
		if v_0_0_1_1.AuxInt != 63 {
			break
		}
		y := v_0_0_1_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64CMPconst {
			break
		}
		if v_0_1.AuxInt != 64 {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpARM64SUB {
			break
		}
		if v_0_1_0.Type != t {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_1_0_0.AuxInt != 64 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_1_0_1.Type != t {
			break
		}
		if v_0_1_0_1.AuxInt != 63 {
			break
		}
		if y != v_0_1_0_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1.Type != t {
			break
		}
		if v_1_1.AuxInt != 63 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (XOR (SRL <typ.UInt64> x (ANDconst <t> [63] y)) (CSEL0 <typ.UInt64> {cc} (SLL x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (ROR x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRL {
			break
		}
		if v_0.Type != typ.UInt64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64ANDconst {
			break
		}
		t := v_0_1.Type
		if v_0_1.AuxInt != 63 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CSEL0 {
			break
		}
		if v_1.Type != typ.UInt64 {
			break
		}
		cc := v_1.Aux
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64SLL {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpARM64SUB {
			break
		}
		if v_1_0_1.Type != t {
			break
		}
		_ = v_1_0_1.Args[1]
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_0_1_0.AuxInt != 64 {
			break
		}
		v_1_0_1_1 := v_1_0_1.Args[1]
		if v_1_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_0_1_1.Type != t {
			break
		}
		if v_1_0_1_1.AuxInt != 63 {
			break
		}
		if y != v_1_0_1_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64CMPconst {
			break
		}
		if v_1_1.AuxInt != 64 {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpARM64SUB {
			break
		}
		if v_1_1_0.Type != t {
			break
		}
		_ = v_1_1_0.Args[1]
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_1_0_0.AuxInt != 64 {
			break
		}
		v_1_1_0_1 := v_1_1_0.Args[1]
		if v_1_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1_0_1.Type != t {
			break
		}
		if v_1_1_0_1.AuxInt != 63 {
			break
		}
		if y != v_1_1_0_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR (CSEL0 <typ.UInt64> {cc} (SLL x (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y))) (CMPconst [64] (SUB <t> (MOVDconst [64]) (ANDconst <t> [63] y)))) (SRL <typ.UInt64> x (ANDconst <t> [63] y)))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (ROR x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64CSEL0 {
			break
		}
		if v_0.Type != typ.UInt64 {
			break
		}
		cc := v_0.Aux
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64SLL {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpARM64SUB {
			break
		}
		t := v_0_0_1.Type
		_ = v_0_0_1.Args[1]
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_0_1_0.AuxInt != 64 {
			break
		}
		v_0_0_1_1 := v_0_0_1.Args[1]
		if v_0_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_0_1_1.Type != t {
			break
		}
		if v_0_0_1_1.AuxInt != 63 {
			break
		}
		y := v_0_0_1_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64CMPconst {
			break
		}
		if v_0_1.AuxInt != 64 {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpARM64SUB {
			break
		}
		if v_0_1_0.Type != t {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_1_0_0.AuxInt != 64 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_1_0_1.Type != t {
			break
		}
		if v_0_1_0_1.AuxInt != 63 {
			break
		}
		if y != v_0_1_0_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRL {
			break
		}
		if v_1.Type != typ.UInt64 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1.Type != t {
			break
		}
		if v_1_1.AuxInt != 63 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR (SLL x (ANDconst <t> [31] y)) (CSEL0 <typ.UInt32> {cc} (SRL <typ.UInt32> (MOVWUreg x) (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (RORW x (NEG <t> y))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64ANDconst {
			break
		}
		t := v_0_1.Type
		if v_0_1.AuxInt != 31 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CSEL0 {
			break
		}
		if v_1.Type != typ.UInt32 {
			break
		}
		cc := v_1.Aux
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64SRL {
			break
		}
		if v_1_0.Type != typ.UInt32 {
			break
		}
		_ = v_1_0.Args[1]
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpARM64MOVWUreg {
			break
		}
		if x != v_1_0_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpARM64SUB {
			break
		}
		if v_1_0_1.Type != t {
			break
		}
		_ = v_1_0_1.Args[1]
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_0_1_0.AuxInt != 32 {
			break
		}
		v_1_0_1_1 := v_1_0_1.Args[1]
		if v_1_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_0_1_1.Type != t {
			break
		}
		if v_1_0_1_1.AuxInt != 31 {
			break
		}
		if y != v_1_0_1_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64CMPconst {
			break
		}
		if v_1_1.AuxInt != 64 {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpARM64SUB {
			break
		}
		if v_1_1_0.Type != t {
			break
		}
		_ = v_1_1_0.Args[1]
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_1_0_0.AuxInt != 32 {
			break
		}
		v_1_1_0_1 := v_1_1_0.Args[1]
		if v_1_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1_0_1.Type != t {
			break
		}
		if v_1_1_0_1.AuxInt != 31 {
			break
		}
		if y != v_1_1_0_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (XOR (CSEL0 <typ.UInt32> {cc} (SRL <typ.UInt32> (MOVWUreg x) (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))) (SLL x (ANDconst <t> [31] y)))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (RORW x (NEG <t> y))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64CSEL0 {
			break
		}
		if v_0.Type != typ.UInt32 {
			break
		}
		cc := v_0.Aux
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64SRL {
			break
		}
		if v_0_0.Type != typ.UInt32 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpARM64SUB {
			break
		}
		t := v_0_0_1.Type
		_ = v_0_0_1.Args[1]
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_0_1_0.AuxInt != 32 {
			break
		}
		v_0_0_1_1 := v_0_0_1.Args[1]
		if v_0_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_0_1_1.Type != t {
			break
		}
		if v_0_0_1_1.AuxInt != 31 {
			break
		}
		y := v_0_0_1_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64CMPconst {
			break
		}
		if v_0_1.AuxInt != 64 {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpARM64SUB {
			break
		}
		if v_0_1_0.Type != t {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_1_0_0.AuxInt != 32 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_1_0_1.Type != t {
			break
		}
		if v_0_1_0_1.AuxInt != 31 {
			break
		}
		if y != v_0_1_0_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1.Type != t {
			break
		}
		if v_1_1.AuxInt != 31 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (XOR (SRL <typ.UInt32> (MOVWUreg x) (ANDconst <t> [31] y)) (CSEL0 <typ.UInt32> {cc} (SLL x (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (RORW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRL {
			break
		}
		if v_0.Type != typ.UInt32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64ANDconst {
			break
		}
		t := v_0_1.Type
		if v_0_1.AuxInt != 31 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64CSEL0 {
			break
		}
		if v_1.Type != typ.UInt32 {
			break
		}
		cc := v_1.Aux
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64SLL {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpARM64SUB {
			break
		}
		if v_1_0_1.Type != t {
			break
		}
		_ = v_1_0_1.Args[1]
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_0_1_0.AuxInt != 32 {
			break
		}
		v_1_0_1_1 := v_1_0_1.Args[1]
		if v_1_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_0_1_1.Type != t {
			break
		}
		if v_1_0_1_1.AuxInt != 31 {
			break
		}
		if y != v_1_0_1_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64CMPconst {
			break
		}
		if v_1_1.AuxInt != 64 {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpARM64SUB {
			break
		}
		if v_1_1_0.Type != t {
			break
		}
		_ = v_1_1_0.Args[1]
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_1_1_0_0.AuxInt != 32 {
			break
		}
		v_1_1_0_1 := v_1_1_0.Args[1]
		if v_1_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1_0_1.Type != t {
			break
		}
		if v_1_1_0_1.AuxInt != 31 {
			break
		}
		if y != v_1_1_0_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR (CSEL0 <typ.UInt32> {cc} (SLL x (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y))) (CMPconst [64] (SUB <t> (MOVDconst [32]) (ANDconst <t> [31] y)))) (SRL <typ.UInt32> (MOVWUreg x) (ANDconst <t> [31] y)))
	// cond: cc.(Op) == OpARM64LessThanU
	// result: (RORW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64CSEL0 {
			break
		}
		if v_0.Type != typ.UInt32 {
			break
		}
		cc := v_0.Aux
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64SLL {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpARM64SUB {
			break
		}
		t := v_0_0_1.Type
		_ = v_0_0_1.Args[1]
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_0_1_0.AuxInt != 32 {
			break
		}
		v_0_0_1_1 := v_0_0_1.Args[1]
		if v_0_0_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_0_1_1.Type != t {
			break
		}
		if v_0_0_1_1.AuxInt != 31 {
			break
		}
		y := v_0_0_1_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARM64CMPconst {
			break
		}
		if v_0_1.AuxInt != 64 {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpARM64SUB {
			break
		}
		if v_0_1_0.Type != t {
			break
		}
		_ = v_0_1_0.Args[1]
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0_1_0_0.AuxInt != 32 {
			break
		}
		v_0_1_0_1 := v_0_1_0.Args[1]
		if v_0_1_0_1.Op != OpARM64ANDconst {
			break
		}
		if v_0_1_0_1.Type != t {
			break
		}
		if v_0_1_0_1.AuxInt != 31 {
			break
		}
		if y != v_0_1_0_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRL {
			break
		}
		if v_1.Type != typ.UInt32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpARM64MOVWUreg {
			break
		}
		if x != v_1_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpARM64ANDconst {
			break
		}
		if v_1_1.Type != t {
			break
		}
		if v_1_1.AuxInt != 31 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(cc.(Op) == OpARM64LessThanU) {
			break
		}
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64XORconst_0(v *Value) bool {
	// match: (XORconst [0] x)
	// cond:
	// result: x
	for {
		if v.AuxInt != 0 {
			break
		}
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (XORconst [-1] x)
	// cond:
	// result: (MVN x)
	for {
		if v.AuxInt != -1 {
			break
		}
		x := v.Args[0]
		v.reset(OpARM64MVN)
		v.AddArg(x)
		return true
	}
	// match: (XORconst [c] (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [c^d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = c ^ d
		return true
	}
	// match: (XORconst [c] (XORconst [d] x))
	// cond:
	// result: (XORconst [c^d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64XORconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARM64XORconst)
		v.AuxInt = c ^ d
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64XORshiftLL_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (XORshiftLL (MOVDconst [c]) x [d])
	// cond:
	// result: (XORconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64XORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftLL x (MOVDconst [c]) [d])
	// cond:
	// result: (XORconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64XORconst)
		v.AuxInt = int64(uint64(c) << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (XORshiftLL [c] (SRLconst x [64-c]) x)
	// cond:
	// result: (RORconst [64-c] x)
	for {
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = 64 - c
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL <t> [c] (UBFX [bfc] x) x)
	// cond: c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)
	// result: (RORWconst [32-c] x)
	for {
		t := v.Type
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := v_0.AuxInt
		if x != v_0.Args[0] {
			break
		}
		if !(c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = 32 - c
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL <typ.UInt16> [8] (UBFX <typ.UInt16> [armBFAuxInt(8, 8)] x) x)
	// cond:
	// result: (REV16W x)
	for {
		if v.Type != typ.UInt16 {
			break
		}
		if v.AuxInt != 8 {
			break
		}
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFX {
			break
		}
		if v_0.Type != typ.UInt16 {
			break
		}
		if v_0.AuxInt != armBFAuxInt(8, 8) {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARM64REV16W)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL [c] (SRLconst x [64-c]) x2)
	// cond:
	// result: (EXTRconst [64-c] x2 x)
	for {
		c := v.AuxInt
		x2 := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARM64EXTRconst)
		v.AuxInt = 64 - c
		v.AddArg(x2)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL <t> [c] (UBFX [bfc] x) x2)
	// cond: c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)
	// result: (EXTRWconst [32-c] x2 x)
	for {
		t := v.Type
		c := v.AuxInt
		x2 := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64UBFX {
			break
		}
		bfc := v_0.AuxInt
		x := v_0.Args[0]
		if !(c < 32 && t.Size() == 4 && bfc == armBFAuxInt(32-c, c)) {
			break
		}
		v.reset(OpARM64EXTRWconst)
		v.AuxInt = 32 - c
		v.AddArg(x2)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64XORshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (XORshiftRA (MOVDconst [c]) x [d])
	// cond:
	// result: (XORconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64XORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRA x (MOVDconst [c]) [d])
	// cond:
	// result: (XORconst x [c>>uint64(d)])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64XORconst)
		v.AuxInt = c >> uint64(d)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64XORshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (XORshiftRL (MOVDconst [c]) x [d])
	// cond:
	// result: (XORconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARM64XORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRL x (MOVDconst [c]) [d])
	// cond:
	// result: (XORconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64XORconst)
		v.AuxInt = int64(uint64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVDconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (XORshiftRL [c] (SLLconst x [64-c]) x)
	// cond:
	// result: (RORconst [ c] x)
	for {
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRL <t> [c] (SLLconst x [32-c]) (MOVWUreg x))
	// cond: c < 32 && t.Size() == 4
	// result: (RORWconst [c] x)
	for {
		t := v.Type
		c := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 32-c {
			break
		}
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWUreg {
			break
		}
		if x != v_1.Args[0] {
			break
		}
		if !(c < 32 && t.Size() == 4) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpAbs_0(v *Value) bool {
	// match: (Abs x)
	// cond:
	// result: (FABSD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FABSD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpAdd16_0(v *Value) bool {
	// match: (Add16 x y)
	// cond:
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpAdd32_0(v *Value) bool {
	// match: (Add32 x y)
	// cond:
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpAdd32F_0(v *Value) bool {
	// match: (Add32F x y)
	// cond:
	// result: (FADDS x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64FADDS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpAdd64_0(v *Value) bool {
	// match: (Add64 x y)
	// cond:
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpAdd64F_0(v *Value) bool {
	// match: (Add64F x y)
	// cond:
	// result: (FADDD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64FADDD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpAdd8_0(v *Value) bool {
	// match: (Add8 x y)
	// cond:
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpAddPtr_0(v *Value) bool {
	// match: (AddPtr x y)
	// cond:
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpAddr_0(v *Value) bool {
	// match: (Addr {sym} base)
	// cond:
	// result: (MOVDaddr {sym} base)
	for {
		sym := v.Aux
		base := v.Args[0]
		v.reset(OpARM64MOVDaddr)
		v.Aux = sym
		v.AddArg(base)
		return true
	}
}
func rewriteValueARM64_OpAnd16_0(v *Value) bool {
	// match: (And16 x y)
	// cond:
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpAnd32_0(v *Value) bool {
	// match: (And32 x y)
	// cond:
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpAnd64_0(v *Value) bool {
	// match: (And64 x y)
	// cond:
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpAnd8_0(v *Value) bool {
	// match: (And8 x y)
	// cond:
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpAndB_0(v *Value) bool {
	// match: (AndB x y)
	// cond:
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpAtomicAdd32_0(v *Value) bool {
	// match: (AtomicAdd32 ptr val mem)
	// cond:
	// result: (LoweredAtomicAdd32 ptr val mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		v.reset(OpARM64LoweredAtomicAdd32)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicAdd32Variant_0(v *Value) bool {
	// match: (AtomicAdd32Variant ptr val mem)
	// cond:
	// result: (LoweredAtomicAdd32Variant ptr val mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		v.reset(OpARM64LoweredAtomicAdd32Variant)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicAdd64_0(v *Value) bool {
	// match: (AtomicAdd64 ptr val mem)
	// cond:
	// result: (LoweredAtomicAdd64 ptr val mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		v.reset(OpARM64LoweredAtomicAdd64)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicAdd64Variant_0(v *Value) bool {
	// match: (AtomicAdd64Variant ptr val mem)
	// cond:
	// result: (LoweredAtomicAdd64Variant ptr val mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		v.reset(OpARM64LoweredAtomicAdd64Variant)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicAnd8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicAnd8 ptr val mem)
	// cond:
	// result: (Select1 (LoweredAtomicAnd8 ptr val mem))
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpARM64LoweredAtomicAnd8, types.NewTuple(typ.UInt8, types.TypeMem))
		v0.AddArg(ptr)
		v0.AddArg(val)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpAtomicCompareAndSwap32_0(v *Value) bool {
	// match: (AtomicCompareAndSwap32 ptr old new_ mem)
	// cond:
	// result: (LoweredAtomicCas32 ptr old new_ mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		old := v.Args[1]
		new_ := v.Args[2]
		v.reset(OpARM64LoweredAtomicCas32)
		v.AddArg(ptr)
		v.AddArg(old)
		v.AddArg(new_)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicCompareAndSwap64_0(v *Value) bool {
	// match: (AtomicCompareAndSwap64 ptr old new_ mem)
	// cond:
	// result: (LoweredAtomicCas64 ptr old new_ mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		old := v.Args[1]
		new_ := v.Args[2]
		v.reset(OpARM64LoweredAtomicCas64)
		v.AddArg(ptr)
		v.AddArg(old)
		v.AddArg(new_)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicExchange32_0(v *Value) bool {
	// match: (AtomicExchange32 ptr val mem)
	// cond:
	// result: (LoweredAtomicExchange32 ptr val mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		v.reset(OpARM64LoweredAtomicExchange32)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicExchange64_0(v *Value) bool {
	// match: (AtomicExchange64 ptr val mem)
	// cond:
	// result: (LoweredAtomicExchange64 ptr val mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		v.reset(OpARM64LoweredAtomicExchange64)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicLoad32_0(v *Value) bool {
	// match: (AtomicLoad32 ptr mem)
	// cond:
	// result: (LDARW ptr mem)
	for {
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64LDARW)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicLoad64_0(v *Value) bool {
	// match: (AtomicLoad64 ptr mem)
	// cond:
	// result: (LDAR ptr mem)
	for {
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64LDAR)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicLoad8_0(v *Value) bool {
	// match: (AtomicLoad8 ptr mem)
	// cond:
	// result: (LDARB ptr mem)
	for {
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64LDARB)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicLoadPtr_0(v *Value) bool {
	// match: (AtomicLoadPtr ptr mem)
	// cond:
	// result: (LDAR ptr mem)
	for {
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64LDAR)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicOr8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (AtomicOr8 ptr val mem)
	// cond:
	// result: (Select1 (LoweredAtomicOr8 ptr val mem))
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpARM64LoweredAtomicOr8, types.NewTuple(typ.UInt8, types.TypeMem))
		v0.AddArg(ptr)
		v0.AddArg(val)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpAtomicStore32_0(v *Value) bool {
	// match: (AtomicStore32 ptr val mem)
	// cond:
	// result: (STLRW ptr val mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		v.reset(OpARM64STLRW)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicStore64_0(v *Value) bool {
	// match: (AtomicStore64 ptr val mem)
	// cond:
	// result: (STLR ptr val mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		v.reset(OpARM64STLR)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicStorePtrNoWB_0(v *Value) bool {
	// match: (AtomicStorePtrNoWB ptr val mem)
	// cond:
	// result: (STLR ptr val mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		v.reset(OpARM64STLR)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAvg64u_0(v *Value) bool {
	b := v.Block
	// match: (Avg64u <t> x y)
	// cond:
	// result: (ADD (SRLconst <t> (SUB <t> x y) [1]) y)
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64ADD)
		v0 := b.NewValue0(v.Pos, OpARM64SRLconst, t)
		v0.AuxInt = 1
		v1 := b.NewValue0(v.Pos, OpARM64SUB, t)
		v1.AddArg(x)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpBitLen32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen32 x)
	// cond:
	// result: (SUB (MOVDconst [32]) (CLZW <typ.Int> x))
	for {
		x := v.Args[0]
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 32
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CLZW, typ.Int)
		v1.AddArg(x)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpBitLen64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen64 x)
	// cond:
	// result: (SUB (MOVDconst [64]) (CLZ <typ.Int> x))
	for {
		x := v.Args[0]
		v.reset(OpARM64SUB)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 64
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CLZ, typ.Int)
		v1.AddArg(x)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpBitRev16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitRev16 x)
	// cond:
	// result: (SRLconst [48] (RBIT <typ.UInt64> x))
	for {
		x := v.Args[0]
		v.reset(OpARM64SRLconst)
		v.AuxInt = 48
		v0 := b.NewValue0(v.Pos, OpARM64RBIT, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpBitRev32_0(v *Value) bool {
	// match: (BitRev32 x)
	// cond:
	// result: (RBITW x)
	for {
		x := v.Args[0]
		v.reset(OpARM64RBITW)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpBitRev64_0(v *Value) bool {
	// match: (BitRev64 x)
	// cond:
	// result: (RBIT x)
	for {
		x := v.Args[0]
		v.reset(OpARM64RBIT)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpBitRev8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitRev8 x)
	// cond:
	// result: (SRLconst [56] (RBIT <typ.UInt64> x))
	for {
		x := v.Args[0]
		v.reset(OpARM64SRLconst)
		v.AuxInt = 56
		v0 := b.NewValue0(v.Pos, OpARM64RBIT, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpBswap32_0(v *Value) bool {
	// match: (Bswap32 x)
	// cond:
	// result: (REVW x)
	for {
		x := v.Args[0]
		v.reset(OpARM64REVW)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpBswap64_0(v *Value) bool {
	// match: (Bswap64 x)
	// cond:
	// result: (REV x)
	for {
		x := v.Args[0]
		v.reset(OpARM64REV)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCeil_0(v *Value) bool {
	// match: (Ceil x)
	// cond:
	// result: (FRINTPD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FRINTPD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpClosureCall_0(v *Value) bool {
	// match: (ClosureCall [argwid] entry closure mem)
	// cond:
	// result: (CALLclosure [argwid] entry closure mem)
	for {
		argwid := v.AuxInt
		mem := v.Args[2]
		entry := v.Args[0]
		closure := v.Args[1]
		v.reset(OpARM64CALLclosure)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(closure)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpCom16_0(v *Value) bool {
	// match: (Com16 x)
	// cond:
	// result: (MVN x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MVN)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCom32_0(v *Value) bool {
	// match: (Com32 x)
	// cond:
	// result: (MVN x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MVN)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCom64_0(v *Value) bool {
	// match: (Com64 x)
	// cond:
	// result: (MVN x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MVN)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCom8_0(v *Value) bool {
	// match: (Com8 x)
	// cond:
	// result: (MVN x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MVN)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCondSelect_0(v *Value) bool {
	b := v.Block
	// match: (CondSelect x y boolval)
	// cond: flagArg(boolval) != nil
	// result: (CSEL {boolval.Op} x y flagArg(boolval))
	for {
		boolval := v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		if !(flagArg(boolval) != nil) {
			break
		}
		v.reset(OpARM64CSEL)
		v.Aux = boolval.Op
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flagArg(boolval))
		return true
	}
	// match: (CondSelect x y boolval)
	// cond: flagArg(boolval) == nil
	// result: (CSEL {OpARM64NotEqual} x y (CMPWconst [0] boolval))
	for {
		boolval := v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		if !(flagArg(boolval) == nil) {
			break
		}
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64NotEqual
		v.AddArg(x)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpARM64CMPWconst, types.TypeFlags)
		v0.AuxInt = 0
		v0.AddArg(boolval)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpConst16_0(v *Value) bool {
	// match: (Const16 [val])
	// cond:
	// result: (MOVDconst [val])
	for {
		val := v.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueARM64_OpConst32_0(v *Value) bool {
	// match: (Const32 [val])
	// cond:
	// result: (MOVDconst [val])
	for {
		val := v.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueARM64_OpConst32F_0(v *Value) bool {
	// match: (Const32F [val])
	// cond:
	// result: (FMOVSconst [val])
	for {
		val := v.AuxInt
		v.reset(OpARM64FMOVSconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueARM64_OpConst64_0(v *Value) bool {
	// match: (Const64 [val])
	// cond:
	// result: (MOVDconst [val])
	for {
		val := v.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueARM64_OpConst64F_0(v *Value) bool {
	// match: (Const64F [val])
	// cond:
	// result: (FMOVDconst [val])
	for {
		val := v.AuxInt
		v.reset(OpARM64FMOVDconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueARM64_OpConst8_0(v *Value) bool {
	// match: (Const8 [val])
	// cond:
	// result: (MOVDconst [val])
	for {
		val := v.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueARM64_OpConstBool_0(v *Value) bool {
	// match: (ConstBool [b])
	// cond:
	// result: (MOVDconst [b])
	for {
		b := v.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = b
		return true
	}
}
func rewriteValueARM64_OpConstNil_0(v *Value) bool {
	// match: (ConstNil)
	// cond:
	// result: (MOVDconst [0])
	for {
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
}
func rewriteValueARM64_OpCtz16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 <t> x)
	// cond:
	// result: (CLZW <t> (RBITW <typ.UInt32> (ORconst <typ.UInt32> [0x10000] x)))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpARM64CLZW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARM64RBITW, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpARM64ORconst, typ.UInt32)
		v1.AuxInt = 0x10000
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpCtz16NonZero_0(v *Value) bool {
	// match: (Ctz16NonZero x)
	// cond:
	// result: (Ctz32 x)
	for {
		x := v.Args[0]
		v.reset(OpCtz32)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCtz32_0(v *Value) bool {
	b := v.Block
	// match: (Ctz32 <t> x)
	// cond:
	// result: (CLZW (RBITW <t> x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpARM64CLZW)
		v0 := b.NewValue0(v.Pos, OpARM64RBITW, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpCtz32NonZero_0(v *Value) bool {
	// match: (Ctz32NonZero x)
	// cond:
	// result: (Ctz32 x)
	for {
		x := v.Args[0]
		v.reset(OpCtz32)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCtz64_0(v *Value) bool {
	b := v.Block
	// match: (Ctz64 <t> x)
	// cond:
	// result: (CLZ (RBIT <t> x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpARM64CLZ)
		v0 := b.NewValue0(v.Pos, OpARM64RBIT, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpCtz64NonZero_0(v *Value) bool {
	// match: (Ctz64NonZero x)
	// cond:
	// result: (Ctz64 x)
	for {
		x := v.Args[0]
		v.reset(OpCtz64)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCtz8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 <t> x)
	// cond:
	// result: (CLZW <t> (RBITW <typ.UInt32> (ORconst <typ.UInt32> [0x100] x)))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpARM64CLZW)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARM64RBITW, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpARM64ORconst, typ.UInt32)
		v1.AuxInt = 0x100
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpCtz8NonZero_0(v *Value) bool {
	// match: (Ctz8NonZero x)
	// cond:
	// result: (Ctz32 x)
	for {
		x := v.Args[0]
		v.reset(OpCtz32)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt32Fto32_0(v *Value) bool {
	// match: (Cvt32Fto32 x)
	// cond:
	// result: (FCVTZSSW x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FCVTZSSW)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt32Fto32U_0(v *Value) bool {
	// match: (Cvt32Fto32U x)
	// cond:
	// result: (FCVTZUSW x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FCVTZUSW)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt32Fto64_0(v *Value) bool {
	// match: (Cvt32Fto64 x)
	// cond:
	// result: (FCVTZSS x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FCVTZSS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt32Fto64F_0(v *Value) bool {
	// match: (Cvt32Fto64F x)
	// cond:
	// result: (FCVTSD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FCVTSD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt32Fto64U_0(v *Value) bool {
	// match: (Cvt32Fto64U x)
	// cond:
	// result: (FCVTZUS x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FCVTZUS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt32Uto32F_0(v *Value) bool {
	// match: (Cvt32Uto32F x)
	// cond:
	// result: (UCVTFWS x)
	for {
		x := v.Args[0]
		v.reset(OpARM64UCVTFWS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt32Uto64F_0(v *Value) bool {
	// match: (Cvt32Uto64F x)
	// cond:
	// result: (UCVTFWD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64UCVTFWD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt32to32F_0(v *Value) bool {
	// match: (Cvt32to32F x)
	// cond:
	// result: (SCVTFWS x)
	for {
		x := v.Args[0]
		v.reset(OpARM64SCVTFWS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt32to64F_0(v *Value) bool {
	// match: (Cvt32to64F x)
	// cond:
	// result: (SCVTFWD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64SCVTFWD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt64Fto32_0(v *Value) bool {
	// match: (Cvt64Fto32 x)
	// cond:
	// result: (FCVTZSDW x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FCVTZSDW)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt64Fto32F_0(v *Value) bool {
	// match: (Cvt64Fto32F x)
	// cond:
	// result: (FCVTDS x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FCVTDS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt64Fto32U_0(v *Value) bool {
	// match: (Cvt64Fto32U x)
	// cond:
	// result: (FCVTZUDW x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FCVTZUDW)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt64Fto64_0(v *Value) bool {
	// match: (Cvt64Fto64 x)
	// cond:
	// result: (FCVTZSD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FCVTZSD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt64Fto64U_0(v *Value) bool {
	// match: (Cvt64Fto64U x)
	// cond:
	// result: (FCVTZUD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FCVTZUD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt64Uto32F_0(v *Value) bool {
	// match: (Cvt64Uto32F x)
	// cond:
	// result: (UCVTFS x)
	for {
		x := v.Args[0]
		v.reset(OpARM64UCVTFS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt64Uto64F_0(v *Value) bool {
	// match: (Cvt64Uto64F x)
	// cond:
	// result: (UCVTFD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64UCVTFD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt64to32F_0(v *Value) bool {
	// match: (Cvt64to32F x)
	// cond:
	// result: (SCVTFS x)
	for {
		x := v.Args[0]
		v.reset(OpARM64SCVTFS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpCvt64to64F_0(v *Value) bool {
	// match: (Cvt64to64F x)
	// cond:
	// result: (SCVTFD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64SCVTFD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpDiv16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 x y)
	// cond:
	// result: (DIVW (SignExt16to32 x) (SignExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64DIVW)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpDiv16u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// cond:
	// result: (UDIVW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64UDIVW)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpDiv32_0(v *Value) bool {
	// match: (Div32 x y)
	// cond:
	// result: (DIVW x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64DIVW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpDiv32F_0(v *Value) bool {
	// match: (Div32F x y)
	// cond:
	// result: (FDIVS x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64FDIVS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpDiv32u_0(v *Value) bool {
	// match: (Div32u x y)
	// cond:
	// result: (UDIVW x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64UDIVW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpDiv64_0(v *Value) bool {
	// match: (Div64 x y)
	// cond:
	// result: (DIV x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64DIV)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpDiv64F_0(v *Value) bool {
	// match: (Div64F x y)
	// cond:
	// result: (FDIVD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64FDIVD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpDiv64u_0(v *Value) bool {
	// match: (Div64u x y)
	// cond:
	// result: (UDIV x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64UDIV)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpDiv8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// cond:
	// result: (DIVW (SignExt8to32 x) (SignExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64DIVW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpDiv8u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// cond:
	// result: (UDIVW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64UDIVW)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpEq16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// cond:
	// result: (Equal (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpEq32_0(v *Value) bool {
	b := v.Block
	// match: (Eq32 x y)
	// cond:
	// result: (Equal (CMPW x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpEq32F_0(v *Value) bool {
	b := v.Block
	// match: (Eq32F x y)
	// cond:
	// result: (Equal (FCMPS x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpEq64_0(v *Value) bool {
	b := v.Block
	// match: (Eq64 x y)
	// cond:
	// result: (Equal (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpEq64F_0(v *Value) bool {
	b := v.Block
	// match: (Eq64F x y)
	// cond:
	// result: (Equal (FCMPD x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpEq8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// cond:
	// result: (Equal (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpEqB_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// cond:
	// result: (XOR (MOVDconst [1]) (XOR <typ.Bool> x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64XOR)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 1
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64XOR, typ.Bool)
		v1.AddArg(x)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpEqPtr_0(v *Value) bool {
	b := v.Block
	// match: (EqPtr x y)
	// cond:
	// result: (Equal (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpFloor_0(v *Value) bool {
	// match: (Floor x)
	// cond:
	// result: (FRINTMD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FRINTMD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpGeq16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq16 x y)
	// cond:
	// result: (GreaterEqual (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGeq16U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq16U x y)
	// cond:
	// result: (GreaterEqualU (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGeq32_0(v *Value) bool {
	b := v.Block
	// match: (Geq32 x y)
	// cond:
	// result: (GreaterEqual (CMPW x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGeq32F_0(v *Value) bool {
	b := v.Block
	// match: (Geq32F x y)
	// cond:
	// result: (GreaterEqualF (FCMPS x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterEqualF)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGeq32U_0(v *Value) bool {
	b := v.Block
	// match: (Geq32U x y)
	// cond:
	// result: (GreaterEqualU (CMPW x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGeq64_0(v *Value) bool {
	b := v.Block
	// match: (Geq64 x y)
	// cond:
	// result: (GreaterEqual (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGeq64F_0(v *Value) bool {
	b := v.Block
	// match: (Geq64F x y)
	// cond:
	// result: (GreaterEqualF (FCMPD x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterEqualF)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGeq64U_0(v *Value) bool {
	b := v.Block
	// match: (Geq64U x y)
	// cond:
	// result: (GreaterEqualU (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGeq8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq8 x y)
	// cond:
	// result: (GreaterEqual (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGeq8U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq8U x y)
	// cond:
	// result: (GreaterEqualU (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGetCallerPC_0(v *Value) bool {
	// match: (GetCallerPC)
	// cond:
	// result: (LoweredGetCallerPC)
	for {
		v.reset(OpARM64LoweredGetCallerPC)
		return true
	}
}
func rewriteValueARM64_OpGetCallerSP_0(v *Value) bool {
	// match: (GetCallerSP)
	// cond:
	// result: (LoweredGetCallerSP)
	for {
		v.reset(OpARM64LoweredGetCallerSP)
		return true
	}
}
func rewriteValueARM64_OpGetClosurePtr_0(v *Value) bool {
	// match: (GetClosurePtr)
	// cond:
	// result: (LoweredGetClosurePtr)
	for {
		v.reset(OpARM64LoweredGetClosurePtr)
		return true
	}
}
func rewriteValueARM64_OpGreater16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater16 x y)
	// cond:
	// result: (GreaterThan (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGreater16U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater16U x y)
	// cond:
	// result: (GreaterThanU (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGreater32_0(v *Value) bool {
	b := v.Block
	// match: (Greater32 x y)
	// cond:
	// result: (GreaterThan (CMPW x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGreater32F_0(v *Value) bool {
	b := v.Block
	// match: (Greater32F x y)
	// cond:
	// result: (GreaterThanF (FCMPS x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterThanF)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGreater32U_0(v *Value) bool {
	b := v.Block
	// match: (Greater32U x y)
	// cond:
	// result: (GreaterThanU (CMPW x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGreater64_0(v *Value) bool {
	b := v.Block
	// match: (Greater64 x y)
	// cond:
	// result: (GreaterThan (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGreater64F_0(v *Value) bool {
	b := v.Block
	// match: (Greater64F x y)
	// cond:
	// result: (GreaterThanF (FCMPD x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterThanF)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGreater64U_0(v *Value) bool {
	b := v.Block
	// match: (Greater64U x y)
	// cond:
	// result: (GreaterThanU (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGreater8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater8 x y)
	// cond:
	// result: (GreaterThan (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGreater8U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater8U x y)
	// cond:
	// result: (GreaterThanU (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64GreaterThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpHmul32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32 x y)
	// cond:
	// result: (SRAconst (MULL <typ.Int64> x y) [32])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRAconst)
		v.AuxInt = 32
		v0 := b.NewValue0(v.Pos, OpARM64MULL, typ.Int64)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpHmul32u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32u x y)
	// cond:
	// result: (SRAconst (UMULL <typ.UInt64> x y) [32])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRAconst)
		v.AuxInt = 32
		v0 := b.NewValue0(v.Pos, OpARM64UMULL, typ.UInt64)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpHmul64_0(v *Value) bool {
	// match: (Hmul64 x y)
	// cond:
	// result: (MULH x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64MULH)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpHmul64u_0(v *Value) bool {
	// match: (Hmul64u x y)
	// cond:
	// result: (UMULH x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64UMULH)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpInterCall_0(v *Value) bool {
	// match: (InterCall [argwid] entry mem)
	// cond:
	// result: (CALLinter [argwid] entry mem)
	for {
		argwid := v.AuxInt
		mem := v.Args[1]
		entry := v.Args[0]
		v.reset(OpARM64CALLinter)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpIsInBounds_0(v *Value) bool {
	b := v.Block
	// match: (IsInBounds idx len)
	// cond:
	// result: (LessThanU (CMP idx len))
	for {
		len := v.Args[1]
		idx := v.Args[0]
		v.reset(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(idx)
		v0.AddArg(len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpIsNonNil_0(v *Value) bool {
	b := v.Block
	// match: (IsNonNil ptr)
	// cond:
	// result: (NotEqual (CMPconst [0] ptr))
	for {
		ptr := v.Args[0]
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v0.AuxInt = 0
		v0.AddArg(ptr)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpIsSliceInBounds_0(v *Value) bool {
	b := v.Block
	// match: (IsSliceInBounds idx len)
	// cond:
	// result: (LessEqualU (CMP idx len))
	for {
		len := v.Args[1]
		idx := v.Args[0]
		v.reset(OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(idx)
		v0.AddArg(len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// cond:
	// result: (LessEqual (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq16U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// cond:
	// result: (LessEqualU (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq32_0(v *Value) bool {
	b := v.Block
	// match: (Leq32 x y)
	// cond:
	// result: (LessEqual (CMPW x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq32F_0(v *Value) bool {
	b := v.Block
	// match: (Leq32F x y)
	// cond:
	// result: (LessEqualF (FCMPS x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessEqualF)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq32U_0(v *Value) bool {
	b := v.Block
	// match: (Leq32U x y)
	// cond:
	// result: (LessEqualU (CMPW x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq64_0(v *Value) bool {
	b := v.Block
	// match: (Leq64 x y)
	// cond:
	// result: (LessEqual (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq64F_0(v *Value) bool {
	b := v.Block
	// match: (Leq64F x y)
	// cond:
	// result: (LessEqualF (FCMPD x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessEqualF)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq64U_0(v *Value) bool {
	b := v.Block
	// match: (Leq64U x y)
	// cond:
	// result: (LessEqualU (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// cond:
	// result: (LessEqual (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq8U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// cond:
	// result: (LessEqualU (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessEqualU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// cond:
	// result: (LessThan (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess16U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// cond:
	// result: (LessThanU (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess32_0(v *Value) bool {
	b := v.Block
	// match: (Less32 x y)
	// cond:
	// result: (LessThan (CMPW x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess32F_0(v *Value) bool {
	b := v.Block
	// match: (Less32F x y)
	// cond:
	// result: (LessThanF (FCMPS x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessThanF)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess32U_0(v *Value) bool {
	b := v.Block
	// match: (Less32U x y)
	// cond:
	// result: (LessThanU (CMPW x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess64_0(v *Value) bool {
	b := v.Block
	// match: (Less64 x y)
	// cond:
	// result: (LessThan (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess64F_0(v *Value) bool {
	b := v.Block
	// match: (Less64F x y)
	// cond:
	// result: (LessThanF (FCMPD x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessThanF)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess64U_0(v *Value) bool {
	b := v.Block
	// match: (Less64U x y)
	// cond:
	// result: (LessThanU (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// cond:
	// result: (LessThan (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessThan)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess8U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// cond:
	// result: (LessThanU (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LessThanU)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLoad_0(v *Value) bool {
	// match: (Load <t> ptr mem)
	// cond: t.IsBoolean()
	// result: (MOVBUload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(t.IsBoolean()) {
			break
		}
		v.reset(OpARM64MOVBUload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is8BitInt(t) && isSigned(t))
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is8BitInt(t) && isSigned(t)) {
			break
		}
		v.reset(OpARM64MOVBload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is8BitInt(t) && !isSigned(t))
	// result: (MOVBUload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is8BitInt(t) && !isSigned(t)) {
			break
		}
		v.reset(OpARM64MOVBUload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is16BitInt(t) && isSigned(t))
	// result: (MOVHload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is16BitInt(t) && isSigned(t)) {
			break
		}
		v.reset(OpARM64MOVHload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is16BitInt(t) && !isSigned(t))
	// result: (MOVHUload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is16BitInt(t) && !isSigned(t)) {
			break
		}
		v.reset(OpARM64MOVHUload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is32BitInt(t) && isSigned(t))
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is32BitInt(t) && isSigned(t)) {
			break
		}
		v.reset(OpARM64MOVWload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is32BitInt(t) && !isSigned(t))
	// result: (MOVWUload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is32BitInt(t) && !isSigned(t)) {
			break
		}
		v.reset(OpARM64MOVWUload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is64BitInt(t) || isPtr(t))
	// result: (MOVDload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is64BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpARM64MOVDload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is32BitFloat(t)
	// result: (FMOVSload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is32BitFloat(t)) {
			break
		}
		v.reset(OpARM64FMOVSload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is64BitFloat(t)
	// result: (FMOVDload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is64BitFloat(t)) {
			break
		}
		v.reset(OpARM64FMOVDload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpLocalAddr_0(v *Value) bool {
	// match: (LocalAddr {sym} base _)
	// cond:
	// result: (MOVDaddr {sym} base)
	for {
		sym := v.Aux
		_ = v.Args[1]
		base := v.Args[0]
		v.reset(OpARM64MOVDaddr)
		v.Aux = sym
		v.AddArg(base)
		return true
	}
}
func rewriteValueARM64_OpLsh16x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpLsh16x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpLsh16x64_0(v *Value) bool {
	b := v.Block
	// match: (Lsh16x64 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = 0
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM64_OpLsh16x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpLsh32x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpLsh32x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpLsh32x64_0(v *Value) bool {
	b := v.Block
	// match: (Lsh32x64 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = 0
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM64_OpLsh32x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpLsh64x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpLsh64x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpLsh64x64_0(v *Value) bool {
	b := v.Block
	// match: (Lsh64x64 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = 0
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM64_OpLsh64x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpLsh8x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpLsh8x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpLsh8x64_0(v *Value) bool {
	b := v.Block
	// match: (Lsh8x64 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = 0
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM64_OpLsh8x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SLL <t> x (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SLL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpMod16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// cond:
	// result: (MODW (SignExt16to32 x) (SignExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64MODW)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpMod16u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// cond:
	// result: (UMODW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64UMODW)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpMod32_0(v *Value) bool {
	// match: (Mod32 x y)
	// cond:
	// result: (MODW x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64MODW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpMod32u_0(v *Value) bool {
	// match: (Mod32u x y)
	// cond:
	// result: (UMODW x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64UMODW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpMod64_0(v *Value) bool {
	// match: (Mod64 x y)
	// cond:
	// result: (MOD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64MOD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpMod64u_0(v *Value) bool {
	// match: (Mod64u x y)
	// cond:
	// result: (UMOD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64UMOD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpMod8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// cond:
	// result: (MODW (SignExt8to32 x) (SignExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64MODW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpMod8u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// cond:
	// result: (UMODW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64UMODW)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpMove_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Move [0] _ _ mem)
	// cond:
	// result: mem
	for {
		if v.AuxInt != 0 {
			break
		}
		mem := v.Args[2]
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	// match: (Move [1] dst src mem)
	// cond:
	// result: (MOVBstore dst (MOVBUload src mem) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARM64MOVBstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARM64MOVBUload, typ.UInt8)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// cond:
	// result: (MOVHstore dst (MOVHUload src mem) mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARM64MOVHstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARM64MOVHUload, typ.UInt16)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [4] dst src mem)
	// cond:
	// result: (MOVWstore dst (MOVWUload src mem) mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARM64MOVWstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUload, typ.UInt32)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [8] dst src mem)
	// cond:
	// result: (MOVDstore dst (MOVDload src mem) mem)
	for {
		if v.AuxInt != 8 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARM64MOVDstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [3] dst src mem)
	// cond:
	// result: (MOVBstore [2] dst (MOVBUload [2] src mem) (MOVHstore dst (MOVHUload src mem) mem))
	for {
		if v.AuxInt != 3 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = 2
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARM64MOVBUload, typ.UInt8)
		v0.AuxInt = 2
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHstore, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpARM64MOVHUload, typ.UInt16)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [5] dst src mem)
	// cond:
	// result: (MOVBstore [4] dst (MOVBUload [4] src mem) (MOVWstore dst (MOVWUload src mem) mem))
	for {
		if v.AuxInt != 5 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = 4
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARM64MOVBUload, typ.UInt8)
		v0.AuxInt = 4
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpARM64MOVWUload, typ.UInt32)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [6] dst src mem)
	// cond:
	// result: (MOVHstore [4] dst (MOVHUload [4] src mem) (MOVWstore dst (MOVWUload src mem) mem))
	for {
		if v.AuxInt != 6 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARM64MOVHstore)
		v.AuxInt = 4
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARM64MOVHUload, typ.UInt16)
		v0.AuxInt = 4
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpARM64MOVWUload, typ.UInt32)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [7] dst src mem)
	// cond:
	// result: (MOVBstore [6] dst (MOVBUload [6] src mem) (MOVHstore [4] dst (MOVHUload [4] src mem) (MOVWstore dst (MOVWUload src mem) mem)))
	for {
		if v.AuxInt != 7 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = 6
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARM64MOVBUload, typ.UInt8)
		v0.AuxInt = 6
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHstore, types.TypeMem)
		v1.AuxInt = 4
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpARM64MOVHUload, typ.UInt16)
		v2.AuxInt = 4
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v3.AddArg(dst)
		v4 := b.NewValue0(v.Pos, OpARM64MOVWUload, typ.UInt32)
		v4.AddArg(src)
		v4.AddArg(mem)
		v3.AddArg(v4)
		v3.AddArg(mem)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Move [12] dst src mem)
	// cond:
	// result: (MOVWstore [8] dst (MOVWUload [8] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if v.AuxInt != 12 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARM64MOVWstore)
		v.AuxInt = 8
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUload, typ.UInt32)
		v0.AuxInt = 8
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpMove_10(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Move [16] dst src mem)
	// cond:
	// result: (MOVDstore [8] dst (MOVDload [8] src mem) (MOVDstore dst (MOVDload src mem) mem))
	for {
		if v.AuxInt != 16 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARM64MOVDstore)
		v.AuxInt = 8
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = 8
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [24] dst src mem)
	// cond:
	// result: (MOVDstore [16] dst (MOVDload [16] src mem) (MOVDstore [8] dst (MOVDload [8] src mem) (MOVDstore dst (MOVDload src mem) mem)))
	for {
		if v.AuxInt != 24 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARM64MOVDstore)
		v.AuxInt = 16
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = 16
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AuxInt = 8
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v2.AuxInt = 8
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v3.AddArg(dst)
		v4 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v4.AddArg(src)
		v4.AddArg(mem)
		v3.AddArg(v4)
		v3.AddArg(mem)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s%8 != 0 && s > 8
	// result: (Move [s%8] (OffPtr <dst.Type> dst [s-s%8]) (OffPtr <src.Type> src [s-s%8]) (Move [s-s%8] dst src mem))
	for {
		s := v.AuxInt
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		if !(s%8 != 0 && s > 8) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = s % 8
		v0 := b.NewValue0(v.Pos, OpOffPtr, dst.Type)
		v0.AuxInt = s - s%8
		v0.AddArg(dst)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, src.Type)
		v1.AuxInt = s - s%8
		v1.AddArg(src)
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpMove, types.TypeMem)
		v2.AuxInt = s - s%8
		v2.AddArg(dst)
		v2.AddArg(src)
		v2.AddArg(mem)
		v.AddArg(v2)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 32 && s <= 16*64 && s%16 == 8 && !config.noDuffDevice
	// result: (MOVDstore [s-8] dst (MOVDload [s-8] src mem) (DUFFCOPY <types.TypeMem> [8*(64-(s-8)/16)] dst src mem))
	for {
		s := v.AuxInt
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		if !(s > 32 && s <= 16*64 && s%16 == 8 && !config.noDuffDevice) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AuxInt = s - 8
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, typ.UInt64)
		v0.AuxInt = s - 8
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64DUFFCOPY, types.TypeMem)
		v1.AuxInt = 8 * (64 - (s-8)/16)
		v1.AddArg(dst)
		v1.AddArg(src)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 32 && s <= 16*64 && s%16 == 0 && !config.noDuffDevice
	// result: (DUFFCOPY [8 * (64 - s/16)] dst src mem)
	for {
		s := v.AuxInt
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		if !(s > 32 && s <= 16*64 && s%16 == 0 && !config.noDuffDevice) {
			break
		}
		v.reset(OpARM64DUFFCOPY)
		v.AuxInt = 8 * (64 - s/16)
		v.AddArg(dst)
		v.AddArg(src)
		v.AddArg(mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 24 && s%8 == 0
	// result: (LoweredMove dst src (ADDconst <src.Type> src [s-8]) mem)
	for {
		s := v.AuxInt
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		if !(s > 24 && s%8 == 0) {
			break
		}
		v.reset(OpARM64LoweredMove)
		v.AddArg(dst)
		v.AddArg(src)
		v0 := b.NewValue0(v.Pos, OpARM64ADDconst, src.Type)
		v0.AuxInt = s - 8
		v0.AddArg(src)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpMul16_0(v *Value) bool {
	// match: (Mul16 x y)
	// cond:
	// result: (MULW x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64MULW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpMul32_0(v *Value) bool {
	// match: (Mul32 x y)
	// cond:
	// result: (MULW x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64MULW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpMul32F_0(v *Value) bool {
	// match: (Mul32F x y)
	// cond:
	// result: (FMULS x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64FMULS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpMul64_0(v *Value) bool {
	// match: (Mul64 x y)
	// cond:
	// result: (MUL x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64MUL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpMul64F_0(v *Value) bool {
	// match: (Mul64F x y)
	// cond:
	// result: (FMULD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64FMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpMul64uhilo_0(v *Value) bool {
	// match: (Mul64uhilo x y)
	// cond:
	// result: (LoweredMuluhilo x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64LoweredMuluhilo)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpMul8_0(v *Value) bool {
	// match: (Mul8 x y)
	// cond:
	// result: (MULW x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64MULW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpNeg16_0(v *Value) bool {
	// match: (Neg16 x)
	// cond:
	// result: (NEG x)
	for {
		x := v.Args[0]
		v.reset(OpARM64NEG)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpNeg32_0(v *Value) bool {
	// match: (Neg32 x)
	// cond:
	// result: (NEG x)
	for {
		x := v.Args[0]
		v.reset(OpARM64NEG)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpNeg32F_0(v *Value) bool {
	// match: (Neg32F x)
	// cond:
	// result: (FNEGS x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FNEGS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpNeg64_0(v *Value) bool {
	// match: (Neg64 x)
	// cond:
	// result: (NEG x)
	for {
		x := v.Args[0]
		v.reset(OpARM64NEG)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpNeg64F_0(v *Value) bool {
	// match: (Neg64F x)
	// cond:
	// result: (FNEGD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FNEGD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpNeg8_0(v *Value) bool {
	// match: (Neg8 x)
	// cond:
	// result: (NEG x)
	for {
		x := v.Args[0]
		v.reset(OpARM64NEG)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpNeq16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// cond:
	// result: (NotEqual (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNeq32_0(v *Value) bool {
	b := v.Block
	// match: (Neq32 x y)
	// cond:
	// result: (NotEqual (CMPW x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNeq32F_0(v *Value) bool {
	b := v.Block
	// match: (Neq32F x y)
	// cond:
	// result: (NotEqual (FCMPS x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNeq64_0(v *Value) bool {
	b := v.Block
	// match: (Neq64 x y)
	// cond:
	// result: (NotEqual (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNeq64F_0(v *Value) bool {
	b := v.Block
	// match: (Neq64F x y)
	// cond:
	// result: (NotEqual (FCMPD x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNeq8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// cond:
	// result: (NotEqual (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNeqB_0(v *Value) bool {
	// match: (NeqB x y)
	// cond:
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpNeqPtr_0(v *Value) bool {
	b := v.Block
	// match: (NeqPtr x y)
	// cond:
	// result: (NotEqual (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64NotEqual)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpNilCheck_0(v *Value) bool {
	// match: (NilCheck ptr mem)
	// cond:
	// result: (LoweredNilCheck ptr mem)
	for {
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64LoweredNilCheck)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpNot_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Not x)
	// cond:
	// result: (XOR (MOVDconst [1]) x)
	for {
		x := v.Args[0]
		v.reset(OpARM64XOR)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpOffPtr_0(v *Value) bool {
	// match: (OffPtr [off] ptr:(SP))
	// cond:
	// result: (MOVDaddr [off] ptr)
	for {
		off := v.AuxInt
		ptr := v.Args[0]
		if ptr.Op != OpSP {
			break
		}
		v.reset(OpARM64MOVDaddr)
		v.AuxInt = off
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// cond:
	// result: (ADDconst [off] ptr)
	for {
		off := v.AuxInt
		ptr := v.Args[0]
		v.reset(OpARM64ADDconst)
		v.AuxInt = off
		v.AddArg(ptr)
		return true
	}
}
func rewriteValueARM64_OpOr16_0(v *Value) bool {
	// match: (Or16 x y)
	// cond:
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpOr32_0(v *Value) bool {
	// match: (Or32 x y)
	// cond:
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpOr64_0(v *Value) bool {
	// match: (Or64 x y)
	// cond:
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpOr8_0(v *Value) bool {
	// match: (Or8 x y)
	// cond:
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpOrB_0(v *Value) bool {
	// match: (OrB x y)
	// cond:
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpPanicBounds_0(v *Value) bool {
	// match: (PanicBounds [kind] x y mem)
	// cond: boundsABI(kind) == 0
	// result: (LoweredPanicBoundsA [kind] x y mem)
	for {
		kind := v.AuxInt
		mem := v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		if !(boundsABI(kind) == 0) {
			break
		}
		v.reset(OpARM64LoweredPanicBoundsA)
		v.AuxInt = kind
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(mem)
		return true
	}
	// match: (PanicBounds [kind] x y mem)
	// cond: boundsABI(kind) == 1
	// result: (LoweredPanicBoundsB [kind] x y mem)
	for {
		kind := v.AuxInt
		mem := v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		if !(boundsABI(kind) == 1) {
			break
		}
		v.reset(OpARM64LoweredPanicBoundsB)
		v.AuxInt = kind
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(mem)
		return true
	}
	// match: (PanicBounds [kind] x y mem)
	// cond: boundsABI(kind) == 2
	// result: (LoweredPanicBoundsC [kind] x y mem)
	for {
		kind := v.AuxInt
		mem := v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		if !(boundsABI(kind) == 2) {
			break
		}
		v.reset(OpARM64LoweredPanicBoundsC)
		v.AuxInt = kind
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpPopCount16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount16 <t> x)
	// cond:
	// result: (FMOVDfpgp <t> (VUADDLV <typ.Float64> (VCNT <typ.Float64> (FMOVDgpfp <typ.Float64> (ZeroExt16to64 x)))))
	for {
		t := v.Type
		x := v.Args[0]
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
func rewriteValueARM64_OpPopCount32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount32 <t> x)
	// cond:
	// result: (FMOVDfpgp <t> (VUADDLV <typ.Float64> (VCNT <typ.Float64> (FMOVDgpfp <typ.Float64> (ZeroExt32to64 x)))))
	for {
		t := v.Type
		x := v.Args[0]
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
func rewriteValueARM64_OpPopCount64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (PopCount64 <t> x)
	// cond:
	// result: (FMOVDfpgp <t> (VUADDLV <typ.Float64> (VCNT <typ.Float64> (FMOVDgpfp <typ.Float64> x))))
	for {
		t := v.Type
		x := v.Args[0]
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
func rewriteValueARM64_OpRotateLeft16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVDconst [c]))
	// cond:
	// result: (Or16 (Lsh16x64 <t> x (MOVDconst [c&15])) (Rsh16Ux64 <t> x (MOVDconst [-c&15])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpLsh16x64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v1.AuxInt = c & 15
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpRsh16Ux64, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = -c & 15
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpRotateLeft32_0(v *Value) bool {
	b := v.Block
	// match: (RotateLeft32 x y)
	// cond:
	// result: (RORW x (NEG <y.Type> y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64RORW)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, y.Type)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpRotateLeft64_0(v *Value) bool {
	b := v.Block
	// match: (RotateLeft64 x y)
	// cond:
	// result: (ROR x (NEG <y.Type> y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64ROR)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64NEG, y.Type)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpRotateLeft8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVDconst [c]))
	// cond:
	// result: (Or8 (Lsh8x64 <t> x (MOVDconst [c&7])) (Rsh8Ux64 <t> x (MOVDconst [-c&7])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpLsh8x64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v1.AuxInt = c & 7
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpRsh8Ux64, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = -c & 7
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpRound_0(v *Value) bool {
	// match: (Round x)
	// cond:
	// result: (FRINTAD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FRINTAD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpRound32F_0(v *Value) bool {
	// match: (Round32F x)
	// cond:
	// result: (LoweredRound32F x)
	for {
		x := v.Args[0]
		v.reset(OpARM64LoweredRound32F)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpRound64F_0(v *Value) bool {
	// match: (Round64F x)
	// cond:
	// result: (LoweredRound64F x)
	for {
		x := v.Args[0]
		v.reset(OpARM64LoweredRound64F)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpRoundToEven_0(v *Value) bool {
	// match: (RoundToEven x)
	// cond:
	// result: (FRINTND x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FRINTND)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpRsh16Ux16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> (ZeroExt16to64 x) (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = 0
		v.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v.AddArg(v4)
		return true
	}
}
func rewriteValueARM64_OpRsh16Ux32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> (ZeroExt16to64 x) (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = 0
		v.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v.AddArg(v4)
		return true
	}
}
func rewriteValueARM64_OpRsh16Ux64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> (ZeroExt16to64 x) y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v3.AddArg(y)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpRsh16Ux8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> (ZeroExt16to64 x) (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = 0
		v.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v.AddArg(v4)
		return true
	}
}
func rewriteValueARM64_OpRsh16x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 x y)
	// cond:
	// result: (SRA (SignExt16to64 x) (CSEL {OpARM64LessThanU} <y.Type> (ZeroExt16to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.Aux = OpARM64LessThanU
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = 63
		v1.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v1.AddArg(v4)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpRsh16x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 x y)
	// cond:
	// result: (SRA (SignExt16to64 x) (CSEL {OpARM64LessThanU} <y.Type> (ZeroExt32to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.Aux = OpARM64LessThanU
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = 63
		v1.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v1.AddArg(v4)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpRsh16x64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 x y)
	// cond:
	// result: (SRA (SignExt16to64 x) (CSEL {OpARM64LessThanU} <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.Aux = OpARM64LessThanU
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = 63
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v3.AddArg(y)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpRsh16x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 x y)
	// cond:
	// result: (SRA (SignExt16to64 x) (CSEL {OpARM64LessThanU} <y.Type> (ZeroExt8to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64 y))))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.Aux = OpARM64LessThanU
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = 63
		v1.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v1.AddArg(v4)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpRsh32Ux16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> (ZeroExt32to64 x) (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = 0
		v.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v.AddArg(v4)
		return true
	}
}
func rewriteValueARM64_OpRsh32Ux32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> (ZeroExt32to64 x) (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = 0
		v.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v.AddArg(v4)
		return true
	}
}
func rewriteValueARM64_OpRsh32Ux64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> (ZeroExt32to64 x) y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v3.AddArg(y)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpRsh32Ux8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> (ZeroExt32to64 x) (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = 0
		v.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v.AddArg(v4)
		return true
	}
}
func rewriteValueARM64_OpRsh32x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 x y)
	// cond:
	// result: (SRA (SignExt32to64 x) (CSEL {OpARM64LessThanU} <y.Type> (ZeroExt16to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.Aux = OpARM64LessThanU
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = 63
		v1.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v1.AddArg(v4)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpRsh32x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 x y)
	// cond:
	// result: (SRA (SignExt32to64 x) (CSEL {OpARM64LessThanU} <y.Type> (ZeroExt32to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.Aux = OpARM64LessThanU
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = 63
		v1.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v1.AddArg(v4)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpRsh32x64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x64 x y)
	// cond:
	// result: (SRA (SignExt32to64 x) (CSEL {OpARM64LessThanU} <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.Aux = OpARM64LessThanU
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = 63
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v3.AddArg(y)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpRsh32x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 x y)
	// cond:
	// result: (SRA (SignExt32to64 x) (CSEL {OpARM64LessThanU} <y.Type> (ZeroExt8to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64 y))))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.Aux = OpARM64LessThanU
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = 63
		v1.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v1.AddArg(v4)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpRsh64Ux16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpRsh64Ux32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpRsh64Ux64_0(v *Value) bool {
	b := v.Block
	// match: (Rsh64Ux64 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = 0
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM64_OpRsh64Ux8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> x (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpRsh64x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 x y)
	// cond:
	// result: (SRA x (CSEL {OpARM64LessThanU} <y.Type> (ZeroExt16to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v0.Aux = OpARM64LessThanU
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = 63
		v0.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v0.AddArg(v3)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpRsh64x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x32 x y)
	// cond:
	// result: (SRA x (CSEL {OpARM64LessThanU} <y.Type> (ZeroExt32to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v0.Aux = OpARM64LessThanU
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = 63
		v0.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v0.AddArg(v3)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpRsh64x64_0(v *Value) bool {
	b := v.Block
	// match: (Rsh64x64 x y)
	// cond:
	// result: (SRA x (CSEL {OpARM64LessThanU} <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v0.Aux = OpARM64LessThanU
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v1.AuxInt = 63
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpRsh64x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 x y)
	// cond:
	// result: (SRA x (CSEL {OpARM64LessThanU} <y.Type> (ZeroExt8to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64 y))))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v0.Aux = OpARM64LessThanU
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = 63
		v0.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v0.AddArg(v3)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpRsh8Ux16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> (ZeroExt8to64 x) (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = 0
		v.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v.AddArg(v4)
		return true
	}
}
func rewriteValueARM64_OpRsh8Ux32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> (ZeroExt8to64 x) (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = 0
		v.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v.AddArg(v4)
		return true
	}
}
func rewriteValueARM64_OpRsh8Ux64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> (ZeroExt8to64 x) y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = 0
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v3.AddArg(y)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM64_OpRsh8Ux8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 <t> x y)
	// cond:
	// result: (CSEL {OpARM64LessThanU} (SRL <t> (ZeroExt8to64 x) (ZeroExt8to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64 y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64CSEL)
		v.Aux = OpARM64LessThanU
		v0 := b.NewValue0(v.Pos, OpARM64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = 0
		v.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v.AddArg(v4)
		return true
	}
}
func rewriteValueARM64_OpRsh8x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 x y)
	// cond:
	// result: (SRA (SignExt8to64 x) (CSEL {OpARM64LessThanU} <y.Type> (ZeroExt16to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.Aux = OpARM64LessThanU
		v2 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = 63
		v1.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v1.AddArg(v4)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpRsh8x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 x y)
	// cond:
	// result: (SRA (SignExt8to64 x) (CSEL {OpARM64LessThanU} <y.Type> (ZeroExt32to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.Aux = OpARM64LessThanU
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = 63
		v1.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v1.AddArg(v4)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpRsh8x64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 x y)
	// cond:
	// result: (SRA (SignExt8to64 x) (CSEL {OpARM64LessThanU} <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.Aux = OpARM64LessThanU
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v2.AuxInt = 63
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v3.AuxInt = 64
		v3.AddArg(y)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpRsh8x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 x y)
	// cond:
	// result: (SRA (SignExt8to64 x) (CSEL {OpARM64LessThanU} <y.Type> (ZeroExt8to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64 y))))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSEL, y.Type)
		v1.Aux = OpARM64LessThanU
		v2 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v2.AddArg(y)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpConst64, y.Type)
		v3.AuxInt = 63
		v1.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
		v4.AuxInt = 64
		v5 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v5.AddArg(y)
		v4.AddArg(v5)
		v1.AddArg(v4)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM64_OpSelect0_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select0 (Add64carry x y c))
	// cond:
	// result: (Select0 <typ.UInt64> (ADCSflags x y (Select1 <types.TypeFlags> (ADDSconstflags [-1] c))))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64carry {
			break
		}
		c := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpARM64ADCSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v0.AddArg(x)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v2 := b.NewValue0(v.Pos, OpARM64ADDSconstflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v2.AuxInt = -1
		v2.AddArg(c)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 (Sub64borrow x y bo))
	// cond:
	// result: (Select0 <typ.UInt64> (SBCSflags x y (Select1 <types.TypeFlags> (NEGSflags bo))))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSub64borrow {
			break
		}
		bo := v_0.Args[2]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpSelect0)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpARM64SBCSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v0.AddArg(x)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v2 := b.NewValue0(v.Pos, OpARM64NEGSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v2.AddArg(bo)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpSelect1_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Select1 (Add64carry x y c))
	// cond:
	// result: (ADCzerocarry <typ.UInt64> (Select1 <types.TypeFlags> (ADCSflags x y (Select1 <types.TypeFlags> (ADDSconstflags [-1] c)))))
	for {
		v_0 := v.Args[0]
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
		v1.AddArg(x)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v3 := b.NewValue0(v.Pos, OpARM64ADDSconstflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v3.AuxInt = -1
		v3.AddArg(c)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Select1 (Sub64borrow x y bo))
	// cond:
	// result: (NEG <typ.UInt64> (NGCzerocarry <typ.UInt64> (Select1 <types.TypeFlags> (SBCSflags x y (Select1 <types.TypeFlags> (NEGSflags bo))))))
	for {
		v_0 := v.Args[0]
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
		v2.AddArg(x)
		v2.AddArg(y)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpARM64NEGSflags, types.NewTuple(typ.UInt64, types.TypeFlags))
		v4.AddArg(bo)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpSignExt16to32_0(v *Value) bool {
	// match: (SignExt16to32 x)
	// cond:
	// result: (MOVHreg x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MOVHreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpSignExt16to64_0(v *Value) bool {
	// match: (SignExt16to64 x)
	// cond:
	// result: (MOVHreg x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MOVHreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpSignExt32to64_0(v *Value) bool {
	// match: (SignExt32to64 x)
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MOVWreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpSignExt8to16_0(v *Value) bool {
	// match: (SignExt8to16 x)
	// cond:
	// result: (MOVBreg x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MOVBreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpSignExt8to32_0(v *Value) bool {
	// match: (SignExt8to32 x)
	// cond:
	// result: (MOVBreg x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MOVBreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpSignExt8to64_0(v *Value) bool {
	// match: (SignExt8to64 x)
	// cond:
	// result: (MOVBreg x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MOVBreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpSlicemask_0(v *Value) bool {
	b := v.Block
	// match: (Slicemask <t> x)
	// cond:
	// result: (SRAconst (NEG <t> x) [63])
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpARM64SRAconst)
		v.AuxInt = 63
		v0 := b.NewValue0(v.Pos, OpARM64NEG, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpSqrt_0(v *Value) bool {
	// match: (Sqrt x)
	// cond:
	// result: (FSQRTD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FSQRTD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpStaticCall_0(v *Value) bool {
	// match: (StaticCall [argwid] {target} mem)
	// cond:
	// result: (CALLstatic [argwid] {target} mem)
	for {
		argwid := v.AuxInt
		target := v.Aux
		mem := v.Args[0]
		v.reset(OpARM64CALLstatic)
		v.AuxInt = argwid
		v.Aux = target
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpStore_0(v *Value) bool {
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 1
	// result: (MOVBstore ptr val mem)
	for {
		t := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		if !(t.(*types.Type).Size() == 1) {
			break
		}
		v.reset(OpARM64MOVBstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 2
	// result: (MOVHstore ptr val mem)
	for {
		t := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		if !(t.(*types.Type).Size() == 2) {
			break
		}
		v.reset(OpARM64MOVHstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 4 && !is32BitFloat(val.Type)
	// result: (MOVWstore ptr val mem)
	for {
		t := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		if !(t.(*types.Type).Size() == 4 && !is32BitFloat(val.Type)) {
			break
		}
		v.reset(OpARM64MOVWstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 8 && !is64BitFloat(val.Type)
	// result: (MOVDstore ptr val mem)
	for {
		t := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		if !(t.(*types.Type).Size() == 8 && !is64BitFloat(val.Type)) {
			break
		}
		v.reset(OpARM64MOVDstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 4 && is32BitFloat(val.Type)
	// result: (FMOVSstore ptr val mem)
	for {
		t := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		if !(t.(*types.Type).Size() == 4 && is32BitFloat(val.Type)) {
			break
		}
		v.reset(OpARM64FMOVSstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 8 && is64BitFloat(val.Type)
	// result: (FMOVDstore ptr val mem)
	for {
		t := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		if !(t.(*types.Type).Size() == 8 && is64BitFloat(val.Type)) {
			break
		}
		v.reset(OpARM64FMOVDstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpSub16_0(v *Value) bool {
	// match: (Sub16 x y)
	// cond:
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpSub32_0(v *Value) bool {
	// match: (Sub32 x y)
	// cond:
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpSub32F_0(v *Value) bool {
	// match: (Sub32F x y)
	// cond:
	// result: (FSUBS x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64FSUBS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpSub64_0(v *Value) bool {
	// match: (Sub64 x y)
	// cond:
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpSub64F_0(v *Value) bool {
	// match: (Sub64F x y)
	// cond:
	// result: (FSUBD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64FSUBD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpSub8_0(v *Value) bool {
	// match: (Sub8 x y)
	// cond:
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpSubPtr_0(v *Value) bool {
	// match: (SubPtr x y)
	// cond:
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpTrunc_0(v *Value) bool {
	// match: (Trunc x)
	// cond:
	// result: (FRINTZD x)
	for {
		x := v.Args[0]
		v.reset(OpARM64FRINTZD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpTrunc16to8_0(v *Value) bool {
	// match: (Trunc16to8 x)
	// cond:
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpTrunc32to16_0(v *Value) bool {
	// match: (Trunc32to16 x)
	// cond:
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpTrunc32to8_0(v *Value) bool {
	// match: (Trunc32to8 x)
	// cond:
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpTrunc64to16_0(v *Value) bool {
	// match: (Trunc64to16 x)
	// cond:
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpTrunc64to32_0(v *Value) bool {
	// match: (Trunc64to32 x)
	// cond:
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpTrunc64to8_0(v *Value) bool {
	// match: (Trunc64to8 x)
	// cond:
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpWB_0(v *Value) bool {
	// match: (WB {fn} destptr srcptr mem)
	// cond:
	// result: (LoweredWB {fn} destptr srcptr mem)
	for {
		fn := v.Aux
		mem := v.Args[2]
		destptr := v.Args[0]
		srcptr := v.Args[1]
		v.reset(OpARM64LoweredWB)
		v.Aux = fn
		v.AddArg(destptr)
		v.AddArg(srcptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpXor16_0(v *Value) bool {
	// match: (Xor16 x y)
	// cond:
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpXor32_0(v *Value) bool {
	// match: (Xor32 x y)
	// cond:
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpXor64_0(v *Value) bool {
	// match: (Xor64 x y)
	// cond:
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpXor8_0(v *Value) bool {
	// match: (Xor8 x y)
	// cond:
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARM64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpZero_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Zero [0] _ mem)
	// cond:
	// result: mem
	for {
		if v.AuxInt != 0 {
			break
		}
		mem := v.Args[1]
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	// match: (Zero [1] ptr mem)
	// cond:
	// result: (MOVBstore ptr (MOVDconst [0]) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [2] ptr mem)
	// cond:
	// result: (MOVHstore ptr (MOVDconst [0]) mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVHstore)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [4] ptr mem)
	// cond:
	// result: (MOVWstore ptr (MOVDconst [0]) mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVWstore)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [8] ptr mem)
	// cond:
	// result: (MOVDstore ptr (MOVDconst [0]) mem)
	for {
		if v.AuxInt != 8 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVDstore)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [3] ptr mem)
	// cond:
	// result: (MOVBstore [2] ptr (MOVDconst [0]) (MOVHstore ptr (MOVDconst [0]) mem))
	for {
		if v.AuxInt != 3 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = 2
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHstore, types.TypeMem)
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [5] ptr mem)
	// cond:
	// result: (MOVBstore [4] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem))
	for {
		if v.AuxInt != 5 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = 4
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [6] ptr mem)
	// cond:
	// result: (MOVHstore [4] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem))
	for {
		if v.AuxInt != 6 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVHstore)
		v.AuxInt = 4
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [7] ptr mem)
	// cond:
	// result: (MOVBstore [6] ptr (MOVDconst [0]) (MOVHstore [4] ptr (MOVDconst [0]) (MOVWstore ptr (MOVDconst [0]) mem)))
	for {
		if v.AuxInt != 7 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = 6
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHstore, types.TypeMem)
		v1.AuxInt = 4
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v3.AddArg(ptr)
		v4 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = 0
		v3.AddArg(v4)
		v3.AddArg(mem)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [9] ptr mem)
	// cond:
	// result: (MOVBstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if v.AuxInt != 9 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = 8
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpZero_10(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Zero [10] ptr mem)
	// cond:
	// result: (MOVHstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if v.AuxInt != 10 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVHstore)
		v.AuxInt = 8
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [11] ptr mem)
	// cond:
	// result: (MOVBstore [10] ptr (MOVDconst [0]) (MOVHstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem)))
	for {
		if v.AuxInt != 11 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = 10
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHstore, types.TypeMem)
		v1.AuxInt = 8
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v3.AddArg(ptr)
		v4 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = 0
		v3.AddArg(v4)
		v3.AddArg(mem)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [12] ptr mem)
	// cond:
	// result: (MOVWstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if v.AuxInt != 12 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVWstore)
		v.AuxInt = 8
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [13] ptr mem)
	// cond:
	// result: (MOVBstore [12] ptr (MOVDconst [0]) (MOVWstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem)))
	for {
		if v.AuxInt != 13 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = 12
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v1.AuxInt = 8
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v3.AddArg(ptr)
		v4 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = 0
		v3.AddArg(v4)
		v3.AddArg(mem)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [14] ptr mem)
	// cond:
	// result: (MOVHstore [12] ptr (MOVDconst [0]) (MOVWstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem)))
	for {
		if v.AuxInt != 14 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVHstore)
		v.AuxInt = 12
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v1.AuxInt = 8
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v3.AddArg(ptr)
		v4 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = 0
		v3.AddArg(v4)
		v3.AddArg(mem)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [15] ptr mem)
	// cond:
	// result: (MOVBstore [14] ptr (MOVDconst [0]) (MOVHstore [12] ptr (MOVDconst [0]) (MOVWstore [8] ptr (MOVDconst [0]) (MOVDstore ptr (MOVDconst [0]) mem))))
	for {
		if v.AuxInt != 15 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64MOVBstore)
		v.AuxInt = 14
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHstore, types.TypeMem)
		v1.AuxInt = 12
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARM64MOVWstore, types.TypeMem)
		v3.AuxInt = 8
		v3.AddArg(ptr)
		v4 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = 0
		v3.AddArg(v4)
		v5 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
		v5.AddArg(ptr)
		v6 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v6.AuxInt = 0
		v5.AddArg(v6)
		v5.AddArg(mem)
		v3.AddArg(v5)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [16] ptr mem)
	// cond:
	// result: (STP [0] ptr (MOVDconst [0]) (MOVDconst [0]) mem)
	for {
		if v.AuxInt != 16 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64STP)
		v.AuxInt = 0
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v1.AuxInt = 0
		v.AddArg(v1)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [32] ptr mem)
	// cond:
	// result: (STP [16] ptr (MOVDconst [0]) (MOVDconst [0]) (STP [0] ptr (MOVDconst [0]) (MOVDconst [0]) mem))
	for {
		if v.AuxInt != 32 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64STP)
		v.AuxInt = 16
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v1.AuxInt = 0
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v2.AuxInt = 0
		v2.AddArg(ptr)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = 0
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = 0
		v2.AddArg(v4)
		v2.AddArg(mem)
		v.AddArg(v2)
		return true
	}
	// match: (Zero [48] ptr mem)
	// cond:
	// result: (STP [32] ptr (MOVDconst [0]) (MOVDconst [0]) (STP [16] ptr (MOVDconst [0]) (MOVDconst [0]) (STP [0] ptr (MOVDconst [0]) (MOVDconst [0]) mem)))
	for {
		if v.AuxInt != 48 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64STP)
		v.AuxInt = 32
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v1.AuxInt = 0
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v2.AuxInt = 16
		v2.AddArg(ptr)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = 0
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = 0
		v2.AddArg(v4)
		v5 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v5.AuxInt = 0
		v5.AddArg(ptr)
		v6 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v6.AuxInt = 0
		v5.AddArg(v6)
		v7 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v7.AuxInt = 0
		v5.AddArg(v7)
		v5.AddArg(mem)
		v2.AddArg(v5)
		v.AddArg(v2)
		return true
	}
	// match: (Zero [64] ptr mem)
	// cond:
	// result: (STP [48] ptr (MOVDconst [0]) (MOVDconst [0]) (STP [32] ptr (MOVDconst [0]) (MOVDconst [0]) (STP [16] ptr (MOVDconst [0]) (MOVDconst [0]) (STP [0] ptr (MOVDconst [0]) (MOVDconst [0]) mem))))
	for {
		if v.AuxInt != 64 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARM64STP)
		v.AuxInt = 48
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v1.AuxInt = 0
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v2.AuxInt = 32
		v2.AddArg(ptr)
		v3 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v3.AuxInt = 0
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v4.AuxInt = 0
		v2.AddArg(v4)
		v5 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v5.AuxInt = 16
		v5.AddArg(ptr)
		v6 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v6.AuxInt = 0
		v5.AddArg(v6)
		v7 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v7.AuxInt = 0
		v5.AddArg(v7)
		v8 := b.NewValue0(v.Pos, OpARM64STP, types.TypeMem)
		v8.AuxInt = 0
		v8.AddArg(ptr)
		v9 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v9.AuxInt = 0
		v8.AddArg(v9)
		v10 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v10.AuxInt = 0
		v8.AddArg(v10)
		v8.AddArg(mem)
		v5.AddArg(v8)
		v2.AddArg(v5)
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValueARM64_OpZero_20(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (Zero [s] ptr mem)
	// cond: s%16 != 0 && s%16 <= 8 && s > 16
	// result: (Zero [8] (OffPtr <ptr.Type> ptr [s-8]) (Zero [s-s%16] ptr mem))
	for {
		s := v.AuxInt
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(s%16 != 0 && s%16 <= 8 && s > 16) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = 8
		v0 := b.NewValue0(v.Pos, OpOffPtr, ptr.Type)
		v0.AuxInt = s - 8
		v0.AddArg(ptr)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v1.AuxInt = s - s%16
		v1.AddArg(ptr)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s%16 != 0 && s%16 > 8 && s > 16
	// result: (Zero [16] (OffPtr <ptr.Type> ptr [s-16]) (Zero [s-s%16] ptr mem))
	for {
		s := v.AuxInt
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(s%16 != 0 && s%16 > 8 && s > 16) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = 16
		v0 := b.NewValue0(v.Pos, OpOffPtr, ptr.Type)
		v0.AuxInt = s - 16
		v0.AddArg(ptr)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v1.AuxInt = s - s%16
		v1.AddArg(ptr)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s%16 == 0 && s > 64 && s <= 16*64 && !config.noDuffDevice
	// result: (DUFFZERO [4 * (64 - s/16)] ptr mem)
	for {
		s := v.AuxInt
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(s%16 == 0 && s > 64 && s <= 16*64 && !config.noDuffDevice) {
			break
		}
		v.reset(OpARM64DUFFZERO)
		v.AuxInt = 4 * (64 - s/16)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s%16 == 0 && (s > 16*64 || config.noDuffDevice)
	// result: (LoweredZero ptr (ADDconst <ptr.Type> [s-16] ptr) mem)
	for {
		s := v.AuxInt
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(s%16 == 0 && (s > 16*64 || config.noDuffDevice)) {
			break
		}
		v.reset(OpARM64LoweredZero)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64ADDconst, ptr.Type)
		v0.AuxInt = s - 16
		v0.AddArg(ptr)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpZeroExt16to32_0(v *Value) bool {
	// match: (ZeroExt16to32 x)
	// cond:
	// result: (MOVHUreg x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MOVHUreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpZeroExt16to64_0(v *Value) bool {
	// match: (ZeroExt16to64 x)
	// cond:
	// result: (MOVHUreg x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MOVHUreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpZeroExt32to64_0(v *Value) bool {
	// match: (ZeroExt32to64 x)
	// cond:
	// result: (MOVWUreg x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MOVWUreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpZeroExt8to16_0(v *Value) bool {
	// match: (ZeroExt8to16 x)
	// cond:
	// result: (MOVBUreg x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MOVBUreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpZeroExt8to32_0(v *Value) bool {
	// match: (ZeroExt8to32 x)
	// cond:
	// result: (MOVBUreg x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MOVBUreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM64_OpZeroExt8to64_0(v *Value) bool {
	// match: (ZeroExt8to64 x)
	// cond:
	// result: (MOVBUreg x)
	for {
		x := v.Args[0]
		v.reset(OpARM64MOVBUreg)
		v.AddArg(x)
		return true
	}
}
func rewriteBlockARM64(b *Block) bool {
	config := b.Func.Config
	typ := &config.Types
	_ = typ
	v := b.Control
	_ = v
	switch b.Kind {
	case BlockARM64EQ:
		// match: (EQ (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (EQ (TSTWconst [c] y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (TST x y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64AND {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64TST, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (TSTW x y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64AND {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64TSTW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (EQ (TSTconst [c] y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (EQ (CMNconst [c] y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (EQ (CMNWconst [c] y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (CMN x y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64ADD {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (CMNW x y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64ADD {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (CMN x y) yes no)
		for v.Op == OpARM64CMP {
			_ = v.Args[1]
			x := v.Args[0]
			z := v.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (EQ (CMNW x y) yes no)
		for v.Op == OpARM64CMPW {
			_ = v.Args[1]
			x := v.Args[0]
			z := v.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] x) yes no)
		// cond:
		// result: (Z x yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			b.Kind = BlockARM64Z
			b.SetControl(x)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPWconst [0] x) yes no)
		// cond:
		// result: (ZW x yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			b.Kind = BlockARM64ZW
			b.SetControl(x)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (EQ (CMN a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (EQ (CMP a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (EQ (CMNW a (MULW <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (EQ (CMPW a (MULW <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64EQ
			v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (TSTconst [c] x) yes no)
		// cond: oneBit(c)
		// result: (TBZ {ntz(c)} x yes no)
		for v.Op == OpARM64TSTconst {
			c := v.AuxInt
			x := v.Args[0]
			if !(oneBit(c)) {
				break
			}
			b.Kind = BlockARM64TBZ
			b.SetControl(x)
			b.Aux = ntz(c)
			return true
		}
		// match: (EQ (TSTWconst [c] x) yes no)
		// cond: oneBit(int64(uint32(c)))
		// result: (TBZ {ntz(int64(uint32(c)))} x yes no)
		for v.Op == OpARM64TSTWconst {
			c := v.AuxInt
			x := v.Args[0]
			if !(oneBit(int64(uint32(c)))) {
				break
			}
			b.Kind = BlockARM64TBZ
			b.SetControl(x)
			b.Aux = ntz(int64(uint32(c)))
			return true
		}
		// match: (EQ (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (EQ (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (EQ (InvertFlags cmp) yes no)
		// cond:
		// result: (EQ cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64EQ
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64FGE:
		// match: (FGE (InvertFlags cmp) yes no)
		// cond:
		// result: (FLE cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64FLE
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64FGT:
		// match: (FGT (InvertFlags cmp) yes no)
		// cond:
		// result: (FLT cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64FLT
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64FLE:
		// match: (FLE (InvertFlags cmp) yes no)
		// cond:
		// result: (FGE cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64FGE
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64FLT:
		// match: (FLT (InvertFlags cmp) yes no)
		// cond:
		// result: (FGT cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64FGT
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64GE:
		// match: (GE (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GE (TSTWconst [c] y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (TST x y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64AND {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64TST, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (TSTW x y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64AND {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64TSTW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GE (TSTconst [c] y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GE (CMNconst [c] y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GE (CMNWconst [c] y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (CMN x y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64ADD {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (CMNW x y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64ADD {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (CMN x y) yes no)
		for v.Op == OpARM64CMP {
			_ = v.Args[1]
			x := v.Args[0]
			z := v.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (GE (CMNW x y) yes no)
		for v.Op == OpARM64CMPW {
			_ = v.Args[1]
			x := v.Args[0]
			z := v.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (GE (CMN a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (GE (CMP a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (GE (CMNW a (MULW <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (GE (CMPW a (MULW <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GE
			v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPWconst [0] x) yes no)
		// cond:
		// result: (TBZ {int64(31)} x yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			b.Kind = BlockARM64TBZ
			b.SetControl(x)
			b.Aux = int64(31)
			return true
		}
		// match: (GE (CMPconst [0] x) yes no)
		// cond:
		// result: (TBZ {int64(63)} x yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			b.Kind = BlockARM64TBZ
			b.SetControl(x)
			b.Aux = int64(63)
			return true
		}
		// match: (GE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (GE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (GE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (GE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (GE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (GE (InvertFlags cmp) yes no)
		// cond:
		// result: (LE cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64LE
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64GT:
		// match: (GT (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GT (TSTWconst [c] y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (TST x y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64AND {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64TST, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (TSTW x y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64AND {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64TSTW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GT (TSTconst [c] y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GT (CMNconst [c] y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (GT (CMNWconst [c] y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (CMN x y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64ADD {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (CMNW x y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64ADD {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (CMN x y) yes no)
		for v.Op == OpARM64CMP {
			_ = v.Args[1]
			x := v.Args[0]
			z := v.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (GT (CMNW x y) yes no)
		for v.Op == OpARM64CMPW {
			_ = v.Args[1]
			x := v.Args[0]
			z := v.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (GT (CMN a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (GT (CMP a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (GT (CMNW a (MULW <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (GT (CMPW a (MULW <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64GT
			v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (GT (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (GT (InvertFlags cmp) yes no)
		// cond:
		// result: (LT cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64LT
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockIf:
		// match: (If (Equal cc) yes no)
		// cond:
		// result: (EQ cc yes no)
		for v.Op == OpARM64Equal {
			cc := v.Args[0]
			b.Kind = BlockARM64EQ
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (NotEqual cc) yes no)
		// cond:
		// result: (NE cc yes no)
		for v.Op == OpARM64NotEqual {
			cc := v.Args[0]
			b.Kind = BlockARM64NE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (LessThan cc) yes no)
		// cond:
		// result: (LT cc yes no)
		for v.Op == OpARM64LessThan {
			cc := v.Args[0]
			b.Kind = BlockARM64LT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (LessThanU cc) yes no)
		// cond:
		// result: (ULT cc yes no)
		for v.Op == OpARM64LessThanU {
			cc := v.Args[0]
			b.Kind = BlockARM64ULT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (LessEqual cc) yes no)
		// cond:
		// result: (LE cc yes no)
		for v.Op == OpARM64LessEqual {
			cc := v.Args[0]
			b.Kind = BlockARM64LE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (LessEqualU cc) yes no)
		// cond:
		// result: (ULE cc yes no)
		for v.Op == OpARM64LessEqualU {
			cc := v.Args[0]
			b.Kind = BlockARM64ULE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (GreaterThan cc) yes no)
		// cond:
		// result: (GT cc yes no)
		for v.Op == OpARM64GreaterThan {
			cc := v.Args[0]
			b.Kind = BlockARM64GT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (GreaterThanU cc) yes no)
		// cond:
		// result: (UGT cc yes no)
		for v.Op == OpARM64GreaterThanU {
			cc := v.Args[0]
			b.Kind = BlockARM64UGT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (GreaterEqual cc) yes no)
		// cond:
		// result: (GE cc yes no)
		for v.Op == OpARM64GreaterEqual {
			cc := v.Args[0]
			b.Kind = BlockARM64GE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (GreaterEqualU cc) yes no)
		// cond:
		// result: (UGE cc yes no)
		for v.Op == OpARM64GreaterEqualU {
			cc := v.Args[0]
			b.Kind = BlockARM64UGE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (LessThanF cc) yes no)
		// cond:
		// result: (FLT cc yes no)
		for v.Op == OpARM64LessThanF {
			cc := v.Args[0]
			b.Kind = BlockARM64FLT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (LessEqualF cc) yes no)
		// cond:
		// result: (FLE cc yes no)
		for v.Op == OpARM64LessEqualF {
			cc := v.Args[0]
			b.Kind = BlockARM64FLE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (GreaterThanF cc) yes no)
		// cond:
		// result: (FGT cc yes no)
		for v.Op == OpARM64GreaterThanF {
			cc := v.Args[0]
			b.Kind = BlockARM64FGT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (GreaterEqualF cc) yes no)
		// cond:
		// result: (FGE cc yes no)
		for v.Op == OpARM64GreaterEqualF {
			cc := v.Args[0]
			b.Kind = BlockARM64FGE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If cond yes no)
		// cond:
		// result: (NZ cond yes no)
		for {
			cond := b.Control
			b.Kind = BlockARM64NZ
			b.SetControl(cond)
			b.Aux = nil
			return true
		}
	case BlockARM64LE:
		// match: (LE (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LE (TSTWconst [c] y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (TST x y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64AND {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64TST, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (TSTW x y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64AND {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64TSTW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LE (TSTconst [c] y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LE (CMNconst [c] y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LE (CMNWconst [c] y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (CMN x y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64ADD {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (CMNW x y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64ADD {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (CMN x y) yes no)
		for v.Op == OpARM64CMP {
			_ = v.Args[1]
			x := v.Args[0]
			z := v.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (LE (CMNW x y) yes no)
		for v.Op == OpARM64CMPW {
			_ = v.Args[1]
			x := v.Args[0]
			z := v.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (LE (CMN a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (LE (CMP a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (LE (CMNW a (MULW <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (LE (CMPW a (MULW <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LE
			v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (LE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (LE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (LE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (LE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (LE (InvertFlags cmp) yes no)
		// cond:
		// result: (GE cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64GE
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64LT:
		// match: (LT (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LT (TSTWconst [c] y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (TST x y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64AND {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64TST, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (TSTW x y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64AND {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64TSTW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LT (TSTconst [c] y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LT (CMNconst [c] y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (LT (CMNWconst [c] y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (CMN x y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64ADD {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (CMNW x y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64ADD {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (CMN x y) yes no)
		for v.Op == OpARM64CMP {
			_ = v.Args[1]
			x := v.Args[0]
			z := v.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (LT (CMNW x y) yes no)
		for v.Op == OpARM64CMPW {
			_ = v.Args[1]
			x := v.Args[0]
			z := v.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (LT (CMN a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (LT (CMP a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (LT (CMNW a (MULW <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (LT (CMPW a (MULW <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64LT
			v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPWconst [0] x) yes no)
		// cond:
		// result: (TBNZ {int64(31)} x yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			b.Kind = BlockARM64TBNZ
			b.SetControl(x)
			b.Aux = int64(31)
			return true
		}
		// match: (LT (CMPconst [0] x) yes no)
		// cond:
		// result: (TBNZ {int64(63)} x yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			b.Kind = BlockARM64TBNZ
			b.SetControl(x)
			b.Aux = int64(63)
			return true
		}
		// match: (LT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (LT (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (LT (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (LT (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (LT (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (LT (InvertFlags cmp) yes no)
		// cond:
		// result: (GT cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64GT
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64NE:
		// match: (NE (CMPWconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (NE (TSTWconst [c] y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64TSTWconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (TST x y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64AND {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64TST, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPWconst [0] z:(AND x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (TSTW x y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64AND {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64TSTW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] x:(ANDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (NE (TSTconst [c] y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ANDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64TSTconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (NE (CMNconst [c] y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64CMNconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPWconst [0] x:(ADDconst [c] y)) yes no)
		// cond: x.Uses == 1
		// result: (NE (CMNWconst [c] y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			if x.Op != OpARM64ADDconst {
				break
			}
			c := x.AuxInt
			y := x.Args[0]
			if !(x.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64CMNWconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (CMN x y) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64ADD {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPWconst [0] z:(ADD x y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (CMNW x y) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64ADD {
				break
			}
			y := z.Args[1]
			x := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMP x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (CMN x y) yes no)
		for v.Op == OpARM64CMP {
			_ = v.Args[1]
			x := v.Args[0]
			z := v.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPW x z:(NEG y)) yes no)
		// cond: z.Uses == 1
		// result: (NE (CMNW x y) yes no)
		for v.Op == OpARM64CMPW {
			_ = v.Args[1]
			x := v.Args[0]
			z := v.Args[1]
			if z.Op != OpARM64NEG {
				break
			}
			y := z.Args[0]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] x) yes no)
		// cond:
		// result: (NZ x yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			b.Kind = BlockARM64NZ
			b.SetControl(x)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPWconst [0] x) yes no)
		// cond:
		// result: (NZW x yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			b.Kind = BlockARM64NZW
			b.SetControl(x)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] z:(MADD a x y)) yes no)
		// cond: z.Uses==1
		// result: (NE (CMN a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MADD {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64CMN, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] z:(MSUB a x y)) yes no)
		// cond: z.Uses==1
		// result: (NE (CMP a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MSUB {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPWconst [0] z:(MADDW a x y)) yes no)
		// cond: z.Uses==1
		// result: (NE (CMNW a (MULW <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MADDW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64CMNW, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPWconst [0] z:(MSUBW a x y)) yes no)
		// cond: z.Uses==1
		// result: (NE (CMPW a (MULW <x.Type> x y)) yes no)
		for v.Op == OpARM64CMPWconst {
			if v.AuxInt != 0 {
				break
			}
			z := v.Args[0]
			if z.Op != OpARM64MSUBW {
				break
			}
			y := z.Args[2]
			a := z.Args[0]
			x := z.Args[1]
			if !(z.Uses == 1) {
				break
			}
			b.Kind = BlockARM64NE
			v0 := b.NewValue0(v.Pos, OpARM64CMPW, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARM64MULW, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (TSTconst [c] x) yes no)
		// cond: oneBit(c)
		// result: (TBNZ {ntz(c)} x yes no)
		for v.Op == OpARM64TSTconst {
			c := v.AuxInt
			x := v.Args[0]
			if !(oneBit(c)) {
				break
			}
			b.Kind = BlockARM64TBNZ
			b.SetControl(x)
			b.Aux = ntz(c)
			return true
		}
		// match: (NE (TSTWconst [c] x) yes no)
		// cond: oneBit(int64(uint32(c)))
		// result: (TBNZ {ntz(int64(uint32(c)))} x yes no)
		for v.Op == OpARM64TSTWconst {
			c := v.AuxInt
			x := v.Args[0]
			if !(oneBit(int64(uint32(c)))) {
				break
			}
			b.Kind = BlockARM64TBNZ
			b.SetControl(x)
			b.Aux = ntz(int64(uint32(c)))
			return true
		}
		// match: (NE (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (NE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (NE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (NE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (NE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (NE (InvertFlags cmp) yes no)
		// cond:
		// result: (NE cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64NE
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64NZ:
		// match: (NZ (Equal cc) yes no)
		// cond:
		// result: (EQ cc yes no)
		for v.Op == OpARM64Equal {
			cc := v.Args[0]
			b.Kind = BlockARM64EQ
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (NotEqual cc) yes no)
		// cond:
		// result: (NE cc yes no)
		for v.Op == OpARM64NotEqual {
			cc := v.Args[0]
			b.Kind = BlockARM64NE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (LessThan cc) yes no)
		// cond:
		// result: (LT cc yes no)
		for v.Op == OpARM64LessThan {
			cc := v.Args[0]
			b.Kind = BlockARM64LT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (LessThanU cc) yes no)
		// cond:
		// result: (ULT cc yes no)
		for v.Op == OpARM64LessThanU {
			cc := v.Args[0]
			b.Kind = BlockARM64ULT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (LessEqual cc) yes no)
		// cond:
		// result: (LE cc yes no)
		for v.Op == OpARM64LessEqual {
			cc := v.Args[0]
			b.Kind = BlockARM64LE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (LessEqualU cc) yes no)
		// cond:
		// result: (ULE cc yes no)
		for v.Op == OpARM64LessEqualU {
			cc := v.Args[0]
			b.Kind = BlockARM64ULE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (GreaterThan cc) yes no)
		// cond:
		// result: (GT cc yes no)
		for v.Op == OpARM64GreaterThan {
			cc := v.Args[0]
			b.Kind = BlockARM64GT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (GreaterThanU cc) yes no)
		// cond:
		// result: (UGT cc yes no)
		for v.Op == OpARM64GreaterThanU {
			cc := v.Args[0]
			b.Kind = BlockARM64UGT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (GreaterEqual cc) yes no)
		// cond:
		// result: (GE cc yes no)
		for v.Op == OpARM64GreaterEqual {
			cc := v.Args[0]
			b.Kind = BlockARM64GE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (GreaterEqualU cc) yes no)
		// cond:
		// result: (UGE cc yes no)
		for v.Op == OpARM64GreaterEqualU {
			cc := v.Args[0]
			b.Kind = BlockARM64UGE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (LessThanF cc) yes no)
		// cond:
		// result: (FLT cc yes no)
		for v.Op == OpARM64LessThanF {
			cc := v.Args[0]
			b.Kind = BlockARM64FLT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (LessEqualF cc) yes no)
		// cond:
		// result: (FLE cc yes no)
		for v.Op == OpARM64LessEqualF {
			cc := v.Args[0]
			b.Kind = BlockARM64FLE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (GreaterThanF cc) yes no)
		// cond:
		// result: (FGT cc yes no)
		for v.Op == OpARM64GreaterThanF {
			cc := v.Args[0]
			b.Kind = BlockARM64FGT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (GreaterEqualF cc) yes no)
		// cond:
		// result: (FGE cc yes no)
		for v.Op == OpARM64GreaterEqualF {
			cc := v.Args[0]
			b.Kind = BlockARM64FGE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NZ (ANDconst [c] x) yes no)
		// cond: oneBit(c)
		// result: (TBNZ {ntz(c)} x yes no)
		for v.Op == OpARM64ANDconst {
			c := v.AuxInt
			x := v.Args[0]
			if !(oneBit(c)) {
				break
			}
			b.Kind = BlockARM64TBNZ
			b.SetControl(x)
			b.Aux = ntz(c)
			return true
		}
		// match: (NZ (MOVDconst [0]) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64MOVDconst {
			if v.AuxInt != 0 {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (NZ (MOVDconst [c]) yes no)
		// cond: c != 0
		// result: (First nil yes no)
		for v.Op == OpARM64MOVDconst {
			c := v.AuxInt
			if !(c != 0) {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
	case BlockARM64NZW:
		// match: (NZW (ANDconst [c] x) yes no)
		// cond: oneBit(int64(uint32(c)))
		// result: (TBNZ {ntz(int64(uint32(c)))} x yes no)
		for v.Op == OpARM64ANDconst {
			c := v.AuxInt
			x := v.Args[0]
			if !(oneBit(int64(uint32(c)))) {
				break
			}
			b.Kind = BlockARM64TBNZ
			b.SetControl(x)
			b.Aux = ntz(int64(uint32(c)))
			return true
		}
		// match: (NZW (MOVDconst [c]) yes no)
		// cond: int32(c) == 0
		// result: (First nil no yes)
		for v.Op == OpARM64MOVDconst {
			c := v.AuxInt
			if !(int32(c) == 0) {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (NZW (MOVDconst [c]) yes no)
		// cond: int32(c) != 0
		// result: (First nil yes no)
		for v.Op == OpARM64MOVDconst {
			c := v.AuxInt
			if !(int32(c) != 0) {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
	case BlockARM64UGE:
		// match: (UGE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (UGE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (UGE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (UGE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (UGE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (UGE (InvertFlags cmp) yes no)
		// cond:
		// result: (ULE cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64ULE
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64UGT:
		// match: (UGT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (UGT (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (UGT (InvertFlags cmp) yes no)
		// cond:
		// result: (ULT cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64ULT
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64ULE:
		// match: (ULE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (ULE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (ULE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (ULE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (ULE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (ULE (InvertFlags cmp) yes no)
		// cond:
		// result: (UGE cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64UGE
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64ULT:
		// match: (ULT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (ULT (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (ULT (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (ULT (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64FlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (ULT (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARM64FlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (ULT (InvertFlags cmp) yes no)
		// cond:
		// result: (UGT cmp yes no)
		for v.Op == OpARM64InvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARM64UGT
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARM64Z:
		// match: (Z (ANDconst [c] x) yes no)
		// cond: oneBit(c)
		// result: (TBZ {ntz(c)} x yes no)
		for v.Op == OpARM64ANDconst {
			c := v.AuxInt
			x := v.Args[0]
			if !(oneBit(c)) {
				break
			}
			b.Kind = BlockARM64TBZ
			b.SetControl(x)
			b.Aux = ntz(c)
			return true
		}
		// match: (Z (MOVDconst [0]) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARM64MOVDconst {
			if v.AuxInt != 0 {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (Z (MOVDconst [c]) yes no)
		// cond: c != 0
		// result: (First nil no yes)
		for v.Op == OpARM64MOVDconst {
			c := v.AuxInt
			if !(c != 0) {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
	case BlockARM64ZW:
		// match: (ZW (ANDconst [c] x) yes no)
		// cond: oneBit(int64(uint32(c)))
		// result: (TBZ {ntz(int64(uint32(c)))} x yes no)
		for v.Op == OpARM64ANDconst {
			c := v.AuxInt
			x := v.Args[0]
			if !(oneBit(int64(uint32(c)))) {
				break
			}
			b.Kind = BlockARM64TBZ
			b.SetControl(x)
			b.Aux = ntz(int64(uint32(c)))
			return true
		}
		// match: (ZW (MOVDconst [c]) yes no)
		// cond: int32(c) == 0
		// result: (First nil yes no)
		for v.Op == OpARM64MOVDconst {
			c := v.AuxInt
			if !(int32(c) == 0) {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (ZW (MOVDconst [c]) yes no)
		// cond: int32(c) != 0
		// result: (First nil no yes)
		for v.Op == OpARM64MOVDconst {
			c := v.AuxInt
			if !(int32(c) != 0) {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
	}
	return false
}
