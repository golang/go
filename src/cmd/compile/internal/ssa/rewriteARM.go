// Code generated from gen/ARM.rules; DO NOT EDIT.
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

func rewriteValueARM(v *Value) bool {
	switch v.Op {
	case OpARMADC:
		return rewriteValueARM_OpARMADC_0(v) || rewriteValueARM_OpARMADC_10(v) || rewriteValueARM_OpARMADC_20(v)
	case OpARMADCconst:
		return rewriteValueARM_OpARMADCconst_0(v)
	case OpARMADCshiftLL:
		return rewriteValueARM_OpARMADCshiftLL_0(v)
	case OpARMADCshiftLLreg:
		return rewriteValueARM_OpARMADCshiftLLreg_0(v)
	case OpARMADCshiftRA:
		return rewriteValueARM_OpARMADCshiftRA_0(v)
	case OpARMADCshiftRAreg:
		return rewriteValueARM_OpARMADCshiftRAreg_0(v)
	case OpARMADCshiftRL:
		return rewriteValueARM_OpARMADCshiftRL_0(v)
	case OpARMADCshiftRLreg:
		return rewriteValueARM_OpARMADCshiftRLreg_0(v)
	case OpARMADD:
		return rewriteValueARM_OpARMADD_0(v) || rewriteValueARM_OpARMADD_10(v)
	case OpARMADDD:
		return rewriteValueARM_OpARMADDD_0(v)
	case OpARMADDF:
		return rewriteValueARM_OpARMADDF_0(v)
	case OpARMADDS:
		return rewriteValueARM_OpARMADDS_0(v) || rewriteValueARM_OpARMADDS_10(v)
	case OpARMADDSshiftLL:
		return rewriteValueARM_OpARMADDSshiftLL_0(v)
	case OpARMADDSshiftLLreg:
		return rewriteValueARM_OpARMADDSshiftLLreg_0(v)
	case OpARMADDSshiftRA:
		return rewriteValueARM_OpARMADDSshiftRA_0(v)
	case OpARMADDSshiftRAreg:
		return rewriteValueARM_OpARMADDSshiftRAreg_0(v)
	case OpARMADDSshiftRL:
		return rewriteValueARM_OpARMADDSshiftRL_0(v)
	case OpARMADDSshiftRLreg:
		return rewriteValueARM_OpARMADDSshiftRLreg_0(v)
	case OpARMADDconst:
		return rewriteValueARM_OpARMADDconst_0(v)
	case OpARMADDshiftLL:
		return rewriteValueARM_OpARMADDshiftLL_0(v)
	case OpARMADDshiftLLreg:
		return rewriteValueARM_OpARMADDshiftLLreg_0(v)
	case OpARMADDshiftRA:
		return rewriteValueARM_OpARMADDshiftRA_0(v)
	case OpARMADDshiftRAreg:
		return rewriteValueARM_OpARMADDshiftRAreg_0(v)
	case OpARMADDshiftRL:
		return rewriteValueARM_OpARMADDshiftRL_0(v)
	case OpARMADDshiftRLreg:
		return rewriteValueARM_OpARMADDshiftRLreg_0(v)
	case OpARMAND:
		return rewriteValueARM_OpARMAND_0(v) || rewriteValueARM_OpARMAND_10(v) || rewriteValueARM_OpARMAND_20(v)
	case OpARMANDconst:
		return rewriteValueARM_OpARMANDconst_0(v)
	case OpARMANDshiftLL:
		return rewriteValueARM_OpARMANDshiftLL_0(v)
	case OpARMANDshiftLLreg:
		return rewriteValueARM_OpARMANDshiftLLreg_0(v)
	case OpARMANDshiftRA:
		return rewriteValueARM_OpARMANDshiftRA_0(v)
	case OpARMANDshiftRAreg:
		return rewriteValueARM_OpARMANDshiftRAreg_0(v)
	case OpARMANDshiftRL:
		return rewriteValueARM_OpARMANDshiftRL_0(v)
	case OpARMANDshiftRLreg:
		return rewriteValueARM_OpARMANDshiftRLreg_0(v)
	case OpARMBFX:
		return rewriteValueARM_OpARMBFX_0(v)
	case OpARMBFXU:
		return rewriteValueARM_OpARMBFXU_0(v)
	case OpARMBIC:
		return rewriteValueARM_OpARMBIC_0(v)
	case OpARMBICconst:
		return rewriteValueARM_OpARMBICconst_0(v)
	case OpARMBICshiftLL:
		return rewriteValueARM_OpARMBICshiftLL_0(v)
	case OpARMBICshiftLLreg:
		return rewriteValueARM_OpARMBICshiftLLreg_0(v)
	case OpARMBICshiftRA:
		return rewriteValueARM_OpARMBICshiftRA_0(v)
	case OpARMBICshiftRAreg:
		return rewriteValueARM_OpARMBICshiftRAreg_0(v)
	case OpARMBICshiftRL:
		return rewriteValueARM_OpARMBICshiftRL_0(v)
	case OpARMBICshiftRLreg:
		return rewriteValueARM_OpARMBICshiftRLreg_0(v)
	case OpARMCMN:
		return rewriteValueARM_OpARMCMN_0(v) || rewriteValueARM_OpARMCMN_10(v)
	case OpARMCMNconst:
		return rewriteValueARM_OpARMCMNconst_0(v)
	case OpARMCMNshiftLL:
		return rewriteValueARM_OpARMCMNshiftLL_0(v)
	case OpARMCMNshiftLLreg:
		return rewriteValueARM_OpARMCMNshiftLLreg_0(v)
	case OpARMCMNshiftRA:
		return rewriteValueARM_OpARMCMNshiftRA_0(v)
	case OpARMCMNshiftRAreg:
		return rewriteValueARM_OpARMCMNshiftRAreg_0(v)
	case OpARMCMNshiftRL:
		return rewriteValueARM_OpARMCMNshiftRL_0(v)
	case OpARMCMNshiftRLreg:
		return rewriteValueARM_OpARMCMNshiftRLreg_0(v)
	case OpARMCMOVWHSconst:
		return rewriteValueARM_OpARMCMOVWHSconst_0(v)
	case OpARMCMOVWLSconst:
		return rewriteValueARM_OpARMCMOVWLSconst_0(v)
	case OpARMCMP:
		return rewriteValueARM_OpARMCMP_0(v) || rewriteValueARM_OpARMCMP_10(v)
	case OpARMCMPD:
		return rewriteValueARM_OpARMCMPD_0(v)
	case OpARMCMPF:
		return rewriteValueARM_OpARMCMPF_0(v)
	case OpARMCMPconst:
		return rewriteValueARM_OpARMCMPconst_0(v)
	case OpARMCMPshiftLL:
		return rewriteValueARM_OpARMCMPshiftLL_0(v)
	case OpARMCMPshiftLLreg:
		return rewriteValueARM_OpARMCMPshiftLLreg_0(v)
	case OpARMCMPshiftRA:
		return rewriteValueARM_OpARMCMPshiftRA_0(v)
	case OpARMCMPshiftRAreg:
		return rewriteValueARM_OpARMCMPshiftRAreg_0(v)
	case OpARMCMPshiftRL:
		return rewriteValueARM_OpARMCMPshiftRL_0(v)
	case OpARMCMPshiftRLreg:
		return rewriteValueARM_OpARMCMPshiftRLreg_0(v)
	case OpARMEqual:
		return rewriteValueARM_OpARMEqual_0(v)
	case OpARMGreaterEqual:
		return rewriteValueARM_OpARMGreaterEqual_0(v)
	case OpARMGreaterEqualU:
		return rewriteValueARM_OpARMGreaterEqualU_0(v)
	case OpARMGreaterThan:
		return rewriteValueARM_OpARMGreaterThan_0(v)
	case OpARMGreaterThanU:
		return rewriteValueARM_OpARMGreaterThanU_0(v)
	case OpARMLessEqual:
		return rewriteValueARM_OpARMLessEqual_0(v)
	case OpARMLessEqualU:
		return rewriteValueARM_OpARMLessEqualU_0(v)
	case OpARMLessThan:
		return rewriteValueARM_OpARMLessThan_0(v)
	case OpARMLessThanU:
		return rewriteValueARM_OpARMLessThanU_0(v)
	case OpARMMOVBUload:
		return rewriteValueARM_OpARMMOVBUload_0(v)
	case OpARMMOVBUloadidx:
		return rewriteValueARM_OpARMMOVBUloadidx_0(v)
	case OpARMMOVBUreg:
		return rewriteValueARM_OpARMMOVBUreg_0(v)
	case OpARMMOVBload:
		return rewriteValueARM_OpARMMOVBload_0(v)
	case OpARMMOVBloadidx:
		return rewriteValueARM_OpARMMOVBloadidx_0(v)
	case OpARMMOVBreg:
		return rewriteValueARM_OpARMMOVBreg_0(v)
	case OpARMMOVBstore:
		return rewriteValueARM_OpARMMOVBstore_0(v)
	case OpARMMOVBstoreidx:
		return rewriteValueARM_OpARMMOVBstoreidx_0(v)
	case OpARMMOVDload:
		return rewriteValueARM_OpARMMOVDload_0(v)
	case OpARMMOVDstore:
		return rewriteValueARM_OpARMMOVDstore_0(v)
	case OpARMMOVFload:
		return rewriteValueARM_OpARMMOVFload_0(v)
	case OpARMMOVFstore:
		return rewriteValueARM_OpARMMOVFstore_0(v)
	case OpARMMOVHUload:
		return rewriteValueARM_OpARMMOVHUload_0(v)
	case OpARMMOVHUloadidx:
		return rewriteValueARM_OpARMMOVHUloadidx_0(v)
	case OpARMMOVHUreg:
		return rewriteValueARM_OpARMMOVHUreg_0(v)
	case OpARMMOVHload:
		return rewriteValueARM_OpARMMOVHload_0(v)
	case OpARMMOVHloadidx:
		return rewriteValueARM_OpARMMOVHloadidx_0(v)
	case OpARMMOVHreg:
		return rewriteValueARM_OpARMMOVHreg_0(v)
	case OpARMMOVHstore:
		return rewriteValueARM_OpARMMOVHstore_0(v)
	case OpARMMOVHstoreidx:
		return rewriteValueARM_OpARMMOVHstoreidx_0(v)
	case OpARMMOVWload:
		return rewriteValueARM_OpARMMOVWload_0(v)
	case OpARMMOVWloadidx:
		return rewriteValueARM_OpARMMOVWloadidx_0(v)
	case OpARMMOVWloadshiftLL:
		return rewriteValueARM_OpARMMOVWloadshiftLL_0(v)
	case OpARMMOVWloadshiftRA:
		return rewriteValueARM_OpARMMOVWloadshiftRA_0(v)
	case OpARMMOVWloadshiftRL:
		return rewriteValueARM_OpARMMOVWloadshiftRL_0(v)
	case OpARMMOVWreg:
		return rewriteValueARM_OpARMMOVWreg_0(v)
	case OpARMMOVWstore:
		return rewriteValueARM_OpARMMOVWstore_0(v)
	case OpARMMOVWstoreidx:
		return rewriteValueARM_OpARMMOVWstoreidx_0(v)
	case OpARMMOVWstoreshiftLL:
		return rewriteValueARM_OpARMMOVWstoreshiftLL_0(v)
	case OpARMMOVWstoreshiftRA:
		return rewriteValueARM_OpARMMOVWstoreshiftRA_0(v)
	case OpARMMOVWstoreshiftRL:
		return rewriteValueARM_OpARMMOVWstoreshiftRL_0(v)
	case OpARMMUL:
		return rewriteValueARM_OpARMMUL_0(v) || rewriteValueARM_OpARMMUL_10(v) || rewriteValueARM_OpARMMUL_20(v)
	case OpARMMULA:
		return rewriteValueARM_OpARMMULA_0(v) || rewriteValueARM_OpARMMULA_10(v) || rewriteValueARM_OpARMMULA_20(v)
	case OpARMMULD:
		return rewriteValueARM_OpARMMULD_0(v)
	case OpARMMULF:
		return rewriteValueARM_OpARMMULF_0(v)
	case OpARMMULS:
		return rewriteValueARM_OpARMMULS_0(v) || rewriteValueARM_OpARMMULS_10(v) || rewriteValueARM_OpARMMULS_20(v)
	case OpARMMVN:
		return rewriteValueARM_OpARMMVN_0(v)
	case OpARMMVNshiftLL:
		return rewriteValueARM_OpARMMVNshiftLL_0(v)
	case OpARMMVNshiftLLreg:
		return rewriteValueARM_OpARMMVNshiftLLreg_0(v)
	case OpARMMVNshiftRA:
		return rewriteValueARM_OpARMMVNshiftRA_0(v)
	case OpARMMVNshiftRAreg:
		return rewriteValueARM_OpARMMVNshiftRAreg_0(v)
	case OpARMMVNshiftRL:
		return rewriteValueARM_OpARMMVNshiftRL_0(v)
	case OpARMMVNshiftRLreg:
		return rewriteValueARM_OpARMMVNshiftRLreg_0(v)
	case OpARMNEGD:
		return rewriteValueARM_OpARMNEGD_0(v)
	case OpARMNEGF:
		return rewriteValueARM_OpARMNEGF_0(v)
	case OpARMNMULD:
		return rewriteValueARM_OpARMNMULD_0(v)
	case OpARMNMULF:
		return rewriteValueARM_OpARMNMULF_0(v)
	case OpARMNotEqual:
		return rewriteValueARM_OpARMNotEqual_0(v)
	case OpARMOR:
		return rewriteValueARM_OpARMOR_0(v) || rewriteValueARM_OpARMOR_10(v)
	case OpARMORconst:
		return rewriteValueARM_OpARMORconst_0(v)
	case OpARMORshiftLL:
		return rewriteValueARM_OpARMORshiftLL_0(v)
	case OpARMORshiftLLreg:
		return rewriteValueARM_OpARMORshiftLLreg_0(v)
	case OpARMORshiftRA:
		return rewriteValueARM_OpARMORshiftRA_0(v)
	case OpARMORshiftRAreg:
		return rewriteValueARM_OpARMORshiftRAreg_0(v)
	case OpARMORshiftRL:
		return rewriteValueARM_OpARMORshiftRL_0(v)
	case OpARMORshiftRLreg:
		return rewriteValueARM_OpARMORshiftRLreg_0(v)
	case OpARMRSB:
		return rewriteValueARM_OpARMRSB_0(v) || rewriteValueARM_OpARMRSB_10(v)
	case OpARMRSBSshiftLL:
		return rewriteValueARM_OpARMRSBSshiftLL_0(v)
	case OpARMRSBSshiftLLreg:
		return rewriteValueARM_OpARMRSBSshiftLLreg_0(v)
	case OpARMRSBSshiftRA:
		return rewriteValueARM_OpARMRSBSshiftRA_0(v)
	case OpARMRSBSshiftRAreg:
		return rewriteValueARM_OpARMRSBSshiftRAreg_0(v)
	case OpARMRSBSshiftRL:
		return rewriteValueARM_OpARMRSBSshiftRL_0(v)
	case OpARMRSBSshiftRLreg:
		return rewriteValueARM_OpARMRSBSshiftRLreg_0(v)
	case OpARMRSBconst:
		return rewriteValueARM_OpARMRSBconst_0(v)
	case OpARMRSBshiftLL:
		return rewriteValueARM_OpARMRSBshiftLL_0(v)
	case OpARMRSBshiftLLreg:
		return rewriteValueARM_OpARMRSBshiftLLreg_0(v)
	case OpARMRSBshiftRA:
		return rewriteValueARM_OpARMRSBshiftRA_0(v)
	case OpARMRSBshiftRAreg:
		return rewriteValueARM_OpARMRSBshiftRAreg_0(v)
	case OpARMRSBshiftRL:
		return rewriteValueARM_OpARMRSBshiftRL_0(v)
	case OpARMRSBshiftRLreg:
		return rewriteValueARM_OpARMRSBshiftRLreg_0(v)
	case OpARMRSCconst:
		return rewriteValueARM_OpARMRSCconst_0(v)
	case OpARMRSCshiftLL:
		return rewriteValueARM_OpARMRSCshiftLL_0(v)
	case OpARMRSCshiftLLreg:
		return rewriteValueARM_OpARMRSCshiftLLreg_0(v)
	case OpARMRSCshiftRA:
		return rewriteValueARM_OpARMRSCshiftRA_0(v)
	case OpARMRSCshiftRAreg:
		return rewriteValueARM_OpARMRSCshiftRAreg_0(v)
	case OpARMRSCshiftRL:
		return rewriteValueARM_OpARMRSCshiftRL_0(v)
	case OpARMRSCshiftRLreg:
		return rewriteValueARM_OpARMRSCshiftRLreg_0(v)
	case OpARMSBC:
		return rewriteValueARM_OpARMSBC_0(v) || rewriteValueARM_OpARMSBC_10(v)
	case OpARMSBCconst:
		return rewriteValueARM_OpARMSBCconst_0(v)
	case OpARMSBCshiftLL:
		return rewriteValueARM_OpARMSBCshiftLL_0(v)
	case OpARMSBCshiftLLreg:
		return rewriteValueARM_OpARMSBCshiftLLreg_0(v)
	case OpARMSBCshiftRA:
		return rewriteValueARM_OpARMSBCshiftRA_0(v)
	case OpARMSBCshiftRAreg:
		return rewriteValueARM_OpARMSBCshiftRAreg_0(v)
	case OpARMSBCshiftRL:
		return rewriteValueARM_OpARMSBCshiftRL_0(v)
	case OpARMSBCshiftRLreg:
		return rewriteValueARM_OpARMSBCshiftRLreg_0(v)
	case OpARMSLL:
		return rewriteValueARM_OpARMSLL_0(v)
	case OpARMSLLconst:
		return rewriteValueARM_OpARMSLLconst_0(v)
	case OpARMSRA:
		return rewriteValueARM_OpARMSRA_0(v)
	case OpARMSRAcond:
		return rewriteValueARM_OpARMSRAcond_0(v)
	case OpARMSRAconst:
		return rewriteValueARM_OpARMSRAconst_0(v)
	case OpARMSRL:
		return rewriteValueARM_OpARMSRL_0(v)
	case OpARMSRLconst:
		return rewriteValueARM_OpARMSRLconst_0(v)
	case OpARMSUB:
		return rewriteValueARM_OpARMSUB_0(v) || rewriteValueARM_OpARMSUB_10(v)
	case OpARMSUBD:
		return rewriteValueARM_OpARMSUBD_0(v)
	case OpARMSUBF:
		return rewriteValueARM_OpARMSUBF_0(v)
	case OpARMSUBS:
		return rewriteValueARM_OpARMSUBS_0(v) || rewriteValueARM_OpARMSUBS_10(v)
	case OpARMSUBSshiftLL:
		return rewriteValueARM_OpARMSUBSshiftLL_0(v)
	case OpARMSUBSshiftLLreg:
		return rewriteValueARM_OpARMSUBSshiftLLreg_0(v)
	case OpARMSUBSshiftRA:
		return rewriteValueARM_OpARMSUBSshiftRA_0(v)
	case OpARMSUBSshiftRAreg:
		return rewriteValueARM_OpARMSUBSshiftRAreg_0(v)
	case OpARMSUBSshiftRL:
		return rewriteValueARM_OpARMSUBSshiftRL_0(v)
	case OpARMSUBSshiftRLreg:
		return rewriteValueARM_OpARMSUBSshiftRLreg_0(v)
	case OpARMSUBconst:
		return rewriteValueARM_OpARMSUBconst_0(v)
	case OpARMSUBshiftLL:
		return rewriteValueARM_OpARMSUBshiftLL_0(v)
	case OpARMSUBshiftLLreg:
		return rewriteValueARM_OpARMSUBshiftLLreg_0(v)
	case OpARMSUBshiftRA:
		return rewriteValueARM_OpARMSUBshiftRA_0(v)
	case OpARMSUBshiftRAreg:
		return rewriteValueARM_OpARMSUBshiftRAreg_0(v)
	case OpARMSUBshiftRL:
		return rewriteValueARM_OpARMSUBshiftRL_0(v)
	case OpARMSUBshiftRLreg:
		return rewriteValueARM_OpARMSUBshiftRLreg_0(v)
	case OpARMTEQ:
		return rewriteValueARM_OpARMTEQ_0(v) || rewriteValueARM_OpARMTEQ_10(v)
	case OpARMTEQconst:
		return rewriteValueARM_OpARMTEQconst_0(v)
	case OpARMTEQshiftLL:
		return rewriteValueARM_OpARMTEQshiftLL_0(v)
	case OpARMTEQshiftLLreg:
		return rewriteValueARM_OpARMTEQshiftLLreg_0(v)
	case OpARMTEQshiftRA:
		return rewriteValueARM_OpARMTEQshiftRA_0(v)
	case OpARMTEQshiftRAreg:
		return rewriteValueARM_OpARMTEQshiftRAreg_0(v)
	case OpARMTEQshiftRL:
		return rewriteValueARM_OpARMTEQshiftRL_0(v)
	case OpARMTEQshiftRLreg:
		return rewriteValueARM_OpARMTEQshiftRLreg_0(v)
	case OpARMTST:
		return rewriteValueARM_OpARMTST_0(v) || rewriteValueARM_OpARMTST_10(v)
	case OpARMTSTconst:
		return rewriteValueARM_OpARMTSTconst_0(v)
	case OpARMTSTshiftLL:
		return rewriteValueARM_OpARMTSTshiftLL_0(v)
	case OpARMTSTshiftLLreg:
		return rewriteValueARM_OpARMTSTshiftLLreg_0(v)
	case OpARMTSTshiftRA:
		return rewriteValueARM_OpARMTSTshiftRA_0(v)
	case OpARMTSTshiftRAreg:
		return rewriteValueARM_OpARMTSTshiftRAreg_0(v)
	case OpARMTSTshiftRL:
		return rewriteValueARM_OpARMTSTshiftRL_0(v)
	case OpARMTSTshiftRLreg:
		return rewriteValueARM_OpARMTSTshiftRLreg_0(v)
	case OpARMXOR:
		return rewriteValueARM_OpARMXOR_0(v) || rewriteValueARM_OpARMXOR_10(v)
	case OpARMXORconst:
		return rewriteValueARM_OpARMXORconst_0(v)
	case OpARMXORshiftLL:
		return rewriteValueARM_OpARMXORshiftLL_0(v)
	case OpARMXORshiftLLreg:
		return rewriteValueARM_OpARMXORshiftLLreg_0(v)
	case OpARMXORshiftRA:
		return rewriteValueARM_OpARMXORshiftRA_0(v)
	case OpARMXORshiftRAreg:
		return rewriteValueARM_OpARMXORshiftRAreg_0(v)
	case OpARMXORshiftRL:
		return rewriteValueARM_OpARMXORshiftRL_0(v)
	case OpARMXORshiftRLreg:
		return rewriteValueARM_OpARMXORshiftRLreg_0(v)
	case OpARMXORshiftRR:
		return rewriteValueARM_OpARMXORshiftRR_0(v)
	case OpAdd16:
		return rewriteValueARM_OpAdd16_0(v)
	case OpAdd32:
		return rewriteValueARM_OpAdd32_0(v)
	case OpAdd32F:
		return rewriteValueARM_OpAdd32F_0(v)
	case OpAdd32carry:
		return rewriteValueARM_OpAdd32carry_0(v)
	case OpAdd32withcarry:
		return rewriteValueARM_OpAdd32withcarry_0(v)
	case OpAdd64F:
		return rewriteValueARM_OpAdd64F_0(v)
	case OpAdd8:
		return rewriteValueARM_OpAdd8_0(v)
	case OpAddPtr:
		return rewriteValueARM_OpAddPtr_0(v)
	case OpAddr:
		return rewriteValueARM_OpAddr_0(v)
	case OpAnd16:
		return rewriteValueARM_OpAnd16_0(v)
	case OpAnd32:
		return rewriteValueARM_OpAnd32_0(v)
	case OpAnd8:
		return rewriteValueARM_OpAnd8_0(v)
	case OpAndB:
		return rewriteValueARM_OpAndB_0(v)
	case OpAvg32u:
		return rewriteValueARM_OpAvg32u_0(v)
	case OpBitLen32:
		return rewriteValueARM_OpBitLen32_0(v)
	case OpBswap32:
		return rewriteValueARM_OpBswap32_0(v)
	case OpClosureCall:
		return rewriteValueARM_OpClosureCall_0(v)
	case OpCom16:
		return rewriteValueARM_OpCom16_0(v)
	case OpCom32:
		return rewriteValueARM_OpCom32_0(v)
	case OpCom8:
		return rewriteValueARM_OpCom8_0(v)
	case OpConst16:
		return rewriteValueARM_OpConst16_0(v)
	case OpConst32:
		return rewriteValueARM_OpConst32_0(v)
	case OpConst32F:
		return rewriteValueARM_OpConst32F_0(v)
	case OpConst64F:
		return rewriteValueARM_OpConst64F_0(v)
	case OpConst8:
		return rewriteValueARM_OpConst8_0(v)
	case OpConstBool:
		return rewriteValueARM_OpConstBool_0(v)
	case OpConstNil:
		return rewriteValueARM_OpConstNil_0(v)
	case OpCtz16:
		return rewriteValueARM_OpCtz16_0(v)
	case OpCtz16NonZero:
		return rewriteValueARM_OpCtz16NonZero_0(v)
	case OpCtz32:
		return rewriteValueARM_OpCtz32_0(v)
	case OpCtz32NonZero:
		return rewriteValueARM_OpCtz32NonZero_0(v)
	case OpCtz8:
		return rewriteValueARM_OpCtz8_0(v)
	case OpCtz8NonZero:
		return rewriteValueARM_OpCtz8NonZero_0(v)
	case OpCvt32Fto32:
		return rewriteValueARM_OpCvt32Fto32_0(v)
	case OpCvt32Fto32U:
		return rewriteValueARM_OpCvt32Fto32U_0(v)
	case OpCvt32Fto64F:
		return rewriteValueARM_OpCvt32Fto64F_0(v)
	case OpCvt32Uto32F:
		return rewriteValueARM_OpCvt32Uto32F_0(v)
	case OpCvt32Uto64F:
		return rewriteValueARM_OpCvt32Uto64F_0(v)
	case OpCvt32to32F:
		return rewriteValueARM_OpCvt32to32F_0(v)
	case OpCvt32to64F:
		return rewriteValueARM_OpCvt32to64F_0(v)
	case OpCvt64Fto32:
		return rewriteValueARM_OpCvt64Fto32_0(v)
	case OpCvt64Fto32F:
		return rewriteValueARM_OpCvt64Fto32F_0(v)
	case OpCvt64Fto32U:
		return rewriteValueARM_OpCvt64Fto32U_0(v)
	case OpDiv16:
		return rewriteValueARM_OpDiv16_0(v)
	case OpDiv16u:
		return rewriteValueARM_OpDiv16u_0(v)
	case OpDiv32:
		return rewriteValueARM_OpDiv32_0(v)
	case OpDiv32F:
		return rewriteValueARM_OpDiv32F_0(v)
	case OpDiv32u:
		return rewriteValueARM_OpDiv32u_0(v)
	case OpDiv64F:
		return rewriteValueARM_OpDiv64F_0(v)
	case OpDiv8:
		return rewriteValueARM_OpDiv8_0(v)
	case OpDiv8u:
		return rewriteValueARM_OpDiv8u_0(v)
	case OpEq16:
		return rewriteValueARM_OpEq16_0(v)
	case OpEq32:
		return rewriteValueARM_OpEq32_0(v)
	case OpEq32F:
		return rewriteValueARM_OpEq32F_0(v)
	case OpEq64F:
		return rewriteValueARM_OpEq64F_0(v)
	case OpEq8:
		return rewriteValueARM_OpEq8_0(v)
	case OpEqB:
		return rewriteValueARM_OpEqB_0(v)
	case OpEqPtr:
		return rewriteValueARM_OpEqPtr_0(v)
	case OpGeq16:
		return rewriteValueARM_OpGeq16_0(v)
	case OpGeq16U:
		return rewriteValueARM_OpGeq16U_0(v)
	case OpGeq32:
		return rewriteValueARM_OpGeq32_0(v)
	case OpGeq32F:
		return rewriteValueARM_OpGeq32F_0(v)
	case OpGeq32U:
		return rewriteValueARM_OpGeq32U_0(v)
	case OpGeq64F:
		return rewriteValueARM_OpGeq64F_0(v)
	case OpGeq8:
		return rewriteValueARM_OpGeq8_0(v)
	case OpGeq8U:
		return rewriteValueARM_OpGeq8U_0(v)
	case OpGetCallerPC:
		return rewriteValueARM_OpGetCallerPC_0(v)
	case OpGetCallerSP:
		return rewriteValueARM_OpGetCallerSP_0(v)
	case OpGetClosurePtr:
		return rewriteValueARM_OpGetClosurePtr_0(v)
	case OpGreater16:
		return rewriteValueARM_OpGreater16_0(v)
	case OpGreater16U:
		return rewriteValueARM_OpGreater16U_0(v)
	case OpGreater32:
		return rewriteValueARM_OpGreater32_0(v)
	case OpGreater32F:
		return rewriteValueARM_OpGreater32F_0(v)
	case OpGreater32U:
		return rewriteValueARM_OpGreater32U_0(v)
	case OpGreater64F:
		return rewriteValueARM_OpGreater64F_0(v)
	case OpGreater8:
		return rewriteValueARM_OpGreater8_0(v)
	case OpGreater8U:
		return rewriteValueARM_OpGreater8U_0(v)
	case OpHmul32:
		return rewriteValueARM_OpHmul32_0(v)
	case OpHmul32u:
		return rewriteValueARM_OpHmul32u_0(v)
	case OpInterCall:
		return rewriteValueARM_OpInterCall_0(v)
	case OpIsInBounds:
		return rewriteValueARM_OpIsInBounds_0(v)
	case OpIsNonNil:
		return rewriteValueARM_OpIsNonNil_0(v)
	case OpIsSliceInBounds:
		return rewriteValueARM_OpIsSliceInBounds_0(v)
	case OpLeq16:
		return rewriteValueARM_OpLeq16_0(v)
	case OpLeq16U:
		return rewriteValueARM_OpLeq16U_0(v)
	case OpLeq32:
		return rewriteValueARM_OpLeq32_0(v)
	case OpLeq32F:
		return rewriteValueARM_OpLeq32F_0(v)
	case OpLeq32U:
		return rewriteValueARM_OpLeq32U_0(v)
	case OpLeq64F:
		return rewriteValueARM_OpLeq64F_0(v)
	case OpLeq8:
		return rewriteValueARM_OpLeq8_0(v)
	case OpLeq8U:
		return rewriteValueARM_OpLeq8U_0(v)
	case OpLess16:
		return rewriteValueARM_OpLess16_0(v)
	case OpLess16U:
		return rewriteValueARM_OpLess16U_0(v)
	case OpLess32:
		return rewriteValueARM_OpLess32_0(v)
	case OpLess32F:
		return rewriteValueARM_OpLess32F_0(v)
	case OpLess32U:
		return rewriteValueARM_OpLess32U_0(v)
	case OpLess64F:
		return rewriteValueARM_OpLess64F_0(v)
	case OpLess8:
		return rewriteValueARM_OpLess8_0(v)
	case OpLess8U:
		return rewriteValueARM_OpLess8U_0(v)
	case OpLoad:
		return rewriteValueARM_OpLoad_0(v)
	case OpLocalAddr:
		return rewriteValueARM_OpLocalAddr_0(v)
	case OpLsh16x16:
		return rewriteValueARM_OpLsh16x16_0(v)
	case OpLsh16x32:
		return rewriteValueARM_OpLsh16x32_0(v)
	case OpLsh16x64:
		return rewriteValueARM_OpLsh16x64_0(v)
	case OpLsh16x8:
		return rewriteValueARM_OpLsh16x8_0(v)
	case OpLsh32x16:
		return rewriteValueARM_OpLsh32x16_0(v)
	case OpLsh32x32:
		return rewriteValueARM_OpLsh32x32_0(v)
	case OpLsh32x64:
		return rewriteValueARM_OpLsh32x64_0(v)
	case OpLsh32x8:
		return rewriteValueARM_OpLsh32x8_0(v)
	case OpLsh8x16:
		return rewriteValueARM_OpLsh8x16_0(v)
	case OpLsh8x32:
		return rewriteValueARM_OpLsh8x32_0(v)
	case OpLsh8x64:
		return rewriteValueARM_OpLsh8x64_0(v)
	case OpLsh8x8:
		return rewriteValueARM_OpLsh8x8_0(v)
	case OpMod16:
		return rewriteValueARM_OpMod16_0(v)
	case OpMod16u:
		return rewriteValueARM_OpMod16u_0(v)
	case OpMod32:
		return rewriteValueARM_OpMod32_0(v)
	case OpMod32u:
		return rewriteValueARM_OpMod32u_0(v)
	case OpMod8:
		return rewriteValueARM_OpMod8_0(v)
	case OpMod8u:
		return rewriteValueARM_OpMod8u_0(v)
	case OpMove:
		return rewriteValueARM_OpMove_0(v)
	case OpMul16:
		return rewriteValueARM_OpMul16_0(v)
	case OpMul32:
		return rewriteValueARM_OpMul32_0(v)
	case OpMul32F:
		return rewriteValueARM_OpMul32F_0(v)
	case OpMul32uhilo:
		return rewriteValueARM_OpMul32uhilo_0(v)
	case OpMul64F:
		return rewriteValueARM_OpMul64F_0(v)
	case OpMul8:
		return rewriteValueARM_OpMul8_0(v)
	case OpNeg16:
		return rewriteValueARM_OpNeg16_0(v)
	case OpNeg32:
		return rewriteValueARM_OpNeg32_0(v)
	case OpNeg32F:
		return rewriteValueARM_OpNeg32F_0(v)
	case OpNeg64F:
		return rewriteValueARM_OpNeg64F_0(v)
	case OpNeg8:
		return rewriteValueARM_OpNeg8_0(v)
	case OpNeq16:
		return rewriteValueARM_OpNeq16_0(v)
	case OpNeq32:
		return rewriteValueARM_OpNeq32_0(v)
	case OpNeq32F:
		return rewriteValueARM_OpNeq32F_0(v)
	case OpNeq64F:
		return rewriteValueARM_OpNeq64F_0(v)
	case OpNeq8:
		return rewriteValueARM_OpNeq8_0(v)
	case OpNeqB:
		return rewriteValueARM_OpNeqB_0(v)
	case OpNeqPtr:
		return rewriteValueARM_OpNeqPtr_0(v)
	case OpNilCheck:
		return rewriteValueARM_OpNilCheck_0(v)
	case OpNot:
		return rewriteValueARM_OpNot_0(v)
	case OpOffPtr:
		return rewriteValueARM_OpOffPtr_0(v)
	case OpOr16:
		return rewriteValueARM_OpOr16_0(v)
	case OpOr32:
		return rewriteValueARM_OpOr32_0(v)
	case OpOr8:
		return rewriteValueARM_OpOr8_0(v)
	case OpOrB:
		return rewriteValueARM_OpOrB_0(v)
	case OpPanicBounds:
		return rewriteValueARM_OpPanicBounds_0(v)
	case OpPanicExtend:
		return rewriteValueARM_OpPanicExtend_0(v)
	case OpRotateLeft16:
		return rewriteValueARM_OpRotateLeft16_0(v)
	case OpRotateLeft32:
		return rewriteValueARM_OpRotateLeft32_0(v)
	case OpRotateLeft8:
		return rewriteValueARM_OpRotateLeft8_0(v)
	case OpRound32F:
		return rewriteValueARM_OpRound32F_0(v)
	case OpRound64F:
		return rewriteValueARM_OpRound64F_0(v)
	case OpRsh16Ux16:
		return rewriteValueARM_OpRsh16Ux16_0(v)
	case OpRsh16Ux32:
		return rewriteValueARM_OpRsh16Ux32_0(v)
	case OpRsh16Ux64:
		return rewriteValueARM_OpRsh16Ux64_0(v)
	case OpRsh16Ux8:
		return rewriteValueARM_OpRsh16Ux8_0(v)
	case OpRsh16x16:
		return rewriteValueARM_OpRsh16x16_0(v)
	case OpRsh16x32:
		return rewriteValueARM_OpRsh16x32_0(v)
	case OpRsh16x64:
		return rewriteValueARM_OpRsh16x64_0(v)
	case OpRsh16x8:
		return rewriteValueARM_OpRsh16x8_0(v)
	case OpRsh32Ux16:
		return rewriteValueARM_OpRsh32Ux16_0(v)
	case OpRsh32Ux32:
		return rewriteValueARM_OpRsh32Ux32_0(v)
	case OpRsh32Ux64:
		return rewriteValueARM_OpRsh32Ux64_0(v)
	case OpRsh32Ux8:
		return rewriteValueARM_OpRsh32Ux8_0(v)
	case OpRsh32x16:
		return rewriteValueARM_OpRsh32x16_0(v)
	case OpRsh32x32:
		return rewriteValueARM_OpRsh32x32_0(v)
	case OpRsh32x64:
		return rewriteValueARM_OpRsh32x64_0(v)
	case OpRsh32x8:
		return rewriteValueARM_OpRsh32x8_0(v)
	case OpRsh8Ux16:
		return rewriteValueARM_OpRsh8Ux16_0(v)
	case OpRsh8Ux32:
		return rewriteValueARM_OpRsh8Ux32_0(v)
	case OpRsh8Ux64:
		return rewriteValueARM_OpRsh8Ux64_0(v)
	case OpRsh8Ux8:
		return rewriteValueARM_OpRsh8Ux8_0(v)
	case OpRsh8x16:
		return rewriteValueARM_OpRsh8x16_0(v)
	case OpRsh8x32:
		return rewriteValueARM_OpRsh8x32_0(v)
	case OpRsh8x64:
		return rewriteValueARM_OpRsh8x64_0(v)
	case OpRsh8x8:
		return rewriteValueARM_OpRsh8x8_0(v)
	case OpSelect0:
		return rewriteValueARM_OpSelect0_0(v)
	case OpSelect1:
		return rewriteValueARM_OpSelect1_0(v)
	case OpSignExt16to32:
		return rewriteValueARM_OpSignExt16to32_0(v)
	case OpSignExt8to16:
		return rewriteValueARM_OpSignExt8to16_0(v)
	case OpSignExt8to32:
		return rewriteValueARM_OpSignExt8to32_0(v)
	case OpSignmask:
		return rewriteValueARM_OpSignmask_0(v)
	case OpSlicemask:
		return rewriteValueARM_OpSlicemask_0(v)
	case OpSqrt:
		return rewriteValueARM_OpSqrt_0(v)
	case OpStaticCall:
		return rewriteValueARM_OpStaticCall_0(v)
	case OpStore:
		return rewriteValueARM_OpStore_0(v)
	case OpSub16:
		return rewriteValueARM_OpSub16_0(v)
	case OpSub32:
		return rewriteValueARM_OpSub32_0(v)
	case OpSub32F:
		return rewriteValueARM_OpSub32F_0(v)
	case OpSub32carry:
		return rewriteValueARM_OpSub32carry_0(v)
	case OpSub32withcarry:
		return rewriteValueARM_OpSub32withcarry_0(v)
	case OpSub64F:
		return rewriteValueARM_OpSub64F_0(v)
	case OpSub8:
		return rewriteValueARM_OpSub8_0(v)
	case OpSubPtr:
		return rewriteValueARM_OpSubPtr_0(v)
	case OpTrunc16to8:
		return rewriteValueARM_OpTrunc16to8_0(v)
	case OpTrunc32to16:
		return rewriteValueARM_OpTrunc32to16_0(v)
	case OpTrunc32to8:
		return rewriteValueARM_OpTrunc32to8_0(v)
	case OpWB:
		return rewriteValueARM_OpWB_0(v)
	case OpXor16:
		return rewriteValueARM_OpXor16_0(v)
	case OpXor32:
		return rewriteValueARM_OpXor32_0(v)
	case OpXor8:
		return rewriteValueARM_OpXor8_0(v)
	case OpZero:
		return rewriteValueARM_OpZero_0(v)
	case OpZeroExt16to32:
		return rewriteValueARM_OpZeroExt16to32_0(v)
	case OpZeroExt8to16:
		return rewriteValueARM_OpZeroExt8to16_0(v)
	case OpZeroExt8to32:
		return rewriteValueARM_OpZeroExt8to32_0(v)
	case OpZeromask:
		return rewriteValueARM_OpZeromask_0(v)
	}
	return false
}
func rewriteValueARM_OpARMADC_0(v *Value) bool {
	// match: (ADC (MOVWconst [c]) x flags)
	// cond:
	// result: (ADCconst [c] x flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMADCconst)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (MOVWconst [c]) flags)
	// cond:
	// result: (ADCconst [c] x flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADCconst)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (MOVWconst [c]) flags)
	// cond:
	// result: (ADCconst [c] x flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADCconst)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	// match: (ADC (MOVWconst [c]) x flags)
	// cond:
	// result: (ADCconst [c] x flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMADCconst)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (SLLconst [c] y) flags)
	// cond:
	// result: (ADCshiftLL x y [c] flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMADCshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (ADC (SLLconst [c] y) x flags)
	// cond:
	// result: (ADCshiftLL x y [c] flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMADCshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (ADC (SLLconst [c] y) x flags)
	// cond:
	// result: (ADCshiftLL x y [c] flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMADCshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (SLLconst [c] y) flags)
	// cond:
	// result: (ADCshiftLL x y [c] flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMADCshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (SRLconst [c] y) flags)
	// cond:
	// result: (ADCshiftRL x y [c] flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMADCshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (ADC (SRLconst [c] y) x flags)
	// cond:
	// result: (ADCshiftRL x y [c] flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMADCshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADC_10(v *Value) bool {
	// match: (ADC (SRLconst [c] y) x flags)
	// cond:
	// result: (ADCshiftRL x y [c] flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMADCshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (SRLconst [c] y) flags)
	// cond:
	// result: (ADCshiftRL x y [c] flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMADCshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (SRAconst [c] y) flags)
	// cond:
	// result: (ADCshiftRA x y [c] flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMADCshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (ADC (SRAconst [c] y) x flags)
	// cond:
	// result: (ADCshiftRA x y [c] flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMADCshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (ADC (SRAconst [c] y) x flags)
	// cond:
	// result: (ADCshiftRA x y [c] flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMADCshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (SRAconst [c] y) flags)
	// cond:
	// result: (ADCshiftRA x y [c] flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMADCshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (SLL y z) flags)
	// cond:
	// result: (ADCshiftLLreg x y z flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMADCshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (ADC (SLL y z) x flags)
	// cond:
	// result: (ADCshiftLLreg x y z flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMADCshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (ADC (SLL y z) x flags)
	// cond:
	// result: (ADCshiftLLreg x y z flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMADCshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (SLL y z) flags)
	// cond:
	// result: (ADCshiftLLreg x y z flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMADCshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADC_20(v *Value) bool {
	// match: (ADC x (SRL y z) flags)
	// cond:
	// result: (ADCshiftRLreg x y z flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMADCshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (ADC (SRL y z) x flags)
	// cond:
	// result: (ADCshiftRLreg x y z flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMADCshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (ADC (SRL y z) x flags)
	// cond:
	// result: (ADCshiftRLreg x y z flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMADCshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (SRL y z) flags)
	// cond:
	// result: (ADCshiftRLreg x y z flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMADCshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (SRA y z) flags)
	// cond:
	// result: (ADCshiftRAreg x y z flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMADCshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (ADC (SRA y z) x flags)
	// cond:
	// result: (ADCshiftRAreg x y z flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMADCshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (ADC (SRA y z) x flags)
	// cond:
	// result: (ADCshiftRAreg x y z flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMADCshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (ADC x (SRA y z) flags)
	// cond:
	// result: (ADCshiftRAreg x y z flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMADCshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCconst_0(v *Value) bool {
	// match: (ADCconst [c] (ADDconst [d] x) flags)
	// cond:
	// result: (ADCconst [int64(int32(c+d))] x flags)
	for {
		c := v.AuxInt
		flags := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMADCconst)
		v.AuxInt = int64(int32(c + d))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	// match: (ADCconst [c] (SUBconst [d] x) flags)
	// cond:
	// result: (ADCconst [int64(int32(c-d))] x flags)
	for {
		c := v.AuxInt
		flags := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMADCconst)
		v.AuxInt = int64(int32(c - d))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (ADCshiftLL (MOVWconst [c]) x [d] flags)
	// cond:
	// result: (ADCconst [c] (SLLconst <x.Type> x [d]) flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMADCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (ADCshiftLL x (MOVWconst [c]) [d] flags)
	// cond:
	// result: (ADCconst x [int64(int32(uint32(c)<<uint64(d)))] flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADCconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (ADCshiftLLreg (MOVWconst [c]) x y flags)
	// cond:
	// result: (ADCconst [c] (SLL <x.Type> x y) flags)
	for {
		flags := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		y := v.Args[2]
		v.reset(OpARMADCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (ADCshiftLLreg x y (MOVWconst [c]) flags)
	// cond:
	// result: (ADCshiftLL x y [c] flags)
	for {
		flags := v.Args[3]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMADCshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (ADCshiftRA (MOVWconst [c]) x [d] flags)
	// cond:
	// result: (ADCconst [c] (SRAconst <x.Type> x [d]) flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMADCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (ADCshiftRA x (MOVWconst [c]) [d] flags)
	// cond:
	// result: (ADCconst x [int64(int32(c)>>uint64(d))] flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADCconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (ADCshiftRAreg (MOVWconst [c]) x y flags)
	// cond:
	// result: (ADCconst [c] (SRA <x.Type> x y) flags)
	for {
		flags := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		y := v.Args[2]
		v.reset(OpARMADCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (ADCshiftRAreg x y (MOVWconst [c]) flags)
	// cond:
	// result: (ADCshiftRA x y [c] flags)
	for {
		flags := v.Args[3]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMADCshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (ADCshiftRL (MOVWconst [c]) x [d] flags)
	// cond:
	// result: (ADCconst [c] (SRLconst <x.Type> x [d]) flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMADCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (ADCshiftRL x (MOVWconst [c]) [d] flags)
	// cond:
	// result: (ADCconst x [int64(int32(uint32(c)>>uint64(d)))] flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADCconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (ADCshiftRLreg (MOVWconst [c]) x y flags)
	// cond:
	// result: (ADCconst [c] (SRL <x.Type> x y) flags)
	for {
		flags := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		y := v.Args[2]
		v.reset(OpARMADCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (ADCshiftRLreg x y (MOVWconst [c]) flags)
	// cond:
	// result: (ADCshiftRL x y [c] flags)
	for {
		flags := v.Args[3]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMADCshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADD_0(v *Value) bool {
	// match: (ADD x (MOVWconst [c]))
	// cond:
	// result: (ADDconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADDconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADD (MOVWconst [c]) x)
	// cond:
	// result: (ADDconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMADDconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADD x (SLLconst [c] y))
	// cond:
	// result: (ADDshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMADDshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD (SLLconst [c] y) x)
	// cond:
	// result: (ADDshiftLL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMADDshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD x (SRLconst [c] y))
	// cond:
	// result: (ADDshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMADDshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD (SRLconst [c] y) x)
	// cond:
	// result: (ADDshiftRL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMADDshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD x (SRAconst [c] y))
	// cond:
	// result: (ADDshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMADDshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD (SRAconst [c] y) x)
	// cond:
	// result: (ADDshiftRA x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMADDshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD x (SLL y z))
	// cond:
	// result: (ADDshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMADDshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (ADD (SLL y z) x)
	// cond:
	// result: (ADDshiftLLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMADDshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADD_10(v *Value) bool {
	b := v.Block
	// match: (ADD x (SRL y z))
	// cond:
	// result: (ADDshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMADDshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (ADD (SRL y z) x)
	// cond:
	// result: (ADDshiftRLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMADDshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (ADD x (SRA y z))
	// cond:
	// result: (ADDshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMADDshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (ADD (SRA y z) x)
	// cond:
	// result: (ADDshiftRAreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMADDshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (ADD x (RSBconst [0] y))
	// cond:
	// result: (SUB x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMRSBconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpARMSUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD (RSBconst [0] y) x)
	// cond:
	// result: (SUB x y)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMRSBconst {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		y := v_0.Args[0]
		v.reset(OpARMSUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD <t> (RSBconst [c] x) (RSBconst [d] y))
	// cond:
	// result: (RSBconst [c+d] (ADD <t> x y))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMRSBconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMRSBconst {
			break
		}
		d := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMRSBconst)
		v.AuxInt = c + d
		v0 := b.NewValue0(v.Pos, OpARMADD, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ADD <t> (RSBconst [d] y) (RSBconst [c] x))
	// cond:
	// result: (RSBconst [c+d] (ADD <t> x y))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMRSBconst {
			break
		}
		d := v_0.AuxInt
		y := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMRSBconst {
			break
		}
		c := v_1.AuxInt
		x := v_1.Args[0]
		v.reset(OpARMRSBconst)
		v.AuxInt = c + d
		v0 := b.NewValue0(v.Pos, OpARMADD, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ADD (MUL x y) a)
	// cond:
	// result: (MULA x y a)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMUL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARMMULA)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(a)
		return true
	}
	// match: (ADD a (MUL x y))
	// cond:
	// result: (MULA x y a)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMUL {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		v.reset(OpARMMULA)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDD_0(v *Value) bool {
	// match: (ADDD a (MULD x y))
	// cond: a.Uses == 1 && objabi.GOARM >= 6
	// result: (MULAD a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Uses == 1 && objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMMULAD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDD (MULD x y) a)
	// cond: a.Uses == 1 && objabi.GOARM >= 6
	// result: (MULAD a x y)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(a.Uses == 1 && objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMMULAD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDD a (NMULD x y))
	// cond: a.Uses == 1 && objabi.GOARM >= 6
	// result: (MULSD a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMNMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Uses == 1 && objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMMULSD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDD (NMULD x y) a)
	// cond: a.Uses == 1 && objabi.GOARM >= 6
	// result: (MULSD a x y)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMNMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(a.Uses == 1 && objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMMULSD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDF_0(v *Value) bool {
	// match: (ADDF a (MULF x y))
	// cond: a.Uses == 1 && objabi.GOARM >= 6
	// result: (MULAF a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMULF {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Uses == 1 && objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMMULAF)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDF (MULF x y) a)
	// cond: a.Uses == 1 && objabi.GOARM >= 6
	// result: (MULAF a x y)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMULF {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(a.Uses == 1 && objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMMULAF)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDF a (NMULF x y))
	// cond: a.Uses == 1 && objabi.GOARM >= 6
	// result: (MULSF a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMNMULF {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Uses == 1 && objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMMULSF)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDF (NMULF x y) a)
	// cond: a.Uses == 1 && objabi.GOARM >= 6
	// result: (MULSF a x y)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMNMULF {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(a.Uses == 1 && objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMMULSF)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDS_0(v *Value) bool {
	// match: (ADDS x (MOVWconst [c]))
	// cond:
	// result: (ADDSconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADDSconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDS (MOVWconst [c]) x)
	// cond:
	// result: (ADDSconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMADDSconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDS x (SLLconst [c] y))
	// cond:
	// result: (ADDSshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMADDSshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDS (SLLconst [c] y) x)
	// cond:
	// result: (ADDSshiftLL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMADDSshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDS x (SRLconst [c] y))
	// cond:
	// result: (ADDSshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMADDSshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDS (SRLconst [c] y) x)
	// cond:
	// result: (ADDSshiftRL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMADDSshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDS x (SRAconst [c] y))
	// cond:
	// result: (ADDSshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMADDSshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDS (SRAconst [c] y) x)
	// cond:
	// result: (ADDSshiftRA x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMADDSshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDS x (SLL y z))
	// cond:
	// result: (ADDSshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMADDSshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (ADDS (SLL y z) x)
	// cond:
	// result: (ADDSshiftLLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMADDSshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDS_10(v *Value) bool {
	// match: (ADDS x (SRL y z))
	// cond:
	// result: (ADDSshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMADDSshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (ADDS (SRL y z) x)
	// cond:
	// result: (ADDSshiftRLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMADDSshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (ADDS x (SRA y z))
	// cond:
	// result: (ADDSshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMADDSshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (ADDS (SRA y z) x)
	// cond:
	// result: (ADDSshiftRAreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMADDSshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDSshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (ADDSshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (ADDSconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMADDSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDSshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (ADDSconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADDSconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDSshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (ADDSshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (ADDSconst [c] (SLL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMADDSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ADDSshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (ADDSshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMADDSshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDSshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (ADDSshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (ADDSconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMADDSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDSshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (ADDSconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADDSconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDSshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (ADDSshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (ADDSconst [c] (SRA <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMADDSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ADDSshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (ADDSshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMADDSshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDSshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (ADDSshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (ADDSconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMADDSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDSshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (ADDSconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADDSconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDSshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (ADDSshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (ADDSconst [c] (SRL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMADDSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ADDSshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (ADDSshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMADDSshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDconst_0(v *Value) bool {
	// match: (ADDconst [off1] (MOVWaddr [off2] {sym} ptr))
	// cond:
	// result: (MOVWaddr [off1+off2] {sym} ptr)
	for {
		off1 := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym := v_0.Aux
		ptr := v_0.Args[0]
		v.reset(OpARMMOVWaddr)
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
	// match: (ADDconst [c] x)
	// cond: !isARMImmRot(uint32(c)) && isARMImmRot(uint32(-c))
	// result: (SUBconst [int64(int32(-c))] x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(!isARMImmRot(uint32(c)) && isARMImmRot(uint32(-c))) {
			break
		}
		v.reset(OpARMSUBconst)
		v.AuxInt = int64(int32(-c))
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] x)
	// cond: objabi.GOARM==7 && !isARMImmRot(uint32(c)) && uint32(c)>0xffff && uint32(-c)<=0xffff
	// result: (SUBconst [int64(int32(-c))] x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(objabi.GOARM == 7 && !isARMImmRot(uint32(c)) && uint32(c) > 0xffff && uint32(-c) <= 0xffff) {
			break
		}
		v.reset(OpARMSUBconst)
		v.AuxInt = int64(int32(-c))
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [int64(int32(c+d))])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(c + d))
		return true
	}
	// match: (ADDconst [c] (ADDconst [d] x))
	// cond:
	// result: (ADDconst [int64(int32(c+d))] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMADDconst)
		v.AuxInt = int64(int32(c + d))
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (SUBconst [d] x))
	// cond:
	// result: (ADDconst [int64(int32(c-d))] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMADDconst)
		v.AuxInt = int64(int32(c - d))
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (RSBconst [d] x))
	// cond:
	// result: (RSBconst [int64(int32(c+d))] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMRSBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMRSBconst)
		v.AuxInt = int64(int32(c + d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDshiftLL_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (ADDconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (ADDconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADDconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL [c] (SRLconst x [32-c]) x)
	// cond:
	// result: (SRRconst [32-c] x)
	for {
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		if v_0.AuxInt != 32-c {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARMSRRconst)
		v.AuxInt = 32 - c
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL <typ.UInt16> [8] (BFXU <typ.UInt16> [armBFAuxInt(8, 8)] x) x)
	// cond:
	// result: (REV16 x)
	for {
		if v.Type != typ.UInt16 {
			break
		}
		if v.AuxInt != 8 {
			break
		}
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMBFXU {
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
		v.reset(OpARMREV16)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL <typ.UInt16> [8] (SRLconst <typ.UInt16> [24] (SLLconst [16] x)) x)
	// cond: objabi.GOARM>=6
	// result: (REV16 x)
	for {
		if v.Type != typ.UInt16 {
			break
		}
		if v.AuxInt != 8 {
			break
		}
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		if v_0.Type != typ.UInt16 {
			break
		}
		if v_0.AuxInt != 24 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARMSLLconst {
			break
		}
		if v_0_0.AuxInt != 16 {
			break
		}
		if x != v_0_0.Args[0] {
			break
		}
		if !(objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMREV16)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (ADDshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (ADDconst [c] (SLL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (ADDshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMADDshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (ADDshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (ADDconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (ADDconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADDconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (ADDshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (ADDconst [c] (SRA <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (ADDshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMADDshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (ADDshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (ADDconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (ADDconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMADDconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftRL [c] (SLLconst x [32-c]) x)
	// cond:
	// result: (SRRconst [ c] x)
	for {
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		if v_0.AuxInt != 32-c {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARMSRRconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (ADDshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (ADDconst [c] (SRL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMADDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (ADDshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMADDshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMAND_0(v *Value) bool {
	// match: (AND x (MOVWconst [c]))
	// cond:
	// result: (ANDconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMANDconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (AND (MOVWconst [c]) x)
	// cond:
	// result: (ANDconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMANDconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (AND x (SLLconst [c] y))
	// cond:
	// result: (ANDshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMANDshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND (SLLconst [c] y) x)
	// cond:
	// result: (ANDshiftLL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMANDshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND x (SRLconst [c] y))
	// cond:
	// result: (ANDshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMANDshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND (SRLconst [c] y) x)
	// cond:
	// result: (ANDshiftRL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMANDshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND x (SRAconst [c] y))
	// cond:
	// result: (ANDshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMANDshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND (SRAconst [c] y) x)
	// cond:
	// result: (ANDshiftRA x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMANDshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND x (SLL y z))
	// cond:
	// result: (ANDshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMANDshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (AND (SLL y z) x)
	// cond:
	// result: (ANDshiftLLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMANDshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMAND_10(v *Value) bool {
	// match: (AND x (SRL y z))
	// cond:
	// result: (ANDshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMANDshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (AND (SRL y z) x)
	// cond:
	// result: (ANDshiftRLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMANDshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (AND x (SRA y z))
	// cond:
	// result: (ANDshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMANDshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (AND (SRA y z) x)
	// cond:
	// result: (ANDshiftRAreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMANDshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
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
		if v_1.Op != OpARMMVN {
			break
		}
		y := v_1.Args[0]
		v.reset(OpARMBIC)
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
		if v_0.Op != OpARMMVN {
			break
		}
		y := v_0.Args[0]
		v.reset(OpARMBIC)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND x (MVNshiftLL y [c]))
	// cond:
	// result: (BICshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMVNshiftLL {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMBICshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND (MVNshiftLL y [c]) x)
	// cond:
	// result: (BICshiftLL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMVNshiftLL {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMBICshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND x (MVNshiftRL y [c]))
	// cond:
	// result: (BICshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMVNshiftRL {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMBICshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMAND_20(v *Value) bool {
	// match: (AND (MVNshiftRL y [c]) x)
	// cond:
	// result: (BICshiftRL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMVNshiftRL {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMBICshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND x (MVNshiftRA y [c]))
	// cond:
	// result: (BICshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMVNshiftRA {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMBICshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND (MVNshiftRA y [c]) x)
	// cond:
	// result: (BICshiftRA x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMVNshiftRA {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMBICshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMANDconst_0(v *Value) bool {
	// match: (ANDconst [0] _)
	// cond:
	// result: (MOVWconst [0])
	for {
		if v.AuxInt != 0 {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (ANDconst [c] x)
	// cond: int32(c)==-1
	// result: x
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] x)
	// cond: !isARMImmRot(uint32(c)) && isARMImmRot(^uint32(c))
	// result: (BICconst [int64(int32(^uint32(c)))] x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(!isARMImmRot(uint32(c)) && isARMImmRot(^uint32(c))) {
			break
		}
		v.reset(OpARMBICconst)
		v.AuxInt = int64(int32(^uint32(c)))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] x)
	// cond: objabi.GOARM==7 && !isARMImmRot(uint32(c)) && uint32(c)>0xffff && ^uint32(c)<=0xffff
	// result: (BICconst [int64(int32(^uint32(c)))] x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(objabi.GOARM == 7 && !isARMImmRot(uint32(c)) && uint32(c) > 0xffff && ^uint32(c) <= 0xffff) {
			break
		}
		v.reset(OpARMBICconst)
		v.AuxInt = int64(int32(^uint32(c)))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [c&d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = c & d
		return true
	}
	// match: (ANDconst [c] (ANDconst [d] x))
	// cond:
	// result: (ANDconst [c&d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMANDconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMANDconst)
		v.AuxInt = c & d
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMANDshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (ANDshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (ANDconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMANDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (ANDconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMANDconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
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
		if y.Op != OpARMSLLconst {
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
func rewriteValueARM_OpARMANDshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (ANDshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (ANDconst [c] (SLL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMANDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (ANDshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMANDshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMANDshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (ANDshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (ANDconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMANDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (ANDconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMANDconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
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
		if y.Op != OpARMSRAconst {
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
func rewriteValueARM_OpARMANDshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (ANDshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (ANDconst [c] (SRA <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMANDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (ANDshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMANDshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMANDshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (ANDshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (ANDconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMANDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (ANDconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMANDconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
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
		if y.Op != OpARMSRLconst {
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
func rewriteValueARM_OpARMANDshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (ANDshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (ANDconst [c] (SRL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMANDconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (ANDshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMANDshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMBFX_0(v *Value) bool {
	// match: (BFX [c] (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [int64(int32(d)<<(32-uint32(c&0xff)-uint32(c>>8))>>(32-uint32(c>>8)))])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(d) << (32 - uint32(c&0xff) - uint32(c>>8)) >> (32 - uint32(c>>8)))
		return true
	}
	return false
}
func rewriteValueARM_OpARMBFXU_0(v *Value) bool {
	// match: (BFXU [c] (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [int64(int32(uint32(d)<<(32-uint32(c&0xff)-uint32(c>>8))>>(32-uint32(c>>8))))])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(uint32(d) << (32 - uint32(c&0xff) - uint32(c>>8)) >> (32 - uint32(c>>8))))
		return true
	}
	return false
}
func rewriteValueARM_OpARMBIC_0(v *Value) bool {
	// match: (BIC x (MOVWconst [c]))
	// cond:
	// result: (BICconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMBICconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (BIC x (SLLconst [c] y))
	// cond:
	// result: (BICshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMBICshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (BIC x (SRLconst [c] y))
	// cond:
	// result: (BICshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMBICshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (BIC x (SRAconst [c] y))
	// cond:
	// result: (BICshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMBICshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (BIC x (SLL y z))
	// cond:
	// result: (BICshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMBICshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (BIC x (SRL y z))
	// cond:
	// result: (BICshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMBICshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (BIC x (SRA y z))
	// cond:
	// result: (BICshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMBICshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (BIC x x)
	// cond:
	// result: (MOVWconst [0])
	for {
		x := v.Args[1]
		if x != v.Args[0] {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICconst_0(v *Value) bool {
	// match: (BICconst [0] x)
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
	// match: (BICconst [c] _)
	// cond: int32(c)==-1
	// result: (MOVWconst [0])
	for {
		c := v.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (BICconst [c] x)
	// cond: !isARMImmRot(uint32(c)) && isARMImmRot(^uint32(c))
	// result: (ANDconst [int64(int32(^uint32(c)))] x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(!isARMImmRot(uint32(c)) && isARMImmRot(^uint32(c))) {
			break
		}
		v.reset(OpARMANDconst)
		v.AuxInt = int64(int32(^uint32(c)))
		v.AddArg(x)
		return true
	}
	// match: (BICconst [c] x)
	// cond: objabi.GOARM==7 && !isARMImmRot(uint32(c)) && uint32(c)>0xffff && ^uint32(c)<=0xffff
	// result: (ANDconst [int64(int32(^uint32(c)))] x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(objabi.GOARM == 7 && !isARMImmRot(uint32(c)) && uint32(c) > 0xffff && ^uint32(c) <= 0xffff) {
			break
		}
		v.reset(OpARMANDconst)
		v.AuxInt = int64(int32(^uint32(c)))
		v.AddArg(x)
		return true
	}
	// match: (BICconst [c] (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [d&^c])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = d &^ c
		return true
	}
	// match: (BICconst [c] (BICconst [d] x))
	// cond:
	// result: (BICconst [int64(int32(c|d))] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMBICconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMBICconst)
		v.AuxInt = int64(int32(c | d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICshiftLL_0(v *Value) bool {
	// match: (BICshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (BICconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMBICconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVWconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICshiftLLreg_0(v *Value) bool {
	// match: (BICshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (BICshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMBICshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICshiftRA_0(v *Value) bool {
	// match: (BICshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (BICconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMBICconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVWconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICshiftRAreg_0(v *Value) bool {
	// match: (BICshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (BICshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMBICshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICshiftRL_0(v *Value) bool {
	// match: (BICshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (BICconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMBICconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVWconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICshiftRLreg_0(v *Value) bool {
	// match: (BICshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (BICshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMBICshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMN_0(v *Value) bool {
	// match: (CMN x (MOVWconst [c]))
	// cond:
	// result: (CMNconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMCMNconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (CMN (MOVWconst [c]) x)
	// cond:
	// result: (CMNconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMCMNconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (CMN x (SLLconst [c] y))
	// cond:
	// result: (CMNshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMCMNshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMN (SLLconst [c] y) x)
	// cond:
	// result: (CMNshiftLL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMCMNshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMN x (SRLconst [c] y))
	// cond:
	// result: (CMNshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMCMNshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMN (SRLconst [c] y) x)
	// cond:
	// result: (CMNshiftRL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMCMNshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMN x (SRAconst [c] y))
	// cond:
	// result: (CMNshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMCMNshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMN (SRAconst [c] y) x)
	// cond:
	// result: (CMNshiftRA x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMCMNshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMN x (SLL y z))
	// cond:
	// result: (CMNshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMCMNshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (CMN (SLL y z) x)
	// cond:
	// result: (CMNshiftLLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMCMNshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMN_10(v *Value) bool {
	// match: (CMN x (SRL y z))
	// cond:
	// result: (CMNshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMCMNshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (CMN (SRL y z) x)
	// cond:
	// result: (CMNshiftRLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMCMNshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (CMN x (SRA y z))
	// cond:
	// result: (CMNshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMCMNshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (CMN (SRA y z) x)
	// cond:
	// result: (CMNshiftRAreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMCMNshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (CMN x (RSBconst [0] y))
	// cond:
	// result: (CMP x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMRSBconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpARMCMP)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMN (RSBconst [0] y) x)
	// cond:
	// result: (CMP x y)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMRSBconst {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		y := v_0.Args[0]
		v.reset(OpARMCMP)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNconst_0(v *Value) bool {
	// match: (CMNconst (MOVWconst [x]) [y])
	// cond: int32(x)==int32(-y)
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) == int32(-y)) {
			break
		}
		v.reset(OpARMFlagEQ)
		return true
	}
	// match: (CMNconst (MOVWconst [x]) [y])
	// cond: int32(x)<int32(-y) && uint32(x)<uint32(-y)
	// result: (FlagLT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) < int32(-y) && uint32(x) < uint32(-y)) {
			break
		}
		v.reset(OpARMFlagLT_ULT)
		return true
	}
	// match: (CMNconst (MOVWconst [x]) [y])
	// cond: int32(x)<int32(-y) && uint32(x)>uint32(-y)
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) < int32(-y) && uint32(x) > uint32(-y)) {
			break
		}
		v.reset(OpARMFlagLT_UGT)
		return true
	}
	// match: (CMNconst (MOVWconst [x]) [y])
	// cond: int32(x)>int32(-y) && uint32(x)<uint32(-y)
	// result: (FlagGT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) > int32(-y) && uint32(x) < uint32(-y)) {
			break
		}
		v.reset(OpARMFlagGT_ULT)
		return true
	}
	// match: (CMNconst (MOVWconst [x]) [y])
	// cond: int32(x)>int32(-y) && uint32(x)>uint32(-y)
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) > int32(-y) && uint32(x) > uint32(-y)) {
			break
		}
		v.reset(OpARMFlagGT_UGT)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (CMNshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (CMNconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMCMNconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (CMNconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMCMNconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (CMNshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (CMNconst [c] (SLL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMCMNconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (CMNshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMCMNshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (CMNshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (CMNconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMCMNconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (CMNconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMCMNconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (CMNshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (CMNconst [c] (SRA <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMCMNconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (CMNshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMCMNshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (CMNshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (CMNconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMCMNconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (CMNconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMCMNconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (CMNshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (CMNconst [c] (SRL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMCMNconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (CMNshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMCMNshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMOVWHSconst_0(v *Value) bool {
	// match: (CMOVWHSconst _ (FlagEQ) [c])
	// cond:
	// result: (MOVWconst [c])
	for {
		c := v.AuxInt
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = c
		return true
	}
	// match: (CMOVWHSconst x (FlagLT_ULT))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (CMOVWHSconst _ (FlagLT_UGT) [c])
	// cond:
	// result: (MOVWconst [c])
	for {
		c := v.AuxInt
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = c
		return true
	}
	// match: (CMOVWHSconst x (FlagGT_ULT))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (CMOVWHSconst _ (FlagGT_UGT) [c])
	// cond:
	// result: (MOVWconst [c])
	for {
		c := v.AuxInt
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = c
		return true
	}
	// match: (CMOVWHSconst x (InvertFlags flags) [c])
	// cond:
	// result: (CMOVWLSconst x flags [c])
	for {
		c := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMInvertFlags {
			break
		}
		flags := v_1.Args[0]
		v.reset(OpARMCMOVWLSconst)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMOVWLSconst_0(v *Value) bool {
	// match: (CMOVWLSconst _ (FlagEQ) [c])
	// cond:
	// result: (MOVWconst [c])
	for {
		c := v.AuxInt
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = c
		return true
	}
	// match: (CMOVWLSconst _ (FlagLT_ULT) [c])
	// cond:
	// result: (MOVWconst [c])
	for {
		c := v.AuxInt
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = c
		return true
	}
	// match: (CMOVWLSconst x (FlagLT_UGT))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (CMOVWLSconst _ (FlagGT_ULT) [c])
	// cond:
	// result: (MOVWconst [c])
	for {
		c := v.AuxInt
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = c
		return true
	}
	// match: (CMOVWLSconst x (FlagGT_UGT))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (CMOVWLSconst x (InvertFlags flags) [c])
	// cond:
	// result: (CMOVWHSconst x flags [c])
	for {
		c := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMInvertFlags {
			break
		}
		flags := v_1.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMP_0(v *Value) bool {
	b := v.Block
	// match: (CMP x (MOVWconst [c]))
	// cond:
	// result: (CMPconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMCMPconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (CMP (MOVWconst [c]) x)
	// cond:
	// result: (InvertFlags (CMPconst [c] x))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = c
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x (SLLconst [c] y))
	// cond:
	// result: (CMPshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMCMPshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMP (SLLconst [c] y) x)
	// cond:
	// result: (InvertFlags (CMPshiftLL x y [c]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPshiftLL, types.TypeFlags)
		v0.AuxInt = c
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x (SRLconst [c] y))
	// cond:
	// result: (CMPshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMCMPshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMP (SRLconst [c] y) x)
	// cond:
	// result: (InvertFlags (CMPshiftRL x y [c]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPshiftRL, types.TypeFlags)
		v0.AuxInt = c
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x (SRAconst [c] y))
	// cond:
	// result: (CMPshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMCMPshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMP (SRAconst [c] y) x)
	// cond:
	// result: (InvertFlags (CMPshiftRA x y [c]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPshiftRA, types.TypeFlags)
		v0.AuxInt = c
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x (SLL y z))
	// cond:
	// result: (CMPshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMCMPshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (CMP (SLL y z) x)
	// cond:
	// result: (InvertFlags (CMPshiftLLreg x y z))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPshiftLLreg, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMP_10(v *Value) bool {
	b := v.Block
	// match: (CMP x (SRL y z))
	// cond:
	// result: (CMPshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMCMPshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (CMP (SRL y z) x)
	// cond:
	// result: (InvertFlags (CMPshiftRLreg x y z))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPshiftRLreg, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x (SRA y z))
	// cond:
	// result: (CMPshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMCMPshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (CMP (SRA y z) x)
	// cond:
	// result: (InvertFlags (CMPshiftRAreg x y z))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPshiftRAreg, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x (RSBconst [0] y))
	// cond:
	// result: (CMN x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMRSBconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpARMCMN)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPD_0(v *Value) bool {
	// match: (CMPD x (MOVDconst [0]))
	// cond:
	// result: (CMPD0 x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpARMCMPD0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPF_0(v *Value) bool {
	// match: (CMPF x (MOVFconst [0]))
	// cond:
	// result: (CMPF0 x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVFconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpARMCMPF0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPconst_0(v *Value) bool {
	// match: (CMPconst (MOVWconst [x]) [y])
	// cond: int32(x)==int32(y)
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) == int32(y)) {
			break
		}
		v.reset(OpARMFlagEQ)
		return true
	}
	// match: (CMPconst (MOVWconst [x]) [y])
	// cond: int32(x)<int32(y) && uint32(x)<uint32(y)
	// result: (FlagLT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) < int32(y) && uint32(x) < uint32(y)) {
			break
		}
		v.reset(OpARMFlagLT_ULT)
		return true
	}
	// match: (CMPconst (MOVWconst [x]) [y])
	// cond: int32(x)<int32(y) && uint32(x)>uint32(y)
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) < int32(y) && uint32(x) > uint32(y)) {
			break
		}
		v.reset(OpARMFlagLT_UGT)
		return true
	}
	// match: (CMPconst (MOVWconst [x]) [y])
	// cond: int32(x)>int32(y) && uint32(x)<uint32(y)
	// result: (FlagGT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) > int32(y) && uint32(x) < uint32(y)) {
			break
		}
		v.reset(OpARMFlagGT_ULT)
		return true
	}
	// match: (CMPconst (MOVWconst [x]) [y])
	// cond: int32(x)>int32(y) && uint32(x)>uint32(y)
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) > int32(y) && uint32(x) > uint32(y)) {
			break
		}
		v.reset(OpARMFlagGT_UGT)
		return true
	}
	// match: (CMPconst (MOVBUreg _) [c])
	// cond: 0xff < c
	// result: (FlagLT_ULT)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVBUreg {
			break
		}
		if !(0xff < c) {
			break
		}
		v.reset(OpARMFlagLT_ULT)
		return true
	}
	// match: (CMPconst (MOVHUreg _) [c])
	// cond: 0xffff < c
	// result: (FlagLT_ULT)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVHUreg {
			break
		}
		if !(0xffff < c) {
			break
		}
		v.reset(OpARMFlagLT_ULT)
		return true
	}
	// match: (CMPconst (ANDconst _ [m]) [n])
	// cond: 0 <= int32(m) && int32(m) < int32(n)
	// result: (FlagLT_ULT)
	for {
		n := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMANDconst {
			break
		}
		m := v_0.AuxInt
		if !(0 <= int32(m) && int32(m) < int32(n)) {
			break
		}
		v.reset(OpARMFlagLT_ULT)
		return true
	}
	// match: (CMPconst (SRLconst _ [c]) [n])
	// cond: 0 <= n && 0 < c && c <= 32 && (1<<uint32(32-c)) <= uint32(n)
	// result: (FlagLT_ULT)
	for {
		n := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		if !(0 <= n && 0 < c && c <= 32 && (1<<uint32(32-c)) <= uint32(n)) {
			break
		}
		v.reset(OpARMFlagLT_ULT)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (CMPshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (InvertFlags (CMPconst [c] (SLLconst <x.Type> x [d])))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = c
		v1 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v1.AuxInt = d
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (CMPconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMCMPconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (CMPshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (InvertFlags (CMPconst [c] (SLL <x.Type> x y)))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = c
		v1 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v1.AddArg(x)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (CMPshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMCMPshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (CMPshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (InvertFlags (CMPconst [c] (SRAconst <x.Type> x [d])))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = c
		v1 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v1.AuxInt = d
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (CMPconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMCMPconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (CMPshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (InvertFlags (CMPconst [c] (SRA <x.Type> x y)))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = c
		v1 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v1.AddArg(x)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (CMPshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMCMPshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (CMPshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (InvertFlags (CMPconst [c] (SRLconst <x.Type> x [d])))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = c
		v1 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v1.AuxInt = d
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (CMPconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMCMPconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (CMPshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (InvertFlags (CMPconst [c] (SRL <x.Type> x y)))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = c
		v1 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v1.AddArg(x)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (CMPshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMCMPshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMEqual_0(v *Value) bool {
	// match: (Equal (FlagEQ))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (Equal (FlagLT_ULT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (Equal (FlagLT_UGT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (Equal (FlagGT_ULT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (Equal (FlagGT_UGT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (Equal (InvertFlags x))
	// cond:
	// result: (Equal x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARMEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMGreaterEqual_0(v *Value) bool {
	// match: (GreaterEqual (FlagEQ))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterEqual (FlagLT_ULT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterEqual (FlagLT_UGT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterEqual (FlagGT_ULT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterEqual (FlagGT_UGT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterEqual (InvertFlags x))
	// cond:
	// result: (LessEqual x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARMLessEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMGreaterEqualU_0(v *Value) bool {
	// match: (GreaterEqualU (FlagEQ))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterEqualU (FlagLT_ULT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterEqualU (FlagLT_UGT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterEqualU (FlagGT_ULT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterEqualU (FlagGT_UGT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterEqualU (InvertFlags x))
	// cond:
	// result: (LessEqualU x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARMLessEqualU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMGreaterThan_0(v *Value) bool {
	// match: (GreaterThan (FlagEQ))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterThan (FlagLT_ULT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterThan (FlagLT_UGT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterThan (FlagGT_ULT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterThan (FlagGT_UGT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterThan (InvertFlags x))
	// cond:
	// result: (LessThan x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARMLessThan)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMGreaterThanU_0(v *Value) bool {
	// match: (GreaterThanU (FlagEQ))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterThanU (FlagLT_ULT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterThanU (FlagLT_UGT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterThanU (FlagGT_ULT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (GreaterThanU (FlagGT_UGT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (GreaterThanU (InvertFlags x))
	// cond:
	// result: (LessThanU x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARMLessThanU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMLessEqual_0(v *Value) bool {
	// match: (LessEqual (FlagEQ))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessEqual (FlagLT_ULT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessEqual (FlagLT_UGT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessEqual (FlagGT_ULT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessEqual (FlagGT_UGT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessEqual (InvertFlags x))
	// cond:
	// result: (GreaterEqual x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARMGreaterEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMLessEqualU_0(v *Value) bool {
	// match: (LessEqualU (FlagEQ))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessEqualU (FlagLT_ULT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessEqualU (FlagLT_UGT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessEqualU (FlagGT_ULT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessEqualU (FlagGT_UGT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessEqualU (InvertFlags x))
	// cond:
	// result: (GreaterEqualU x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARMGreaterEqualU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMLessThan_0(v *Value) bool {
	// match: (LessThan (FlagEQ))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessThan (FlagLT_ULT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessThan (FlagLT_UGT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessThan (FlagGT_ULT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessThan (FlagGT_UGT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessThan (InvertFlags x))
	// cond:
	// result: (GreaterThan x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARMGreaterThan)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMLessThanU_0(v *Value) bool {
	// match: (LessThanU (FlagEQ))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessThanU (FlagLT_ULT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessThanU (FlagLT_UGT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessThanU (FlagGT_ULT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (LessThanU (FlagGT_UGT))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (LessThanU (InvertFlags x))
	// cond:
	// result: (GreaterThanU x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARMGreaterThanU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBUload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVBUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond:
	// result: (MOVBUload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVBUload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBUload [off1] {sym} (SUBconst [off2] ptr) mem)
	// cond:
	// result: (MOVBUload [off1-off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVBUload)
		v.AuxInt = off1 - off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBUload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2)
	// result: (MOVBUload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpARMMOVBUload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBUload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVBUreg x)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVBstore {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[2]
		ptr2 := v_1.Args[0]
		x := v_1.Args[1]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARMMOVBUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUload [0] {sym} (ADD ptr idx) mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVBUloadidx ptr idx mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVBUloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBUload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVWconst [int64(read8(sym, off))])
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
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(read8(sym, off))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBUloadidx_0(v *Value) bool {
	// match: (MOVBUloadidx ptr idx (MOVBstoreidx ptr2 idx x _))
	// cond: isSamePtr(ptr, ptr2)
	// result: (MOVBUreg x)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVBstoreidx {
			break
		}
		_ = v_2.Args[3]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] {
			break
		}
		x := v_2.Args[2]
		if !(isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARMMOVBUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUloadidx ptr (MOVWconst [c]) mem)
	// cond:
	// result: (MOVBUload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMMOVBUload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBUloadidx (MOVWconst [c]) ptr mem)
	// cond:
	// result: (MOVBUload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARMMOVBUload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBUreg_0(v *Value) bool {
	// match: (MOVBUreg x:(MOVBUload _ _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVBUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (ANDconst [c] x))
	// cond:
	// result: (ANDconst [c&0xff] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMANDconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMANDconst)
		v.AuxInt = c & 0xff
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVBUreg {
			break
		}
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (MOVWconst [c]))
	// cond:
	// result: (MOVWconst [int64(uint8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(uint8(c))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVBload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond:
	// result: (MOVBload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVBload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBload [off1] {sym} (SUBconst [off2] ptr) mem)
	// cond:
	// result: (MOVBload [off1-off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVBload)
		v.AuxInt = off1 - off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2)
	// result: (MOVBload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpARMMOVBload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVBreg x)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVBstore {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[2]
		ptr2 := v_1.Args[0]
		x := v_1.Args[1]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARMMOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBload [0] {sym} (ADD ptr idx) mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVBloadidx ptr idx mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVBloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBloadidx_0(v *Value) bool {
	// match: (MOVBloadidx ptr idx (MOVBstoreidx ptr2 idx x _))
	// cond: isSamePtr(ptr, ptr2)
	// result: (MOVBreg x)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVBstoreidx {
			break
		}
		_ = v_2.Args[3]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] {
			break
		}
		x := v_2.Args[2]
		if !(isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARMMOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBloadidx ptr (MOVWconst [c]) mem)
	// cond:
	// result: (MOVBload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMMOVBload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBloadidx (MOVWconst [c]) ptr mem)
	// cond:
	// result: (MOVBload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARMMOVBload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBreg_0(v *Value) bool {
	// match: (MOVBreg x:(MOVBload _ _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVBload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (ANDconst [c] x))
	// cond: c & 0x80 == 0
	// result: (ANDconst [c&0x7f] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMANDconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		if !(c&0x80 == 0) {
			break
		}
		v.reset(OpARMANDconst)
		v.AuxInt = c & 0x7f
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVBreg {
			break
		}
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (MOVWconst [c]))
	// cond:
	// result: (MOVWconst [int64(int8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int8(c))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBstore_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVBstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond:
	// result: (MOVBstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		v.reset(OpARMMOVBstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off1] {sym} (SUBconst [off2] ptr) val mem)
	// cond:
	// result: (MOVBstore [off1-off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		v.reset(OpARMMOVBstore)
		v.AuxInt = off1 - off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2)
	// result: (MOVBstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpARMMOVBstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(val)
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
		if v_1.Op != OpARMMOVBreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARMMOVBstore)
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
		if v_1.Op != OpARMMOVBUreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARMMOVBstore)
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
		if v_1.Op != OpARMMOVHreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARMMOVBstore)
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
		if v_1.Op != OpARMMOVHUreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARMMOVBstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [0] {sym} (ADD ptr idx) val mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVBstoreidx ptr idx val mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVBstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBstoreidx_0(v *Value) bool {
	// match: (MOVBstoreidx ptr (MOVWconst [c]) val mem)
	// cond:
	// result: (MOVBstore [c] ptr val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARMMOVBstore)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx (MOVWconst [c]) ptr val mem)
	// cond:
	// result: (MOVBstore [c] ptr val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		val := v.Args[2]
		v.reset(OpARMMOVBstore)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVDload_0(v *Value) bool {
	// match: (MOVDload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond:
	// result: (MOVDload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVDload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDload [off1] {sym} (SUBconst [off2] ptr) mem)
	// cond:
	// result: (MOVDload [off1-off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVDload)
		v.AuxInt = off1 - off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2)
	// result: (MOVDload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpARMMOVDload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDload [off] {sym} ptr (MOVDstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: x
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVDstore {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[2]
		ptr2 := v_1.Args[0]
		x := v_1.Args[1]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVDstore_0(v *Value) bool {
	// match: (MOVDstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond:
	// result: (MOVDstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		v.reset(OpARMMOVDstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (SUBconst [off2] ptr) val mem)
	// cond:
	// result: (MOVDstore [off1-off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		v.reset(OpARMMOVDstore)
		v.AuxInt = off1 - off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2)
	// result: (MOVDstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpARMMOVDstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVFload_0(v *Value) bool {
	// match: (MOVFload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond:
	// result: (MOVFload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVFload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVFload [off1] {sym} (SUBconst [off2] ptr) mem)
	// cond:
	// result: (MOVFload [off1-off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVFload)
		v.AuxInt = off1 - off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVFload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2)
	// result: (MOVFload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpARMMOVFload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVFload [off] {sym} ptr (MOVFstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: x
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVFstore {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[2]
		ptr2 := v_1.Args[0]
		x := v_1.Args[1]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVFstore_0(v *Value) bool {
	// match: (MOVFstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond:
	// result: (MOVFstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		v.reset(OpARMMOVFstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVFstore [off1] {sym} (SUBconst [off2] ptr) val mem)
	// cond:
	// result: (MOVFstore [off1-off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		v.reset(OpARMMOVFstore)
		v.AuxInt = off1 - off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVFstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2)
	// result: (MOVFstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpARMMOVFstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHUload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVHUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond:
	// result: (MOVHUload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVHUload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUload [off1] {sym} (SUBconst [off2] ptr) mem)
	// cond:
	// result: (MOVHUload [off1-off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVHUload)
		v.AuxInt = off1 - off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2)
	// result: (MOVHUload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpARMMOVHUload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUload [off] {sym} ptr (MOVHstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVHUreg x)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVHstore {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[2]
		ptr2 := v_1.Args[0]
		x := v_1.Args[1]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARMMOVHUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUload [0] {sym} (ADD ptr idx) mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVHUloadidx ptr idx mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVHUloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVWconst [int64(read16(sym, off, config.BigEndian))])
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
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(read16(sym, off, config.BigEndian))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHUloadidx_0(v *Value) bool {
	// match: (MOVHUloadidx ptr idx (MOVHstoreidx ptr2 idx x _))
	// cond: isSamePtr(ptr, ptr2)
	// result: (MOVHUreg x)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVHstoreidx {
			break
		}
		_ = v_2.Args[3]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] {
			break
		}
		x := v_2.Args[2]
		if !(isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARMMOVHUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUloadidx ptr (MOVWconst [c]) mem)
	// cond:
	// result: (MOVHUload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMMOVHUload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUloadidx (MOVWconst [c]) ptr mem)
	// cond:
	// result: (MOVHUload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARMMOVHUload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHUreg_0(v *Value) bool {
	// match: (MOVHUreg x:(MOVBUload _ _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVBUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVHUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (ANDconst [c] x))
	// cond:
	// result: (ANDconst [c&0xffff] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMANDconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMANDconst)
		v.AuxInt = c & 0xffff
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVBUreg {
			break
		}
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVHUreg {
			break
		}
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (MOVWconst [c]))
	// cond:
	// result: (MOVWconst [int64(uint16(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(uint16(c))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVHload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond:
	// result: (MOVHload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVHload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHload [off1] {sym} (SUBconst [off2] ptr) mem)
	// cond:
	// result: (MOVHload [off1-off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVHload)
		v.AuxInt = off1 - off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2)
	// result: (MOVHload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpARMMOVHload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHload [off] {sym} ptr (MOVHstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVHreg x)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVHstore {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[2]
		ptr2 := v_1.Args[0]
		x := v_1.Args[1]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARMMOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHload [0] {sym} (ADD ptr idx) mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVHloadidx ptr idx mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVHloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHloadidx_0(v *Value) bool {
	// match: (MOVHloadidx ptr idx (MOVHstoreidx ptr2 idx x _))
	// cond: isSamePtr(ptr, ptr2)
	// result: (MOVHreg x)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVHstoreidx {
			break
		}
		_ = v_2.Args[3]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] {
			break
		}
		x := v_2.Args[2]
		if !(isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpARMMOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHloadidx ptr (MOVWconst [c]) mem)
	// cond:
	// result: (MOVHload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMMOVHload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHloadidx (MOVWconst [c]) ptr mem)
	// cond:
	// result: (MOVHload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARMMOVHload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHreg_0(v *Value) bool {
	// match: (MOVHreg x:(MOVBload _ _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVBload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVBUload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVHload {
			break
		}
		_ = x.Args[1]
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (ANDconst [c] x))
	// cond: c & 0x8000 == 0
	// result: (ANDconst [c&0x7fff] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMANDconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		if !(c&0x8000 == 0) {
			break
		}
		v.reset(OpARMANDconst)
		v.AuxInt = c & 0x7fff
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVBreg {
			break
		}
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVBUreg {
			break
		}
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// cond:
	// result: (MOVWreg x)
	for {
		x := v.Args[0]
		if x.Op != OpARMMOVHreg {
			break
		}
		v.reset(OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (MOVWconst [c]))
	// cond:
	// result: (MOVWconst [int64(int16(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int16(c))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHstore_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVHstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond:
	// result: (MOVHstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		v.reset(OpARMMOVHstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [off1] {sym} (SUBconst [off2] ptr) val mem)
	// cond:
	// result: (MOVHstore [off1-off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		v.reset(OpARMMOVHstore)
		v.AuxInt = off1 - off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2)
	// result: (MOVHstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpARMMOVHstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(val)
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
		if v_1.Op != OpARMMOVHreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARMMOVHstore)
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
		if v_1.Op != OpARMMOVHUreg {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARMMOVHstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [0] {sym} (ADD ptr idx) val mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVHstoreidx ptr idx val mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVHstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHstoreidx_0(v *Value) bool {
	// match: (MOVHstoreidx ptr (MOVWconst [c]) val mem)
	// cond:
	// result: (MOVHstore [c] ptr val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARMMOVHstore)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstoreidx (MOVWconst [c]) ptr val mem)
	// cond:
	// result: (MOVHstore [c] ptr val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		val := v.Args[2]
		v.reset(OpARMMOVHstore)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWload_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVWload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond:
	// result: (MOVWload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVWload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off1] {sym} (SUBconst [off2] ptr) mem)
	// cond:
	// result: (MOVWload [off1-off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		v.reset(OpARMMOVWload)
		v.AuxInt = off1 - off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2)
	// result: (MOVWload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpARMMOVWload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off] {sym} ptr (MOVWstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: x
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWstore {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		_ = v_1.Args[2]
		ptr2 := v_1.Args[0]
		x := v_1.Args[1]
		if !(sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MOVWload [0] {sym} (ADD ptr idx) mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVWloadidx ptr idx mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVWloadidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [0] {sym} (ADDshiftLL ptr idx [c]) mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVWloadshiftLL ptr idx [c] mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDshiftLL {
			break
		}
		c := v_0.AuxInt
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVWloadshiftLL)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [0] {sym} (ADDshiftRL ptr idx [c]) mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVWloadshiftRL ptr idx [c] mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDshiftRL {
			break
		}
		c := v_0.AuxInt
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVWloadshiftRL)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [0] {sym} (ADDshiftRA ptr idx [c]) mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVWloadshiftRA ptr idx [c] mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDshiftRA {
			break
		}
		c := v_0.AuxInt
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVWloadshiftRA)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off] {sym} (SB) _)
	// cond: symIsRO(sym)
	// result: (MOVWconst [int64(int32(read32(sym, off, config.BigEndian)))])
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
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(read32(sym, off, config.BigEndian)))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWloadidx_0(v *Value) bool {
	// match: (MOVWloadidx ptr idx (MOVWstoreidx ptr2 idx x _))
	// cond: isSamePtr(ptr, ptr2)
	// result: x
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWstoreidx {
			break
		}
		_ = v_2.Args[3]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] {
			break
		}
		x := v_2.Args[2]
		if !(isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MOVWloadidx ptr (MOVWconst [c]) mem)
	// cond:
	// result: (MOVWload [c] ptr mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMMOVWload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx (MOVWconst [c]) ptr mem)
	// cond:
	// result: (MOVWload [c] ptr mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		v.reset(OpARMMOVWload)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx ptr (SLLconst idx [c]) mem)
	// cond:
	// result: (MOVWloadshiftLL ptr idx [c] mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		v.reset(OpARMMOVWloadshiftLL)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx (SLLconst idx [c]) ptr mem)
	// cond:
	// result: (MOVWloadshiftLL ptr idx [c] mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		idx := v_0.Args[0]
		ptr := v.Args[1]
		v.reset(OpARMMOVWloadshiftLL)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx ptr (SRLconst idx [c]) mem)
	// cond:
	// result: (MOVWloadshiftRL ptr idx [c] mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		v.reset(OpARMMOVWloadshiftRL)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx (SRLconst idx [c]) ptr mem)
	// cond:
	// result: (MOVWloadshiftRL ptr idx [c] mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		idx := v_0.Args[0]
		ptr := v.Args[1]
		v.reset(OpARMMOVWloadshiftRL)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx ptr (SRAconst idx [c]) mem)
	// cond:
	// result: (MOVWloadshiftRA ptr idx [c] mem)
	for {
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		v.reset(OpARMMOVWloadshiftRA)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx (SRAconst idx [c]) ptr mem)
	// cond:
	// result: (MOVWloadshiftRA ptr idx [c] mem)
	for {
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		idx := v_0.Args[0]
		ptr := v.Args[1]
		v.reset(OpARMMOVWloadshiftRA)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWloadshiftLL_0(v *Value) bool {
	// match: (MOVWloadshiftLL ptr idx [c] (MOVWstoreshiftLL ptr2 idx [d] x _))
	// cond: c==d && isSamePtr(ptr, ptr2)
	// result: x
	for {
		c := v.AuxInt
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWstoreshiftLL {
			break
		}
		d := v_2.AuxInt
		_ = v_2.Args[3]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] {
			break
		}
		x := v_2.Args[2]
		if !(c == d && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MOVWloadshiftLL ptr (MOVWconst [c]) [d] mem)
	// cond:
	// result: (MOVWload [int64(uint32(c)<<uint64(d))] ptr mem)
	for {
		d := v.AuxInt
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMMOVWload)
		v.AuxInt = int64(uint32(c) << uint64(d))
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWloadshiftRA_0(v *Value) bool {
	// match: (MOVWloadshiftRA ptr idx [c] (MOVWstoreshiftRA ptr2 idx [d] x _))
	// cond: c==d && isSamePtr(ptr, ptr2)
	// result: x
	for {
		c := v.AuxInt
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWstoreshiftRA {
			break
		}
		d := v_2.AuxInt
		_ = v_2.Args[3]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] {
			break
		}
		x := v_2.Args[2]
		if !(c == d && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MOVWloadshiftRA ptr (MOVWconst [c]) [d] mem)
	// cond:
	// result: (MOVWload [int64(int32(c)>>uint64(d))] ptr mem)
	for {
		d := v.AuxInt
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMMOVWload)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWloadshiftRL_0(v *Value) bool {
	// match: (MOVWloadshiftRL ptr idx [c] (MOVWstoreshiftRL ptr2 idx [d] x _))
	// cond: c==d && isSamePtr(ptr, ptr2)
	// result: x
	for {
		c := v.AuxInt
		_ = v.Args[2]
		ptr := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWstoreshiftRL {
			break
		}
		d := v_2.AuxInt
		_ = v_2.Args[3]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] {
			break
		}
		x := v_2.Args[2]
		if !(c == d && isSamePtr(ptr, ptr2)) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MOVWloadshiftRL ptr (MOVWconst [c]) [d] mem)
	// cond:
	// result: (MOVWload [int64(uint32(c)>>uint64(d))] ptr mem)
	for {
		d := v.AuxInt
		mem := v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMMOVWload)
		v.AuxInt = int64(uint32(c) >> uint64(d))
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWreg_0(v *Value) bool {
	// match: (MOVWreg x)
	// cond: x.Uses == 1
	// result: (MOVWnop x)
	for {
		x := v.Args[0]
		if !(x.Uses == 1) {
			break
		}
		v.reset(OpARMMOVWnop)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (MOVWconst [c]))
	// cond:
	// result: (MOVWconst [c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = c
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWstore_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	// match: (MOVWstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond:
	// result: (MOVWstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		v.reset(OpARMMOVWstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (SUBconst [off2] ptr) val mem)
	// cond:
	// result: (MOVWstore [off1-off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		v.reset(OpARMMOVWstore)
		v.AuxInt = off1 - off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2)
	// result: (MOVWstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpARMMOVWstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [0] {sym} (ADD ptr idx) val mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVWstoreidx ptr idx val mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVWstoreidx)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [0] {sym} (ADDshiftLL ptr idx [c]) val mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVWstoreshiftLL ptr idx [c] val mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDshiftLL {
			break
		}
		c := v_0.AuxInt
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVWstoreshiftLL)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [0] {sym} (ADDshiftRL ptr idx [c]) val mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVWstoreshiftRL ptr idx [c] val mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDshiftRL {
			break
		}
		c := v_0.AuxInt
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVWstoreshiftRL)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [0] {sym} (ADDshiftRA ptr idx [c]) val mem)
	// cond: sym == nil && !config.nacl
	// result: (MOVWstoreshiftRA ptr idx [c] val mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDshiftRA {
			break
		}
		c := v_0.AuxInt
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v.Args[1]
		if !(sym == nil && !config.nacl) {
			break
		}
		v.reset(OpARMMOVWstoreshiftRA)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWstoreidx_0(v *Value) bool {
	// match: (MOVWstoreidx ptr (MOVWconst [c]) val mem)
	// cond:
	// result: (MOVWstore [c] ptr val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARMMOVWstore)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx (MOVWconst [c]) ptr val mem)
	// cond:
	// result: (MOVWstore [c] ptr val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		ptr := v.Args[1]
		val := v.Args[2]
		v.reset(OpARMMOVWstore)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx ptr (SLLconst idx [c]) val mem)
	// cond:
	// result: (MOVWstoreshiftLL ptr idx [c] val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		v.reset(OpARMMOVWstoreshiftLL)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx (SLLconst idx [c]) ptr val mem)
	// cond:
	// result: (MOVWstoreshiftLL ptr idx [c] val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		idx := v_0.Args[0]
		ptr := v.Args[1]
		val := v.Args[2]
		v.reset(OpARMMOVWstoreshiftLL)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx ptr (SRLconst idx [c]) val mem)
	// cond:
	// result: (MOVWstoreshiftRL ptr idx [c] val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		v.reset(OpARMMOVWstoreshiftRL)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx (SRLconst idx [c]) ptr val mem)
	// cond:
	// result: (MOVWstoreshiftRL ptr idx [c] val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		idx := v_0.Args[0]
		ptr := v.Args[1]
		val := v.Args[2]
		v.reset(OpARMMOVWstoreshiftRL)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx ptr (SRAconst idx [c]) val mem)
	// cond:
	// result: (MOVWstoreshiftRA ptr idx [c] val mem)
	for {
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		v.reset(OpARMMOVWstoreshiftRA)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx (SRAconst idx [c]) ptr val mem)
	// cond:
	// result: (MOVWstoreshiftRA ptr idx [c] val mem)
	for {
		mem := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		idx := v_0.Args[0]
		ptr := v.Args[1]
		val := v.Args[2]
		v.reset(OpARMMOVWstoreshiftRA)
		v.AuxInt = c
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWstoreshiftLL_0(v *Value) bool {
	// match: (MOVWstoreshiftLL ptr (MOVWconst [c]) [d] val mem)
	// cond:
	// result: (MOVWstore [int64(uint32(c)<<uint64(d))] ptr val mem)
	for {
		d := v.AuxInt
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARMMOVWstore)
		v.AuxInt = int64(uint32(c) << uint64(d))
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWstoreshiftRA_0(v *Value) bool {
	// match: (MOVWstoreshiftRA ptr (MOVWconst [c]) [d] val mem)
	// cond:
	// result: (MOVWstore [int64(int32(c)>>uint64(d))] ptr val mem)
	for {
		d := v.AuxInt
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARMMOVWstore)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWstoreshiftRL_0(v *Value) bool {
	// match: (MOVWstoreshiftRL ptr (MOVWconst [c]) [d] val mem)
	// cond:
	// result: (MOVWstore [int64(uint32(c)>>uint64(d))] ptr val mem)
	for {
		d := v.AuxInt
		mem := v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		val := v.Args[2]
		v.reset(OpARMMOVWstore)
		v.AuxInt = int64(uint32(c) >> uint64(d))
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMUL_0(v *Value) bool {
	// match: (MUL x (MOVWconst [c]))
	// cond: int32(c) == -1
	// result: (RSBconst [0] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARMRSBconst)
		v.AuxInt = 0
		v.AddArg(x)
		return true
	}
	// match: (MUL (MOVWconst [c]) x)
	// cond: int32(c) == -1
	// result: (RSBconst [0] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARMRSBconst)
		v.AuxInt = 0
		v.AddArg(x)
		return true
	}
	// match: (MUL _ (MOVWconst [0]))
	// cond:
	// result: (MOVWconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (MUL (MOVWconst [0]) _)
	// cond:
	// result: (MOVWconst [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (MUL x (MOVWconst [1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
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
	// match: (MUL (MOVWconst [1]) x)
	// cond:
	// result: x
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
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
	// match: (MUL x (MOVWconst [c]))
	// cond: isPowerOfTwo(c)
	// result: (SLLconst [log2(c)] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = log2(c)
		v.AddArg(x)
		return true
	}
	// match: (MUL (MOVWconst [c]) x)
	// cond: isPowerOfTwo(c)
	// result: (SLLconst [log2(c)] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = log2(c)
		v.AddArg(x)
		return true
	}
	// match: (MUL x (MOVWconst [c]))
	// cond: isPowerOfTwo(c-1) && int32(c) >= 3
	// result: (ADDshiftLL x x [log2(c-1)])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARMADDshiftLL)
		v.AuxInt = log2(c - 1)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	// match: (MUL (MOVWconst [c]) x)
	// cond: isPowerOfTwo(c-1) && int32(c) >= 3
	// result: (ADDshiftLL x x [log2(c-1)])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARMADDshiftLL)
		v.AuxInt = log2(c - 1)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMUL_10(v *Value) bool {
	b := v.Block
	// match: (MUL x (MOVWconst [c]))
	// cond: isPowerOfTwo(c+1) && int32(c) >= 7
	// result: (RSBshiftLL x x [log2(c+1)])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARMRSBshiftLL)
		v.AuxInt = log2(c + 1)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	// match: (MUL (MOVWconst [c]) x)
	// cond: isPowerOfTwo(c+1) && int32(c) >= 7
	// result: (RSBshiftLL x x [log2(c+1)])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARMRSBshiftLL)
		v.AuxInt = log2(c + 1)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	// match: (MUL x (MOVWconst [c]))
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (SLLconst [log2(c/3)] (ADDshiftLL <x.Type> x x [1]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = log2(c / 3)
		v0 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v0.AuxInt = 1
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL (MOVWconst [c]) x)
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (SLLconst [log2(c/3)] (ADDshiftLL <x.Type> x x [1]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = log2(c / 3)
		v0 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v0.AuxInt = 1
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL x (MOVWconst [c]))
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (SLLconst [log2(c/5)] (ADDshiftLL <x.Type> x x [2]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = log2(c / 5)
		v0 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL (MOVWconst [c]) x)
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (SLLconst [log2(c/5)] (ADDshiftLL <x.Type> x x [2]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = log2(c / 5)
		v0 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v0.AuxInt = 2
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL x (MOVWconst [c]))
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (SLLconst [log2(c/7)] (RSBshiftLL <x.Type> x x [3]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = log2(c / 7)
		v0 := b.NewValue0(v.Pos, OpARMRSBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL (MOVWconst [c]) x)
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (SLLconst [log2(c/7)] (RSBshiftLL <x.Type> x x [3]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = log2(c / 7)
		v0 := b.NewValue0(v.Pos, OpARMRSBshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL x (MOVWconst [c]))
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (SLLconst [log2(c/9)] (ADDshiftLL <x.Type> x x [3]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = log2(c / 9)
		v0 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MUL (MOVWconst [c]) x)
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (SLLconst [log2(c/9)] (ADDshiftLL <x.Type> x x [3]))
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = log2(c / 9)
		v0 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v0.AuxInt = 3
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMUL_20(v *Value) bool {
	// match: (MUL (MOVWconst [c]) (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [int64(int32(c*d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(c * d))
		return true
	}
	// match: (MUL (MOVWconst [d]) (MOVWconst [c]))
	// cond:
	// result: (MOVWconst [int64(int32(c*d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(c * d))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMULA_0(v *Value) bool {
	b := v.Block
	// match: (MULA x (MOVWconst [c]) a)
	// cond: int32(c) == -1
	// result: (SUB a x)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARMSUB)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MULA _ (MOVWconst [0]) a)
	// cond:
	// result: a
	for {
		a := v.Args[2]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
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
	// match: (MULA x (MOVWconst [1]) a)
	// cond:
	// result: (ADD x a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpARMADD)
		v.AddArg(x)
		v.AddArg(a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: isPowerOfTwo(c)
	// result: (ADD (SLLconst <x.Type> [log2(c)] x) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: isPowerOfTwo(c-1) && int32(c) >= 3
	// result: (ADD (ADDshiftLL <x.Type> x x [log2(c-1)]) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: isPowerOfTwo(c+1) && int32(c) >= 7
	// result: (ADD (RSBshiftLL <x.Type> x x [log2(c+1)]) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMRSBshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (ADD (SLLconst <x.Type> [log2(c/3)] (ADDshiftLL <x.Type> x x [1])) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 3)
		v1 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v1.AuxInt = 1
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (ADD (SLLconst <x.Type> [log2(c/5)] (ADDshiftLL <x.Type> x x [2])) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 5)
		v1 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v1.AuxInt = 2
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (ADD (SLLconst <x.Type> [log2(c/7)] (RSBshiftLL <x.Type> x x [3])) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 7)
		v1 := b.NewValue0(v.Pos, OpARMRSBshiftLL, x.Type)
		v1.AuxInt = 3
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (ADD (SLLconst <x.Type> [log2(c/9)] (ADDshiftLL <x.Type> x x [3])) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 9)
		v1 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v1.AuxInt = 3
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMULA_10(v *Value) bool {
	b := v.Block
	// match: (MULA (MOVWconst [c]) x a)
	// cond: int32(c) == -1
	// result: (SUB a x)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARMSUB)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MULA (MOVWconst [0]) _ a)
	// cond:
	// result: a
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = a.Type
		v.AddArg(a)
		return true
	}
	// match: (MULA (MOVWconst [1]) x a)
	// cond:
	// result: (ADD x a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		x := v.Args[1]
		v.reset(OpARMADD)
		v.AddArg(x)
		v.AddArg(a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: isPowerOfTwo(c)
	// result: (ADD (SLLconst <x.Type> [log2(c)] x) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: isPowerOfTwo(c-1) && int32(c) >= 3
	// result: (ADD (ADDshiftLL <x.Type> x x [log2(c-1)]) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: isPowerOfTwo(c+1) && int32(c) >= 7
	// result: (ADD (RSBshiftLL <x.Type> x x [log2(c+1)]) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMRSBshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (ADD (SLLconst <x.Type> [log2(c/3)] (ADDshiftLL <x.Type> x x [1])) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 3)
		v1 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v1.AuxInt = 1
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (ADD (SLLconst <x.Type> [log2(c/5)] (ADDshiftLL <x.Type> x x [2])) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 5)
		v1 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v1.AuxInt = 2
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (ADD (SLLconst <x.Type> [log2(c/7)] (RSBshiftLL <x.Type> x x [3])) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 7)
		v1 := b.NewValue0(v.Pos, OpARMRSBshiftLL, x.Type)
		v1.AuxInt = 3
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (ADD (SLLconst <x.Type> [log2(c/9)] (ADDshiftLL <x.Type> x x [3])) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 9)
		v1 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v1.AuxInt = 3
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMULA_20(v *Value) bool {
	// match: (MULA (MOVWconst [c]) (MOVWconst [d]) a)
	// cond:
	// result: (ADDconst [int64(int32(c*d))] a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARMADDconst)
		v.AuxInt = int64(int32(c * d))
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMULD_0(v *Value) bool {
	// match: (MULD (NEGD x) y)
	// cond: objabi.GOARM >= 6
	// result: (NMULD x y)
	for {
		y := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMNEGD {
			break
		}
		x := v_0.Args[0]
		if !(objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMNMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (MULD y (NEGD x))
	// cond: objabi.GOARM >= 6
	// result: (NMULD x y)
	for {
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMNEGD {
			break
		}
		x := v_1.Args[0]
		if !(objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMNMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMULF_0(v *Value) bool {
	// match: (MULF (NEGF x) y)
	// cond: objabi.GOARM >= 6
	// result: (NMULF x y)
	for {
		y := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMNEGF {
			break
		}
		x := v_0.Args[0]
		if !(objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMNMULF)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (MULF y (NEGF x))
	// cond: objabi.GOARM >= 6
	// result: (NMULF x y)
	for {
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMNEGF {
			break
		}
		x := v_1.Args[0]
		if !(objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMNMULF)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMULS_0(v *Value) bool {
	b := v.Block
	// match: (MULS x (MOVWconst [c]) a)
	// cond: int32(c) == -1
	// result: (ADD a x)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARMADD)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MULS _ (MOVWconst [0]) a)
	// cond:
	// result: a
	for {
		a := v.Args[2]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
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
	// match: (MULS x (MOVWconst [1]) a)
	// cond:
	// result: (RSB x a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpARMRSB)
		v.AddArg(x)
		v.AddArg(a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: isPowerOfTwo(c)
	// result: (RSB (SLLconst <x.Type> [log2(c)] x) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: isPowerOfTwo(c-1) && int32(c) >= 3
	// result: (RSB (ADDshiftLL <x.Type> x x [log2(c-1)]) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: isPowerOfTwo(c+1) && int32(c) >= 7
	// result: (RSB (RSBshiftLL <x.Type> x x [log2(c+1)]) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMRSBshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (RSB (SLLconst <x.Type> [log2(c/3)] (ADDshiftLL <x.Type> x x [1])) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 3)
		v1 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v1.AuxInt = 1
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (RSB (SLLconst <x.Type> [log2(c/5)] (ADDshiftLL <x.Type> x x [2])) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 5)
		v1 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v1.AuxInt = 2
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (RSB (SLLconst <x.Type> [log2(c/7)] (RSBshiftLL <x.Type> x x [3])) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 7)
		v1 := b.NewValue0(v.Pos, OpARMRSBshiftLL, x.Type)
		v1.AuxInt = 3
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (RSB (SLLconst <x.Type> [log2(c/9)] (ADDshiftLL <x.Type> x x [3])) a)
	for {
		a := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 9)
		v1 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v1.AuxInt = 3
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMULS_10(v *Value) bool {
	b := v.Block
	// match: (MULS (MOVWconst [c]) x a)
	// cond: int32(c) == -1
	// result: (ADD a x)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARMADD)
		v.AddArg(a)
		v.AddArg(x)
		return true
	}
	// match: (MULS (MOVWconst [0]) _ a)
	// cond:
	// result: a
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = a.Type
		v.AddArg(a)
		return true
	}
	// match: (MULS (MOVWconst [1]) x a)
	// cond:
	// result: (RSB x a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		x := v.Args[1]
		v.reset(OpARMRSB)
		v.AddArg(x)
		v.AddArg(a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: isPowerOfTwo(c)
	// result: (RSB (SLLconst <x.Type> [log2(c)] x) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: isPowerOfTwo(c-1) && int32(c) >= 3
	// result: (RSB (ADDshiftLL <x.Type> x x [log2(c-1)]) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: isPowerOfTwo(c+1) && int32(c) >= 7
	// result: (RSB (RSBshiftLL <x.Type> x x [log2(c+1)]) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(isPowerOfTwo(c+1) && int32(c) >= 7) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMRSBshiftLL, x.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)
	// result: (RSB (SLLconst <x.Type> [log2(c/3)] (ADDshiftLL <x.Type> x x [1])) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(c%3 == 0 && isPowerOfTwo(c/3) && is32Bit(c)) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 3)
		v1 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v1.AuxInt = 1
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)
	// result: (RSB (SLLconst <x.Type> [log2(c/5)] (ADDshiftLL <x.Type> x x [2])) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(c%5 == 0 && isPowerOfTwo(c/5) && is32Bit(c)) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 5)
		v1 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v1.AuxInt = 2
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)
	// result: (RSB (SLLconst <x.Type> [log2(c/7)] (RSBshiftLL <x.Type> x x [3])) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(c%7 == 0 && isPowerOfTwo(c/7) && is32Bit(c)) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 7)
		v1 := b.NewValue0(v.Pos, OpARMRSBshiftLL, x.Type)
		v1.AuxInt = 3
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)
	// result: (RSB (SLLconst <x.Type> [log2(c/9)] (ADDshiftLL <x.Type> x x [3])) a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(c%9 == 0 && isPowerOfTwo(c/9) && is32Bit(c)) {
			break
		}
		v.reset(OpARMRSB)
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = log2(c / 9)
		v1 := b.NewValue0(v.Pos, OpARMADDshiftLL, x.Type)
		v1.AuxInt = 3
		v1.AddArg(x)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMULS_20(v *Value) bool {
	// match: (MULS (MOVWconst [c]) (MOVWconst [d]) a)
	// cond:
	// result: (SUBconst [int64(int32(c*d))] a)
	for {
		a := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		d := v_1.AuxInt
		v.reset(OpARMSUBconst)
		v.AuxInt = int64(int32(c * d))
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVN_0(v *Value) bool {
	// match: (MVN (MOVWconst [c]))
	// cond:
	// result: (MOVWconst [^c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = ^c
		return true
	}
	// match: (MVN (SLLconst [c] x))
	// cond:
	// result: (MVNshiftLL x [c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMMVNshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (MVN (SRLconst [c] x))
	// cond:
	// result: (MVNshiftRL x [c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMMVNshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (MVN (SRAconst [c] x))
	// cond:
	// result: (MVNshiftRA x [c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMMVNshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (MVN (SLL x y))
	// cond:
	// result: (MVNshiftLLreg x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARMMVNshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (MVN (SRL x y))
	// cond:
	// result: (MVNshiftRLreg x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARMMVNshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (MVN (SRA x y))
	// cond:
	// result: (MVNshiftRAreg x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpARMMVNshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVNshiftLL_0(v *Value) bool {
	// match: (MVNshiftLL (MOVWconst [c]) [d])
	// cond:
	// result: (MOVWconst [^int64(uint32(c)<<uint64(d))])
	for {
		d := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = ^int64(uint32(c) << uint64(d))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVNshiftLLreg_0(v *Value) bool {
	// match: (MVNshiftLLreg x (MOVWconst [c]))
	// cond:
	// result: (MVNshiftLL x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMMVNshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVNshiftRA_0(v *Value) bool {
	// match: (MVNshiftRA (MOVWconst [c]) [d])
	// cond:
	// result: (MOVWconst [^int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = ^int64(int32(c) >> uint64(d))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVNshiftRAreg_0(v *Value) bool {
	// match: (MVNshiftRAreg x (MOVWconst [c]))
	// cond:
	// result: (MVNshiftRA x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMMVNshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVNshiftRL_0(v *Value) bool {
	// match: (MVNshiftRL (MOVWconst [c]) [d])
	// cond:
	// result: (MOVWconst [^int64(uint32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = ^int64(uint32(c) >> uint64(d))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVNshiftRLreg_0(v *Value) bool {
	// match: (MVNshiftRLreg x (MOVWconst [c]))
	// cond:
	// result: (MVNshiftRL x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMMVNshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMNEGD_0(v *Value) bool {
	// match: (NEGD (MULD x y))
	// cond: objabi.GOARM >= 6
	// result: (NMULD x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMNMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMNEGF_0(v *Value) bool {
	// match: (NEGF (MULF x y))
	// cond: objabi.GOARM >= 6
	// result: (NMULF x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMMULF {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMNMULF)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMNMULD_0(v *Value) bool {
	// match: (NMULD (NEGD x) y)
	// cond:
	// result: (MULD x y)
	for {
		y := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMNEGD {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARMMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (NMULD y (NEGD x))
	// cond:
	// result: (MULD x y)
	for {
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMNEGD {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARMMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMNMULF_0(v *Value) bool {
	// match: (NMULF (NEGF x) y)
	// cond:
	// result: (MULF x y)
	for {
		y := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMNEGF {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARMMULF)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (NMULF y (NEGF x))
	// cond:
	// result: (MULF x y)
	for {
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMNEGF {
			break
		}
		x := v_1.Args[0]
		v.reset(OpARMMULF)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMNotEqual_0(v *Value) bool {
	// match: (NotEqual (FlagEQ))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (NotEqual (FlagLT_ULT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (NotEqual (FlagLT_UGT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (NotEqual (FlagGT_ULT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (NotEqual (FlagGT_UGT))
	// cond:
	// result: (MOVWconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 1
		return true
	}
	// match: (NotEqual (InvertFlags x))
	// cond:
	// result: (NotEqual x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpARMNotEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMOR_0(v *Value) bool {
	// match: (OR x (MOVWconst [c]))
	// cond:
	// result: (ORconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (OR (MOVWconst [c]) x)
	// cond:
	// result: (ORconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (OR x (SLLconst [c] y))
	// cond:
	// result: (ORshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMORshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR (SLLconst [c] y) x)
	// cond:
	// result: (ORshiftLL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMORshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR x (SRLconst [c] y))
	// cond:
	// result: (ORshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMORshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR (SRLconst [c] y) x)
	// cond:
	// result: (ORshiftRL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMORshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR x (SRAconst [c] y))
	// cond:
	// result: (ORshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMORshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR (SRAconst [c] y) x)
	// cond:
	// result: (ORshiftRA x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMORshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR x (SLL y z))
	// cond:
	// result: (ORshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMORshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (OR (SLL y z) x)
	// cond:
	// result: (ORshiftLLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMORshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMOR_10(v *Value) bool {
	// match: (OR x (SRL y z))
	// cond:
	// result: (ORshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMORshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (OR (SRL y z) x)
	// cond:
	// result: (ORshiftRLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMORshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (OR x (SRA y z))
	// cond:
	// result: (ORshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMORshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (OR (SRA y z) x)
	// cond:
	// result: (ORshiftRAreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMORshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
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
	return false
}
func rewriteValueARM_OpARMORconst_0(v *Value) bool {
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
	// match: (ORconst [c] _)
	// cond: int32(c)==-1
	// result: (MOVWconst [-1])
	for {
		c := v.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = -1
		return true
	}
	// match: (ORconst [c] (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [c|d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = c | d
		return true
	}
	// match: (ORconst [c] (ORconst [d] x))
	// cond:
	// result: (ORconst [c|d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMORconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMORconst)
		v.AuxInt = c | d
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMORshiftLL_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ORshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (ORconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (ORconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMORconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL [c] (SRLconst x [32-c]) x)
	// cond:
	// result: (SRRconst [32-c] x)
	for {
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		if v_0.AuxInt != 32-c {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARMSRRconst)
		v.AuxInt = 32 - c
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL <typ.UInt16> [8] (BFXU <typ.UInt16> [armBFAuxInt(8, 8)] x) x)
	// cond:
	// result: (REV16 x)
	for {
		if v.Type != typ.UInt16 {
			break
		}
		if v.AuxInt != 8 {
			break
		}
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMBFXU {
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
		v.reset(OpARMREV16)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL <typ.UInt16> [8] (SRLconst <typ.UInt16> [24] (SLLconst [16] x)) x)
	// cond: objabi.GOARM>=6
	// result: (REV16 x)
	for {
		if v.Type != typ.UInt16 {
			break
		}
		if v.AuxInt != 8 {
			break
		}
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		if v_0.Type != typ.UInt16 {
			break
		}
		if v_0.AuxInt != 24 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARMSLLconst {
			break
		}
		if v_0_0.AuxInt != 16 {
			break
		}
		if x != v_0_0.Args[0] {
			break
		}
		if !(objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMREV16)
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
		if y.Op != OpARMSLLconst {
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
func rewriteValueARM_OpARMORshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (ORshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (ORconst [c] (SLL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (ORshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMORshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMORshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (ORshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (ORconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (ORconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMORconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
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
		if y.Op != OpARMSRAconst {
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
func rewriteValueARM_OpARMORshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (ORshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (ORconst [c] (SRA <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (ORshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMORshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMORshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (ORshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (ORconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (ORconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMORconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRL [c] (SLLconst x [32-c]) x)
	// cond:
	// result: (SRRconst [ c] x)
	for {
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		if v_0.AuxInt != 32-c {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARMSRRconst)
		v.AuxInt = c
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
		if y.Op != OpARMSRLconst {
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
func rewriteValueARM_OpARMORshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (ORshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (ORconst [c] (SRL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (ORshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMORshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSB_0(v *Value) bool {
	// match: (RSB (MOVWconst [c]) x)
	// cond:
	// result: (SUBconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMSUBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (RSB x (MOVWconst [c]))
	// cond:
	// result: (RSBconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMRSBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (RSB x (SLLconst [c] y))
	// cond:
	// result: (RSBshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMRSBshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RSB (SLLconst [c] y) x)
	// cond:
	// result: (SUBshiftLL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMSUBshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RSB x (SRLconst [c] y))
	// cond:
	// result: (RSBshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMRSBshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RSB (SRLconst [c] y) x)
	// cond:
	// result: (SUBshiftRL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMSUBshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RSB x (SRAconst [c] y))
	// cond:
	// result: (RSBshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMRSBshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RSB (SRAconst [c] y) x)
	// cond:
	// result: (SUBshiftRA x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMSUBshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RSB x (SLL y z))
	// cond:
	// result: (RSBshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMRSBshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (RSB (SLL y z) x)
	// cond:
	// result: (SUBshiftLLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMSUBshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSB_10(v *Value) bool {
	// match: (RSB x (SRL y z))
	// cond:
	// result: (RSBshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMRSBshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (RSB (SRL y z) x)
	// cond:
	// result: (SUBshiftRLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMSUBshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (RSB x (SRA y z))
	// cond:
	// result: (RSBshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMRSBshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (RSB (SRA y z) x)
	// cond:
	// result: (SUBshiftRAreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMSUBshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (RSB x x)
	// cond:
	// result: (MOVWconst [0])
	for {
		x := v.Args[1]
		if x != v.Args[0] {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (RSB (MUL x y) a)
	// cond: objabi.GOARM == 7
	// result: (MULS x y a)
	for {
		a := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMUL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(objabi.GOARM == 7) {
			break
		}
		v.reset(OpARMMULS)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBSshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (RSBSshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (SUBSconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMSUBSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (RSBSshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (RSBSconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMRSBSconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBSshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (RSBSshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (SUBSconst [c] (SLL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMSUBSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (RSBSshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (RSBSshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMRSBSshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBSshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (RSBSshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (SUBSconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMSUBSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (RSBSshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (RSBSconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMRSBSconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBSshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (RSBSshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (SUBSconst [c] (SRA <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMSUBSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (RSBSshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (RSBSshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMRSBSshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBSshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (RSBSshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (SUBSconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMSUBSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (RSBSshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (RSBSconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMRSBSconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBSshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (RSBSshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (SUBSconst [c] (SRL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMSUBSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (RSBSshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (RSBSshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMRSBSshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBconst_0(v *Value) bool {
	// match: (RSBconst [c] (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [int64(int32(c-d))])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(c - d))
		return true
	}
	// match: (RSBconst [c] (RSBconst [d] x))
	// cond:
	// result: (ADDconst [int64(int32(c-d))] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMRSBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMADDconst)
		v.AuxInt = int64(int32(c - d))
		v.AddArg(x)
		return true
	}
	// match: (RSBconst [c] (ADDconst [d] x))
	// cond:
	// result: (RSBconst [int64(int32(c-d))] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMRSBconst)
		v.AuxInt = int64(int32(c - d))
		v.AddArg(x)
		return true
	}
	// match: (RSBconst [c] (SUBconst [d] x))
	// cond:
	// result: (RSBconst [int64(int32(c+d))] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMRSBconst)
		v.AuxInt = int64(int32(c + d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (RSBshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (SUBconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMSUBconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (RSBshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (RSBconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMRSBconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (RSBshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVWconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (RSBshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (SUBconst [c] (SLL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMSUBconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (RSBshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (RSBshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMRSBshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (RSBshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (SUBconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMSUBconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (RSBshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (RSBconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMRSBconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (RSBshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVWconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (RSBshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (SUBconst [c] (SRA <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMSUBconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (RSBshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (RSBshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMRSBshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (RSBshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (SUBconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMSUBconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (RSBshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (RSBconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMRSBconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (RSBshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVWconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (RSBshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (SUBconst [c] (SRL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMSUBconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (RSBshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (RSBshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMRSBshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCconst_0(v *Value) bool {
	// match: (RSCconst [c] (ADDconst [d] x) flags)
	// cond:
	// result: (RSCconst [int64(int32(c-d))] x flags)
	for {
		c := v.AuxInt
		flags := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMRSCconst)
		v.AuxInt = int64(int32(c - d))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	// match: (RSCconst [c] (SUBconst [d] x) flags)
	// cond:
	// result: (RSCconst [int64(int32(c+d))] x flags)
	for {
		c := v.AuxInt
		flags := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMRSCconst)
		v.AuxInt = int64(int32(c + d))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (RSCshiftLL (MOVWconst [c]) x [d] flags)
	// cond:
	// result: (SBCconst [c] (SLLconst <x.Type> x [d]) flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMSBCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (RSCshiftLL x (MOVWconst [c]) [d] flags)
	// cond:
	// result: (RSCconst x [int64(int32(uint32(c)<<uint64(d)))] flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMRSCconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (RSCshiftLLreg (MOVWconst [c]) x y flags)
	// cond:
	// result: (SBCconst [c] (SLL <x.Type> x y) flags)
	for {
		flags := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		y := v.Args[2]
		v.reset(OpARMSBCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (RSCshiftLLreg x y (MOVWconst [c]) flags)
	// cond:
	// result: (RSCshiftLL x y [c] flags)
	for {
		flags := v.Args[3]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMRSCshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (RSCshiftRA (MOVWconst [c]) x [d] flags)
	// cond:
	// result: (SBCconst [c] (SRAconst <x.Type> x [d]) flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMSBCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (RSCshiftRA x (MOVWconst [c]) [d] flags)
	// cond:
	// result: (RSCconst x [int64(int32(c)>>uint64(d))] flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMRSCconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (RSCshiftRAreg (MOVWconst [c]) x y flags)
	// cond:
	// result: (SBCconst [c] (SRA <x.Type> x y) flags)
	for {
		flags := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		y := v.Args[2]
		v.reset(OpARMSBCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (RSCshiftRAreg x y (MOVWconst [c]) flags)
	// cond:
	// result: (RSCshiftRA x y [c] flags)
	for {
		flags := v.Args[3]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMRSCshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (RSCshiftRL (MOVWconst [c]) x [d] flags)
	// cond:
	// result: (SBCconst [c] (SRLconst <x.Type> x [d]) flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMSBCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (RSCshiftRL x (MOVWconst [c]) [d] flags)
	// cond:
	// result: (RSCconst x [int64(int32(uint32(c)>>uint64(d)))] flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMRSCconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (RSCshiftRLreg (MOVWconst [c]) x y flags)
	// cond:
	// result: (SBCconst [c] (SRL <x.Type> x y) flags)
	for {
		flags := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		y := v.Args[2]
		v.reset(OpARMSBCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (RSCshiftRLreg x y (MOVWconst [c]) flags)
	// cond:
	// result: (RSCshiftRL x y [c] flags)
	for {
		flags := v.Args[3]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMRSCshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBC_0(v *Value) bool {
	// match: (SBC (MOVWconst [c]) x flags)
	// cond:
	// result: (RSCconst [c] x flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMRSCconst)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	// match: (SBC x (MOVWconst [c]) flags)
	// cond:
	// result: (SBCconst [c] x flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSBCconst)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	// match: (SBC x (SLLconst [c] y) flags)
	// cond:
	// result: (SBCshiftLL x y [c] flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMSBCshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (SBC (SLLconst [c] y) x flags)
	// cond:
	// result: (RSCshiftLL x y [c] flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMRSCshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (SBC x (SRLconst [c] y) flags)
	// cond:
	// result: (SBCshiftRL x y [c] flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMSBCshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (SBC (SRLconst [c] y) x flags)
	// cond:
	// result: (RSCshiftRL x y [c] flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMRSCshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (SBC x (SRAconst [c] y) flags)
	// cond:
	// result: (SBCshiftRA x y [c] flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMSBCshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (SBC (SRAconst [c] y) x flags)
	// cond:
	// result: (RSCshiftRA x y [c] flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMRSCshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	// match: (SBC x (SLL y z) flags)
	// cond:
	// result: (SBCshiftLLreg x y z flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMSBCshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (SBC (SLL y z) x flags)
	// cond:
	// result: (RSCshiftLLreg x y z flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMRSCshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBC_10(v *Value) bool {
	// match: (SBC x (SRL y z) flags)
	// cond:
	// result: (SBCshiftRLreg x y z flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMSBCshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (SBC (SRL y z) x flags)
	// cond:
	// result: (RSCshiftRLreg x y z flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMRSCshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (SBC x (SRA y z) flags)
	// cond:
	// result: (SBCshiftRAreg x y z flags)
	for {
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMSBCshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	// match: (SBC (SRA y z) x flags)
	// cond:
	// result: (RSCshiftRAreg x y z flags)
	for {
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARMRSCshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCconst_0(v *Value) bool {
	// match: (SBCconst [c] (ADDconst [d] x) flags)
	// cond:
	// result: (SBCconst [int64(int32(c-d))] x flags)
	for {
		c := v.AuxInt
		flags := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMSBCconst)
		v.AuxInt = int64(int32(c - d))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	// match: (SBCconst [c] (SUBconst [d] x) flags)
	// cond:
	// result: (SBCconst [int64(int32(c+d))] x flags)
	for {
		c := v.AuxInt
		flags := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMSBCconst)
		v.AuxInt = int64(int32(c + d))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (SBCshiftLL (MOVWconst [c]) x [d] flags)
	// cond:
	// result: (RSCconst [c] (SLLconst <x.Type> x [d]) flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMRSCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (SBCshiftLL x (MOVWconst [c]) [d] flags)
	// cond:
	// result: (SBCconst x [int64(int32(uint32(c)<<uint64(d)))] flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSBCconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (SBCshiftLLreg (MOVWconst [c]) x y flags)
	// cond:
	// result: (RSCconst [c] (SLL <x.Type> x y) flags)
	for {
		flags := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		y := v.Args[2]
		v.reset(OpARMRSCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (SBCshiftLLreg x y (MOVWconst [c]) flags)
	// cond:
	// result: (SBCshiftLL x y [c] flags)
	for {
		flags := v.Args[3]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMSBCshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (SBCshiftRA (MOVWconst [c]) x [d] flags)
	// cond:
	// result: (RSCconst [c] (SRAconst <x.Type> x [d]) flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMRSCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (SBCshiftRA x (MOVWconst [c]) [d] flags)
	// cond:
	// result: (SBCconst x [int64(int32(c)>>uint64(d))] flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSBCconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (SBCshiftRAreg (MOVWconst [c]) x y flags)
	// cond:
	// result: (RSCconst [c] (SRA <x.Type> x y) flags)
	for {
		flags := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		y := v.Args[2]
		v.reset(OpARMRSCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (SBCshiftRAreg x y (MOVWconst [c]) flags)
	// cond:
	// result: (SBCshiftRA x y [c] flags)
	for {
		flags := v.Args[3]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMSBCshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (SBCshiftRL (MOVWconst [c]) x [d] flags)
	// cond:
	// result: (RSCconst [c] (SRLconst <x.Type> x [d]) flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMRSCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (SBCshiftRL x (MOVWconst [c]) [d] flags)
	// cond:
	// result: (SBCconst x [int64(int32(uint32(c)>>uint64(d)))] flags)
	for {
		d := v.AuxInt
		flags := v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSBCconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (SBCshiftRLreg (MOVWconst [c]) x y flags)
	// cond:
	// result: (RSCconst [c] (SRL <x.Type> x y) flags)
	for {
		flags := v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		y := v.Args[2]
		v.reset(OpARMRSCconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v.AddArg(flags)
		return true
	}
	// match: (SBCshiftRLreg x y (MOVWconst [c]) flags)
	// cond:
	// result: (SBCshiftRL x y [c] flags)
	for {
		flags := v.Args[3]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMSBCshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSLL_0(v *Value) bool {
	// match: (SLL x (MOVWconst [c]))
	// cond:
	// result: (SLLconst x [c&31])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSLLconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSLLconst_0(v *Value) bool {
	// match: (SLLconst [c] (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [int64(int32(uint32(d)<<uint64(c)))])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(uint32(d) << uint64(c)))
		return true
	}
	return false
}
func rewriteValueARM_OpARMSRA_0(v *Value) bool {
	// match: (SRA x (MOVWconst [c]))
	// cond:
	// result: (SRAconst x [c&31])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSRAconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSRAcond_0(v *Value) bool {
	// match: (SRAcond x _ (FlagEQ))
	// cond:
	// result: (SRAconst x [31])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpARMFlagEQ {
			break
		}
		v.reset(OpARMSRAconst)
		v.AuxInt = 31
		v.AddArg(x)
		return true
	}
	// match: (SRAcond x y (FlagLT_ULT))
	// cond:
	// result: (SRA x y)
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMFlagLT_ULT {
			break
		}
		v.reset(OpARMSRA)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SRAcond x _ (FlagLT_UGT))
	// cond:
	// result: (SRAconst x [31])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpARMFlagLT_UGT {
			break
		}
		v.reset(OpARMSRAconst)
		v.AuxInt = 31
		v.AddArg(x)
		return true
	}
	// match: (SRAcond x y (FlagGT_ULT))
	// cond:
	// result: (SRA x y)
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMFlagGT_ULT {
			break
		}
		v.reset(OpARMSRA)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SRAcond x _ (FlagGT_UGT))
	// cond:
	// result: (SRAconst x [31])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpARMFlagGT_UGT {
			break
		}
		v.reset(OpARMSRAconst)
		v.AuxInt = 31
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSRAconst_0(v *Value) bool {
	// match: (SRAconst [c] (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [int64(int32(d)>>uint64(c))])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(d) >> uint64(c))
		return true
	}
	// match: (SRAconst (SLLconst x [c]) [d])
	// cond: objabi.GOARM==7 && uint64(d)>=uint64(c) && uint64(d)<=31
	// result: (BFX [(d-c)|(32-d)<<8] x)
	for {
		d := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		if !(objabi.GOARM == 7 && uint64(d) >= uint64(c) && uint64(d) <= 31) {
			break
		}
		v.reset(OpARMBFX)
		v.AuxInt = (d - c) | (32-d)<<8
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSRL_0(v *Value) bool {
	// match: (SRL x (MOVWconst [c]))
	// cond:
	// result: (SRLconst x [c&31])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSRLconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSRLconst_0(v *Value) bool {
	// match: (SRLconst [c] (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [int64(int32(uint32(d)>>uint64(c)))])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(uint32(d) >> uint64(c)))
		return true
	}
	// match: (SRLconst (SLLconst x [c]) [d])
	// cond: objabi.GOARM==7 && uint64(d)>=uint64(c) && uint64(d)<=31
	// result: (BFXU [(d-c)|(32-d)<<8] x)
	for {
		d := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		if !(objabi.GOARM == 7 && uint64(d) >= uint64(c) && uint64(d) <= 31) {
			break
		}
		v.reset(OpARMBFXU)
		v.AuxInt = (d - c) | (32-d)<<8
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUB_0(v *Value) bool {
	// match: (SUB (MOVWconst [c]) x)
	// cond:
	// result: (RSBconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMRSBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (SUB x (MOVWconst [c]))
	// cond:
	// result: (SUBconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSUBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (SUB x (SLLconst [c] y))
	// cond:
	// result: (SUBshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMSUBshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUB (SLLconst [c] y) x)
	// cond:
	// result: (RSBshiftLL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMRSBshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUB x (SRLconst [c] y))
	// cond:
	// result: (SUBshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMSUBshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUB (SRLconst [c] y) x)
	// cond:
	// result: (RSBshiftRL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMRSBshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUB x (SRAconst [c] y))
	// cond:
	// result: (SUBshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMSUBshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUB (SRAconst [c] y) x)
	// cond:
	// result: (RSBshiftRA x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMRSBshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUB x (SLL y z))
	// cond:
	// result: (SUBshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMSUBshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (SUB (SLL y z) x)
	// cond:
	// result: (RSBshiftLLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMRSBshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUB_10(v *Value) bool {
	// match: (SUB x (SRL y z))
	// cond:
	// result: (SUBshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMSUBshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (SUB (SRL y z) x)
	// cond:
	// result: (RSBshiftRLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMRSBshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (SUB x (SRA y z))
	// cond:
	// result: (SUBshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMSUBshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (SUB (SRA y z) x)
	// cond:
	// result: (RSBshiftRAreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMRSBshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (SUB x x)
	// cond:
	// result: (MOVWconst [0])
	for {
		x := v.Args[1]
		if x != v.Args[0] {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (SUB a (MUL x y))
	// cond: objabi.GOARM == 7
	// result: (MULS x y a)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMUL {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(objabi.GOARM == 7) {
			break
		}
		v.reset(OpARMMULS)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBD_0(v *Value) bool {
	// match: (SUBD a (MULD x y))
	// cond: a.Uses == 1 && objabi.GOARM >= 6
	// result: (MULSD a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Uses == 1 && objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMMULSD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUBD a (NMULD x y))
	// cond: a.Uses == 1 && objabi.GOARM >= 6
	// result: (MULAD a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMNMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Uses == 1 && objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMMULAD)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBF_0(v *Value) bool {
	// match: (SUBF a (MULF x y))
	// cond: a.Uses == 1 && objabi.GOARM >= 6
	// result: (MULSF a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMULF {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Uses == 1 && objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMMULSF)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUBF a (NMULF x y))
	// cond: a.Uses == 1 && objabi.GOARM >= 6
	// result: (MULAF a x y)
	for {
		_ = v.Args[1]
		a := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMNMULF {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Uses == 1 && objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMMULAF)
		v.AddArg(a)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBS_0(v *Value) bool {
	// match: (SUBS x (MOVWconst [c]))
	// cond:
	// result: (SUBSconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSUBSconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (SUBS x (SLLconst [c] y))
	// cond:
	// result: (SUBSshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMSUBSshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUBS (SLLconst [c] y) x)
	// cond:
	// result: (RSBSshiftLL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMRSBSshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUBS x (SRLconst [c] y))
	// cond:
	// result: (SUBSshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMSUBSshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUBS (SRLconst [c] y) x)
	// cond:
	// result: (RSBSshiftRL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMRSBSshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUBS x (SRAconst [c] y))
	// cond:
	// result: (SUBSshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMSUBSshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUBS (SRAconst [c] y) x)
	// cond:
	// result: (RSBSshiftRA x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMRSBSshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SUBS x (SLL y z))
	// cond:
	// result: (SUBSshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMSUBSshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (SUBS (SLL y z) x)
	// cond:
	// result: (RSBSshiftLLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMRSBSshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (SUBS x (SRL y z))
	// cond:
	// result: (SUBSshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMSUBSshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBS_10(v *Value) bool {
	// match: (SUBS (SRL y z) x)
	// cond:
	// result: (RSBSshiftRLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMRSBSshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (SUBS x (SRA y z))
	// cond:
	// result: (SUBSshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMSUBSshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (SUBS (SRA y z) x)
	// cond:
	// result: (RSBSshiftRAreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMRSBSshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBSshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (SUBSshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (RSBSconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMRSBSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBSshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (SUBSconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSUBSconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBSshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (SUBSshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (RSBSconst [c] (SLL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMRSBSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SUBSshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (SUBSshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMSUBSshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBSshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (SUBSshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (RSBSconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMRSBSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBSshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (SUBSconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSUBSconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBSshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (SUBSshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (RSBSconst [c] (SRA <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMRSBSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SUBSshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (SUBSshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMSUBSshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBSshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (SUBSshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (RSBSconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMRSBSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBSshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (SUBSconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSUBSconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBSshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (SUBSshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (RSBSconst [c] (SRL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMRSBSconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SUBSshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (SUBSshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMSUBSshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBconst_0(v *Value) bool {
	// match: (SUBconst [off1] (MOVWaddr [off2] {sym} ptr))
	// cond:
	// result: (MOVWaddr [off2-off1] {sym} ptr)
	for {
		off1 := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWaddr {
			break
		}
		off2 := v_0.AuxInt
		sym := v_0.Aux
		ptr := v_0.Args[0]
		v.reset(OpARMMOVWaddr)
		v.AuxInt = off2 - off1
		v.Aux = sym
		v.AddArg(ptr)
		return true
	}
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
	// match: (SUBconst [c] x)
	// cond: !isARMImmRot(uint32(c)) && isARMImmRot(uint32(-c))
	// result: (ADDconst [int64(int32(-c))] x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(!isARMImmRot(uint32(c)) && isARMImmRot(uint32(-c))) {
			break
		}
		v.reset(OpARMADDconst)
		v.AuxInt = int64(int32(-c))
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] x)
	// cond: objabi.GOARM==7 && !isARMImmRot(uint32(c)) && uint32(c)>0xffff && uint32(-c)<=0xffff
	// result: (ANDconst [int64(int32(-c))] x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(objabi.GOARM == 7 && !isARMImmRot(uint32(c)) && uint32(c) > 0xffff && uint32(-c) <= 0xffff) {
			break
		}
		v.reset(OpARMANDconst)
		v.AuxInt = int64(int32(-c))
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [int64(int32(d-c))])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(d - c))
		return true
	}
	// match: (SUBconst [c] (SUBconst [d] x))
	// cond:
	// result: (ADDconst [int64(int32(-c-d))] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMSUBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMADDconst)
		v.AuxInt = int64(int32(-c - d))
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] (ADDconst [d] x))
	// cond:
	// result: (ADDconst [int64(int32(-c+d))] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMADDconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMADDconst)
		v.AuxInt = int64(int32(-c + d))
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] (RSBconst [d] x))
	// cond:
	// result: (RSBconst [int64(int32(-c+d))] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMRSBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMRSBconst)
		v.AuxInt = int64(int32(-c + d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (SUBshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (RSBconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMRSBconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (SUBconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSUBconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVWconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (SUBshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (RSBconst [c] (SLL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMRSBconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SUBshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (SUBshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMSUBshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (SUBshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (RSBconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMRSBconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (SUBconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSUBconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVWconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (SUBshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (RSBconst [c] (SRA <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMRSBconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SUBshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (SUBshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMSUBshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (SUBshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (RSBconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMRSBconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (SUBconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSUBconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVWconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (SUBshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (RSBconst [c] (SRL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMRSBconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SUBshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (SUBshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMSUBshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQ_0(v *Value) bool {
	// match: (TEQ x (MOVWconst [c]))
	// cond:
	// result: (TEQconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMTEQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (TEQ (MOVWconst [c]) x)
	// cond:
	// result: (TEQconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMTEQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (TEQ x (SLLconst [c] y))
	// cond:
	// result: (TEQshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMTEQshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (TEQ (SLLconst [c] y) x)
	// cond:
	// result: (TEQshiftLL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMTEQshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (TEQ x (SRLconst [c] y))
	// cond:
	// result: (TEQshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMTEQshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (TEQ (SRLconst [c] y) x)
	// cond:
	// result: (TEQshiftRL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMTEQshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (TEQ x (SRAconst [c] y))
	// cond:
	// result: (TEQshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMTEQshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (TEQ (SRAconst [c] y) x)
	// cond:
	// result: (TEQshiftRA x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMTEQshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (TEQ x (SLL y z))
	// cond:
	// result: (TEQshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMTEQshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (TEQ (SLL y z) x)
	// cond:
	// result: (TEQshiftLLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMTEQshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQ_10(v *Value) bool {
	// match: (TEQ x (SRL y z))
	// cond:
	// result: (TEQshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMTEQshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (TEQ (SRL y z) x)
	// cond:
	// result: (TEQshiftRLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMTEQshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (TEQ x (SRA y z))
	// cond:
	// result: (TEQshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMTEQshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (TEQ (SRA y z) x)
	// cond:
	// result: (TEQshiftRAreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMTEQshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQconst_0(v *Value) bool {
	// match: (TEQconst (MOVWconst [x]) [y])
	// cond: int32(x^y)==0
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x^y) == 0) {
			break
		}
		v.reset(OpARMFlagEQ)
		return true
	}
	// match: (TEQconst (MOVWconst [x]) [y])
	// cond: int32(x^y)<0
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x^y) < 0) {
			break
		}
		v.reset(OpARMFlagLT_UGT)
		return true
	}
	// match: (TEQconst (MOVWconst [x]) [y])
	// cond: int32(x^y)>0
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x^y) > 0) {
			break
		}
		v.reset(OpARMFlagGT_UGT)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (TEQshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (TEQconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMTEQconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TEQshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (TEQconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMTEQconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (TEQshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (TEQconst [c] (SLL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMTEQconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (TEQshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (TEQshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMTEQshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (TEQshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (TEQconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMTEQconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TEQshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (TEQconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMTEQconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (TEQshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (TEQconst [c] (SRA <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMTEQconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (TEQshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (TEQshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMTEQshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (TEQshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (TEQconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMTEQconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TEQshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (TEQconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMTEQconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (TEQshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (TEQconst [c] (SRL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMTEQconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (TEQshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (TEQshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMTEQshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTST_0(v *Value) bool {
	// match: (TST x (MOVWconst [c]))
	// cond:
	// result: (TSTconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMTSTconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (TST (MOVWconst [c]) x)
	// cond:
	// result: (TSTconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMTSTconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (TST x (SLLconst [c] y))
	// cond:
	// result: (TSTshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMTSTshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (TST (SLLconst [c] y) x)
	// cond:
	// result: (TSTshiftLL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMTSTshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (TST x (SRLconst [c] y))
	// cond:
	// result: (TSTshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMTSTshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (TST (SRLconst [c] y) x)
	// cond:
	// result: (TSTshiftRL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMTSTshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (TST x (SRAconst [c] y))
	// cond:
	// result: (TSTshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMTSTshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (TST (SRAconst [c] y) x)
	// cond:
	// result: (TSTshiftRA x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMTSTshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (TST x (SLL y z))
	// cond:
	// result: (TSTshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMTSTshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (TST (SLL y z) x)
	// cond:
	// result: (TSTshiftLLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMTSTshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTST_10(v *Value) bool {
	// match: (TST x (SRL y z))
	// cond:
	// result: (TSTshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMTSTshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (TST (SRL y z) x)
	// cond:
	// result: (TSTshiftRLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMTSTshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (TST x (SRA y z))
	// cond:
	// result: (TSTshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMTSTshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (TST (SRA y z) x)
	// cond:
	// result: (TSTshiftRAreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMTSTshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTconst_0(v *Value) bool {
	// match: (TSTconst (MOVWconst [x]) [y])
	// cond: int32(x&y)==0
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x&y) == 0) {
			break
		}
		v.reset(OpARMFlagEQ)
		return true
	}
	// match: (TSTconst (MOVWconst [x]) [y])
	// cond: int32(x&y)<0
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x&y) < 0) {
			break
		}
		v.reset(OpARMFlagLT_UGT)
		return true
	}
	// match: (TSTconst (MOVWconst [x]) [y])
	// cond: int32(x&y)>0
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x&y) > 0) {
			break
		}
		v.reset(OpARMFlagGT_UGT)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTshiftLL_0(v *Value) bool {
	b := v.Block
	// match: (TSTshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (TSTconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMTSTconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (TSTconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMTSTconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (TSTshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (TSTconst [c] (SLL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMTSTconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (TSTshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMTSTshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (TSTshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (TSTconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMTSTconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (TSTconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMTSTconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (TSTshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (TSTconst [c] (SRA <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMTSTconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (TSTshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMTSTshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (TSTshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (TSTconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMTSTconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (TSTconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMTSTconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (TSTshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (TSTconst [c] (SRL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMTSTconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (TSTshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMTSTshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXOR_0(v *Value) bool {
	// match: (XOR x (MOVWconst [c]))
	// cond:
	// result: (XORconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMXORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XOR (MOVWconst [c]) x)
	// cond:
	// result: (XORconst [c] x)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMXORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XOR x (SLLconst [c] y))
	// cond:
	// result: (XORshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMXORshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR (SLLconst [c] y) x)
	// cond:
	// result: (XORshiftLL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMXORshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR x (SRLconst [c] y))
	// cond:
	// result: (XORshiftRL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMXORshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR (SRLconst [c] y) x)
	// cond:
	// result: (XORshiftRL x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMXORshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR x (SRAconst [c] y))
	// cond:
	// result: (XORshiftRA x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMXORshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR (SRAconst [c] y) x)
	// cond:
	// result: (XORshiftRA x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMXORshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR x (SRRconst [c] y))
	// cond:
	// result: (XORshiftRR x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRRconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARMXORshiftRR)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR (SRRconst [c] y) x)
	// cond:
	// result: (XORshiftRR x y [c])
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRRconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		v.reset(OpARMXORshiftRR)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXOR_10(v *Value) bool {
	// match: (XOR x (SLL y z))
	// cond:
	// result: (XORshiftLLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMXORshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (XOR (SLL y z) x)
	// cond:
	// result: (XORshiftLLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMXORshiftLLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (XOR x (SRL y z))
	// cond:
	// result: (XORshiftRLreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMXORshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (XOR (SRL y z) x)
	// cond:
	// result: (XORshiftRLreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMXORshiftRLreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (XOR x (SRA y z))
	// cond:
	// result: (XORshiftRAreg x y z)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.reset(OpARMXORshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (XOR (SRA y z) x)
	// cond:
	// result: (XORshiftRAreg x y z)
	for {
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		v.reset(OpARMXORshiftRAreg)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(z)
		return true
	}
	// match: (XOR x x)
	// cond:
	// result: (MOVWconst [0])
	for {
		x := v.Args[1]
		if x != v.Args[0] {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORconst_0(v *Value) bool {
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
	// match: (XORconst [c] (MOVWconst [d]))
	// cond:
	// result: (MOVWconst [c^d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = c ^ d
		return true
	}
	// match: (XORconst [c] (XORconst [d] x))
	// cond:
	// result: (XORconst [c^d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARMXORconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpARMXORconst)
		v.AuxInt = c ^ d
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftLL_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (XORshiftLL (MOVWconst [c]) x [d])
	// cond:
	// result: (XORconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMXORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftLL x (MOVWconst [c]) [d])
	// cond:
	// result: (XORconst x [int64(int32(uint32(c)<<uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMXORconst)
		v.AuxInt = int64(int32(uint32(c) << uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL [c] (SRLconst x [32-c]) x)
	// cond:
	// result: (SRRconst [32-c] x)
	for {
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		if v_0.AuxInt != 32-c {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARMSRRconst)
		v.AuxInt = 32 - c
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL <typ.UInt16> [8] (BFXU <typ.UInt16> [armBFAuxInt(8, 8)] x) x)
	// cond:
	// result: (REV16 x)
	for {
		if v.Type != typ.UInt16 {
			break
		}
		if v.AuxInt != 8 {
			break
		}
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMBFXU {
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
		v.reset(OpARMREV16)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL <typ.UInt16> [8] (SRLconst <typ.UInt16> [24] (SLLconst [16] x)) x)
	// cond: objabi.GOARM>=6
	// result: (REV16 x)
	for {
		if v.Type != typ.UInt16 {
			break
		}
		if v.AuxInt != 8 {
			break
		}
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSRLconst {
			break
		}
		if v_0.Type != typ.UInt16 {
			break
		}
		if v_0.AuxInt != 24 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARMSLLconst {
			break
		}
		if v_0_0.AuxInt != 16 {
			break
		}
		if x != v_0_0.Args[0] {
			break
		}
		if !(objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMREV16)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL x (SLLconst x [c]) [d])
	// cond: c==d
	// result: (MOVWconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftLLreg_0(v *Value) bool {
	b := v.Block
	// match: (XORshiftLLreg (MOVWconst [c]) x y)
	// cond:
	// result: (XORconst [c] (SLL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMXORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftLLreg x y (MOVWconst [c]))
	// cond:
	// result: (XORshiftLL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMXORshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftRA_0(v *Value) bool {
	b := v.Block
	// match: (XORshiftRA (MOVWconst [c]) x [d])
	// cond:
	// result: (XORconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMXORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRAconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRA x (MOVWconst [c]) [d])
	// cond:
	// result: (XORconst x [int64(int32(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMXORconst)
		v.AuxInt = int64(int32(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRA x (SRAconst x [c]) [d])
	// cond: c==d
	// result: (MOVWconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRAconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftRAreg_0(v *Value) bool {
	b := v.Block
	// match: (XORshiftRAreg (MOVWconst [c]) x y)
	// cond:
	// result: (XORconst [c] (SRA <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMXORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRA, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRAreg x y (MOVWconst [c]))
	// cond:
	// result: (XORshiftRA x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMXORshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftRL_0(v *Value) bool {
	b := v.Block
	// match: (XORshiftRL (MOVWconst [c]) x [d])
	// cond:
	// result: (XORconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMXORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRL x (MOVWconst [c]) [d])
	// cond:
	// result: (XORconst x [int64(int32(uint32(c)>>uint64(d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMXORconst)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRL [c] (SLLconst x [32-c]) x)
	// cond:
	// result: (SRRconst [ c] x)
	for {
		c := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMSLLconst {
			break
		}
		if v_0.AuxInt != 32-c {
			break
		}
		if x != v_0.Args[0] {
			break
		}
		v.reset(OpARMSRRconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRL x (SRLconst x [c]) [d])
	// cond: c==d
	// result: (MOVWconst [0])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMSRLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(c == d) {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftRLreg_0(v *Value) bool {
	b := v.Block
	// match: (XORshiftRLreg (MOVWconst [c]) x y)
	// cond:
	// result: (XORconst [c] (SRL <x.Type> x y))
	for {
		y := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARMXORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRLreg x y (MOVWconst [c]))
	// cond:
	// result: (XORshiftRL x y [c])
	for {
		_ = v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARMMOVWconst {
			break
		}
		c := v_2.AuxInt
		v.reset(OpARMXORshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftRR_0(v *Value) bool {
	b := v.Block
	// match: (XORshiftRR (MOVWconst [c]) x [d])
	// cond:
	// result: (XORconst [c] (SRRconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		x := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpARMXORconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpARMSRRconst, x.Type)
		v0.AuxInt = d
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRR x (MOVWconst [c]) [d])
	// cond:
	// result: (XORconst x [int64(int32(uint32(c)>>uint64(d)|uint32(c)<<uint64(32-d)))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMXORconst)
		v.AuxInt = int64(int32(uint32(c)>>uint64(d) | uint32(c)<<uint64(32-d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpAdd16_0(v *Value) bool {
	// match: (Add16 x y)
	// cond:
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpAdd32_0(v *Value) bool {
	// match: (Add32 x y)
	// cond:
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpAdd32F_0(v *Value) bool {
	// match: (Add32F x y)
	// cond:
	// result: (ADDF x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMADDF)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpAdd32carry_0(v *Value) bool {
	// match: (Add32carry x y)
	// cond:
	// result: (ADDS x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMADDS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpAdd32withcarry_0(v *Value) bool {
	// match: (Add32withcarry x y c)
	// cond:
	// result: (ADC x y c)
	for {
		c := v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARMADC)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(c)
		return true
	}
}
func rewriteValueARM_OpAdd64F_0(v *Value) bool {
	// match: (Add64F x y)
	// cond:
	// result: (ADDD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMADDD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpAdd8_0(v *Value) bool {
	// match: (Add8 x y)
	// cond:
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpAddPtr_0(v *Value) bool {
	// match: (AddPtr x y)
	// cond:
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpAddr_0(v *Value) bool {
	// match: (Addr {sym} base)
	// cond:
	// result: (MOVWaddr {sym} base)
	for {
		sym := v.Aux
		base := v.Args[0]
		v.reset(OpARMMOVWaddr)
		v.Aux = sym
		v.AddArg(base)
		return true
	}
}
func rewriteValueARM_OpAnd16_0(v *Value) bool {
	// match: (And16 x y)
	// cond:
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMAND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpAnd32_0(v *Value) bool {
	// match: (And32 x y)
	// cond:
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMAND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpAnd8_0(v *Value) bool {
	// match: (And8 x y)
	// cond:
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMAND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpAndB_0(v *Value) bool {
	// match: (AndB x y)
	// cond:
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMAND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpAvg32u_0(v *Value) bool {
	b := v.Block
	// match: (Avg32u <t> x y)
	// cond:
	// result: (ADD (SRLconst <t> (SUB <t> x y) [1]) y)
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMADD)
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, t)
		v0.AuxInt = 1
		v1 := b.NewValue0(v.Pos, OpARMSUB, t)
		v1.AddArg(x)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpBitLen32_0(v *Value) bool {
	b := v.Block
	// match: (BitLen32 <t> x)
	// cond:
	// result: (RSBconst [32] (CLZ <t> x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpARMRSBconst)
		v.AuxInt = 32
		v0 := b.NewValue0(v.Pos, OpARMCLZ, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpBswap32_0(v *Value) bool {
	b := v.Block
	// match: (Bswap32 <t> x)
	// cond: objabi.GOARM==5
	// result: (XOR <t> (SRLconst <t> (BICconst <t> (XOR <t> x (SRRconst <t> [16] x)) [0xff0000]) [8]) (SRRconst <t> x [8]))
	for {
		t := v.Type
		x := v.Args[0]
		if !(objabi.GOARM == 5) {
			break
		}
		v.reset(OpARMXOR)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARMSRLconst, t)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpARMBICconst, t)
		v1.AuxInt = 0xff0000
		v2 := b.NewValue0(v.Pos, OpARMXOR, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpARMSRRconst, t)
		v3.AuxInt = 16
		v3.AddArg(x)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		v4 := b.NewValue0(v.Pos, OpARMSRRconst, t)
		v4.AuxInt = 8
		v4.AddArg(x)
		v.AddArg(v4)
		return true
	}
	// match: (Bswap32 x)
	// cond: objabi.GOARM>=6
	// result: (REV x)
	for {
		x := v.Args[0]
		if !(objabi.GOARM >= 6) {
			break
		}
		v.reset(OpARMREV)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpClosureCall_0(v *Value) bool {
	// match: (ClosureCall [argwid] entry closure mem)
	// cond:
	// result: (CALLclosure [argwid] entry closure mem)
	for {
		argwid := v.AuxInt
		mem := v.Args[2]
		entry := v.Args[0]
		closure := v.Args[1]
		v.reset(OpARMCALLclosure)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(closure)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM_OpCom16_0(v *Value) bool {
	// match: (Com16 x)
	// cond:
	// result: (MVN x)
	for {
		x := v.Args[0]
		v.reset(OpARMMVN)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpCom32_0(v *Value) bool {
	// match: (Com32 x)
	// cond:
	// result: (MVN x)
	for {
		x := v.Args[0]
		v.reset(OpARMMVN)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpCom8_0(v *Value) bool {
	// match: (Com8 x)
	// cond:
	// result: (MVN x)
	for {
		x := v.Args[0]
		v.reset(OpARMMVN)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpConst16_0(v *Value) bool {
	// match: (Const16 [val])
	// cond:
	// result: (MOVWconst [val])
	for {
		val := v.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueARM_OpConst32_0(v *Value) bool {
	// match: (Const32 [val])
	// cond:
	// result: (MOVWconst [val])
	for {
		val := v.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueARM_OpConst32F_0(v *Value) bool {
	// match: (Const32F [val])
	// cond:
	// result: (MOVFconst [val])
	for {
		val := v.AuxInt
		v.reset(OpARMMOVFconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueARM_OpConst64F_0(v *Value) bool {
	// match: (Const64F [val])
	// cond:
	// result: (MOVDconst [val])
	for {
		val := v.AuxInt
		v.reset(OpARMMOVDconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueARM_OpConst8_0(v *Value) bool {
	// match: (Const8 [val])
	// cond:
	// result: (MOVWconst [val])
	for {
		val := v.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueARM_OpConstBool_0(v *Value) bool {
	// match: (ConstBool [b])
	// cond:
	// result: (MOVWconst [b])
	for {
		b := v.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = b
		return true
	}
}
func rewriteValueARM_OpConstNil_0(v *Value) bool {
	// match: (ConstNil)
	// cond:
	// result: (MOVWconst [0])
	for {
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
}
func rewriteValueARM_OpCtz16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 <t> x)
	// cond: objabi.GOARM<=6
	// result: (RSBconst [32] (CLZ <t> (SUBconst <typ.UInt32> (AND <typ.UInt32> (ORconst <typ.UInt32> [0x10000] x) (RSBconst <typ.UInt32> [0] (ORconst <typ.UInt32> [0x10000] x))) [1])))
	for {
		t := v.Type
		x := v.Args[0]
		if !(objabi.GOARM <= 6) {
			break
		}
		v.reset(OpARMRSBconst)
		v.AuxInt = 32
		v0 := b.NewValue0(v.Pos, OpARMCLZ, t)
		v1 := b.NewValue0(v.Pos, OpARMSUBconst, typ.UInt32)
		v1.AuxInt = 1
		v2 := b.NewValue0(v.Pos, OpARMAND, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpARMORconst, typ.UInt32)
		v3.AuxInt = 0x10000
		v3.AddArg(x)
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARMRSBconst, typ.UInt32)
		v4.AuxInt = 0
		v5 := b.NewValue0(v.Pos, OpARMORconst, typ.UInt32)
		v5.AuxInt = 0x10000
		v5.AddArg(x)
		v4.AddArg(v5)
		v2.AddArg(v4)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Ctz16 <t> x)
	// cond: objabi.GOARM==7
	// result: (CLZ <t> (RBIT <typ.UInt32> (ORconst <typ.UInt32> [0x10000] x)))
	for {
		t := v.Type
		x := v.Args[0]
		if !(objabi.GOARM == 7) {
			break
		}
		v.reset(OpARMCLZ)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARMRBIT, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpARMORconst, typ.UInt32)
		v1.AuxInt = 0x10000
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpCtz16NonZero_0(v *Value) bool {
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
func rewriteValueARM_OpCtz32_0(v *Value) bool {
	b := v.Block
	// match: (Ctz32 <t> x)
	// cond: objabi.GOARM<=6
	// result: (RSBconst [32] (CLZ <t> (SUBconst <t> (AND <t> x (RSBconst <t> [0] x)) [1])))
	for {
		t := v.Type
		x := v.Args[0]
		if !(objabi.GOARM <= 6) {
			break
		}
		v.reset(OpARMRSBconst)
		v.AuxInt = 32
		v0 := b.NewValue0(v.Pos, OpARMCLZ, t)
		v1 := b.NewValue0(v.Pos, OpARMSUBconst, t)
		v1.AuxInt = 1
		v2 := b.NewValue0(v.Pos, OpARMAND, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpARMRSBconst, t)
		v3.AuxInt = 0
		v3.AddArg(x)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Ctz32 <t> x)
	// cond: objabi.GOARM==7
	// result: (CLZ <t> (RBIT <t> x))
	for {
		t := v.Type
		x := v.Args[0]
		if !(objabi.GOARM == 7) {
			break
		}
		v.reset(OpARMCLZ)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARMRBIT, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpCtz32NonZero_0(v *Value) bool {
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
func rewriteValueARM_OpCtz8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 <t> x)
	// cond: objabi.GOARM<=6
	// result: (RSBconst [32] (CLZ <t> (SUBconst <typ.UInt32> (AND <typ.UInt32> (ORconst <typ.UInt32> [0x100] x) (RSBconst <typ.UInt32> [0] (ORconst <typ.UInt32> [0x100] x))) [1])))
	for {
		t := v.Type
		x := v.Args[0]
		if !(objabi.GOARM <= 6) {
			break
		}
		v.reset(OpARMRSBconst)
		v.AuxInt = 32
		v0 := b.NewValue0(v.Pos, OpARMCLZ, t)
		v1 := b.NewValue0(v.Pos, OpARMSUBconst, typ.UInt32)
		v1.AuxInt = 1
		v2 := b.NewValue0(v.Pos, OpARMAND, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpARMORconst, typ.UInt32)
		v3.AuxInt = 0x100
		v3.AddArg(x)
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpARMRSBconst, typ.UInt32)
		v4.AuxInt = 0
		v5 := b.NewValue0(v.Pos, OpARMORconst, typ.UInt32)
		v5.AuxInt = 0x100
		v5.AddArg(x)
		v4.AddArg(v5)
		v2.AddArg(v4)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Ctz8 <t> x)
	// cond: objabi.GOARM==7
	// result: (CLZ <t> (RBIT <typ.UInt32> (ORconst <typ.UInt32> [0x100] x)))
	for {
		t := v.Type
		x := v.Args[0]
		if !(objabi.GOARM == 7) {
			break
		}
		v.reset(OpARMCLZ)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpARMRBIT, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpARMORconst, typ.UInt32)
		v1.AuxInt = 0x100
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpCtz8NonZero_0(v *Value) bool {
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
func rewriteValueARM_OpCvt32Fto32_0(v *Value) bool {
	// match: (Cvt32Fto32 x)
	// cond:
	// result: (MOVFW x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVFW)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpCvt32Fto32U_0(v *Value) bool {
	// match: (Cvt32Fto32U x)
	// cond:
	// result: (MOVFWU x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVFWU)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpCvt32Fto64F_0(v *Value) bool {
	// match: (Cvt32Fto64F x)
	// cond:
	// result: (MOVFD x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVFD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpCvt32Uto32F_0(v *Value) bool {
	// match: (Cvt32Uto32F x)
	// cond:
	// result: (MOVWUF x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVWUF)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpCvt32Uto64F_0(v *Value) bool {
	// match: (Cvt32Uto64F x)
	// cond:
	// result: (MOVWUD x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVWUD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpCvt32to32F_0(v *Value) bool {
	// match: (Cvt32to32F x)
	// cond:
	// result: (MOVWF x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVWF)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpCvt32to64F_0(v *Value) bool {
	// match: (Cvt32to64F x)
	// cond:
	// result: (MOVWD x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVWD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpCvt64Fto32_0(v *Value) bool {
	// match: (Cvt64Fto32 x)
	// cond:
	// result: (MOVDW x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVDW)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpCvt64Fto32F_0(v *Value) bool {
	// match: (Cvt64Fto32F x)
	// cond:
	// result: (MOVDF x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVDF)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpCvt64Fto32U_0(v *Value) bool {
	// match: (Cvt64Fto32U x)
	// cond:
	// result: (MOVDWU x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVDWU)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpDiv16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 x y)
	// cond:
	// result: (Div32 (SignExt16to32 x) (SignExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpDiv32)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpDiv16u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// cond:
	// result: (Div32u (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpDiv32u)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpDiv32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32 x y)
	// cond:
	// result: (SUB (XOR <typ.UInt32> (Select0 <typ.UInt32> (CALLudiv (SUB <typ.UInt32> (XOR x <typ.UInt32> (Signmask x)) (Signmask x)) (SUB <typ.UInt32> (XOR y <typ.UInt32> (Signmask y)) (Signmask y)))) (Signmask (XOR <typ.UInt32> x y))) (Signmask (XOR <typ.UInt32> x y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSUB)
		v0 := b.NewValue0(v.Pos, OpARMXOR, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpSelect0, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpARMCALLudiv, types.NewTuple(typ.UInt32, typ.UInt32))
		v3 := b.NewValue0(v.Pos, OpARMSUB, typ.UInt32)
		v4 := b.NewValue0(v.Pos, OpARMXOR, typ.UInt32)
		v4.AddArg(x)
		v5 := b.NewValue0(v.Pos, OpSignmask, typ.Int32)
		v5.AddArg(x)
		v4.AddArg(v5)
		v3.AddArg(v4)
		v6 := b.NewValue0(v.Pos, OpSignmask, typ.Int32)
		v6.AddArg(x)
		v3.AddArg(v6)
		v2.AddArg(v3)
		v7 := b.NewValue0(v.Pos, OpARMSUB, typ.UInt32)
		v8 := b.NewValue0(v.Pos, OpARMXOR, typ.UInt32)
		v8.AddArg(y)
		v9 := b.NewValue0(v.Pos, OpSignmask, typ.Int32)
		v9.AddArg(y)
		v8.AddArg(v9)
		v7.AddArg(v8)
		v10 := b.NewValue0(v.Pos, OpSignmask, typ.Int32)
		v10.AddArg(y)
		v7.AddArg(v10)
		v2.AddArg(v7)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v11 := b.NewValue0(v.Pos, OpSignmask, typ.Int32)
		v12 := b.NewValue0(v.Pos, OpARMXOR, typ.UInt32)
		v12.AddArg(x)
		v12.AddArg(y)
		v11.AddArg(v12)
		v0.AddArg(v11)
		v.AddArg(v0)
		v13 := b.NewValue0(v.Pos, OpSignmask, typ.Int32)
		v14 := b.NewValue0(v.Pos, OpARMXOR, typ.UInt32)
		v14.AddArg(x)
		v14.AddArg(y)
		v13.AddArg(v14)
		v.AddArg(v13)
		return true
	}
}
func rewriteValueARM_OpDiv32F_0(v *Value) bool {
	// match: (Div32F x y)
	// cond:
	// result: (DIVF x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMDIVF)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpDiv32u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32u x y)
	// cond:
	// result: (Select0 <typ.UInt32> (CALLudiv x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpSelect0)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, OpARMCALLudiv, types.NewTuple(typ.UInt32, typ.UInt32))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpDiv64F_0(v *Value) bool {
	// match: (Div64F x y)
	// cond:
	// result: (DIVD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMDIVD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpDiv8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// cond:
	// result: (Div32 (SignExt8to32 x) (SignExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpDiv32)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpDiv8u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// cond:
	// result: (Div32u (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpDiv32u)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpEq16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// cond:
	// result: (Equal (CMP (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpEq32_0(v *Value) bool {
	b := v.Block
	// match: (Eq32 x y)
	// cond:
	// result: (Equal (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpEq32F_0(v *Value) bool {
	b := v.Block
	// match: (Eq32F x y)
	// cond:
	// result: (Equal (CMPF x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMPF, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpEq64F_0(v *Value) bool {
	b := v.Block
	// match: (Eq64F x y)
	// cond:
	// result: (Equal (CMPD x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMPD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpEq8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// cond:
	// result: (Equal (CMP (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpEqB_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// cond:
	// result: (XORconst [1] (XOR <typ.Bool> x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMXORconst)
		v.AuxInt = 1
		v0 := b.NewValue0(v.Pos, OpARMXOR, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpEqPtr_0(v *Value) bool {
	b := v.Block
	// match: (EqPtr x y)
	// cond:
	// result: (Equal (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpGeq16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq16 x y)
	// cond:
	// result: (GreaterEqual (CMP (SignExt16to32 x) (SignExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpGeq16U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq16U x y)
	// cond:
	// result: (GreaterEqualU (CMP (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterEqualU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpGeq32_0(v *Value) bool {
	b := v.Block
	// match: (Geq32 x y)
	// cond:
	// result: (GreaterEqual (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpGeq32F_0(v *Value) bool {
	b := v.Block
	// match: (Geq32F x y)
	// cond:
	// result: (GreaterEqual (CMPF x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMPF, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpGeq32U_0(v *Value) bool {
	b := v.Block
	// match: (Geq32U x y)
	// cond:
	// result: (GreaterEqualU (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterEqualU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpGeq64F_0(v *Value) bool {
	b := v.Block
	// match: (Geq64F x y)
	// cond:
	// result: (GreaterEqual (CMPD x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMPD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpGeq8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq8 x y)
	// cond:
	// result: (GreaterEqual (CMP (SignExt8to32 x) (SignExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpGeq8U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq8U x y)
	// cond:
	// result: (GreaterEqualU (CMP (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterEqualU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpGetCallerPC_0(v *Value) bool {
	// match: (GetCallerPC)
	// cond:
	// result: (LoweredGetCallerPC)
	for {
		v.reset(OpARMLoweredGetCallerPC)
		return true
	}
}
func rewriteValueARM_OpGetCallerSP_0(v *Value) bool {
	// match: (GetCallerSP)
	// cond:
	// result: (LoweredGetCallerSP)
	for {
		v.reset(OpARMLoweredGetCallerSP)
		return true
	}
}
func rewriteValueARM_OpGetClosurePtr_0(v *Value) bool {
	// match: (GetClosurePtr)
	// cond:
	// result: (LoweredGetClosurePtr)
	for {
		v.reset(OpARMLoweredGetClosurePtr)
		return true
	}
}
func rewriteValueARM_OpGreater16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater16 x y)
	// cond:
	// result: (GreaterThan (CMP (SignExt16to32 x) (SignExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterThan)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpGreater16U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater16U x y)
	// cond:
	// result: (GreaterThanU (CMP (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterThanU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpGreater32_0(v *Value) bool {
	b := v.Block
	// match: (Greater32 x y)
	// cond:
	// result: (GreaterThan (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterThan)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpGreater32F_0(v *Value) bool {
	b := v.Block
	// match: (Greater32F x y)
	// cond:
	// result: (GreaterThan (CMPF x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterThan)
		v0 := b.NewValue0(v.Pos, OpARMCMPF, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpGreater32U_0(v *Value) bool {
	b := v.Block
	// match: (Greater32U x y)
	// cond:
	// result: (GreaterThanU (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterThanU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpGreater64F_0(v *Value) bool {
	b := v.Block
	// match: (Greater64F x y)
	// cond:
	// result: (GreaterThan (CMPD x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterThan)
		v0 := b.NewValue0(v.Pos, OpARMCMPD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpGreater8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater8 x y)
	// cond:
	// result: (GreaterThan (CMP (SignExt8to32 x) (SignExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterThan)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpGreater8U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Greater8U x y)
	// cond:
	// result: (GreaterThanU (CMP (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterThanU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpHmul32_0(v *Value) bool {
	// match: (Hmul32 x y)
	// cond:
	// result: (HMUL x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMHMUL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpHmul32u_0(v *Value) bool {
	// match: (Hmul32u x y)
	// cond:
	// result: (HMULU x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMHMULU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpInterCall_0(v *Value) bool {
	// match: (InterCall [argwid] entry mem)
	// cond:
	// result: (CALLinter [argwid] entry mem)
	for {
		argwid := v.AuxInt
		mem := v.Args[1]
		entry := v.Args[0]
		v.reset(OpARMCALLinter)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM_OpIsInBounds_0(v *Value) bool {
	b := v.Block
	// match: (IsInBounds idx len)
	// cond:
	// result: (LessThanU (CMP idx len))
	for {
		len := v.Args[1]
		idx := v.Args[0]
		v.reset(OpARMLessThanU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(idx)
		v0.AddArg(len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpIsNonNil_0(v *Value) bool {
	b := v.Block
	// match: (IsNonNil ptr)
	// cond:
	// result: (NotEqual (CMPconst [0] ptr))
	for {
		ptr := v.Args[0]
		v.reset(OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = 0
		v0.AddArg(ptr)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpIsSliceInBounds_0(v *Value) bool {
	b := v.Block
	// match: (IsSliceInBounds idx len)
	// cond:
	// result: (LessEqualU (CMP idx len))
	for {
		len := v.Args[1]
		idx := v.Args[0]
		v.reset(OpARMLessEqualU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(idx)
		v0.AddArg(len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// cond:
	// result: (LessEqual (CMP (SignExt16to32 x) (SignExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMLessEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpLeq16U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// cond:
	// result: (LessEqualU (CMP (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMLessEqualU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpLeq32_0(v *Value) bool {
	b := v.Block
	// match: (Leq32 x y)
	// cond:
	// result: (LessEqual (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMLessEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq32F_0(v *Value) bool {
	b := v.Block
	// match: (Leq32F x y)
	// cond:
	// result: (GreaterEqual (CMPF y x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMPF, types.TypeFlags)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq32U_0(v *Value) bool {
	b := v.Block
	// match: (Leq32U x y)
	// cond:
	// result: (LessEqualU (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMLessEqualU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq64F_0(v *Value) bool {
	b := v.Block
	// match: (Leq64F x y)
	// cond:
	// result: (GreaterEqual (CMPD y x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMPD, types.TypeFlags)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// cond:
	// result: (LessEqual (CMP (SignExt8to32 x) (SignExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMLessEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpLeq8U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// cond:
	// result: (LessEqualU (CMP (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMLessEqualU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpLess16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// cond:
	// result: (LessThan (CMP (SignExt16to32 x) (SignExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMLessThan)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpLess16U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// cond:
	// result: (LessThanU (CMP (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMLessThanU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpLess32_0(v *Value) bool {
	b := v.Block
	// match: (Less32 x y)
	// cond:
	// result: (LessThan (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMLessThan)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLess32F_0(v *Value) bool {
	b := v.Block
	// match: (Less32F x y)
	// cond:
	// result: (GreaterThan (CMPF y x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterThan)
		v0 := b.NewValue0(v.Pos, OpARMCMPF, types.TypeFlags)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLess32U_0(v *Value) bool {
	b := v.Block
	// match: (Less32U x y)
	// cond:
	// result: (LessThanU (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMLessThanU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLess64F_0(v *Value) bool {
	b := v.Block
	// match: (Less64F x y)
	// cond:
	// result: (GreaterThan (CMPD y x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMGreaterThan)
		v0 := b.NewValue0(v.Pos, OpARMCMPD, types.TypeFlags)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLess8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// cond:
	// result: (LessThan (CMP (SignExt8to32 x) (SignExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMLessThan)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpLess8U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// cond:
	// result: (LessThanU (CMP (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMLessThanU)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpLoad_0(v *Value) bool {
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
		v.reset(OpARMMOVBUload)
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
		v.reset(OpARMMOVBload)
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
		v.reset(OpARMMOVBUload)
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
		v.reset(OpARMMOVHload)
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
		v.reset(OpARMMOVHUload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is32BitInt(t) || isPtr(t))
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is32BitInt(t) || isPtr(t)) {
			break
		}
		v.reset(OpARMMOVWload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is32BitFloat(t)
	// result: (MOVFload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is32BitFloat(t)) {
			break
		}
		v.reset(OpARMMOVFload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is64BitFloat(t)
	// result: (MOVDload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is64BitFloat(t)) {
			break
		}
		v.reset(OpARMMOVDload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpLocalAddr_0(v *Value) bool {
	// match: (LocalAddr {sym} base _)
	// cond:
	// result: (MOVWaddr {sym} base)
	for {
		sym := v.Aux
		_ = v.Args[1]
		base := v.Args[0]
		v.reset(OpARMMOVWaddr)
		v.Aux = sym
		v.AddArg(base)
		return true
	}
}
func rewriteValueARM_OpLsh16x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 x y)
	// cond:
	// result: (CMOVWHSconst (SLL <x.Type> x (ZeroExt16to32 y)) (CMPconst [256] (ZeroExt16to32 y)) [0])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = 0
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = 256
		v3 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM_OpLsh16x32_0(v *Value) bool {
	b := v.Block
	// match: (Lsh16x32 x y)
	// cond:
	// result: (CMOVWHSconst (SLL <x.Type> x y) (CMPconst [256] y) [0])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = 0
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = 256
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpLsh16x64_0(v *Value) bool {
	// match: (Lsh16x64 x (Const64 [c]))
	// cond: uint64(c) < 16
	// result: (SLLconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 16) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (Lsh16x64 _ (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 16) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpLsh16x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 x y)
	// cond:
	// result: (SLL x (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSLL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLsh32x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 x y)
	// cond:
	// result: (CMOVWHSconst (SLL <x.Type> x (ZeroExt16to32 y)) (CMPconst [256] (ZeroExt16to32 y)) [0])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = 0
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = 256
		v3 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM_OpLsh32x32_0(v *Value) bool {
	b := v.Block
	// match: (Lsh32x32 x y)
	// cond:
	// result: (CMOVWHSconst (SLL <x.Type> x y) (CMPconst [256] y) [0])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = 0
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = 256
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpLsh32x64_0(v *Value) bool {
	// match: (Lsh32x64 x (Const64 [c]))
	// cond: uint64(c) < 32
	// result: (SLLconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 32) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (Lsh32x64 _ (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 32) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpLsh32x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 x y)
	// cond:
	// result: (SLL x (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSLL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLsh8x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 x y)
	// cond:
	// result: (CMOVWHSconst (SLL <x.Type> x (ZeroExt16to32 y)) (CMPconst [256] (ZeroExt16to32 y)) [0])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = 0
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = 256
		v3 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM_OpLsh8x32_0(v *Value) bool {
	b := v.Block
	// match: (Lsh8x32 x y)
	// cond:
	// result: (CMOVWHSconst (SLL <x.Type> x y) (CMPconst [256] y) [0])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = 0
		v0 := b.NewValue0(v.Pos, OpARMSLL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = 256
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpLsh8x64_0(v *Value) bool {
	// match: (Lsh8x64 x (Const64 [c]))
	// cond: uint64(c) < 8
	// result: (SLLconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 8) {
			break
		}
		v.reset(OpARMSLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (Lsh8x64 _ (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (Const8 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 8) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpLsh8x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 x y)
	// cond:
	// result: (SLL x (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSLL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpMod16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// cond:
	// result: (Mod32 (SignExt16to32 x) (SignExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpMod32)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpMod16u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// cond:
	// result: (Mod32u (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpMod32u)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpMod32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32 x y)
	// cond:
	// result: (SUB (XOR <typ.UInt32> (Select1 <typ.UInt32> (CALLudiv (SUB <typ.UInt32> (XOR <typ.UInt32> x (Signmask x)) (Signmask x)) (SUB <typ.UInt32> (XOR <typ.UInt32> y (Signmask y)) (Signmask y)))) (Signmask x)) (Signmask x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSUB)
		v0 := b.NewValue0(v.Pos, OpARMXOR, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpSelect1, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpARMCALLudiv, types.NewTuple(typ.UInt32, typ.UInt32))
		v3 := b.NewValue0(v.Pos, OpARMSUB, typ.UInt32)
		v4 := b.NewValue0(v.Pos, OpARMXOR, typ.UInt32)
		v4.AddArg(x)
		v5 := b.NewValue0(v.Pos, OpSignmask, typ.Int32)
		v5.AddArg(x)
		v4.AddArg(v5)
		v3.AddArg(v4)
		v6 := b.NewValue0(v.Pos, OpSignmask, typ.Int32)
		v6.AddArg(x)
		v3.AddArg(v6)
		v2.AddArg(v3)
		v7 := b.NewValue0(v.Pos, OpARMSUB, typ.UInt32)
		v8 := b.NewValue0(v.Pos, OpARMXOR, typ.UInt32)
		v8.AddArg(y)
		v9 := b.NewValue0(v.Pos, OpSignmask, typ.Int32)
		v9.AddArg(y)
		v8.AddArg(v9)
		v7.AddArg(v8)
		v10 := b.NewValue0(v.Pos, OpSignmask, typ.Int32)
		v10.AddArg(y)
		v7.AddArg(v10)
		v2.AddArg(v7)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v11 := b.NewValue0(v.Pos, OpSignmask, typ.Int32)
		v11.AddArg(x)
		v0.AddArg(v11)
		v.AddArg(v0)
		v12 := b.NewValue0(v.Pos, OpSignmask, typ.Int32)
		v12.AddArg(x)
		v.AddArg(v12)
		return true
	}
}
func rewriteValueARM_OpMod32u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32u x y)
	// cond:
	// result: (Select1 <typ.UInt32> (CALLudiv x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpSelect1)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, OpARMCALLudiv, types.NewTuple(typ.UInt32, typ.UInt32))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpMod8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// cond:
	// result: (Mod32 (SignExt8to32 x) (SignExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpMod32)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpMod8u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// cond:
	// result: (Mod32u (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpMod32u)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpMove_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
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
		v.reset(OpARMMOVBstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARMMOVBUload, typ.UInt8)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [2] {t} dst src mem)
	// cond: t.(*types.Type).Alignment()%2 == 0
	// result: (MOVHstore dst (MOVHUload src mem) mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		t := v.Aux
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		if !(t.(*types.Type).Alignment()%2 == 0) {
			break
		}
		v.reset(OpARMMOVHstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARMMOVHUload, typ.UInt16)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// cond:
	// result: (MOVBstore [1] dst (MOVBUload [1] src mem) (MOVBstore dst (MOVBUload src mem) mem))
	for {
		if v.AuxInt != 2 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARMMOVBstore)
		v.AuxInt = 1
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARMMOVBUload, typ.UInt8)
		v0.AuxInt = 1
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMMOVBstore, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpARMMOVBUload, typ.UInt8)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [4] {t} dst src mem)
	// cond: t.(*types.Type).Alignment()%4 == 0
	// result: (MOVWstore dst (MOVWload src mem) mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		t := v.Aux
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		if !(t.(*types.Type).Alignment()%4 == 0) {
			break
		}
		v.reset(OpARMMOVWstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARMMOVWload, typ.UInt32)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [4] {t} dst src mem)
	// cond: t.(*types.Type).Alignment()%2 == 0
	// result: (MOVHstore [2] dst (MOVHUload [2] src mem) (MOVHstore dst (MOVHUload src mem) mem))
	for {
		if v.AuxInt != 4 {
			break
		}
		t := v.Aux
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		if !(t.(*types.Type).Alignment()%2 == 0) {
			break
		}
		v.reset(OpARMMOVHstore)
		v.AuxInt = 2
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARMMOVHUload, typ.UInt16)
		v0.AuxInt = 2
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMMOVHstore, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpARMMOVHUload, typ.UInt16)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [4] dst src mem)
	// cond:
	// result: (MOVBstore [3] dst (MOVBUload [3] src mem) (MOVBstore [2] dst (MOVBUload [2] src mem) (MOVBstore [1] dst (MOVBUload [1] src mem) (MOVBstore dst (MOVBUload src mem) mem))))
	for {
		if v.AuxInt != 4 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARMMOVBstore)
		v.AuxInt = 3
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARMMOVBUload, typ.UInt8)
		v0.AuxInt = 3
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMMOVBstore, types.TypeMem)
		v1.AuxInt = 2
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpARMMOVBUload, typ.UInt8)
		v2.AuxInt = 2
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARMMOVBstore, types.TypeMem)
		v3.AuxInt = 1
		v3.AddArg(dst)
		v4 := b.NewValue0(v.Pos, OpARMMOVBUload, typ.UInt8)
		v4.AuxInt = 1
		v4.AddArg(src)
		v4.AddArg(mem)
		v3.AddArg(v4)
		v5 := b.NewValue0(v.Pos, OpARMMOVBstore, types.TypeMem)
		v5.AddArg(dst)
		v6 := b.NewValue0(v.Pos, OpARMMOVBUload, typ.UInt8)
		v6.AddArg(src)
		v6.AddArg(mem)
		v5.AddArg(v6)
		v5.AddArg(mem)
		v3.AddArg(v5)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Move [3] dst src mem)
	// cond:
	// result: (MOVBstore [2] dst (MOVBUload [2] src mem) (MOVBstore [1] dst (MOVBUload [1] src mem) (MOVBstore dst (MOVBUload src mem) mem)))
	for {
		if v.AuxInt != 3 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpARMMOVBstore)
		v.AuxInt = 2
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpARMMOVBUload, typ.UInt8)
		v0.AuxInt = 2
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMMOVBstore, types.TypeMem)
		v1.AuxInt = 1
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpARMMOVBUload, typ.UInt8)
		v2.AuxInt = 1
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARMMOVBstore, types.TypeMem)
		v3.AddArg(dst)
		v4 := b.NewValue0(v.Pos, OpARMMOVBUload, typ.UInt8)
		v4.AddArg(src)
		v4.AddArg(mem)
		v3.AddArg(v4)
		v3.AddArg(mem)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Move [s] {t} dst src mem)
	// cond: s%4 == 0 && s > 4 && s <= 512 && t.(*types.Type).Alignment()%4 == 0 && !config.noDuffDevice
	// result: (DUFFCOPY [8 * (128 - s/4)] dst src mem)
	for {
		s := v.AuxInt
		t := v.Aux
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		if !(s%4 == 0 && s > 4 && s <= 512 && t.(*types.Type).Alignment()%4 == 0 && !config.noDuffDevice) {
			break
		}
		v.reset(OpARMDUFFCOPY)
		v.AuxInt = 8 * (128 - s/4)
		v.AddArg(dst)
		v.AddArg(src)
		v.AddArg(mem)
		return true
	}
	// match: (Move [s] {t} dst src mem)
	// cond: (s > 512 || config.noDuffDevice) || t.(*types.Type).Alignment()%4 != 0
	// result: (LoweredMove [t.(*types.Type).Alignment()] dst src (ADDconst <src.Type> src [s-moveSize(t.(*types.Type).Alignment(), config)]) mem)
	for {
		s := v.AuxInt
		t := v.Aux
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		if !((s > 512 || config.noDuffDevice) || t.(*types.Type).Alignment()%4 != 0) {
			break
		}
		v.reset(OpARMLoweredMove)
		v.AuxInt = t.(*types.Type).Alignment()
		v.AddArg(dst)
		v.AddArg(src)
		v0 := b.NewValue0(v.Pos, OpARMADDconst, src.Type)
		v0.AuxInt = s - moveSize(t.(*types.Type).Alignment(), config)
		v0.AddArg(src)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpMul16_0(v *Value) bool {
	// match: (Mul16 x y)
	// cond:
	// result: (MUL x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMMUL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpMul32_0(v *Value) bool {
	// match: (Mul32 x y)
	// cond:
	// result: (MUL x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMMUL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpMul32F_0(v *Value) bool {
	// match: (Mul32F x y)
	// cond:
	// result: (MULF x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMMULF)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpMul32uhilo_0(v *Value) bool {
	// match: (Mul32uhilo x y)
	// cond:
	// result: (MULLU x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMMULLU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpMul64F_0(v *Value) bool {
	// match: (Mul64F x y)
	// cond:
	// result: (MULD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpMul8_0(v *Value) bool {
	// match: (Mul8 x y)
	// cond:
	// result: (MUL x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMMUL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpNeg16_0(v *Value) bool {
	// match: (Neg16 x)
	// cond:
	// result: (RSBconst [0] x)
	for {
		x := v.Args[0]
		v.reset(OpARMRSBconst)
		v.AuxInt = 0
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpNeg32_0(v *Value) bool {
	// match: (Neg32 x)
	// cond:
	// result: (RSBconst [0] x)
	for {
		x := v.Args[0]
		v.reset(OpARMRSBconst)
		v.AuxInt = 0
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpNeg32F_0(v *Value) bool {
	// match: (Neg32F x)
	// cond:
	// result: (NEGF x)
	for {
		x := v.Args[0]
		v.reset(OpARMNEGF)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpNeg64F_0(v *Value) bool {
	// match: (Neg64F x)
	// cond:
	// result: (NEGD x)
	for {
		x := v.Args[0]
		v.reset(OpARMNEGD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpNeg8_0(v *Value) bool {
	// match: (Neg8 x)
	// cond:
	// result: (RSBconst [0] x)
	for {
		x := v.Args[0]
		v.reset(OpARMRSBconst)
		v.AuxInt = 0
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpNeq16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// cond:
	// result: (NotEqual (CMP (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpNeq32_0(v *Value) bool {
	b := v.Block
	// match: (Neq32 x y)
	// cond:
	// result: (NotEqual (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpNeq32F_0(v *Value) bool {
	b := v.Block
	// match: (Neq32F x y)
	// cond:
	// result: (NotEqual (CMPF x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMPF, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpNeq64F_0(v *Value) bool {
	b := v.Block
	// match: (Neq64F x y)
	// cond:
	// result: (NotEqual (CMPD x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMPD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpNeq8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// cond:
	// result: (NotEqual (CMP (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
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
func rewriteValueARM_OpNeqB_0(v *Value) bool {
	// match: (NeqB x y)
	// cond:
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMXOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpNeqPtr_0(v *Value) bool {
	b := v.Block
	// match: (NeqPtr x y)
	// cond:
	// result: (NotEqual (CMP x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpNilCheck_0(v *Value) bool {
	// match: (NilCheck ptr mem)
	// cond:
	// result: (LoweredNilCheck ptr mem)
	for {
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARMLoweredNilCheck)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM_OpNot_0(v *Value) bool {
	// match: (Not x)
	// cond:
	// result: (XORconst [1] x)
	for {
		x := v.Args[0]
		v.reset(OpARMXORconst)
		v.AuxInt = 1
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpOffPtr_0(v *Value) bool {
	// match: (OffPtr [off] ptr:(SP))
	// cond:
	// result: (MOVWaddr [off] ptr)
	for {
		off := v.AuxInt
		ptr := v.Args[0]
		if ptr.Op != OpSP {
			break
		}
		v.reset(OpARMMOVWaddr)
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
		v.reset(OpARMADDconst)
		v.AuxInt = off
		v.AddArg(ptr)
		return true
	}
}
func rewriteValueARM_OpOr16_0(v *Value) bool {
	// match: (Or16 x y)
	// cond:
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpOr32_0(v *Value) bool {
	// match: (Or32 x y)
	// cond:
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpOr8_0(v *Value) bool {
	// match: (Or8 x y)
	// cond:
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpOrB_0(v *Value) bool {
	// match: (OrB x y)
	// cond:
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpPanicBounds_0(v *Value) bool {
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
		v.reset(OpARMLoweredPanicBoundsA)
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
		v.reset(OpARMLoweredPanicBoundsB)
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
		v.reset(OpARMLoweredPanicBoundsC)
		v.AuxInt = kind
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpPanicExtend_0(v *Value) bool {
	// match: (PanicExtend [kind] hi lo y mem)
	// cond: boundsABI(kind) == 0
	// result: (LoweredPanicExtendA [kind] hi lo y mem)
	for {
		kind := v.AuxInt
		mem := v.Args[3]
		hi := v.Args[0]
		lo := v.Args[1]
		y := v.Args[2]
		if !(boundsABI(kind) == 0) {
			break
		}
		v.reset(OpARMLoweredPanicExtendA)
		v.AuxInt = kind
		v.AddArg(hi)
		v.AddArg(lo)
		v.AddArg(y)
		v.AddArg(mem)
		return true
	}
	// match: (PanicExtend [kind] hi lo y mem)
	// cond: boundsABI(kind) == 1
	// result: (LoweredPanicExtendB [kind] hi lo y mem)
	for {
		kind := v.AuxInt
		mem := v.Args[3]
		hi := v.Args[0]
		lo := v.Args[1]
		y := v.Args[2]
		if !(boundsABI(kind) == 1) {
			break
		}
		v.reset(OpARMLoweredPanicExtendB)
		v.AuxInt = kind
		v.AddArg(hi)
		v.AddArg(lo)
		v.AddArg(y)
		v.AddArg(mem)
		return true
	}
	// match: (PanicExtend [kind] hi lo y mem)
	// cond: boundsABI(kind) == 2
	// result: (LoweredPanicExtendC [kind] hi lo y mem)
	for {
		kind := v.AuxInt
		mem := v.Args[3]
		hi := v.Args[0]
		lo := v.Args[1]
		y := v.Args[2]
		if !(boundsABI(kind) == 2) {
			break
		}
		v.reset(OpARMLoweredPanicExtendC)
		v.AuxInt = kind
		v.AddArg(hi)
		v.AddArg(lo)
		v.AddArg(y)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpRotateLeft16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVWconst [c]))
	// cond:
	// result: (Or16 (Lsh16x32 <t> x (MOVWconst [c&15])) (Rsh16Ux32 <t> x (MOVWconst [-c&15])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpLsh16x32, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v1.AuxInt = c & 15
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpRsh16Ux32, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v3.AuxInt = -c & 15
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValueARM_OpRotateLeft32_0(v *Value) bool {
	// match: (RotateLeft32 x (MOVWconst [c]))
	// cond:
	// result: (SRRconst [-c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARMSRRconst)
		v.AuxInt = -c & 31
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpRotateLeft8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVWconst [c]))
	// cond:
	// result: (Or8 (Lsh8x32 <t> x (MOVWconst [c&7])) (Rsh8Ux32 <t> x (MOVWconst [-c&7])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARMMOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpLsh8x32, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v1.AuxInt = c & 7
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpRsh8Ux32, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v3.AuxInt = -c & 7
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValueARM_OpRound32F_0(v *Value) bool {
	// match: (Round32F x)
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
func rewriteValueARM_OpRound64F_0(v *Value) bool {
	// match: (Round64F x)
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
func rewriteValueARM_OpRsh16Ux16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 x y)
	// cond:
	// result: (CMOVWHSconst (SRL <x.Type> (ZeroExt16to32 x) (ZeroExt16to32 y)) (CMPconst [256] (ZeroExt16to32 y)) [0])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = 0
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v3.AuxInt = 256
		v4 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM_OpRsh16Ux32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 x y)
	// cond:
	// result: (CMOVWHSconst (SRL <x.Type> (ZeroExt16to32 x) y) (CMPconst [256] y) [0])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = 0
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = 256
		v2.AddArg(y)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM_OpRsh16Ux64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 x (Const64 [c]))
	// cond: uint64(c) < 16
	// result: (SRLconst (SLLconst <typ.UInt32> x [16]) [c+16])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 16) {
			break
		}
		v.reset(OpARMSRLconst)
		v.AuxInt = c + 16
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, typ.UInt32)
		v0.AuxInt = 16
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 16) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpRsh16Ux8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 x y)
	// cond:
	// result: (SRL (ZeroExt16to32 x) (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpRsh16x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 x y)
	// cond:
	// result: (SRAcond (SignExt16to32 x) (ZeroExt16to32 y) (CMPconst [256] (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSRAcond)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = 256
		v3 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM_OpRsh16x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 x y)
	// cond:
	// result: (SRAcond (SignExt16to32 x) y (CMPconst [256] y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSRAcond)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = 256
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpRsh16x64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 x (Const64 [c]))
	// cond: uint64(c) < 16
	// result: (SRAconst (SLLconst <typ.UInt32> x [16]) [c+16])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 16) {
			break
		}
		v.reset(OpARMSRAconst)
		v.AuxInt = c + 16
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, typ.UInt32)
		v0.AuxInt = 16
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x64 x (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (SRAconst (SLLconst <typ.UInt32> x [16]) [31])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 16) {
			break
		}
		v.reset(OpARMSRAconst)
		v.AuxInt = 31
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, typ.UInt32)
		v0.AuxInt = 16
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpRsh16x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 x y)
	// cond:
	// result: (SRA (SignExt16to32 x) (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpRsh32Ux16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 x y)
	// cond:
	// result: (CMOVWHSconst (SRL <x.Type> x (ZeroExt16to32 y)) (CMPconst [256] (ZeroExt16to32 y)) [0])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = 0
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = 256
		v3 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM_OpRsh32Ux32_0(v *Value) bool {
	b := v.Block
	// match: (Rsh32Ux32 x y)
	// cond:
	// result: (CMOVWHSconst (SRL <x.Type> x y) (CMPconst [256] y) [0])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = 0
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = 256
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpRsh32Ux64_0(v *Value) bool {
	// match: (Rsh32Ux64 x (Const64 [c]))
	// cond: uint64(c) < 32
	// result: (SRLconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 32) {
			break
		}
		v.reset(OpARMSRLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (Rsh32Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 32) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpRsh32Ux8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 x y)
	// cond:
	// result: (SRL x (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSRL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpRsh32x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 x y)
	// cond:
	// result: (SRAcond x (ZeroExt16to32 y) (CMPconst [256] (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSRAcond)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = 256
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpRsh32x32_0(v *Value) bool {
	b := v.Block
	// match: (Rsh32x32 x y)
	// cond:
	// result: (SRAcond x y (CMPconst [256] y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSRAcond)
		v.AddArg(x)
		v.AddArg(y)
		v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = 256
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpRsh32x64_0(v *Value) bool {
	// match: (Rsh32x64 x (Const64 [c]))
	// cond: uint64(c) < 32
	// result: (SRAconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 32) {
			break
		}
		v.reset(OpARMSRAconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (Rsh32x64 x (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (SRAconst x [31])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 32) {
			break
		}
		v.reset(OpARMSRAconst)
		v.AuxInt = 31
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpRsh32x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 x y)
	// cond:
	// result: (SRA x (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSRA)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpRsh8Ux16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 x y)
	// cond:
	// result: (CMOVWHSconst (SRL <x.Type> (ZeroExt8to32 x) (ZeroExt16to32 y)) (CMPconst [256] (ZeroExt16to32 y)) [0])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = 0
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v3.AuxInt = 256
		v4 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueARM_OpRsh8Ux32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 x y)
	// cond:
	// result: (CMOVWHSconst (SRL <x.Type> (ZeroExt8to32 x) y) (CMPconst [256] y) [0])
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMCMOVWHSconst)
		v.AuxInt = 0
		v0 := b.NewValue0(v.Pos, OpARMSRL, x.Type)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = 256
		v2.AddArg(y)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM_OpRsh8Ux64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 x (Const64 [c]))
	// cond: uint64(c) < 8
	// result: (SRLconst (SLLconst <typ.UInt32> x [24]) [c+24])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 8) {
			break
		}
		v.reset(OpARMSRLconst)
		v.AuxInt = c + 24
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, typ.UInt32)
		v0.AuxInt = 24
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (Const8 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 8) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM_OpRsh8Ux8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 x y)
	// cond:
	// result: (SRL (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSRL)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpRsh8x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 x y)
	// cond:
	// result: (SRAcond (SignExt8to32 x) (ZeroExt16to32 y) (CMPconst [256] (ZeroExt16to32 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSRAcond)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = 256
		v3 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueARM_OpRsh8x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 x y)
	// cond:
	// result: (SRAcond (SignExt8to32 x) y (CMPconst [256] y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSRAcond)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = 256
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpRsh8x64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 x (Const64 [c]))
	// cond: uint64(c) < 8
	// result: (SRAconst (SLLconst <typ.UInt32> x [24]) [c+24])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 8) {
			break
		}
		v.reset(OpARMSRAconst)
		v.AuxInt = c + 24
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, typ.UInt32)
		v0.AuxInt = 24
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8x64 x (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (SRAconst (SLLconst <typ.UInt32> x [24]) [31])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 8) {
			break
		}
		v.reset(OpARMSRAconst)
		v.AuxInt = 31
		v0 := b.NewValue0(v.Pos, OpARMSLLconst, typ.UInt32)
		v0.AuxInt = 24
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpRsh8x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 x y)
	// cond:
	// result: (SRA (SignExt8to32 x) (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueARM_OpSelect0_0(v *Value) bool {
	// match: (Select0 (CALLudiv x (MOVWconst [1])))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMCALLudiv {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARMMOVWconst {
			break
		}
		if v_0_1.AuxInt != 1 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Select0 (CALLudiv x (MOVWconst [c])))
	// cond: isPowerOfTwo(c)
	// result: (SRLconst [log2(c)] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMCALLudiv {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARMMOVWconst {
			break
		}
		c := v_0_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARMSRLconst)
		v.AuxInt = log2(c)
		v.AddArg(x)
		return true
	}
	// match: (Select0 (CALLudiv (MOVWconst [c]) (MOVWconst [d])))
	// cond:
	// result: (MOVWconst [int64(int32(uint32(c)/uint32(d)))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMCALLudiv {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0_0.AuxInt
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARMMOVWconst {
			break
		}
		d := v_0_1.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(uint32(c) / uint32(d)))
		return true
	}
	return false
}
func rewriteValueARM_OpSelect1_0(v *Value) bool {
	// match: (Select1 (CALLudiv _ (MOVWconst [1])))
	// cond:
	// result: (MOVWconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMCALLudiv {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARMMOVWconst {
			break
		}
		if v_0_1.AuxInt != 1 {
			break
		}
		v.reset(OpARMMOVWconst)
		v.AuxInt = 0
		return true
	}
	// match: (Select1 (CALLudiv x (MOVWconst [c])))
	// cond: isPowerOfTwo(c)
	// result: (ANDconst [c-1] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMCALLudiv {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARMMOVWconst {
			break
		}
		c := v_0_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARMANDconst)
		v.AuxInt = c - 1
		v.AddArg(x)
		return true
	}
	// match: (Select1 (CALLudiv (MOVWconst [c]) (MOVWconst [d])))
	// cond:
	// result: (MOVWconst [int64(int32(uint32(c)%uint32(d)))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpARMCALLudiv {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARMMOVWconst {
			break
		}
		c := v_0_0.AuxInt
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpARMMOVWconst {
			break
		}
		d := v_0_1.AuxInt
		v.reset(OpARMMOVWconst)
		v.AuxInt = int64(int32(uint32(c) % uint32(d)))
		return true
	}
	return false
}
func rewriteValueARM_OpSignExt16to32_0(v *Value) bool {
	// match: (SignExt16to32 x)
	// cond:
	// result: (MOVHreg x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVHreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpSignExt8to16_0(v *Value) bool {
	// match: (SignExt8to16 x)
	// cond:
	// result: (MOVBreg x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVBreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpSignExt8to32_0(v *Value) bool {
	// match: (SignExt8to32 x)
	// cond:
	// result: (MOVBreg x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVBreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpSignmask_0(v *Value) bool {
	// match: (Signmask x)
	// cond:
	// result: (SRAconst x [31])
	for {
		x := v.Args[0]
		v.reset(OpARMSRAconst)
		v.AuxInt = 31
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpSlicemask_0(v *Value) bool {
	b := v.Block
	// match: (Slicemask <t> x)
	// cond:
	// result: (SRAconst (RSBconst <t> [0] x) [31])
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpARMSRAconst)
		v.AuxInt = 31
		v0 := b.NewValue0(v.Pos, OpARMRSBconst, t)
		v0.AuxInt = 0
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpSqrt_0(v *Value) bool {
	// match: (Sqrt x)
	// cond:
	// result: (SQRTD x)
	for {
		x := v.Args[0]
		v.reset(OpARMSQRTD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpStaticCall_0(v *Value) bool {
	// match: (StaticCall [argwid] {target} mem)
	// cond:
	// result: (CALLstatic [argwid] {target} mem)
	for {
		argwid := v.AuxInt
		target := v.Aux
		mem := v.Args[0]
		v.reset(OpARMCALLstatic)
		v.AuxInt = argwid
		v.Aux = target
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM_OpStore_0(v *Value) bool {
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
		v.reset(OpARMMOVBstore)
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
		v.reset(OpARMMOVHstore)
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
		v.reset(OpARMMOVWstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 4 && is32BitFloat(val.Type)
	// result: (MOVFstore ptr val mem)
	for {
		t := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		if !(t.(*types.Type).Size() == 4 && is32BitFloat(val.Type)) {
			break
		}
		v.reset(OpARMMOVFstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 8 && is64BitFloat(val.Type)
	// result: (MOVDstore ptr val mem)
	for {
		t := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		if !(t.(*types.Type).Size() == 8 && is64BitFloat(val.Type)) {
			break
		}
		v.reset(OpARMMOVDstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpSub16_0(v *Value) bool {
	// match: (Sub16 x y)
	// cond:
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpSub32_0(v *Value) bool {
	// match: (Sub32 x y)
	// cond:
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpSub32F_0(v *Value) bool {
	// match: (Sub32F x y)
	// cond:
	// result: (SUBF x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSUBF)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpSub32carry_0(v *Value) bool {
	// match: (Sub32carry x y)
	// cond:
	// result: (SUBS x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSUBS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpSub32withcarry_0(v *Value) bool {
	// match: (Sub32withcarry x y c)
	// cond:
	// result: (SBC x y c)
	for {
		c := v.Args[2]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARMSBC)
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(c)
		return true
	}
}
func rewriteValueARM_OpSub64F_0(v *Value) bool {
	// match: (Sub64F x y)
	// cond:
	// result: (SUBD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSUBD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpSub8_0(v *Value) bool {
	// match: (Sub8 x y)
	// cond:
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpSubPtr_0(v *Value) bool {
	// match: (SubPtr x y)
	// cond:
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMSUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpTrunc16to8_0(v *Value) bool {
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
func rewriteValueARM_OpTrunc32to16_0(v *Value) bool {
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
func rewriteValueARM_OpTrunc32to8_0(v *Value) bool {
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
func rewriteValueARM_OpWB_0(v *Value) bool {
	// match: (WB {fn} destptr srcptr mem)
	// cond:
	// result: (LoweredWB {fn} destptr srcptr mem)
	for {
		fn := v.Aux
		mem := v.Args[2]
		destptr := v.Args[0]
		srcptr := v.Args[1]
		v.reset(OpARMLoweredWB)
		v.Aux = fn
		v.AddArg(destptr)
		v.AddArg(srcptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM_OpXor16_0(v *Value) bool {
	// match: (Xor16 x y)
	// cond:
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMXOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpXor32_0(v *Value) bool {
	// match: (Xor32 x y)
	// cond:
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMXOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpXor8_0(v *Value) bool {
	// match: (Xor8 x y)
	// cond:
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpARMXOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM_OpZero_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
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
	// result: (MOVBstore ptr (MOVWconst [0]) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARMMOVBstore)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [2] {t} ptr mem)
	// cond: t.(*types.Type).Alignment()%2 == 0
	// result: (MOVHstore ptr (MOVWconst [0]) mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		t := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(t.(*types.Type).Alignment()%2 == 0) {
			break
		}
		v.reset(OpARMMOVHstore)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [2] ptr mem)
	// cond:
	// result: (MOVBstore [1] ptr (MOVWconst [0]) (MOVBstore [0] ptr (MOVWconst [0]) mem))
	for {
		if v.AuxInt != 2 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARMMOVBstore)
		v.AuxInt = 1
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMMOVBstore, types.TypeMem)
		v1.AuxInt = 0
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// cond: t.(*types.Type).Alignment()%4 == 0
	// result: (MOVWstore ptr (MOVWconst [0]) mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		t := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(t.(*types.Type).Alignment()%4 == 0) {
			break
		}
		v.reset(OpARMMOVWstore)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// cond: t.(*types.Type).Alignment()%2 == 0
	// result: (MOVHstore [2] ptr (MOVWconst [0]) (MOVHstore [0] ptr (MOVWconst [0]) mem))
	for {
		if v.AuxInt != 4 {
			break
		}
		t := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(t.(*types.Type).Alignment()%2 == 0) {
			break
		}
		v.reset(OpARMMOVHstore)
		v.AuxInt = 2
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMMOVHstore, types.TypeMem)
		v1.AuxInt = 0
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [4] ptr mem)
	// cond:
	// result: (MOVBstore [3] ptr (MOVWconst [0]) (MOVBstore [2] ptr (MOVWconst [0]) (MOVBstore [1] ptr (MOVWconst [0]) (MOVBstore [0] ptr (MOVWconst [0]) mem))))
	for {
		if v.AuxInt != 4 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARMMOVBstore)
		v.AuxInt = 3
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMMOVBstore, types.TypeMem)
		v1.AuxInt = 2
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARMMOVBstore, types.TypeMem)
		v3.AuxInt = 1
		v3.AddArg(ptr)
		v4 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v4.AuxInt = 0
		v3.AddArg(v4)
		v5 := b.NewValue0(v.Pos, OpARMMOVBstore, types.TypeMem)
		v5.AuxInt = 0
		v5.AddArg(ptr)
		v6 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v6.AuxInt = 0
		v5.AddArg(v6)
		v5.AddArg(mem)
		v3.AddArg(v5)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [3] ptr mem)
	// cond:
	// result: (MOVBstore [2] ptr (MOVWconst [0]) (MOVBstore [1] ptr (MOVWconst [0]) (MOVBstore [0] ptr (MOVWconst [0]) mem)))
	for {
		if v.AuxInt != 3 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpARMMOVBstore)
		v.AuxInt = 2
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMMOVBstore, types.TypeMem)
		v1.AuxInt = 1
		v1.AddArg(ptr)
		v2 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpARMMOVBstore, types.TypeMem)
		v3.AuxInt = 0
		v3.AddArg(ptr)
		v4 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v4.AuxInt = 0
		v3.AddArg(v4)
		v3.AddArg(mem)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [s] {t} ptr mem)
	// cond: s%4 == 0 && s > 4 && s <= 512 && t.(*types.Type).Alignment()%4 == 0 && !config.noDuffDevice
	// result: (DUFFZERO [4 * (128 - s/4)] ptr (MOVWconst [0]) mem)
	for {
		s := v.AuxInt
		t := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(s%4 == 0 && s > 4 && s <= 512 && t.(*types.Type).Alignment()%4 == 0 && !config.noDuffDevice) {
			break
		}
		v.reset(OpARMDUFFZERO)
		v.AuxInt = 4 * (128 - s/4)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [s] {t} ptr mem)
	// cond: (s > 512 || config.noDuffDevice) || t.(*types.Type).Alignment()%4 != 0
	// result: (LoweredZero [t.(*types.Type).Alignment()] ptr (ADDconst <ptr.Type> ptr [s-moveSize(t.(*types.Type).Alignment(), config)]) (MOVWconst [0]) mem)
	for {
		s := v.AuxInt
		t := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		if !((s > 512 || config.noDuffDevice) || t.(*types.Type).Alignment()%4 != 0) {
			break
		}
		v.reset(OpARMLoweredZero)
		v.AuxInt = t.(*types.Type).Alignment()
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARMADDconst, ptr.Type)
		v0.AuxInt = s - moveSize(t.(*types.Type).Alignment(), config)
		v0.AddArg(ptr)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARMMOVWconst, typ.UInt32)
		v1.AuxInt = 0
		v.AddArg(v1)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpZeroExt16to32_0(v *Value) bool {
	// match: (ZeroExt16to32 x)
	// cond:
	// result: (MOVHUreg x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVHUreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpZeroExt8to16_0(v *Value) bool {
	// match: (ZeroExt8to16 x)
	// cond:
	// result: (MOVBUreg x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVBUreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpZeroExt8to32_0(v *Value) bool {
	// match: (ZeroExt8to32 x)
	// cond:
	// result: (MOVBUreg x)
	for {
		x := v.Args[0]
		v.reset(OpARMMOVBUreg)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpZeromask_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Zeromask x)
	// cond:
	// result: (SRAconst (RSBshiftRL <typ.Int32> x x [1]) [31])
	for {
		x := v.Args[0]
		v.reset(OpARMSRAconst)
		v.AuxInt = 31
		v0 := b.NewValue0(v.Pos, OpARMRSBshiftRL, typ.Int32)
		v0.AuxInt = 1
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteBlockARM(b *Block) bool {
	config := b.Func.Config
	typ := &config.Types
	_ = typ
	v := b.Control
	_ = v
	switch b.Kind {
	case BlockARMEQ:
		// match: (EQ (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (EQ (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (EQ (InvertFlags cmp) yes no)
		// cond:
		// result: (EQ cmp yes no)
		for v.Op == OpARMInvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARMEQ
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUB x y)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMP x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUB {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(MULS x y a)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMP a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMMULS {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARMMUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADD x y)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMN x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADD {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(MULA x y a)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMN a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMMULA {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMN, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARMMUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMNconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(AND x y)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TST x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMAND {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTST, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTSTconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(XOR x y)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQ x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXOR {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTEQ, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTEQconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMEQ
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
	case BlockARMGE:
		// match: (GE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (GE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (GE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (GE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (GE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (GE (InvertFlags cmp) yes no)
		// cond:
		// result: (LE cmp yes no)
		for v.Op == OpARMInvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARMLE
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(SUB x y)) yes no)
		// cond: l.Uses==1
		// result: (GE (CMP x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUB {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(MULS x y a)) yes no)
		// cond: l.Uses==1
		// result: (GE (CMP a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMMULS {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARMMUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(SUBconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GE (CMPconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(SUBshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GE (CMPshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(SUBshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GE (CMPshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(SUBshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GE (CMPshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(SUBshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GE (CMPshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(SUBshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GE (CMPshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(SUBshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GE (CMPshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ADD x y)) yes no)
		// cond: l.Uses==1
		// result: (GE (CMN x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADD {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(MULA x y a)) yes no)
		// cond: l.Uses==1
		// result: (GE (CMN a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMMULA {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMN, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARMMUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GE (CMNconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMNconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GE (CMNshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GE (CMNshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GE (CMNshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GE (CMNshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GE (CMNshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GE (CMNshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(AND x y)) yes no)
		// cond: l.Uses==1
		// result: (GE (TST x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMAND {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTST, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GE (TSTconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTSTconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GE (TSTshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GE (TSTshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GE (TSTshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GE (TSTshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GE (TSTshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GE (TSTshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(XOR x y)) yes no)
		// cond: l.Uses==1
		// result: (GE (TEQ x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXOR {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTEQ, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(XORconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GE (TEQconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTEQconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(XORshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GE (TEQshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(XORshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GE (TEQshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(XORshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GE (TEQshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(XORshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GE (TEQshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(XORshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GE (TEQshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GE (CMPconst [0] l:(XORshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GE (TEQshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
	case BlockARMGT:
		// match: (GT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (GT (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (GT (InvertFlags cmp) yes no)
		// cond:
		// result: (LT cmp yes no)
		for v.Op == OpARMInvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARMLT
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(SUB x y)) yes no)
		// cond: l.Uses==1
		// result: (GT (CMP x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUB {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(MULS x y a)) yes no)
		// cond: l.Uses==1
		// result: (GT (CMP a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMMULS {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARMMUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(SUBconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GT (CMPconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(SUBshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GT (CMPshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(SUBshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GT (CMPshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(SUBshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GT (CMPshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(SUBshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GT (CMPshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(SUBshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GT (CMPshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(SUBshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GT (CMPshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ADD x y)) yes no)
		// cond: l.Uses==1
		// result: (GT (CMN x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADD {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GT (CMNconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMNconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GT (CMNshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GT (CMNshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GT (CMNshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GT (CMNshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GT (CMNshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GT (CMNshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(AND x y)) yes no)
		// cond: l.Uses==1
		// result: (GT (TST x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMAND {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTST, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(MULA x y a)) yes no)
		// cond: l.Uses==1
		// result: (GT (CMN a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMMULA {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMCMN, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARMMUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GT (TSTconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTSTconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GT (TSTshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GT (TSTshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GT (TSTshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GT (TSTshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GT (TSTshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GT (TSTshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(XOR x y)) yes no)
		// cond: l.Uses==1
		// result: (GT (TEQ x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXOR {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTEQ, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(XORconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GT (TEQconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTEQconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(XORshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GT (TEQshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(XORshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GT (TEQshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(XORshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GT (TEQshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(XORshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GT (TEQshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(XORshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GT (TEQshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (GT (CMPconst [0] l:(XORshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GT (TEQshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMGT
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
	case BlockIf:
		// match: (If (Equal cc) yes no)
		// cond:
		// result: (EQ cc yes no)
		for v.Op == OpARMEqual {
			cc := v.Args[0]
			b.Kind = BlockARMEQ
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (NotEqual cc) yes no)
		// cond:
		// result: (NE cc yes no)
		for v.Op == OpARMNotEqual {
			cc := v.Args[0]
			b.Kind = BlockARMNE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (LessThan cc) yes no)
		// cond:
		// result: (LT cc yes no)
		for v.Op == OpARMLessThan {
			cc := v.Args[0]
			b.Kind = BlockARMLT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (LessThanU cc) yes no)
		// cond:
		// result: (ULT cc yes no)
		for v.Op == OpARMLessThanU {
			cc := v.Args[0]
			b.Kind = BlockARMULT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (LessEqual cc) yes no)
		// cond:
		// result: (LE cc yes no)
		for v.Op == OpARMLessEqual {
			cc := v.Args[0]
			b.Kind = BlockARMLE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (LessEqualU cc) yes no)
		// cond:
		// result: (ULE cc yes no)
		for v.Op == OpARMLessEqualU {
			cc := v.Args[0]
			b.Kind = BlockARMULE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (GreaterThan cc) yes no)
		// cond:
		// result: (GT cc yes no)
		for v.Op == OpARMGreaterThan {
			cc := v.Args[0]
			b.Kind = BlockARMGT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (GreaterThanU cc) yes no)
		// cond:
		// result: (UGT cc yes no)
		for v.Op == OpARMGreaterThanU {
			cc := v.Args[0]
			b.Kind = BlockARMUGT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (GreaterEqual cc) yes no)
		// cond:
		// result: (GE cc yes no)
		for v.Op == OpARMGreaterEqual {
			cc := v.Args[0]
			b.Kind = BlockARMGE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If (GreaterEqualU cc) yes no)
		// cond:
		// result: (UGE cc yes no)
		for v.Op == OpARMGreaterEqualU {
			cc := v.Args[0]
			b.Kind = BlockARMUGE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (If cond yes no)
		// cond:
		// result: (NE (CMPconst [0] cond) yes no)
		for {
			cond := b.Control
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
			v0.AuxInt = 0
			v0.AddArg(cond)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
	case BlockARMLE:
		// match: (LE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (LE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (LE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (LE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (LE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (LE (InvertFlags cmp) yes no)
		// cond:
		// result: (GE cmp yes no)
		for v.Op == OpARMInvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARMGE
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(SUB x y)) yes no)
		// cond: l.Uses==1
		// result: (LE (CMP x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUB {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(MULS x y a)) yes no)
		// cond: l.Uses==1
		// result: (LE (CMP a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMMULS {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARMMUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(SUBconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LE (CMPconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(SUBshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LE (CMPshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(SUBshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LE (CMPshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(SUBshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LE (CMPshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(SUBshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LE (CMPshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(SUBshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LE (CMPshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(SUBshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LE (CMPshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ADD x y)) yes no)
		// cond: l.Uses==1
		// result: (LE (CMN x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADD {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(MULA x y a)) yes no)
		// cond: l.Uses==1
		// result: (LE (CMN a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMMULA {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMN, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARMMUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LE (CMNconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMNconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LE (CMNshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LE (CMNshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LE (CMNshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LE (CMNshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LE (CMNshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LE (CMNshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(AND x y)) yes no)
		// cond: l.Uses==1
		// result: (LE (TST x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMAND {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTST, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LE (TSTconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTSTconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LE (TSTshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LE (TSTshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LE (TSTshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LE (TSTshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LE (TSTshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LE (TSTshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(XOR x y)) yes no)
		// cond: l.Uses==1
		// result: (LE (TEQ x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXOR {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTEQ, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(XORconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LE (TEQconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTEQconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(XORshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LE (TEQshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(XORshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LE (TEQshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(XORshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LE (TEQshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(XORshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LE (TEQshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(XORshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LE (TEQshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LE (CMPconst [0] l:(XORshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LE (TEQshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
	case BlockARMLT:
		// match: (LT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (LT (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (LT (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (LT (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (LT (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (LT (InvertFlags cmp) yes no)
		// cond:
		// result: (GT cmp yes no)
		for v.Op == OpARMInvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARMGT
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(SUB x y)) yes no)
		// cond: l.Uses==1
		// result: (LT (CMP x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUB {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(MULS x y a)) yes no)
		// cond: l.Uses==1
		// result: (LT (CMP a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMMULS {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARMMUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(SUBconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LT (CMPconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(SUBshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LT (CMPshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(SUBshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LT (CMPshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(SUBshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LT (CMPshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(SUBshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LT (CMPshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(SUBshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LT (CMPshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(SUBshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LT (CMPshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ADD x y)) yes no)
		// cond: l.Uses==1
		// result: (LT (CMN x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADD {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(MULA x y a)) yes no)
		// cond: l.Uses==1
		// result: (LT (CMN a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMMULA {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMN, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARMMUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LT (CMNconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMNconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LT (CMNshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LT (CMNshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LT (CMNshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LT (CMNshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LT (CMNshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LT (CMNshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(AND x y)) yes no)
		// cond: l.Uses==1
		// result: (LT (TST x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMAND {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTST, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LT (TSTconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTSTconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LT (TSTshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LT (TSTshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LT (TSTshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LT (TSTshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LT (TSTshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LT (TSTshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(XOR x y)) yes no)
		// cond: l.Uses==1
		// result: (LT (TEQ x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXOR {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTEQ, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(XORconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LT (TEQconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTEQconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(XORshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LT (TEQshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(XORshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LT (TEQshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(XORshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LT (TEQshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(XORshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LT (TEQshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(XORshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LT (TEQshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (LT (CMPconst [0] l:(XORshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LT (TEQshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMLT
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
	case BlockARMNE:
		// match: (NE (CMPconst [0] (Equal cc)) yes no)
		// cond:
		// result: (EQ cc yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			v_0 := v.Args[0]
			if v_0.Op != OpARMEqual {
				break
			}
			cc := v_0.Args[0]
			b.Kind = BlockARMEQ
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] (NotEqual cc)) yes no)
		// cond:
		// result: (NE cc yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			v_0 := v.Args[0]
			if v_0.Op != OpARMNotEqual {
				break
			}
			cc := v_0.Args[0]
			b.Kind = BlockARMNE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] (LessThan cc)) yes no)
		// cond:
		// result: (LT cc yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			v_0 := v.Args[0]
			if v_0.Op != OpARMLessThan {
				break
			}
			cc := v_0.Args[0]
			b.Kind = BlockARMLT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] (LessThanU cc)) yes no)
		// cond:
		// result: (ULT cc yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			v_0 := v.Args[0]
			if v_0.Op != OpARMLessThanU {
				break
			}
			cc := v_0.Args[0]
			b.Kind = BlockARMULT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] (LessEqual cc)) yes no)
		// cond:
		// result: (LE cc yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			v_0 := v.Args[0]
			if v_0.Op != OpARMLessEqual {
				break
			}
			cc := v_0.Args[0]
			b.Kind = BlockARMLE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] (LessEqualU cc)) yes no)
		// cond:
		// result: (ULE cc yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			v_0 := v.Args[0]
			if v_0.Op != OpARMLessEqualU {
				break
			}
			cc := v_0.Args[0]
			b.Kind = BlockARMULE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] (GreaterThan cc)) yes no)
		// cond:
		// result: (GT cc yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			v_0 := v.Args[0]
			if v_0.Op != OpARMGreaterThan {
				break
			}
			cc := v_0.Args[0]
			b.Kind = BlockARMGT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] (GreaterThanU cc)) yes no)
		// cond:
		// result: (UGT cc yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			v_0 := v.Args[0]
			if v_0.Op != OpARMGreaterThanU {
				break
			}
			cc := v_0.Args[0]
			b.Kind = BlockARMUGT
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] (GreaterEqual cc)) yes no)
		// cond:
		// result: (GE cc yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			v_0 := v.Args[0]
			if v_0.Op != OpARMGreaterEqual {
				break
			}
			cc := v_0.Args[0]
			b.Kind = BlockARMGE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] (GreaterEqualU cc)) yes no)
		// cond:
		// result: (UGE cc yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			v_0 := v.Args[0]
			if v_0.Op != OpARMGreaterEqualU {
				break
			}
			cc := v_0.Args[0]
			b.Kind = BlockARMUGE
			b.SetControl(cc)
			b.Aux = nil
			return true
		}
		// match: (NE (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (NE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (NE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (NE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (NE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (NE (InvertFlags cmp) yes no)
		// cond:
		// result: (NE cmp yes no)
		for v.Op == OpARMInvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARMNE
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(SUB x y)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMP x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUB {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(MULS x y a)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMP a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMMULS {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMP, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARMMUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMPconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMSUBshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMPshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ADD x y)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMN x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADD {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMN, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(MULA x y a)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMN a (MUL <x.Type> x y)) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMMULA {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMN, types.TypeFlags)
			v0.AddArg(a)
			v1 := b.NewValue0(v.Pos, OpARMMUL, x.Type)
			v1.AddArg(x)
			v1.AddArg(y)
			v0.AddArg(v1)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMNconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMADDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMCMNshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(AND x y)) yes no)
		// cond: l.Uses==1
		// result: (NE (TST x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMAND {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTST, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTSTconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMANDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTSTshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(XOR x y)) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQ x y) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXOR {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTEQ, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(XORconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQconst [c] x) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORconst {
				break
			}
			c := l.AuxInt
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTEQconst, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(XORshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQshiftLL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftLL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftLL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(XORshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQshiftRL x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRL {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRL, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(XORshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQshiftRA x y [c]) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRA {
				break
			}
			c := l.AuxInt
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRA, types.TypeFlags)
			v0.AuxInt = c
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(XORshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQshiftLLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftLLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(XORshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQshiftRLreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRLreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
		// match: (NE (CMPconst [0] l:(XORshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQshiftRAreg x y z) yes no)
		for v.Op == OpARMCMPconst {
			if v.AuxInt != 0 {
				break
			}
			l := v.Args[0]
			if l.Op != OpARMXORshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			b.Kind = BlockARMNE
			v0 := b.NewValue0(v.Pos, OpARMTEQshiftRAreg, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			v0.AddArg(z)
			b.SetControl(v0)
			b.Aux = nil
			return true
		}
	case BlockARMUGE:
		// match: (UGE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (UGE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (UGE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (UGE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (UGE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (UGE (InvertFlags cmp) yes no)
		// cond:
		// result: (ULE cmp yes no)
		for v.Op == OpARMInvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARMULE
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARMUGT:
		// match: (UGT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (UGT (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (UGT (InvertFlags cmp) yes no)
		// cond:
		// result: (ULT cmp yes no)
		for v.Op == OpARMInvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARMULT
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARMULE:
		// match: (ULE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (ULE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (ULE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (ULE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (ULE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (ULE (InvertFlags cmp) yes no)
		// cond:
		// result: (UGE cmp yes no)
		for v.Op == OpARMInvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARMUGE
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	case BlockARMULT:
		// match: (ULT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagEQ {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (ULT (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagLT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (ULT (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagLT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (ULT (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for v.Op == OpARMFlagGT_ULT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			return true
		}
		// match: (ULT (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for v.Op == OpARMFlagGT_UGT {
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.Aux = nil
			b.swapSuccessors()
			return true
		}
		// match: (ULT (InvertFlags cmp) yes no)
		// cond:
		// result: (UGT cmp yes no)
		for v.Op == OpARMInvertFlags {
			cmp := v.Args[0]
			b.Kind = BlockARMUGT
			b.SetControl(cmp)
			b.Aux = nil
			return true
		}
	}
	return false
}
