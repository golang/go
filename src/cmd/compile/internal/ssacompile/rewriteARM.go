// Code generated from _gen/ARM.rules using 'go generate'; DO NOT EDIT.

package ssacompile

import "internal/buildcfg"
import "cmd/compile/internal/types"
import "cmd/compile/internal/ssa/block"
import "cmd/compile/internal/ssa/ssaop"
import "cmd/compile/internal/ssa"

func rewriteValueARM(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpARMADC:
		return rewriteValueARM_OpARMADC(v)
	case ssaop.OpARMADCconst:
		return rewriteValueARM_OpARMADCconst(v)
	case ssaop.OpARMADCshiftLL:
		return rewriteValueARM_OpARMADCshiftLL(v)
	case ssaop.OpARMADCshiftLLreg:
		return rewriteValueARM_OpARMADCshiftLLreg(v)
	case ssaop.OpARMADCshiftRA:
		return rewriteValueARM_OpARMADCshiftRA(v)
	case ssaop.OpARMADCshiftRAreg:
		return rewriteValueARM_OpARMADCshiftRAreg(v)
	case ssaop.OpARMADCshiftRL:
		return rewriteValueARM_OpARMADCshiftRL(v)
	case ssaop.OpARMADCshiftRLreg:
		return rewriteValueARM_OpARMADCshiftRLreg(v)
	case ssaop.OpARMADD:
		return rewriteValueARM_OpARMADD(v)
	case ssaop.OpARMADDD:
		return rewriteValueARM_OpARMADDD(v)
	case ssaop.OpARMADDF:
		return rewriteValueARM_OpARMADDF(v)
	case ssaop.OpARMADDS:
		return rewriteValueARM_OpARMADDS(v)
	case ssaop.OpARMADDSshiftLL:
		return rewriteValueARM_OpARMADDSshiftLL(v)
	case ssaop.OpARMADDSshiftLLreg:
		return rewriteValueARM_OpARMADDSshiftLLreg(v)
	case ssaop.OpARMADDSshiftRA:
		return rewriteValueARM_OpARMADDSshiftRA(v)
	case ssaop.OpARMADDSshiftRAreg:
		return rewriteValueARM_OpARMADDSshiftRAreg(v)
	case ssaop.OpARMADDSshiftRL:
		return rewriteValueARM_OpARMADDSshiftRL(v)
	case ssaop.OpARMADDSshiftRLreg:
		return rewriteValueARM_OpARMADDSshiftRLreg(v)
	case ssaop.OpARMADDconst:
		return rewriteValueARM_OpARMADDconst(v)
	case ssaop.OpARMADDshiftLL:
		return rewriteValueARM_OpARMADDshiftLL(v)
	case ssaop.OpARMADDshiftLLreg:
		return rewriteValueARM_OpARMADDshiftLLreg(v)
	case ssaop.OpARMADDshiftRA:
		return rewriteValueARM_OpARMADDshiftRA(v)
	case ssaop.OpARMADDshiftRAreg:
		return rewriteValueARM_OpARMADDshiftRAreg(v)
	case ssaop.OpARMADDshiftRL:
		return rewriteValueARM_OpARMADDshiftRL(v)
	case ssaop.OpARMADDshiftRLreg:
		return rewriteValueARM_OpARMADDshiftRLreg(v)
	case ssaop.OpARMAND:
		return rewriteValueARM_OpARMAND(v)
	case ssaop.OpARMANDconst:
		return rewriteValueARM_OpARMANDconst(v)
	case ssaop.OpARMANDshiftLL:
		return rewriteValueARM_OpARMANDshiftLL(v)
	case ssaop.OpARMANDshiftLLreg:
		return rewriteValueARM_OpARMANDshiftLLreg(v)
	case ssaop.OpARMANDshiftRA:
		return rewriteValueARM_OpARMANDshiftRA(v)
	case ssaop.OpARMANDshiftRAreg:
		return rewriteValueARM_OpARMANDshiftRAreg(v)
	case ssaop.OpARMANDshiftRL:
		return rewriteValueARM_OpARMANDshiftRL(v)
	case ssaop.OpARMANDshiftRLreg:
		return rewriteValueARM_OpARMANDshiftRLreg(v)
	case ssaop.OpARMBFX:
		return rewriteValueARM_OpARMBFX(v)
	case ssaop.OpARMBFXU:
		return rewriteValueARM_OpARMBFXU(v)
	case ssaop.OpARMBIC:
		return rewriteValueARM_OpARMBIC(v)
	case ssaop.OpARMBICconst:
		return rewriteValueARM_OpARMBICconst(v)
	case ssaop.OpARMBICshiftLL:
		return rewriteValueARM_OpARMBICshiftLL(v)
	case ssaop.OpARMBICshiftLLreg:
		return rewriteValueARM_OpARMBICshiftLLreg(v)
	case ssaop.OpARMBICshiftRA:
		return rewriteValueARM_OpARMBICshiftRA(v)
	case ssaop.OpARMBICshiftRAreg:
		return rewriteValueARM_OpARMBICshiftRAreg(v)
	case ssaop.OpARMBICshiftRL:
		return rewriteValueARM_OpARMBICshiftRL(v)
	case ssaop.OpARMBICshiftRLreg:
		return rewriteValueARM_OpARMBICshiftRLreg(v)
	case ssaop.OpARMCMN:
		return rewriteValueARM_OpARMCMN(v)
	case ssaop.OpARMCMNconst:
		return rewriteValueARM_OpARMCMNconst(v)
	case ssaop.OpARMCMNshiftLL:
		return rewriteValueARM_OpARMCMNshiftLL(v)
	case ssaop.OpARMCMNshiftLLreg:
		return rewriteValueARM_OpARMCMNshiftLLreg(v)
	case ssaop.OpARMCMNshiftRA:
		return rewriteValueARM_OpARMCMNshiftRA(v)
	case ssaop.OpARMCMNshiftRAreg:
		return rewriteValueARM_OpARMCMNshiftRAreg(v)
	case ssaop.OpARMCMNshiftRL:
		return rewriteValueARM_OpARMCMNshiftRL(v)
	case ssaop.OpARMCMNshiftRLreg:
		return rewriteValueARM_OpARMCMNshiftRLreg(v)
	case ssaop.OpARMCMOVWHSconst:
		return rewriteValueARM_OpARMCMOVWHSconst(v)
	case ssaop.OpARMCMOVWLSconst:
		return rewriteValueARM_OpARMCMOVWLSconst(v)
	case ssaop.OpARMCMP:
		return rewriteValueARM_OpARMCMP(v)
	case ssaop.OpARMCMPD:
		return rewriteValueARM_OpARMCMPD(v)
	case ssaop.OpARMCMPF:
		return rewriteValueARM_OpARMCMPF(v)
	case ssaop.OpARMCMPconst:
		return rewriteValueARM_OpARMCMPconst(v)
	case ssaop.OpARMCMPshiftLL:
		return rewriteValueARM_OpARMCMPshiftLL(v)
	case ssaop.OpARMCMPshiftLLreg:
		return rewriteValueARM_OpARMCMPshiftLLreg(v)
	case ssaop.OpARMCMPshiftRA:
		return rewriteValueARM_OpARMCMPshiftRA(v)
	case ssaop.OpARMCMPshiftRAreg:
		return rewriteValueARM_OpARMCMPshiftRAreg(v)
	case ssaop.OpARMCMPshiftRL:
		return rewriteValueARM_OpARMCMPshiftRL(v)
	case ssaop.OpARMCMPshiftRLreg:
		return rewriteValueARM_OpARMCMPshiftRLreg(v)
	case ssaop.OpARMEqual:
		return rewriteValueARM_OpARMEqual(v)
	case ssaop.OpARMGreaterEqual:
		return rewriteValueARM_OpARMGreaterEqual(v)
	case ssaop.OpARMGreaterEqualU:
		return rewriteValueARM_OpARMGreaterEqualU(v)
	case ssaop.OpARMGreaterThan:
		return rewriteValueARM_OpARMGreaterThan(v)
	case ssaop.OpARMGreaterThanU:
		return rewriteValueARM_OpARMGreaterThanU(v)
	case ssaop.OpARMLessEqual:
		return rewriteValueARM_OpARMLessEqual(v)
	case ssaop.OpARMLessEqualU:
		return rewriteValueARM_OpARMLessEqualU(v)
	case ssaop.OpARMLessThan:
		return rewriteValueARM_OpARMLessThan(v)
	case ssaop.OpARMLessThanU:
		return rewriteValueARM_OpARMLessThanU(v)
	case ssaop.OpARMLoweredPanicBoundsRC:
		return rewriteValueARM_OpARMLoweredPanicBoundsRC(v)
	case ssaop.OpARMLoweredPanicBoundsRR:
		return rewriteValueARM_OpARMLoweredPanicBoundsRR(v)
	case ssaop.OpARMLoweredPanicExtendRC:
		return rewriteValueARM_OpARMLoweredPanicExtendRC(v)
	case ssaop.OpARMLoweredPanicExtendRR:
		return rewriteValueARM_OpARMLoweredPanicExtendRR(v)
	case ssaop.OpARMMOVBUload:
		return rewriteValueARM_OpARMMOVBUload(v)
	case ssaop.OpARMMOVBUloadidx:
		return rewriteValueARM_OpARMMOVBUloadidx(v)
	case ssaop.OpARMMOVBUreg:
		return rewriteValueARM_OpARMMOVBUreg(v)
	case ssaop.OpARMMOVBload:
		return rewriteValueARM_OpARMMOVBload(v)
	case ssaop.OpARMMOVBloadidx:
		return rewriteValueARM_OpARMMOVBloadidx(v)
	case ssaop.OpARMMOVBreg:
		return rewriteValueARM_OpARMMOVBreg(v)
	case ssaop.OpARMMOVBstore:
		return rewriteValueARM_OpARMMOVBstore(v)
	case ssaop.OpARMMOVBstoreidx:
		return rewriteValueARM_OpARMMOVBstoreidx(v)
	case ssaop.OpARMMOVDload:
		return rewriteValueARM_OpARMMOVDload(v)
	case ssaop.OpARMMOVDstore:
		return rewriteValueARM_OpARMMOVDstore(v)
	case ssaop.OpARMMOVFload:
		return rewriteValueARM_OpARMMOVFload(v)
	case ssaop.OpARMMOVFstore:
		return rewriteValueARM_OpARMMOVFstore(v)
	case ssaop.OpARMMOVHUload:
		return rewriteValueARM_OpARMMOVHUload(v)
	case ssaop.OpARMMOVHUloadidx:
		return rewriteValueARM_OpARMMOVHUloadidx(v)
	case ssaop.OpARMMOVHUreg:
		return rewriteValueARM_OpARMMOVHUreg(v)
	case ssaop.OpARMMOVHload:
		return rewriteValueARM_OpARMMOVHload(v)
	case ssaop.OpARMMOVHloadidx:
		return rewriteValueARM_OpARMMOVHloadidx(v)
	case ssaop.OpARMMOVHreg:
		return rewriteValueARM_OpARMMOVHreg(v)
	case ssaop.OpARMMOVHstore:
		return rewriteValueARM_OpARMMOVHstore(v)
	case ssaop.OpARMMOVHstoreidx:
		return rewriteValueARM_OpARMMOVHstoreidx(v)
	case ssaop.OpARMMOVWload:
		return rewriteValueARM_OpARMMOVWload(v)
	case ssaop.OpARMMOVWloadidx:
		return rewriteValueARM_OpARMMOVWloadidx(v)
	case ssaop.OpARMMOVWloadshiftLL:
		return rewriteValueARM_OpARMMOVWloadshiftLL(v)
	case ssaop.OpARMMOVWloadshiftRA:
		return rewriteValueARM_OpARMMOVWloadshiftRA(v)
	case ssaop.OpARMMOVWloadshiftRL:
		return rewriteValueARM_OpARMMOVWloadshiftRL(v)
	case ssaop.OpARMMOVWnop:
		return rewriteValueARM_OpARMMOVWnop(v)
	case ssaop.OpARMMOVWreg:
		return rewriteValueARM_OpARMMOVWreg(v)
	case ssaop.OpARMMOVWstore:
		return rewriteValueARM_OpARMMOVWstore(v)
	case ssaop.OpARMMOVWstoreidx:
		return rewriteValueARM_OpARMMOVWstoreidx(v)
	case ssaop.OpARMMOVWstoreshiftLL:
		return rewriteValueARM_OpARMMOVWstoreshiftLL(v)
	case ssaop.OpARMMOVWstoreshiftRA:
		return rewriteValueARM_OpARMMOVWstoreshiftRA(v)
	case ssaop.OpARMMOVWstoreshiftRL:
		return rewriteValueARM_OpARMMOVWstoreshiftRL(v)
	case ssaop.OpARMMUL:
		return rewriteValueARM_OpARMMUL(v)
	case ssaop.OpARMMULA:
		return rewriteValueARM_OpARMMULA(v)
	case ssaop.OpARMMULD:
		return rewriteValueARM_OpARMMULD(v)
	case ssaop.OpARMMULF:
		return rewriteValueARM_OpARMMULF(v)
	case ssaop.OpARMMULS:
		return rewriteValueARM_OpARMMULS(v)
	case ssaop.OpARMMVN:
		return rewriteValueARM_OpARMMVN(v)
	case ssaop.OpARMMVNshiftLL:
		return rewriteValueARM_OpARMMVNshiftLL(v)
	case ssaop.OpARMMVNshiftLLreg:
		return rewriteValueARM_OpARMMVNshiftLLreg(v)
	case ssaop.OpARMMVNshiftRA:
		return rewriteValueARM_OpARMMVNshiftRA(v)
	case ssaop.OpARMMVNshiftRAreg:
		return rewriteValueARM_OpARMMVNshiftRAreg(v)
	case ssaop.OpARMMVNshiftRL:
		return rewriteValueARM_OpARMMVNshiftRL(v)
	case ssaop.OpARMMVNshiftRLreg:
		return rewriteValueARM_OpARMMVNshiftRLreg(v)
	case ssaop.OpARMNEGD:
		return rewriteValueARM_OpARMNEGD(v)
	case ssaop.OpARMNEGF:
		return rewriteValueARM_OpARMNEGF(v)
	case ssaop.OpARMNMULD:
		return rewriteValueARM_OpARMNMULD(v)
	case ssaop.OpARMNMULF:
		return rewriteValueARM_OpARMNMULF(v)
	case ssaop.OpARMNotEqual:
		return rewriteValueARM_OpARMNotEqual(v)
	case ssaop.OpARMOR:
		return rewriteValueARM_OpARMOR(v)
	case ssaop.OpARMORconst:
		return rewriteValueARM_OpARMORconst(v)
	case ssaop.OpARMORshiftLL:
		return rewriteValueARM_OpARMORshiftLL(v)
	case ssaop.OpARMORshiftLLreg:
		return rewriteValueARM_OpARMORshiftLLreg(v)
	case ssaop.OpARMORshiftRA:
		return rewriteValueARM_OpARMORshiftRA(v)
	case ssaop.OpARMORshiftRAreg:
		return rewriteValueARM_OpARMORshiftRAreg(v)
	case ssaop.OpARMORshiftRL:
		return rewriteValueARM_OpARMORshiftRL(v)
	case ssaop.OpARMORshiftRLreg:
		return rewriteValueARM_OpARMORshiftRLreg(v)
	case ssaop.OpARMRSB:
		return rewriteValueARM_OpARMRSB(v)
	case ssaop.OpARMRSBSshiftLL:
		return rewriteValueARM_OpARMRSBSshiftLL(v)
	case ssaop.OpARMRSBSshiftLLreg:
		return rewriteValueARM_OpARMRSBSshiftLLreg(v)
	case ssaop.OpARMRSBSshiftRA:
		return rewriteValueARM_OpARMRSBSshiftRA(v)
	case ssaop.OpARMRSBSshiftRAreg:
		return rewriteValueARM_OpARMRSBSshiftRAreg(v)
	case ssaop.OpARMRSBSshiftRL:
		return rewriteValueARM_OpARMRSBSshiftRL(v)
	case ssaop.OpARMRSBSshiftRLreg:
		return rewriteValueARM_OpARMRSBSshiftRLreg(v)
	case ssaop.OpARMRSBconst:
		return rewriteValueARM_OpARMRSBconst(v)
	case ssaop.OpARMRSBshiftLL:
		return rewriteValueARM_OpARMRSBshiftLL(v)
	case ssaop.OpARMRSBshiftLLreg:
		return rewriteValueARM_OpARMRSBshiftLLreg(v)
	case ssaop.OpARMRSBshiftRA:
		return rewriteValueARM_OpARMRSBshiftRA(v)
	case ssaop.OpARMRSBshiftRAreg:
		return rewriteValueARM_OpARMRSBshiftRAreg(v)
	case ssaop.OpARMRSBshiftRL:
		return rewriteValueARM_OpARMRSBshiftRL(v)
	case ssaop.OpARMRSBshiftRLreg:
		return rewriteValueARM_OpARMRSBshiftRLreg(v)
	case ssaop.OpARMRSCconst:
		return rewriteValueARM_OpARMRSCconst(v)
	case ssaop.OpARMRSCshiftLL:
		return rewriteValueARM_OpARMRSCshiftLL(v)
	case ssaop.OpARMRSCshiftLLreg:
		return rewriteValueARM_OpARMRSCshiftLLreg(v)
	case ssaop.OpARMRSCshiftRA:
		return rewriteValueARM_OpARMRSCshiftRA(v)
	case ssaop.OpARMRSCshiftRAreg:
		return rewriteValueARM_OpARMRSCshiftRAreg(v)
	case ssaop.OpARMRSCshiftRL:
		return rewriteValueARM_OpARMRSCshiftRL(v)
	case ssaop.OpARMRSCshiftRLreg:
		return rewriteValueARM_OpARMRSCshiftRLreg(v)
	case ssaop.OpARMSBC:
		return rewriteValueARM_OpARMSBC(v)
	case ssaop.OpARMSBCconst:
		return rewriteValueARM_OpARMSBCconst(v)
	case ssaop.OpARMSBCshiftLL:
		return rewriteValueARM_OpARMSBCshiftLL(v)
	case ssaop.OpARMSBCshiftLLreg:
		return rewriteValueARM_OpARMSBCshiftLLreg(v)
	case ssaop.OpARMSBCshiftRA:
		return rewriteValueARM_OpARMSBCshiftRA(v)
	case ssaop.OpARMSBCshiftRAreg:
		return rewriteValueARM_OpARMSBCshiftRAreg(v)
	case ssaop.OpARMSBCshiftRL:
		return rewriteValueARM_OpARMSBCshiftRL(v)
	case ssaop.OpARMSBCshiftRLreg:
		return rewriteValueARM_OpARMSBCshiftRLreg(v)
	case ssaop.OpARMSLL:
		return rewriteValueARM_OpARMSLL(v)
	case ssaop.OpARMSLLconst:
		return rewriteValueARM_OpARMSLLconst(v)
	case ssaop.OpARMSRA:
		return rewriteValueARM_OpARMSRA(v)
	case ssaop.OpARMSRAcond:
		return rewriteValueARM_OpARMSRAcond(v)
	case ssaop.OpARMSRAconst:
		return rewriteValueARM_OpARMSRAconst(v)
	case ssaop.OpARMSRL:
		return rewriteValueARM_OpARMSRL(v)
	case ssaop.OpARMSRLconst:
		return rewriteValueARM_OpARMSRLconst(v)
	case ssaop.OpARMSRR:
		return rewriteValueARM_OpARMSRR(v)
	case ssaop.OpARMSUB:
		return rewriteValueARM_OpARMSUB(v)
	case ssaop.OpARMSUBD:
		return rewriteValueARM_OpARMSUBD(v)
	case ssaop.OpARMSUBF:
		return rewriteValueARM_OpARMSUBF(v)
	case ssaop.OpARMSUBS:
		return rewriteValueARM_OpARMSUBS(v)
	case ssaop.OpARMSUBSshiftLL:
		return rewriteValueARM_OpARMSUBSshiftLL(v)
	case ssaop.OpARMSUBSshiftLLreg:
		return rewriteValueARM_OpARMSUBSshiftLLreg(v)
	case ssaop.OpARMSUBSshiftRA:
		return rewriteValueARM_OpARMSUBSshiftRA(v)
	case ssaop.OpARMSUBSshiftRAreg:
		return rewriteValueARM_OpARMSUBSshiftRAreg(v)
	case ssaop.OpARMSUBSshiftRL:
		return rewriteValueARM_OpARMSUBSshiftRL(v)
	case ssaop.OpARMSUBSshiftRLreg:
		return rewriteValueARM_OpARMSUBSshiftRLreg(v)
	case ssaop.OpARMSUBconst:
		return rewriteValueARM_OpARMSUBconst(v)
	case ssaop.OpARMSUBshiftLL:
		return rewriteValueARM_OpARMSUBshiftLL(v)
	case ssaop.OpARMSUBshiftLLreg:
		return rewriteValueARM_OpARMSUBshiftLLreg(v)
	case ssaop.OpARMSUBshiftRA:
		return rewriteValueARM_OpARMSUBshiftRA(v)
	case ssaop.OpARMSUBshiftRAreg:
		return rewriteValueARM_OpARMSUBshiftRAreg(v)
	case ssaop.OpARMSUBshiftRL:
		return rewriteValueARM_OpARMSUBshiftRL(v)
	case ssaop.OpARMSUBshiftRLreg:
		return rewriteValueARM_OpARMSUBshiftRLreg(v)
	case ssaop.OpARMTEQ:
		return rewriteValueARM_OpARMTEQ(v)
	case ssaop.OpARMTEQconst:
		return rewriteValueARM_OpARMTEQconst(v)
	case ssaop.OpARMTEQshiftLL:
		return rewriteValueARM_OpARMTEQshiftLL(v)
	case ssaop.OpARMTEQshiftLLreg:
		return rewriteValueARM_OpARMTEQshiftLLreg(v)
	case ssaop.OpARMTEQshiftRA:
		return rewriteValueARM_OpARMTEQshiftRA(v)
	case ssaop.OpARMTEQshiftRAreg:
		return rewriteValueARM_OpARMTEQshiftRAreg(v)
	case ssaop.OpARMTEQshiftRL:
		return rewriteValueARM_OpARMTEQshiftRL(v)
	case ssaop.OpARMTEQshiftRLreg:
		return rewriteValueARM_OpARMTEQshiftRLreg(v)
	case ssaop.OpARMTST:
		return rewriteValueARM_OpARMTST(v)
	case ssaop.OpARMTSTconst:
		return rewriteValueARM_OpARMTSTconst(v)
	case ssaop.OpARMTSTshiftLL:
		return rewriteValueARM_OpARMTSTshiftLL(v)
	case ssaop.OpARMTSTshiftLLreg:
		return rewriteValueARM_OpARMTSTshiftLLreg(v)
	case ssaop.OpARMTSTshiftRA:
		return rewriteValueARM_OpARMTSTshiftRA(v)
	case ssaop.OpARMTSTshiftRAreg:
		return rewriteValueARM_OpARMTSTshiftRAreg(v)
	case ssaop.OpARMTSTshiftRL:
		return rewriteValueARM_OpARMTSTshiftRL(v)
	case ssaop.OpARMTSTshiftRLreg:
		return rewriteValueARM_OpARMTSTshiftRLreg(v)
	case ssaop.OpARMXOR:
		return rewriteValueARM_OpARMXOR(v)
	case ssaop.OpARMXORconst:
		return rewriteValueARM_OpARMXORconst(v)
	case ssaop.OpARMXORshiftLL:
		return rewriteValueARM_OpARMXORshiftLL(v)
	case ssaop.OpARMXORshiftLLreg:
		return rewriteValueARM_OpARMXORshiftLLreg(v)
	case ssaop.OpARMXORshiftRA:
		return rewriteValueARM_OpARMXORshiftRA(v)
	case ssaop.OpARMXORshiftRAreg:
		return rewriteValueARM_OpARMXORshiftRAreg(v)
	case ssaop.OpARMXORshiftRL:
		return rewriteValueARM_OpARMXORshiftRL(v)
	case ssaop.OpARMXORshiftRLreg:
		return rewriteValueARM_OpARMXORshiftRLreg(v)
	case ssaop.OpARMXORshiftRR:
		return rewriteValueARM_OpARMXORshiftRR(v)
	case ssaop.OpAbs:
		v.Op = ssaop.OpARMABSD
		return true
	case ssaop.OpAdd16:
		v.Op = ssaop.OpARMADD
		return true
	case ssaop.OpAdd32:
		v.Op = ssaop.OpARMADD
		return true
	case ssaop.OpAdd32F:
		v.Op = ssaop.OpARMADDF
		return true
	case ssaop.OpAdd32carry:
		v.Op = ssaop.OpARMADDS
		return true
	case ssaop.OpAdd32carrywithcarry:
		v.Op = ssaop.OpARMADCS
		return true
	case ssaop.OpAdd32withcarry:
		v.Op = ssaop.OpARMADC
		return true
	case ssaop.OpAdd64F:
		v.Op = ssaop.OpARMADDD
		return true
	case ssaop.OpAdd8:
		v.Op = ssaop.OpARMADD
		return true
	case ssaop.OpAddPtr:
		v.Op = ssaop.OpARMADD
		return true
	case ssaop.OpAddr:
		return rewriteValueARM_OpAddr(v)
	case ssaop.OpAnd16:
		v.Op = ssaop.OpARMAND
		return true
	case ssaop.OpAnd32:
		v.Op = ssaop.OpARMAND
		return true
	case ssaop.OpAnd8:
		v.Op = ssaop.OpARMAND
		return true
	case ssaop.OpAndB:
		v.Op = ssaop.OpARMAND
		return true
	case ssaop.OpAvg32u:
		return rewriteValueARM_OpAvg32u(v)
	case ssaop.OpBitLen16:
		return rewriteValueARM_OpBitLen16(v)
	case ssaop.OpBitLen32:
		return rewriteValueARM_OpBitLen32(v)
	case ssaop.OpBitLen8:
		return rewriteValueARM_OpBitLen8(v)
	case ssaop.OpBswap32:
		return rewriteValueARM_OpBswap32(v)
	case ssaop.OpClosureCall:
		v.Op = ssaop.OpARMCALLclosure
		return true
	case ssaop.OpCom16:
		v.Op = ssaop.OpARMMVN
		return true
	case ssaop.OpCom32:
		v.Op = ssaop.OpARMMVN
		return true
	case ssaop.OpCom8:
		v.Op = ssaop.OpARMMVN
		return true
	case ssaop.OpConst16:
		return rewriteValueARM_OpConst16(v)
	case ssaop.OpConst32:
		return rewriteValueARM_OpConst32(v)
	case ssaop.OpConst32F:
		return rewriteValueARM_OpConst32F(v)
	case ssaop.OpConst64F:
		return rewriteValueARM_OpConst64F(v)
	case ssaop.OpConst8:
		return rewriteValueARM_OpConst8(v)
	case ssaop.OpConstBool:
		return rewriteValueARM_OpConstBool(v)
	case ssaop.OpConstNil:
		return rewriteValueARM_OpConstNil(v)
	case ssaop.OpCtz16:
		return rewriteValueARM_OpCtz16(v)
	case ssaop.OpCtz16NonZero:
		v.Op = ssaop.OpCtz32
		return true
	case ssaop.OpCtz32:
		return rewriteValueARM_OpCtz32(v)
	case ssaop.OpCtz32NonZero:
		v.Op = ssaop.OpCtz32
		return true
	case ssaop.OpCtz8:
		return rewriteValueARM_OpCtz8(v)
	case ssaop.OpCtz8NonZero:
		v.Op = ssaop.OpCtz32
		return true
	case ssaop.OpCvt32Fto32:
		v.Op = ssaop.OpARMMOVFW
		return true
	case ssaop.OpCvt32Fto32U:
		v.Op = ssaop.OpARMMOVFWU
		return true
	case ssaop.OpCvt32Fto64F:
		v.Op = ssaop.OpARMMOVFD
		return true
	case ssaop.OpCvt32Uto32F:
		v.Op = ssaop.OpARMMOVWUF
		return true
	case ssaop.OpCvt32Uto64F:
		v.Op = ssaop.OpARMMOVWUD
		return true
	case ssaop.OpCvt32to32F:
		v.Op = ssaop.OpARMMOVWF
		return true
	case ssaop.OpCvt32to64F:
		v.Op = ssaop.OpARMMOVWD
		return true
	case ssaop.OpCvt64Fto32:
		v.Op = ssaop.OpARMMOVDW
		return true
	case ssaop.OpCvt64Fto32F:
		v.Op = ssaop.OpARMMOVDF
		return true
	case ssaop.OpCvt64Fto32U:
		v.Op = ssaop.OpARMMOVDWU
		return true
	case ssaop.OpCvtBoolToUint8:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpDiv16:
		return rewriteValueARM_OpDiv16(v)
	case ssaop.OpDiv16u:
		return rewriteValueARM_OpDiv16u(v)
	case ssaop.OpDiv32:
		return rewriteValueARM_OpDiv32(v)
	case ssaop.OpDiv32F:
		v.Op = ssaop.OpARMDIVF
		return true
	case ssaop.OpDiv32u:
		return rewriteValueARM_OpDiv32u(v)
	case ssaop.OpDiv64F:
		v.Op = ssaop.OpARMDIVD
		return true
	case ssaop.OpDiv8:
		return rewriteValueARM_OpDiv8(v)
	case ssaop.OpDiv8u:
		return rewriteValueARM_OpDiv8u(v)
	case ssaop.OpEq16:
		return rewriteValueARM_OpEq16(v)
	case ssaop.OpEq32:
		return rewriteValueARM_OpEq32(v)
	case ssaop.OpEq32F:
		return rewriteValueARM_OpEq32F(v)
	case ssaop.OpEq64F:
		return rewriteValueARM_OpEq64F(v)
	case ssaop.OpEq8:
		return rewriteValueARM_OpEq8(v)
	case ssaop.OpEqB:
		return rewriteValueARM_OpEqB(v)
	case ssaop.OpEqPtr:
		return rewriteValueARM_OpEqPtr(v)
	case ssaop.OpFMA:
		return rewriteValueARM_OpFMA(v)
	case ssaop.OpGetCallerPC:
		v.Op = ssaop.OpARMLoweredGetCallerPC
		return true
	case ssaop.OpGetCallerSP:
		v.Op = ssaop.OpARMLoweredGetCallerSP
		return true
	case ssaop.OpGetClosurePtr:
		v.Op = ssaop.OpARMLoweredGetClosurePtr
		return true
	case ssaop.OpHmul32:
		v.Op = ssaop.OpARMHMUL
		return true
	case ssaop.OpHmul32u:
		v.Op = ssaop.OpARMHMULU
		return true
	case ssaop.OpInterCall:
		v.Op = ssaop.OpARMCALLinter
		return true
	case ssaop.OpIsInBounds:
		return rewriteValueARM_OpIsInBounds(v)
	case ssaop.OpIsNonNil:
		return rewriteValueARM_OpIsNonNil(v)
	case ssaop.OpIsSliceInBounds:
		return rewriteValueARM_OpIsSliceInBounds(v)
	case ssaop.OpLeq16:
		return rewriteValueARM_OpLeq16(v)
	case ssaop.OpLeq16U:
		return rewriteValueARM_OpLeq16U(v)
	case ssaop.OpLeq32:
		return rewriteValueARM_OpLeq32(v)
	case ssaop.OpLeq32F:
		return rewriteValueARM_OpLeq32F(v)
	case ssaop.OpLeq32U:
		return rewriteValueARM_OpLeq32U(v)
	case ssaop.OpLeq64F:
		return rewriteValueARM_OpLeq64F(v)
	case ssaop.OpLeq8:
		return rewriteValueARM_OpLeq8(v)
	case ssaop.OpLeq8U:
		return rewriteValueARM_OpLeq8U(v)
	case ssaop.OpLess16:
		return rewriteValueARM_OpLess16(v)
	case ssaop.OpLess16U:
		return rewriteValueARM_OpLess16U(v)
	case ssaop.OpLess32:
		return rewriteValueARM_OpLess32(v)
	case ssaop.OpLess32F:
		return rewriteValueARM_OpLess32F(v)
	case ssaop.OpLess32U:
		return rewriteValueARM_OpLess32U(v)
	case ssaop.OpLess64F:
		return rewriteValueARM_OpLess64F(v)
	case ssaop.OpLess8:
		return rewriteValueARM_OpLess8(v)
	case ssaop.OpLess8U:
		return rewriteValueARM_OpLess8U(v)
	case ssaop.OpLoad:
		return rewriteValueARM_OpLoad(v)
	case ssaop.OpLocalAddr:
		return rewriteValueARM_OpLocalAddr(v)
	case ssaop.OpLsh16x16:
		return rewriteValueARM_OpLsh16x16(v)
	case ssaop.OpLsh16x32:
		return rewriteValueARM_OpLsh16x32(v)
	case ssaop.OpLsh16x64:
		return rewriteValueARM_OpLsh16x64(v)
	case ssaop.OpLsh16x8:
		return rewriteValueARM_OpLsh16x8(v)
	case ssaop.OpLsh32x16:
		return rewriteValueARM_OpLsh32x16(v)
	case ssaop.OpLsh32x32:
		return rewriteValueARM_OpLsh32x32(v)
	case ssaop.OpLsh32x64:
		return rewriteValueARM_OpLsh32x64(v)
	case ssaop.OpLsh32x8:
		return rewriteValueARM_OpLsh32x8(v)
	case ssaop.OpLsh8x16:
		return rewriteValueARM_OpLsh8x16(v)
	case ssaop.OpLsh8x32:
		return rewriteValueARM_OpLsh8x32(v)
	case ssaop.OpLsh8x64:
		return rewriteValueARM_OpLsh8x64(v)
	case ssaop.OpLsh8x8:
		return rewriteValueARM_OpLsh8x8(v)
	case ssaop.OpMod16:
		return rewriteValueARM_OpMod16(v)
	case ssaop.OpMod16u:
		return rewriteValueARM_OpMod16u(v)
	case ssaop.OpMod32:
		return rewriteValueARM_OpMod32(v)
	case ssaop.OpMod32u:
		return rewriteValueARM_OpMod32u(v)
	case ssaop.OpMod8:
		return rewriteValueARM_OpMod8(v)
	case ssaop.OpMod8u:
		return rewriteValueARM_OpMod8u(v)
	case ssaop.OpMove:
		return rewriteValueARM_OpMove(v)
	case ssaop.OpMul16:
		v.Op = ssaop.OpARMMUL
		return true
	case ssaop.OpMul32:
		v.Op = ssaop.OpARMMUL
		return true
	case ssaop.OpMul32F:
		v.Op = ssaop.OpARMMULF
		return true
	case ssaop.OpMul32uhilo:
		v.Op = ssaop.OpARMMULLU
		return true
	case ssaop.OpMul64F:
		v.Op = ssaop.OpARMMULD
		return true
	case ssaop.OpMul8:
		v.Op = ssaop.OpARMMUL
		return true
	case ssaop.OpNeg16:
		return rewriteValueARM_OpNeg16(v)
	case ssaop.OpNeg32:
		return rewriteValueARM_OpNeg32(v)
	case ssaop.OpNeg32F:
		v.Op = ssaop.OpARMNEGF
		return true
	case ssaop.OpNeg64F:
		v.Op = ssaop.OpARMNEGD
		return true
	case ssaop.OpNeg8:
		return rewriteValueARM_OpNeg8(v)
	case ssaop.OpNeq16:
		return rewriteValueARM_OpNeq16(v)
	case ssaop.OpNeq32:
		return rewriteValueARM_OpNeq32(v)
	case ssaop.OpNeq32F:
		return rewriteValueARM_OpNeq32F(v)
	case ssaop.OpNeq64F:
		return rewriteValueARM_OpNeq64F(v)
	case ssaop.OpNeq8:
		return rewriteValueARM_OpNeq8(v)
	case ssaop.OpNeqB:
		v.Op = ssaop.OpARMXOR
		return true
	case ssaop.OpNeqPtr:
		return rewriteValueARM_OpNeqPtr(v)
	case ssaop.OpNilCheck:
		v.Op = ssaop.OpARMLoweredNilCheck
		return true
	case ssaop.OpNot:
		return rewriteValueARM_OpNot(v)
	case ssaop.OpOffPtr:
		return rewriteValueARM_OpOffPtr(v)
	case ssaop.OpOr16:
		v.Op = ssaop.OpARMOR
		return true
	case ssaop.OpOr32:
		v.Op = ssaop.OpARMOR
		return true
	case ssaop.OpOr8:
		v.Op = ssaop.OpARMOR
		return true
	case ssaop.OpOrB:
		v.Op = ssaop.OpARMOR
		return true
	case ssaop.OpPanicBounds:
		v.Op = ssaop.OpARMLoweredPanicBoundsRR
		return true
	case ssaop.OpPanicExtend:
		v.Op = ssaop.OpARMLoweredPanicExtendRR
		return true
	case ssaop.OpRotateLeft16:
		return rewriteValueARM_OpRotateLeft16(v)
	case ssaop.OpRotateLeft32:
		return rewriteValueARM_OpRotateLeft32(v)
	case ssaop.OpRotateLeft8:
		return rewriteValueARM_OpRotateLeft8(v)
	case ssaop.OpRound32F:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpRound64F:
		v.Op = ssaop.OpCopy
		return true
	case ssaop.OpRsh16Ux16:
		return rewriteValueARM_OpRsh16Ux16(v)
	case ssaop.OpRsh16Ux32:
		return rewriteValueARM_OpRsh16Ux32(v)
	case ssaop.OpRsh16Ux64:
		return rewriteValueARM_OpRsh16Ux64(v)
	case ssaop.OpRsh16Ux8:
		return rewriteValueARM_OpRsh16Ux8(v)
	case ssaop.OpRsh16x16:
		return rewriteValueARM_OpRsh16x16(v)
	case ssaop.OpRsh16x32:
		return rewriteValueARM_OpRsh16x32(v)
	case ssaop.OpRsh16x64:
		return rewriteValueARM_OpRsh16x64(v)
	case ssaop.OpRsh16x8:
		return rewriteValueARM_OpRsh16x8(v)
	case ssaop.OpRsh32Ux16:
		return rewriteValueARM_OpRsh32Ux16(v)
	case ssaop.OpRsh32Ux32:
		return rewriteValueARM_OpRsh32Ux32(v)
	case ssaop.OpRsh32Ux64:
		return rewriteValueARM_OpRsh32Ux64(v)
	case ssaop.OpRsh32Ux8:
		return rewriteValueARM_OpRsh32Ux8(v)
	case ssaop.OpRsh32x16:
		return rewriteValueARM_OpRsh32x16(v)
	case ssaop.OpRsh32x32:
		return rewriteValueARM_OpRsh32x32(v)
	case ssaop.OpRsh32x64:
		return rewriteValueARM_OpRsh32x64(v)
	case ssaop.OpRsh32x8:
		return rewriteValueARM_OpRsh32x8(v)
	case ssaop.OpRsh8Ux16:
		return rewriteValueARM_OpRsh8Ux16(v)
	case ssaop.OpRsh8Ux32:
		return rewriteValueARM_OpRsh8Ux32(v)
	case ssaop.OpRsh8Ux64:
		return rewriteValueARM_OpRsh8Ux64(v)
	case ssaop.OpRsh8Ux8:
		return rewriteValueARM_OpRsh8Ux8(v)
	case ssaop.OpRsh8x16:
		return rewriteValueARM_OpRsh8x16(v)
	case ssaop.OpRsh8x32:
		return rewriteValueARM_OpRsh8x32(v)
	case ssaop.OpRsh8x64:
		return rewriteValueARM_OpRsh8x64(v)
	case ssaop.OpRsh8x8:
		return rewriteValueARM_OpRsh8x8(v)
	case ssaop.OpSelect0:
		return rewriteValueARM_OpSelect0(v)
	case ssaop.OpSelect1:
		return rewriteValueARM_OpSelect1(v)
	case ssaop.OpSignExt16to32:
		v.Op = ssaop.OpARMMOVHreg
		return true
	case ssaop.OpSignExt8to16:
		v.Op = ssaop.OpARMMOVBreg
		return true
	case ssaop.OpSignExt8to32:
		v.Op = ssaop.OpARMMOVBreg
		return true
	case ssaop.OpSignmask:
		return rewriteValueARM_OpSignmask(v)
	case ssaop.OpSlicemask:
		return rewriteValueARM_OpSlicemask(v)
	case ssaop.OpSqrt:
		v.Op = ssaop.OpARMSQRTD
		return true
	case ssaop.OpSqrt32:
		v.Op = ssaop.OpARMSQRTF
		return true
	case ssaop.OpStaticCall:
		v.Op = ssaop.OpARMCALLstatic
		return true
	case ssaop.OpStore:
		return rewriteValueARM_OpStore(v)
	case ssaop.OpSub16:
		v.Op = ssaop.OpARMSUB
		return true
	case ssaop.OpSub32:
		v.Op = ssaop.OpARMSUB
		return true
	case ssaop.OpSub32F:
		v.Op = ssaop.OpARMSUBF
		return true
	case ssaop.OpSub32carry:
		v.Op = ssaop.OpARMSUBS
		return true
	case ssaop.OpSub32withcarry:
		v.Op = ssaop.OpARMSBC
		return true
	case ssaop.OpSub64F:
		v.Op = ssaop.OpARMSUBD
		return true
	case ssaop.OpSub8:
		v.Op = ssaop.OpARMSUB
		return true
	case ssaop.OpSubPtr:
		v.Op = ssaop.OpARMSUB
		return true
	case ssaop.OpTailCall:
		v.Op = ssaop.OpARMCALLtail
		return true
	case ssaop.OpTailCallInter:
		v.Op = ssaop.OpARMCALLtailinter
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
	case ssaop.OpWB:
		v.Op = ssaop.OpARMLoweredWB
		return true
	case ssaop.OpXor16:
		v.Op = ssaop.OpARMXOR
		return true
	case ssaop.OpXor32:
		v.Op = ssaop.OpARMXOR
		return true
	case ssaop.OpXor8:
		v.Op = ssaop.OpARMXOR
		return true
	case ssaop.OpZero:
		return rewriteValueARM_OpZero(v)
	case ssaop.OpZeroExt16to32:
		v.Op = ssaop.OpARMMOVHUreg
		return true
	case ssaop.OpZeroExt8to16:
		v.Op = ssaop.OpARMMOVBUreg
		return true
	case ssaop.OpZeroExt8to32:
		v.Op = ssaop.OpARMMOVBUreg
		return true
	case ssaop.OpZeromask:
		return rewriteValueARM_OpZeromask(v)
	}
	return false
}
func rewriteValueARM_OpARMADC(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADC (MOVWconst [c]) x flags)
	// result: (ADCconst [c] x flags)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_0.AuxInt)
			x := v_1
			flags := v_2
			v.Reset(ssaop.OpARMADCconst)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, flags)
			return true
		}
		break
	}
	// match: (ADC x (SLLconst [c] y) flags)
	// result: (ADCshiftLL x y [c] flags)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			flags := v_2
			v.Reset(ssaop.OpARMADCshiftLL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg3(x, y, flags)
			return true
		}
		break
	}
	// match: (ADC x (SRLconst [c] y) flags)
	// result: (ADCshiftRL x y [c] flags)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			flags := v_2
			v.Reset(ssaop.OpARMADCshiftRL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg3(x, y, flags)
			return true
		}
		break
	}
	// match: (ADC x (SRAconst [c] y) flags)
	// result: (ADCshiftRA x y [c] flags)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRAconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			flags := v_2
			v.Reset(ssaop.OpARMADCshiftRA)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg3(x, y, flags)
			return true
		}
		break
	}
	// match: (ADC x (SLL y z) flags)
	// result: (ADCshiftLLreg x y z flags)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			flags := v_2
			v.Reset(ssaop.OpARMADCshiftLLreg)
			v.AddArg4(x, y, z, flags)
			return true
		}
		break
	}
	// match: (ADC x (SRL y z) flags)
	// result: (ADCshiftRLreg x y z flags)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			flags := v_2
			v.Reset(ssaop.OpARMADCshiftRLreg)
			v.AddArg4(x, y, z, flags)
			return true
		}
		break
	}
	// match: (ADC x (SRA y z) flags)
	// result: (ADCshiftRAreg x y z flags)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRA {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			flags := v_2
			v.Reset(ssaop.OpARMADCshiftRAreg)
			v.AddArg4(x, y, z, flags)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMADCconst(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADCconst [c] (ADDconst [d] x) flags)
	// result: (ADCconst [c+d] x flags)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		flags := v_1
		v.Reset(ssaop.OpARMADCconst)
		v.AuxInt = Int32ToAuxInt(c + d)
		v.AddArg2(x, flags)
		return true
	}
	// match: (ADCconst [c] (SUBconst [d] x) flags)
	// result: (ADCconst [c-d] x flags)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		flags := v_1
		v.Reset(ssaop.OpARMADCconst)
		v.AuxInt = Int32ToAuxInt(c - d)
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCshiftLL(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADCshiftLL (MOVWconst [c]) x [d] flags)
	// result: (ADCconst [c] (SLLconst <x.Type> x [d]) flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMADCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (ADCshiftLL x (MOVWconst [c]) [d] flags)
	// result: (ADCconst x [c<<uint64(d)] flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		flags := v_2
		v.Reset(ssaop.OpARMADCconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCshiftLLreg(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADCshiftLLreg (MOVWconst [c]) x y flags)
	// result: (ADCconst [c] (SLL <x.Type> x y) flags)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		flags := v_3
		v.Reset(ssaop.OpARMADCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (ADCshiftLLreg x y (MOVWconst [c]) flags)
	// cond: 0 <= c && c < 32
	// result: (ADCshiftLL x y [c] flags)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		flags := v_3
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMADCshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCshiftRA(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADCshiftRA (MOVWconst [c]) x [d] flags)
	// result: (ADCconst [c] (SRAconst <x.Type> x [d]) flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMADCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (ADCshiftRA x (MOVWconst [c]) [d] flags)
	// result: (ADCconst x [c>>uint64(d)] flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		flags := v_2
		v.Reset(ssaop.OpARMADCconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCshiftRAreg(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADCshiftRAreg (MOVWconst [c]) x y flags)
	// result: (ADCconst [c] (SRA <x.Type> x y) flags)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		flags := v_3
		v.Reset(ssaop.OpARMADCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (ADCshiftRAreg x y (MOVWconst [c]) flags)
	// cond: 0 <= c && c < 32
	// result: (ADCshiftRA x y [c] flags)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		flags := v_3
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMADCshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCshiftRL(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADCshiftRL (MOVWconst [c]) x [d] flags)
	// result: (ADCconst [c] (SRLconst <x.Type> x [d]) flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMADCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (ADCshiftRL x (MOVWconst [c]) [d] flags)
	// result: (ADCconst x [int32(uint32(c)>>uint64(d))] flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		flags := v_2
		v.Reset(ssaop.OpARMADCconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADCshiftRLreg(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADCshiftRLreg (MOVWconst [c]) x y flags)
	// result: (ADCconst [c] (SRL <x.Type> x y) flags)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		flags := v_3
		v.Reset(ssaop.OpARMADCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (ADCshiftRLreg x y (MOVWconst [c]) flags)
	// cond: 0 <= c && c < 32
	// result: (ADCshiftRL x y [c] flags)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		flags := v_3
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMADCshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADD x (MOVWconst <t> [c]))
	// cond: !t.IsPtr()
	// result: (ADDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			t := v_1.Type
			c := AuxIntToInt32(v_1.AuxInt)
			if !(!t.IsPtr()) {
				continue
			}
			v.Reset(ssaop.OpARMADDconst)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADD x (SLLconst [c] y))
	// result: (ADDshiftLL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMADDshiftLL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADD x (SRLconst [c] y))
	// result: (ADDshiftRL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMADDshiftRL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADD x (SRAconst [c] y))
	// result: (ADDshiftRA x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRAconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMADDshiftRA)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADD x (SLL y z))
	// result: (ADDshiftLLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMADDshiftLLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (ADD x (SRL y z))
	// result: (ADDshiftRLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMADDshiftRLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (ADD x (SRA y z))
	// result: (ADDshiftRAreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRA {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMADDshiftRAreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (ADD x (RSBconst [0] y))
	// result: (SUB x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMRSBconst || AuxIntToInt32(v_1.AuxInt) != 0 {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMSUB)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADD <t> (RSBconst [c] x) (RSBconst [d] y))
	// result: (RSBconst [c+d] (ADD <t> x y))
	for {
		t := v.Type
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARMRSBconst {
				continue
			}
			c := AuxIntToInt32(v_0.AuxInt)
			x := v_0.Args[0]
			if v_1.Op != ssaop.OpARMRSBconst {
				continue
			}
			d := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMRSBconst)
			v.AuxInt = Int32ToAuxInt(c + d)
			v0 := b.NewValue0(v.Pos, ssaop.OpARMADD, t)
			v0.AddArg2(x, y)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (ADD (MUL x y) a)
	// result: (MULA x y a)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARMMUL {
				continue
			}
			y := v_0.Args[1]
			x := v_0.Args[0]
			a := v_1
			v.Reset(ssaop.OpARMMULA)
			v.AddArg3(x, y, a)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMADDD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDD a (MULD x y))
	// cond: a.Uses == 1 && buildcfg.GOARM.Version >= 6
	// result: (MULAD a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != ssaop.OpARMMULD {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Uses == 1 && buildcfg.GOARM.Version >= 6) {
				continue
			}
			v.Reset(ssaop.OpARMMULAD)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (ADDD a (NMULD x y))
	// cond: a.Uses == 1 && buildcfg.GOARM.Version >= 6
	// result: (MULSD a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != ssaop.OpARMNMULD {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Uses == 1 && buildcfg.GOARM.Version >= 6) {
				continue
			}
			v.Reset(ssaop.OpARMMULSD)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMADDF(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDF a (MULF x y))
	// cond: a.Uses == 1 && buildcfg.GOARM.Version >= 6
	// result: (MULAF a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != ssaop.OpARMMULF {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Uses == 1 && buildcfg.GOARM.Version >= 6) {
				continue
			}
			v.Reset(ssaop.OpARMMULAF)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	// match: (ADDF a (NMULF x y))
	// cond: a.Uses == 1 && buildcfg.GOARM.Version >= 6
	// result: (MULSF a x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			a := v_0
			if v_1.Op != ssaop.OpARMNMULF {
				continue
			}
			y := v_1.Args[1]
			x := v_1.Args[0]
			if !(a.Uses == 1 && buildcfg.GOARM.Version >= 6) {
				continue
			}
			v.Reset(ssaop.OpARMMULSF)
			v.AddArg3(a, x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMADDS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (ADDS x (MOVWconst [c]))
	// result: (ADDSconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.OpARMADDSconst)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (ADDS x (SLLconst [c] y))
	// result: (ADDSshiftLL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMADDSshiftLL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDS x (SRLconst [c] y))
	// result: (ADDSshiftRL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMADDSshiftRL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDS x (SRAconst [c] y))
	// result: (ADDSshiftRA x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRAconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMADDSshiftRA)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (ADDS x (SLL y z))
	// result: (ADDSshiftLLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMADDSshiftLLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (ADDS x (SRL y z))
	// result: (ADDSshiftRLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMADDSshiftRLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (ADDS x (SRA y z))
	// result: (ADDSshiftRAreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRA {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMADDSshiftRAreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMADDSshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDSshiftLL (MOVWconst [c]) x [d])
	// result: (ADDSconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMADDSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDSshiftLL x (MOVWconst [c]) [d])
	// result: (ADDSconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMADDSconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDSshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDSshiftLLreg (MOVWconst [c]) x y)
	// result: (ADDSconst [c] (SLL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMADDSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (ADDSshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (ADDSshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMADDSshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDSshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDSshiftRA (MOVWconst [c]) x [d])
	// result: (ADDSconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMADDSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDSshiftRA x (MOVWconst [c]) [d])
	// result: (ADDSconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMADDSconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDSshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDSshiftRAreg (MOVWconst [c]) x y)
	// result: (ADDSconst [c] (SRA <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMADDSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (ADDSshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (ADDSshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMADDSshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDSshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDSshiftRL (MOVWconst [c]) x [d])
	// result: (ADDSconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMADDSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDSshiftRL x (MOVWconst [c]) [d])
	// result: (ADDSconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMADDSconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDSshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDSshiftRLreg (MOVWconst [c]) x y)
	// result: (ADDSconst [c] (SRL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMADDSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (ADDSshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (ADDSshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMADDSshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ADDconst [off1] (MOVWaddr [off2] {sym} ptr))
	// result: (MOVWaddr [off1+off2] {sym} ptr)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		v.Reset(ssaop.OpARMMOVWaddr)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg(ptr)
		return true
	}
	// match: (ADDconst [0] x)
	// result: x
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ADDconst [c] x)
	// cond: !isARMImmRot(uint32(c)) && isARMImmRot(uint32(-c))
	// result: (SUBconst [-c] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(!isARMImmRot(uint32(c)) && isARMImmRot(uint32(-c))) {
			break
		}
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] x)
	// cond: buildcfg.GOARM.Version==7 && !isARMImmRot(uint32(c)) && uint32(c)>0xffff && uint32(-c)<=0xffff
	// result: (SUBconst [-c] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(buildcfg.GOARM.Version == 7 && !isARMImmRot(uint32(c)) && uint32(c) > 0xffff && uint32(-c) <= 0xffff) {
			break
		}
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (MOVWconst [d]))
	// result: (MOVWconst [c+d])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(c + d)
		return true
	}
	// match: (ADDconst [c] (ADDconst [d] x))
	// result: (ADDconst [c+d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (SUBconst [d] x))
	// result: (ADDconst [c-d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	// match: (ADDconst [c] (RSBconst [d] x))
	// result: (RSBconst [c+d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMRSBconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ADDshiftLL (MOVWconst [c]) x [d])
	// result: (ADDconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftLL x (MOVWconst [c]) [d])
	// result: (ADDconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL <typ.UInt16> [8] (BFXU <typ.UInt16> [int32(ArmBFAuxInt(8, 8))] x) x)
	// result: (REV16 x)
	for {
		if v.Type != typ.UInt16 || AuxIntToInt32(v.AuxInt) != 8 || v_0.Op != ssaop.OpARMBFXU || v_0.Type != typ.UInt16 || AuxIntToInt32(v_0.AuxInt) != int32(ArmBFAuxInt(8, 8)) {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMREV16)
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL <typ.UInt16> [8] (SRLconst <typ.UInt16> [24] (SLLconst [16] x)) x)
	// cond: buildcfg.GOARM.Version>=6
	// result: (REV16 x)
	for {
		if v.Type != typ.UInt16 || AuxIntToInt32(v.AuxInt) != 8 || v_0.Op != ssaop.OpARMSRLconst || v_0.Type != typ.UInt16 || AuxIntToInt32(v_0.AuxInt) != 24 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARMSLLconst || AuxIntToInt32(v_0_0.AuxInt) != 16 {
			break
		}
		x := v_0_0.Args[0]
		if x != v_1 || !(buildcfg.GOARM.Version >= 6) {
			break
		}
		v.Reset(ssaop.OpARMREV16)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDshiftLLreg (MOVWconst [c]) x y)
	// result: (ADDconst [c] (SLL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (ADDshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMADDshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDshiftRA (MOVWconst [c]) x [d])
	// result: (ADDconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRA x (MOVWconst [c]) [d])
	// result: (ADDconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDshiftRAreg (MOVWconst [c]) x y)
	// result: (ADDconst [c] (SRA <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (ADDshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMADDshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDshiftRL (MOVWconst [c]) x [d])
	// result: (ADDconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRL x (MOVWconst [c]) [d])
	// result: (ADDconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMADDshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ADDshiftRLreg (MOVWconst [c]) x y)
	// result: (ADDconst [c] (SRL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (ADDshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (ADDshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMADDshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMAND(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (AND x (MOVWconst [c]))
	// result: (ANDconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.OpARMANDconst)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (AND x (SLLconst [c] y))
	// result: (ANDshiftLL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMANDshiftLL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (AND x (SRLconst [c] y))
	// result: (ANDshiftRL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMANDshiftRL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (AND x (SRAconst [c] y))
	// result: (ANDshiftRA x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRAconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMANDshiftRA)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (AND x (SLL y z))
	// result: (ANDshiftLLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMANDshiftLLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (AND x (SRL y z))
	// result: (ANDshiftRLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMANDshiftRLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (AND x (SRA y z))
	// result: (ANDshiftRAreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRA {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMANDshiftRAreg)
			v.AddArg3(x, y, z)
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
			if v_1.Op != ssaop.OpARMMVN {
				continue
			}
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMBIC)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (AND x (MVNshiftLL y [c]))
	// result: (BICshiftLL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMVNshiftLL {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMBICshiftLL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (AND x (MVNshiftRL y [c]))
	// result: (BICshiftRL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMVNshiftRL {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMBICshiftRL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (AND x (MVNshiftRA y [c]))
	// result: (BICshiftRA x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMVNshiftRA {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMBICshiftRA)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMANDconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ANDconst [0] _)
	// result: (MOVWconst [0])
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	// match: (ANDconst [c] x)
	// cond: int32(c)==-1
	// result: x
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(int32(c) == -1) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (ANDconst [c] x)
	// cond: !isARMImmRot(uint32(c)) && isARMImmRot(^uint32(c))
	// result: (BICconst [int32(^uint32(c))] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(!isARMImmRot(uint32(c)) && isARMImmRot(^uint32(c))) {
			break
		}
		v.Reset(ssaop.OpARMBICconst)
		v.AuxInt = Int32ToAuxInt(int32(^uint32(c)))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] x)
	// cond: buildcfg.GOARM.Version==7 && !isARMImmRot(uint32(c)) && uint32(c)>0xffff && ^uint32(c)<=0xffff
	// result: (BICconst [int32(^uint32(c))] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(buildcfg.GOARM.Version == 7 && !isARMImmRot(uint32(c)) && uint32(c) > 0xffff && ^uint32(c) <= 0xffff) {
			break
		}
		v.Reset(ssaop.OpARMBICconst)
		v.AuxInt = Int32ToAuxInt(int32(^uint32(c)))
		v.AddArg(x)
		return true
	}
	// match: (ANDconst [c] (MOVWconst [d]))
	// result: (MOVWconst [c&d])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(c & d)
		return true
	}
	// match: (ANDconst [c] (ANDconst [d] x))
	// result: (ANDconst [c&d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMANDconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c & d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMANDshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftLL (MOVWconst [c]) x [d])
	// result: (ANDconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftLL x (MOVWconst [c]) [d])
	// result: (ANDconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftLL y:(SLLconst x [c]) x [c])
	// result: y
	for {
		c := AuxIntToInt32(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARMSLLconst || AuxIntToInt32(y.AuxInt) != c {
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
func rewriteValueARM_OpARMANDshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftLLreg (MOVWconst [c]) x y)
	// result: (ANDconst [c] (SLL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (ANDshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMANDshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMANDshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftRA (MOVWconst [c]) x [d])
	// result: (ANDconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRA x (MOVWconst [c]) [d])
	// result: (ANDconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftRA y:(SRAconst x [c]) x [c])
	// result: y
	for {
		c := AuxIntToInt32(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARMSRAconst || AuxIntToInt32(y.AuxInt) != c {
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
func rewriteValueARM_OpARMANDshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftRAreg (MOVWconst [c]) x y)
	// result: (ANDconst [c] (SRA <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (ANDshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMANDshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMANDshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftRL (MOVWconst [c]) x [d])
	// result: (ANDconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRL x (MOVWconst [c]) [d])
	// result: (ANDconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ANDshiftRL y:(SRLconst x [c]) x [c])
	// result: y
	for {
		c := AuxIntToInt32(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARMSRLconst || AuxIntToInt32(y.AuxInt) != c {
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
func rewriteValueARM_OpARMANDshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ANDshiftRLreg (MOVWconst [c]) x y)
	// result: (ANDconst [c] (SRL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (ANDshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (ANDshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMANDshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMBFX(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (BFX [c] (MOVWconst [d]))
	// result: (MOVWconst [d<<(32-uint32(c&0xff)-uint32(c>>8))>>(32-uint32(c>>8))])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(d << (32 - uint32(c&0xff) - uint32(c>>8)) >> (32 - uint32(c>>8)))
		return true
	}
	return false
}
func rewriteValueARM_OpARMBFXU(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (BFXU [c] (MOVWconst [d]))
	// result: (MOVWconst [int32(uint32(d)<<(32-uint32(c&0xff)-uint32(c>>8))>>(32-uint32(c>>8)))])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(d) << (32 - uint32(c&0xff) - uint32(c>>8)) >> (32 - uint32(c>>8))))
		return true
	}
	return false
}
func rewriteValueARM_OpARMBIC(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BIC x (MOVWconst [c]))
	// result: (BICconst [c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMBICconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (BIC x (SLLconst [c] y))
	// result: (BICshiftLL x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMBICshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (BIC x (SRLconst [c] y))
	// result: (BICshiftRL x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMBICshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (BIC x (SRAconst [c] y))
	// result: (BICshiftRA x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMBICshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (BIC x (SLL y z))
	// result: (BICshiftLLreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMBICshiftLLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (BIC x (SRL y z))
	// result: (BICshiftRLreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMBICshiftRLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (BIC x (SRA y z))
	// result: (BICshiftRAreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMBICshiftRAreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (BIC x x)
	// result: (MOVWconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (BICconst [0] x)
	// result: x
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (BICconst [c] _)
	// cond: int32(c)==-1
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if !(int32(c) == -1) {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	// match: (BICconst [c] x)
	// cond: !isARMImmRot(uint32(c)) && isARMImmRot(^uint32(c))
	// result: (ANDconst [int32(^uint32(c))] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(!isARMImmRot(uint32(c)) && isARMImmRot(^uint32(c))) {
			break
		}
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(int32(^uint32(c)))
		v.AddArg(x)
		return true
	}
	// match: (BICconst [c] x)
	// cond: buildcfg.GOARM.Version==7 && !isARMImmRot(uint32(c)) && uint32(c)>0xffff && ^uint32(c)<=0xffff
	// result: (ANDconst [int32(^uint32(c))] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(buildcfg.GOARM.Version == 7 && !isARMImmRot(uint32(c)) && uint32(c) > 0xffff && ^uint32(c) <= 0xffff) {
			break
		}
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(int32(^uint32(c)))
		v.AddArg(x)
		return true
	}
	// match: (BICconst [c] (MOVWconst [d]))
	// result: (MOVWconst [d&^c])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(d &^ c)
		return true
	}
	// match: (BICconst [c] (BICconst [d] x))
	// result: (BICconst [c|d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMBICconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMBICconst)
		v.AuxInt = Int32ToAuxInt(c | d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftLL x (MOVWconst [c]) [d])
	// result: (BICconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMBICconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftLL (SLLconst x [c]) x [c])
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSLLconst || AuxIntToInt32(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (BICshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMBICshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftRA x (MOVWconst [c]) [d])
	// result: (BICconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMBICconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftRA (SRAconst x [c]) x [c])
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSRAconst || AuxIntToInt32(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (BICshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMBICshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftRL x (MOVWconst [c]) [d])
	// result: (BICconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMBICconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (BICshiftRL (SRLconst x [c]) x [c])
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSRLconst || AuxIntToInt32(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMBICshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (BICshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (BICshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMBICshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMN(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMN x (MOVWconst [c]))
	// result: (CMNconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.OpARMCMNconst)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (CMN x (SLLconst [c] y))
	// result: (CMNshiftLL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMCMNshiftLL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (CMN x (SRLconst [c] y))
	// result: (CMNshiftRL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMCMNshiftRL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (CMN x (SRAconst [c] y))
	// result: (CMNshiftRA x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRAconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMCMNshiftRA)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (CMN x (SLL y z))
	// result: (CMNshiftLLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMCMNshiftLLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (CMN x (SRL y z))
	// result: (CMNshiftRLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMCMNshiftRLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (CMN x (SRA y z))
	// result: (CMNshiftRAreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRA {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMCMNshiftRAreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMCMNconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (CMNconst (MOVWconst [x]) [y])
	// result: (FlagConstant [AddFlags32(x,y)])
	for {
		y := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		x := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMFlagConstant)
		v.AuxInt = FlagConstantToAuxInt(AddFlags32(x, y))
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMNshiftLL (MOVWconst [c]) x [d])
	// result: (CMNconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMCMNconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftLL x (MOVWconst [c]) [d])
	// result: (CMNconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMCMNconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMNshiftLLreg (MOVWconst [c]) x y)
	// result: (CMNconst [c] (SLL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMCMNconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (CMNshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMCMNshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMNshiftRA (MOVWconst [c]) x [d])
	// result: (CMNconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMCMNconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRA x (MOVWconst [c]) [d])
	// result: (CMNconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMCMNconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMNshiftRAreg (MOVWconst [c]) x y)
	// result: (CMNconst [c] (SRA <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMCMNconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (CMNshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMCMNshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMNshiftRL (MOVWconst [c]) x [d])
	// result: (CMNconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMCMNconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRL x (MOVWconst [c]) [d])
	// result: (CMNconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMCMNconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMNshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMNshiftRLreg (MOVWconst [c]) x y)
	// result: (CMNconst [c] (SRL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMCMNconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (CMNshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (CMNshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMCMNshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMOVWHSconst(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVWHSconst _ (FlagConstant [fc]) [c])
	// cond: fc.Uge()
	// result: (MOVWconst [c])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_1.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_1.AuxInt)
		if !(fc.Uge()) {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(c)
		return true
	}
	// match: (CMOVWHSconst x (FlagConstant [fc]) [c])
	// cond: fc.Ult()
	// result: x
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_1.AuxInt)
		if !(fc.Ult()) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (CMOVWHSconst x (InvertFlags flags) [c])
	// result: (CMOVWLSconst x flags [c])
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMInvertFlags {
			break
		}
		flags := v_1.Args[0]
		v.Reset(ssaop.OpARMCMOVWLSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMOVWLSconst(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMOVWLSconst _ (FlagConstant [fc]) [c])
	// cond: fc.Ule()
	// result: (MOVWconst [c])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_1.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_1.AuxInt)
		if !(fc.Ule()) {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(c)
		return true
	}
	// match: (CMOVWLSconst x (FlagConstant [fc]) [c])
	// cond: fc.Ugt()
	// result: x
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_1.AuxInt)
		if !(fc.Ugt()) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (CMOVWLSconst x (InvertFlags flags) [c])
	// result: (CMOVWHSconst x flags [c])
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMInvertFlags {
			break
		}
		flags := v_1.Args[0]
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMP(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMP x (MOVWconst [c]))
	// result: (CMPconst [c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMCMPconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (CMP (MOVWconst [c]) x)
	// result: (InvertFlags (CMPconst [c] x))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = Int32ToAuxInt(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x y)
	// cond: CanonLessThan(x,y)
	// result: (InvertFlags (CMP y x))
	for {
		x := v_0
		y := v_1
		if !(CanonLessThan(x, y)) {
			break
		}
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x (SLLconst [c] y))
	// result: (CMPshiftLL x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMCMPshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMP (SLLconst [c] y) x)
	// result: (InvertFlags (CMPshiftLL x y [c]))
	for {
		if v_0.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPshiftLL, types.TypeFlags)
		v0.AuxInt = Int32ToAuxInt(c)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x (SRLconst [c] y))
	// result: (CMPshiftRL x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMCMPshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMP (SRLconst [c] y) x)
	// result: (InvertFlags (CMPshiftRL x y [c]))
	for {
		if v_0.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPshiftRL, types.TypeFlags)
		v0.AuxInt = Int32ToAuxInt(c)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x (SRAconst [c] y))
	// result: (CMPshiftRA x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMCMPshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (CMP (SRAconst [c] y) x)
	// result: (InvertFlags (CMPshiftRA x y [c]))
	for {
		if v_0.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPshiftRA, types.TypeFlags)
		v0.AuxInt = Int32ToAuxInt(c)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x (SLL y z))
	// result: (CMPshiftLLreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMCMPshiftLLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (CMP (SLL y z) x)
	// result: (InvertFlags (CMPshiftLLreg x y z))
	for {
		if v_0.Op != ssaop.OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPshiftLLreg, types.TypeFlags)
		v0.AddArg3(x, y, z)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x (SRL y z))
	// result: (CMPshiftRLreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMCMPshiftRLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (CMP (SRL y z) x)
	// result: (InvertFlags (CMPshiftRLreg x y z))
	for {
		if v_0.Op != ssaop.OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPshiftRLreg, types.TypeFlags)
		v0.AddArg3(x, y, z)
		v.AddArg(v0)
		return true
	}
	// match: (CMP x (SRA y z))
	// result: (CMPshiftRAreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMCMPshiftRAreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (CMP (SRA y z) x)
	// result: (InvertFlags (CMPshiftRAreg x y z))
	for {
		if v_0.Op != ssaop.OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPshiftRAreg, types.TypeFlags)
		v0.AddArg3(x, y, z)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPD x (MOVDconst [0]))
	// result: (CMPD0 x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVDconst || AuxIntToFloat64(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpARMCMPD0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPF(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (CMPF x (MOVFconst [0]))
	// result: (CMPF0 x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVFconst || AuxIntToFloat64(v_1.AuxInt) != 0 {
			break
		}
		v.Reset(ssaop.OpARMCMPF0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (CMPconst (MOVWconst [x]) [y])
	// result: (FlagConstant [SubFlags32(x,y)])
	for {
		y := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		x := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMFlagConstant)
		v.AuxInt = FlagConstantToAuxInt(SubFlags32(x, y))
		return true
	}
	// match: (CMPconst (MOVBUreg _) [c])
	// cond: 0xff < c
	// result: (FlagConstant [SubFlags32(0, 1)])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVBUreg || !(0xff < c) {
			break
		}
		v.Reset(ssaop.OpARMFlagConstant)
		v.AuxInt = FlagConstantToAuxInt(SubFlags32(0, 1))
		return true
	}
	// match: (CMPconst (MOVHUreg _) [c])
	// cond: 0xffff < c
	// result: (FlagConstant [SubFlags32(0, 1)])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVHUreg || !(0xffff < c) {
			break
		}
		v.Reset(ssaop.OpARMFlagConstant)
		v.AuxInt = FlagConstantToAuxInt(SubFlags32(0, 1))
		return true
	}
	// match: (CMPconst (ANDconst _ [m]) [n])
	// cond: 0 <= m && m < n
	// result: (FlagConstant [SubFlags32(0, 1)])
	for {
		n := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMANDconst {
			break
		}
		m := AuxIntToInt32(v_0.AuxInt)
		if !(0 <= m && m < n) {
			break
		}
		v.Reset(ssaop.OpARMFlagConstant)
		v.AuxInt = FlagConstantToAuxInt(SubFlags32(0, 1))
		return true
	}
	// match: (CMPconst (SRLconst _ [c]) [n])
	// cond: 0 <= n && 0 < c && c <= 32 && (1<<uint32(32-c)) <= uint32(n)
	// result: (FlagConstant [SubFlags32(0, 1)])
	for {
		n := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		if !(0 <= n && 0 < c && c <= 32 && (1<<uint32(32-c)) <= uint32(n)) {
			break
		}
		v.Reset(ssaop.OpARMFlagConstant)
		v.AuxInt = FlagConstantToAuxInt(SubFlags32(0, 1))
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPshiftLL (MOVWconst [c]) x [d])
	// result: (InvertFlags (CMPconst [c] (SLLconst <x.Type> x [d])))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = Int32ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v1.AuxInt = Int32ToAuxInt(d)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftLL x (MOVWconst [c]) [d])
	// result: (CMPconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMCMPconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPshiftLLreg (MOVWconst [c]) x y)
	// result: (InvertFlags (CMPconst [c] (SLL <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = Int32ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (CMPshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMCMPshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPshiftRA (MOVWconst [c]) x [d])
	// result: (InvertFlags (CMPconst [c] (SRAconst <x.Type> x [d])))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = Int32ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v1.AuxInt = Int32ToAuxInt(d)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRA x (MOVWconst [c]) [d])
	// result: (CMPconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMCMPconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPshiftRAreg (MOVWconst [c]) x y)
	// result: (InvertFlags (CMPconst [c] (SRA <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = Int32ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (CMPshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMCMPshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPshiftRL (MOVWconst [c]) x [d])
	// result: (InvertFlags (CMPconst [c] (SRLconst <x.Type> x [d])))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = Int32ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v1.AuxInt = Int32ToAuxInt(d)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRL x (MOVWconst [c]) [d])
	// result: (CMPconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMCMPconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMCMPshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (CMPshiftRLreg (MOVWconst [c]) x y)
	// result: (InvertFlags (CMPconst [c] (SRL <x.Type> x y)))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMInvertFlags)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = Int32ToAuxInt(c)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (CMPshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (CMPshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMCMPshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMEqual(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Equal (FlagConstant [fc]))
	// result: (MOVWconst [B2i32(fc.Eq())])
	for {
		if v_0.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(B2i32(fc.Eq()))
		return true
	}
	// match: (Equal (InvertFlags x))
	// result: (Equal x)
	for {
		if v_0.Op != ssaop.OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMGreaterEqual(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterEqual (FlagConstant [fc]))
	// result: (MOVWconst [B2i32(fc.Ge())])
	for {
		if v_0.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(B2i32(fc.Ge()))
		return true
	}
	// match: (GreaterEqual (InvertFlags x))
	// result: (LessEqual x)
	for {
		if v_0.Op != ssaop.OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMLessEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMGreaterEqualU(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterEqualU (FlagConstant [fc]))
	// result: (MOVWconst [B2i32(fc.Uge())])
	for {
		if v_0.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(B2i32(fc.Uge()))
		return true
	}
	// match: (GreaterEqualU (InvertFlags x))
	// result: (LessEqualU x)
	for {
		if v_0.Op != ssaop.OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMLessEqualU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMGreaterThan(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterThan (FlagConstant [fc]))
	// result: (MOVWconst [B2i32(fc.Gt())])
	for {
		if v_0.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(B2i32(fc.Gt()))
		return true
	}
	// match: (GreaterThan (InvertFlags x))
	// result: (LessThan x)
	for {
		if v_0.Op != ssaop.OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMLessThan)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMGreaterThanU(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (GreaterThanU (FlagConstant [fc]))
	// result: (MOVWconst [B2i32(fc.Ugt())])
	for {
		if v_0.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(B2i32(fc.Ugt()))
		return true
	}
	// match: (GreaterThanU (InvertFlags x))
	// result: (LessThanU x)
	for {
		if v_0.Op != ssaop.OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMLessThanU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMLessEqual(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (LessEqual (FlagConstant [fc]))
	// result: (MOVWconst [B2i32(fc.Le())])
	for {
		if v_0.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(B2i32(fc.Le()))
		return true
	}
	// match: (LessEqual (InvertFlags x))
	// result: (GreaterEqual x)
	for {
		if v_0.Op != ssaop.OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMGreaterEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMLessEqualU(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (LessEqualU (FlagConstant [fc]))
	// result: (MOVWconst [B2i32(fc.Ule())])
	for {
		if v_0.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(B2i32(fc.Ule()))
		return true
	}
	// match: (LessEqualU (InvertFlags x))
	// result: (GreaterEqualU x)
	for {
		if v_0.Op != ssaop.OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMGreaterEqualU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMLessThan(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (LessThan (FlagConstant [fc]))
	// result: (MOVWconst [B2i32(fc.Lt())])
	for {
		if v_0.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(B2i32(fc.Lt()))
		return true
	}
	// match: (LessThan (InvertFlags x))
	// result: (GreaterThan x)
	for {
		if v_0.Op != ssaop.OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMGreaterThan)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMLessThanU(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (LessThanU (FlagConstant [fc]))
	// result: (MOVWconst [B2i32(fc.Ult())])
	for {
		if v_0.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(B2i32(fc.Ult()))
		return true
	}
	// match: (LessThanU (InvertFlags x))
	// result: (GreaterThanU x)
	for {
		if v_0.Op != ssaop.OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMGreaterThanU)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMLoweredPanicBoundsRC(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRC [kind] {p} (MOVWconst [c]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:int64(c), Cy:p.C}} mem)
	for {
		kind := AuxIntToInt64(v.AuxInt)
		p := AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		mem := v_1
		v.Reset(ssaop.OpARMLoweredPanicBoundsCC)
		v.AuxInt = Int64ToAuxInt(kind)
		v.Aux = PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: int64(c), Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMLoweredPanicBoundsRR(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicBoundsRR [kind] x (MOVWconst [c]) mem)
	// result: (LoweredPanicBoundsRC [kind] x {ssa.PanicBoundsC{C:int64(c)}} mem)
	for {
		kind := AuxIntToInt64(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpARMLoweredPanicBoundsRC)
		v.AuxInt = Int64ToAuxInt(kind)
		v.Aux = PanicBoundsCToAux(ssa.PanicBoundsC{C: int64(c)})
		v.AddArg2(x, mem)
		return true
	}
	// match: (LoweredPanicBoundsRR [kind] (MOVWconst [c]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {ssa.PanicBoundsC{C:int64(c)}} y mem)
	for {
		kind := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_1
		mem := v_2
		v.Reset(ssaop.OpARMLoweredPanicBoundsCR)
		v.AuxInt = Int64ToAuxInt(kind)
		v.Aux = PanicBoundsCToAux(ssa.PanicBoundsC{C: int64(c)})
		v.AddArg2(y, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMLoweredPanicExtendRC(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicExtendRC [kind] {p} (MOVWconst [hi]) (MOVWconst [lo]) mem)
	// result: (LoweredPanicBoundsCC [kind] {ssa.PanicBoundsCC{Cx:int64(hi)<<32+int64(uint32(lo)), Cy:p.C}} mem)
	for {
		kind := AuxIntToInt64(v.AuxInt)
		p := AuxToPanicBoundsC(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		hi := AuxIntToInt32(v_0.AuxInt)
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		lo := AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpARMLoweredPanicBoundsCC)
		v.AuxInt = Int64ToAuxInt(kind)
		v.Aux = PanicBoundsCCToAux(ssa.PanicBoundsCC{Cx: int64(hi)<<32 + int64(uint32(lo)), Cy: p.C})
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMLoweredPanicExtendRR(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (LoweredPanicExtendRR [kind] hi lo (MOVWconst [c]) mem)
	// result: (LoweredPanicExtendRC [kind] hi lo {ssa.PanicBoundsC{C:int64(c)}} mem)
	for {
		kind := AuxIntToInt64(v.AuxInt)
		hi := v_0
		lo := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		mem := v_3
		v.Reset(ssaop.OpARMLoweredPanicExtendRC)
		v.AuxInt = Int64ToAuxInt(kind)
		v.Aux = PanicBoundsCToAux(ssa.PanicBoundsC{C: int64(c)})
		v.AddArg3(hi, lo, mem)
		return true
	}
	// match: (LoweredPanicExtendRR [kind] (MOVWconst [hi]) (MOVWconst [lo]) y mem)
	// result: (LoweredPanicBoundsCR [kind] {ssa.PanicBoundsC{C:int64(hi)<<32 + int64(uint32(lo))}} y mem)
	for {
		kind := AuxIntToInt64(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		hi := AuxIntToInt32(v_0.AuxInt)
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		lo := AuxIntToInt32(v_1.AuxInt)
		y := v_2
		mem := v_3
		v.Reset(ssaop.OpARMLoweredPanicBoundsCR)
		v.AuxInt = Int64ToAuxInt(kind)
		v.Aux = PanicBoundsCToAux(ssa.PanicBoundsC{C: int64(hi)<<32 + int64(uint32(lo))})
		v.AddArg2(y, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// result: (MOVBUload [off1+off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVBUload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off1] {sym} (SUBconst [off2] ptr) mem)
	// result: (MOVBUload [off1-off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVBUload)
		v.AuxInt = Int32ToAuxInt(off1 - off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2)
	// result: (MOVBUload [off1+off2] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVBUload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVBUreg x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVBstore {
			break
		}
		off2 := AuxIntToInt32(v_1.AuxInt)
		sym2 := AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVBUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUload [0] {sym} (ADD ptr idx) mem)
	// cond: sym == nil
	// result: (MOVBUloadidx ptr idx mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVBUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVBUload [off] {sym} (SB) _)
	// cond: SymIsRO(sym)
	// result: (MOVWconst [int32(Read8(sym, int64(off)))])
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(Read8(sym, int64(off))))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBUloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBUloadidx ptr idx (MOVBstoreidx ptr2 idx x _))
	// cond: ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVBUreg x)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARMMOVBstoreidx {
			break
		}
		x := v_2.Args[2]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] || !(ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVBUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUloadidx ptr (MOVWconst [c]) mem)
	// result: (MOVBUload [c] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpARMMOVBUload)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBUloadidx (MOVWconst [c]) ptr mem)
	// result: (MOVBUload [c] ptr mem)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVBUload)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBUreg x:(MOVBUload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVBUload {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (ANDconst [c] x))
	// result: (ANDconst [c&0xff] x)
	for {
		if v_0.Op != ssaop.OpARMANDconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c & 0xff)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg x:(MOVBUreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVBUreg {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBUreg (MOVWconst [c]))
	// result: (MOVWconst [int32(uint8(c))])
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(uint8(c)))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBload [off1] {sym} (ADDconst [off2] ptr) mem)
	// result: (MOVBload [off1+off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVBload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off1] {sym} (SUBconst [off2] ptr) mem)
	// result: (MOVBload [off1-off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVBload)
		v.AuxInt = Int32ToAuxInt(off1 - off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2)
	// result: (MOVBload [off1+off2] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVBload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVBreg x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVBstore {
			break
		}
		off2 := AuxIntToInt32(v_1.AuxInt)
		sym2 := AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBload [0] {sym} (ADD ptr idx) mem)
	// cond: sym == nil
	// result: (MOVBloadidx ptr idx mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVBloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVBload [off] {sym} (SB) _)
	// cond: SymIsRO(sym)
	// result: (MOVWconst [int32(int8(Read8(sym, int64(off))))])
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(int8(Read8(sym, int64(off)))))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBloadidx ptr idx (MOVBstoreidx ptr2 idx x _))
	// cond: ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVBreg x)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARMMOVBstoreidx {
			break
		}
		x := v_2.Args[2]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] || !(ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVBreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBloadidx ptr (MOVWconst [c]) mem)
	// result: (MOVBload [c] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpARMMOVBload)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVBloadidx (MOVWconst [c]) ptr mem)
	// result: (MOVBload [c] ptr mem)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVBload)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVBreg x:(MOVBload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVBload {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (ANDconst [c] x))
	// cond: c & 0x80 == 0
	// result: (ANDconst [c&0x7f] x)
	for {
		if v_0.Op != ssaop.OpARMANDconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c&0x80 == 0) {
			break
		}
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c & 0x7f)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg x:(MOVBreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVBreg {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVBreg (MOVWconst [c]))
	// result: (MOVWconst [int32(int8(c))])
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(int8(c)))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// result: (MOVBstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVBstore)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym} (SUBconst [off2] ptr) val mem)
	// result: (MOVBstore [off1-off2] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVBstore)
		v.AuxInt = Int32ToAuxInt(off1 - off2)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: CanMergeSym(sym1,sym2)
	// result: (MOVBstore [off1+off2] {MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVBstore)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
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
		if v_1.Op != ssaop.OpARMMOVBreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARMMOVBstore)
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
		if v_1.Op != ssaop.OpARMMOVBUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARMMOVBstore)
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
		if v_1.Op != ssaop.OpARMMOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARMMOVBstore)
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
		if v_1.Op != ssaop.OpARMMOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARMMOVBstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVBstore [0] {sym} (ADD ptr idx) val mem)
	// cond: sym == nil
	// result: (MOVBstoreidx ptr idx val mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVBstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVBstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVBstoreidx ptr (MOVWconst [c]) val mem)
	// result: (MOVBstore [c] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVBstore)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVBstoreidx (MOVWconst [c]) ptr val mem)
	// result: (MOVBstore [c] ptr val mem)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		ptr := v_1
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVBstore)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVDload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDload [off1] {sym} (ADDconst [off2] ptr) mem)
	// result: (MOVDload [off1+off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVDload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off1] {sym} (SUBconst [off2] ptr) mem)
	// result: (MOVDload [off1-off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVDload)
		v.AuxInt = Int32ToAuxInt(off1 - off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2)
	// result: (MOVDload [off1+off2] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVDload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVDload [off] {sym} ptr (MOVDstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: x
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVDstore {
			break
		}
		off2 := AuxIntToInt32(v_1.AuxInt)
		sym2 := AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVDstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVDstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// result: (MOVDstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVDstore)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (SUBconst [off2] ptr) val mem)
	// result: (MOVDstore [off1-off2] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVDstore)
		v.AuxInt = Int32ToAuxInt(off1 - off2)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVDstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: CanMergeSym(sym1,sym2)
	// result: (MOVDstore [off1+off2] {MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVDstore)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVFload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVFload [off1] {sym} (ADDconst [off2] ptr) mem)
	// result: (MOVFload [off1+off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVFload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVFload [off1] {sym} (SUBconst [off2] ptr) mem)
	// result: (MOVFload [off1-off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVFload)
		v.AuxInt = Int32ToAuxInt(off1 - off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVFload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2)
	// result: (MOVFload [off1+off2] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVFload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVFload [off] {sym} ptr (MOVFstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: x
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVFstore {
			break
		}
		off2 := AuxIntToInt32(v_1.AuxInt)
		sym2 := AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVFstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVFstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// result: (MOVFstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVFstore)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstore [off1] {sym} (SUBconst [off2] ptr) val mem)
	// result: (MOVFstore [off1-off2] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVFstore)
		v.AuxInt = Int32ToAuxInt(off1 - off2)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVFstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: CanMergeSym(sym1,sym2)
	// result: (MOVFstore [off1+off2] {MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVFstore)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHUload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// result: (MOVHUload [off1+off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVHUload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off1] {sym} (SUBconst [off2] ptr) mem)
	// result: (MOVHUload [off1-off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVHUload)
		v.AuxInt = Int32ToAuxInt(off1 - off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2)
	// result: (MOVHUload [off1+off2] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVHUload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} ptr (MOVHstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVHUreg x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVHstore {
			break
		}
		off2 := AuxIntToInt32(v_1.AuxInt)
		sym2 := AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVHUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUload [0] {sym} (ADD ptr idx) mem)
	// cond: sym == nil
	// result: (MOVHUloadidx ptr idx mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVHUloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHUload [off] {sym} (SB) _)
	// cond: SymIsRO(sym)
	// result: (MOVWconst [int32(Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHUloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHUloadidx ptr idx (MOVHstoreidx ptr2 idx x _))
	// cond: ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVHUreg x)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARMMOVHstoreidx {
			break
		}
		x := v_2.Args[2]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] || !(ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVHUreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUloadidx ptr (MOVWconst [c]) mem)
	// result: (MOVHUload [c] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpARMMOVHUload)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHUloadidx (MOVWconst [c]) ptr mem)
	// result: (MOVHUload [c] ptr mem)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVHUload)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHUreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHUreg x:(MOVBUload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVBUload {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVHUload {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (ANDconst [c] x))
	// result: (ANDconst [c&0xffff] x)
	for {
		if v_0.Op != ssaop.OpARMANDconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c & 0xffff)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVBUreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVBUreg {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg x:(MOVHUreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVHUreg {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHUreg (MOVWconst [c]))
	// result: (MOVWconst [int32(uint16(c))])
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(uint16(c)))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVHload [off1] {sym} (ADDconst [off2] ptr) mem)
	// result: (MOVHload [off1+off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVHload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off1] {sym} (SUBconst [off2] ptr) mem)
	// result: (MOVHload [off1-off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVHload)
		v.AuxInt = Int32ToAuxInt(off1 - off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2)
	// result: (MOVHload [off1+off2] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVHload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHload [off] {sym} ptr (MOVHstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVHreg x)
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVHstore {
			break
		}
		off2 := AuxIntToInt32(v_1.AuxInt)
		sym2 := AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHload [0] {sym} (ADD ptr idx) mem)
	// cond: sym == nil
	// result: (MOVHloadidx ptr idx mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVHloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVHload [off] {sym} (SB) _)
	// cond: SymIsRO(sym)
	// result: (MOVWconst [int32(int16(Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder)))])
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(int16(Read16(sym, int64(off), config.Ctxt.Arch.ByteOrder))))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHloadidx ptr idx (MOVHstoreidx ptr2 idx x _))
	// cond: ssa.IsSamePtr(ptr, ptr2)
	// result: (MOVHreg x)
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARMMOVHstoreidx {
			break
		}
		x := v_2.Args[2]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] || !(ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVHreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHloadidx ptr (MOVWconst [c]) mem)
	// result: (MOVHload [c] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpARMMOVHload)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVHloadidx (MOVWconst [c]) ptr mem)
	// result: (MOVHload [c] ptr mem)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVHload)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVHreg x:(MOVBload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVBload {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVBUload {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHload _ _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVHload {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (ANDconst [c] x))
	// cond: c & 0x8000 == 0
	// result: (ANDconst [c&0x7fff] x)
	for {
		if v_0.Op != ssaop.OpARMANDconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(c&0x8000 == 0) {
			break
		}
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c & 0x7fff)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVBreg {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVBUreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVBUreg {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg x:(MOVHreg _))
	// result: (MOVWreg x)
	for {
		x := v_0
		if x.Op != ssaop.OpARMMOVHreg {
			break
		}
		v.Reset(ssaop.OpARMMOVWreg)
		v.AddArg(x)
		return true
	}
	// match: (MOVHreg (MOVWconst [c]))
	// result: (MOVWconst [int32(int16(c))])
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(int16(c)))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// result: (MOVHstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVHstore)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off1] {sym} (SUBconst [off2] ptr) val mem)
	// result: (MOVHstore [off1-off2] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVHstore)
		v.AuxInt = Int32ToAuxInt(off1 - off2)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: CanMergeSym(sym1,sym2)
	// result: (MOVHstore [off1+off2] {MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVHstore)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
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
		if v_1.Op != ssaop.OpARMMOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARMMOVHstore)
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
		if v_1.Op != ssaop.OpARMMOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARMMOVHstore)
		v.AuxInt = Int32ToAuxInt(off)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, x, mem)
		return true
	}
	// match: (MOVHstore [0] {sym} (ADD ptr idx) val mem)
	// cond: sym == nil
	// result: (MOVHstoreidx ptr idx val mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVHstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVHstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVHstoreidx ptr (MOVWconst [c]) val mem)
	// result: (MOVHstore [c] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVHstore)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVHstoreidx (MOVWconst [c]) ptr val mem)
	// result: (MOVHstore [c] ptr val mem)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		ptr := v_1
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVHstore)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWload(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	config := b.Func.Config
	// match: (MOVWload [off1] {sym} (ADDconst [off2] ptr) mem)
	// result: (MOVWload [off1+off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVWload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off1] {sym} (SUBconst [off2] ptr) mem)
	// result: (MOVWload [off1-off2] {sym} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		mem := v_1
		v.Reset(ssaop.OpARMMOVWload)
		v.AuxInt = Int32ToAuxInt(off1 - off2)
		v.Aux = SymToAux(sym)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) mem)
	// cond: CanMergeSym(sym1,sym2)
	// result: (MOVWload [off1+off2] {MergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		mem := v_1
		if !(CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVWload)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWload [off] {sym} ptr (MOVWstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)
	// result: x
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWstore {
			break
		}
		off2 := AuxIntToInt32(v_1.AuxInt)
		sym2 := AuxToSym(v_1.Aux)
		x := v_1.Args[1]
		ptr2 := v_1.Args[0]
		if !(sym == sym2 && off == off2 && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWload [0] {sym} (ADD ptr idx) mem)
	// cond: sym == nil
	// result: (MOVWloadidx ptr idx mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVWloadidx)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWload [0] {sym} (ADDshiftLL ptr idx [c]) mem)
	// cond: sym == nil
	// result: (MOVWloadshiftLL ptr idx [c] mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDshiftLL {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVWloadshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWload [0] {sym} (ADDshiftRL ptr idx [c]) mem)
	// cond: sym == nil
	// result: (MOVWloadshiftRL ptr idx [c] mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDshiftRL {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVWloadshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWload [0] {sym} (ADDshiftRA ptr idx [c]) mem)
	// cond: sym == nil
	// result: (MOVWloadshiftRA ptr idx [c] mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDshiftRA {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_1
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVWloadshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWload [off] {sym} (SB) _)
	// cond: SymIsRO(sym)
	// result: (MOVWconst [int32(Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder))])
	for {
		off := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpSB || !(SymIsRO(sym)) {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(Read32(sym, int64(off), config.Ctxt.Arch.ByteOrder)))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWloadidx(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWloadidx ptr idx (MOVWstoreidx ptr2 idx x _))
	// cond: ssa.IsSamePtr(ptr, ptr2)
	// result: x
	for {
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARMMOVWstoreidx {
			break
		}
		x := v_2.Args[2]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] || !(ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWloadidx ptr (MOVWconst [c]) mem)
	// result: (MOVWload [c] ptr mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpARMMOVWload)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWloadidx (MOVWconst [c]) ptr mem)
	// result: (MOVWload [c] ptr mem)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVWload)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (MOVWloadidx ptr (SLLconst idx [c]) mem)
	// result: (MOVWloadshiftLL ptr idx [c] mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		idx := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARMMOVWloadshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWloadidx (SLLconst idx [c]) ptr mem)
	// result: (MOVWloadshiftLL ptr idx [c] mem)
	for {
		if v_0.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVWloadshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWloadidx ptr (SRLconst idx [c]) mem)
	// result: (MOVWloadshiftRL ptr idx [c] mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		idx := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARMMOVWloadshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWloadidx (SRLconst idx [c]) ptr mem)
	// result: (MOVWloadshiftRL ptr idx [c] mem)
	for {
		if v_0.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVWloadshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWloadidx ptr (SRAconst idx [c]) mem)
	// result: (MOVWloadshiftRA ptr idx [c] mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		idx := v_1.Args[0]
		mem := v_2
		v.Reset(ssaop.OpARMMOVWloadshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	// match: (MOVWloadidx (SRAconst idx [c]) ptr mem)
	// result: (MOVWloadshiftRA ptr idx [c] mem)
	for {
		if v_0.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		idx := v_0.Args[0]
		ptr := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVWloadshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, idx, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWloadshiftLL(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWloadshiftLL ptr idx [c] (MOVWstoreshiftLL ptr2 idx [d] x _))
	// cond: c==d && ssa.IsSamePtr(ptr, ptr2)
	// result: x
	for {
		c := AuxIntToInt32(v.AuxInt)
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARMMOVWstoreshiftLL {
			break
		}
		d := AuxIntToInt32(v_2.AuxInt)
		x := v_2.Args[2]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] || !(c == d && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWloadshiftLL ptr (MOVWconst [c]) [d] mem)
	// result: (MOVWload [int32(uint32(c)<<uint64(d))] ptr mem)
	for {
		d := AuxIntToInt32(v.AuxInt)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpARMMOVWload)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) << uint64(d)))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWloadshiftRA(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWloadshiftRA ptr idx [c] (MOVWstoreshiftRA ptr2 idx [d] x _))
	// cond: c==d && ssa.IsSamePtr(ptr, ptr2)
	// result: x
	for {
		c := AuxIntToInt32(v.AuxInt)
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARMMOVWstoreshiftRA {
			break
		}
		d := AuxIntToInt32(v_2.AuxInt)
		x := v_2.Args[2]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] || !(c == d && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWloadshiftRA ptr (MOVWconst [c]) [d] mem)
	// result: (MOVWload [c>>uint64(d)] ptr mem)
	for {
		d := AuxIntToInt32(v.AuxInt)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpARMMOVWload)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWloadshiftRL(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWloadshiftRL ptr idx [c] (MOVWstoreshiftRL ptr2 idx [d] x _))
	// cond: c==d && ssa.IsSamePtr(ptr, ptr2)
	// result: x
	for {
		c := AuxIntToInt32(v.AuxInt)
		ptr := v_0
		idx := v_1
		if v_2.Op != ssaop.OpARMMOVWstoreshiftRL {
			break
		}
		d := AuxIntToInt32(v_2.AuxInt)
		x := v_2.Args[2]
		ptr2 := v_2.Args[0]
		if idx != v_2.Args[1] || !(c == d && ssa.IsSamePtr(ptr, ptr2)) {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (MOVWloadshiftRL ptr (MOVWconst [c]) [d] mem)
	// result: (MOVWload [int32(uint32(c)>>uint64(d))] ptr mem)
	for {
		d := AuxIntToInt32(v.AuxInt)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		mem := v_2
		v.Reset(ssaop.OpARMMOVWload)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWnop(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWnop (MOVWconst [c]))
	// result: (MOVWconst [c])
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWreg(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MOVWreg x)
	// cond: x.Uses == 1
	// result: (MOVWnop x)
	for {
		x := v_0
		if !(x.Uses == 1) {
			break
		}
		v.Reset(ssaop.OpARMMOVWnop)
		v.AddArg(x)
		return true
	}
	// match: (MOVWreg (MOVWconst [c]))
	// result: (MOVWconst [c])
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(c)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWstore(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// result: (MOVWstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVWstore)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (SUBconst [off2] ptr) val mem)
	// result: (MOVWstore [off1-off2] {sym} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVWstore)
		v.AuxInt = Int32ToAuxInt(off1 - off2)
		v.Aux = SymToAux(sym)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (MOVWaddr [off2] {sym2} ptr) val mem)
	// cond: CanMergeSym(sym1,sym2)
	// result: (MOVWstore [off1+off2] {MergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		sym1 := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym2 := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(CanMergeSym(sym1, sym2)) {
			break
		}
		v.Reset(ssaop.OpARMMOVWstore)
		v.AuxInt = Int32ToAuxInt(off1 + off2)
		v.Aux = SymToAux(MergeSym(sym1, sym2))
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstore [0] {sym} (ADD ptr idx) val mem)
	// cond: sym == nil
	// result: (MOVWstoreidx ptr idx val mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADD {
			break
		}
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVWstoreidx)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstore [0] {sym} (ADDshiftLL ptr idx [c]) val mem)
	// cond: sym == nil
	// result: (MOVWstoreshiftLL ptr idx [c] val mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDshiftLL {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVWstoreshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstore [0] {sym} (ADDshiftRL ptr idx [c]) val mem)
	// cond: sym == nil
	// result: (MOVWstoreshiftRL ptr idx [c] val mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDshiftRL {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVWstoreshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstore [0] {sym} (ADDshiftRA ptr idx [c]) val mem)
	// cond: sym == nil
	// result: (MOVWstoreshiftRA ptr idx [c] val mem)
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		sym := AuxToSym(v.Aux)
		if v_0.Op != ssaop.OpARMADDshiftRA {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		idx := v_0.Args[1]
		ptr := v_0.Args[0]
		val := v_1
		mem := v_2
		if !(sym == nil) {
			break
		}
		v.Reset(ssaop.OpARMMOVWstoreshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWstoreidx(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreidx ptr (MOVWconst [c]) val mem)
	// result: (MOVWstore [c] ptr val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVWstore)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstoreidx (MOVWconst [c]) ptr val mem)
	// result: (MOVWstore [c] ptr val mem)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		ptr := v_1
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVWstore)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(ptr, val, mem)
		return true
	}
	// match: (MOVWstoreidx ptr (SLLconst idx [c]) val mem)
	// result: (MOVWstoreshiftLL ptr idx [c] val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVWstoreshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx (SLLconst idx [c]) ptr val mem)
	// result: (MOVWstoreshiftLL ptr idx [c] val mem)
	for {
		if v_0.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVWstoreshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx ptr (SRLconst idx [c]) val mem)
	// result: (MOVWstoreshiftRL ptr idx [c] val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVWstoreshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx (SRLconst idx [c]) ptr val mem)
	// result: (MOVWstoreshiftRL ptr idx [c] val mem)
	for {
		if v_0.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVWstoreshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx ptr (SRAconst idx [c]) val mem)
	// result: (MOVWstoreshiftRA ptr idx [c] val mem)
	for {
		ptr := v_0
		if v_1.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		idx := v_1.Args[0]
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVWstoreshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	// match: (MOVWstoreidx (SRAconst idx [c]) ptr val mem)
	// result: (MOVWstoreshiftRA ptr idx [c] val mem)
	for {
		if v_0.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		idx := v_0.Args[0]
		ptr := v_1
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVWstoreshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg4(ptr, idx, val, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWstoreshiftLL(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreshiftLL ptr (MOVWconst [c]) [d] val mem)
	// result: (MOVWstore [int32(uint32(c)<<uint64(d))] ptr val mem)
	for {
		d := AuxIntToInt32(v.AuxInt)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVWstore)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) << uint64(d)))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWstoreshiftRA(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreshiftRA ptr (MOVWconst [c]) [d] val mem)
	// result: (MOVWstore [c>>uint64(d)] ptr val mem)
	for {
		d := AuxIntToInt32(v.AuxInt)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVWstore)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMOVWstoreshiftRL(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MOVWstoreshiftRL ptr (MOVWconst [c]) [d] val mem)
	// result: (MOVWstore [int32(uint32(c)>>uint64(d))] ptr val mem)
	for {
		d := AuxIntToInt32(v.AuxInt)
		ptr := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		val := v_2
		mem := v_3
		v.Reset(ssaop.OpARMMOVWstore)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMUL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MUL x (MOVWconst [c]))
	// cond: int32(c) == -1
	// result: (RSBconst [0] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			if !(int32(c) == -1) {
				continue
			}
			v.Reset(ssaop.OpARMRSBconst)
			v.AuxInt = Int32ToAuxInt(0)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MUL _ (MOVWconst [0]))
	// result: (MOVWconst [0])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_1.Op != ssaop.OpARMMOVWconst || AuxIntToInt32(v_1.AuxInt) != 0 {
				continue
			}
			v.Reset(ssaop.OpARMMOVWconst)
			v.AuxInt = Int32ToAuxInt(0)
			return true
		}
		break
	}
	// match: (MUL x (MOVWconst [1]))
	// result: x
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst || AuxIntToInt32(v_1.AuxInt) != 1 {
				continue
			}
			v.CopyOf(x)
			return true
		}
		break
	}
	// match: (MUL x (MOVWconst [c]))
	// cond: IsPowerOfTwo(c)
	// result: (SLLconst [int32(Log32(c))] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			if !(IsPowerOfTwo(c)) {
				continue
			}
			v.Reset(ssaop.OpARMSLLconst)
			v.AuxInt = Int32ToAuxInt(int32(Log32(c)))
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (MUL x (MOVWconst [c]))
	// cond: IsPowerOfTwo(c-1) && c >= 3
	// result: (ADDshiftLL x x [int32(Log32(c-1))])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			if !(IsPowerOfTwo(c-1) && c >= 3) {
				continue
			}
			v.Reset(ssaop.OpARMADDshiftLL)
			v.AuxInt = Int32ToAuxInt(int32(Log32(c - 1)))
			v.AddArg2(x, x)
			return true
		}
		break
	}
	// match: (MUL x (MOVWconst [c]))
	// cond: IsPowerOfTwo(c+1) && c >= 7
	// result: (RSBshiftLL x x [int32(Log32(c+1))])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			if !(IsPowerOfTwo(c+1) && c >= 7) {
				continue
			}
			v.Reset(ssaop.OpARMRSBshiftLL)
			v.AuxInt = Int32ToAuxInt(int32(Log32(c + 1)))
			v.AddArg2(x, x)
			return true
		}
		break
	}
	// match: (MUL x (MOVWconst [c]))
	// cond: c%3 == 0 && IsPowerOfTwo(c/3)
	// result: (SLLconst [int32(Log32(c/3))] (ADDshiftLL <x.Type> x x [1]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			if !(c%3 == 0 && IsPowerOfTwo(c/3)) {
				continue
			}
			v.Reset(ssaop.OpARMSLLconst)
			v.AuxInt = Int32ToAuxInt(int32(Log32(c / 3)))
			v0 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
			v0.AuxInt = Int32ToAuxInt(1)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MUL x (MOVWconst [c]))
	// cond: c%5 == 0 && IsPowerOfTwo(c/5)
	// result: (SLLconst [int32(Log32(c/5))] (ADDshiftLL <x.Type> x x [2]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			if !(c%5 == 0 && IsPowerOfTwo(c/5)) {
				continue
			}
			v.Reset(ssaop.OpARMSLLconst)
			v.AuxInt = Int32ToAuxInt(int32(Log32(c / 5)))
			v0 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
			v0.AuxInt = Int32ToAuxInt(2)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MUL x (MOVWconst [c]))
	// cond: c%7 == 0 && IsPowerOfTwo(c/7)
	// result: (SLLconst [int32(Log32(c/7))] (RSBshiftLL <x.Type> x x [3]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			if !(c%7 == 0 && IsPowerOfTwo(c/7)) {
				continue
			}
			v.Reset(ssaop.OpARMSLLconst)
			v.AuxInt = Int32ToAuxInt(int32(Log32(c / 7)))
			v0 := b.NewValue0(v.Pos, ssaop.OpARMRSBshiftLL, x.Type)
			v0.AuxInt = Int32ToAuxInt(3)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MUL x (MOVWconst [c]))
	// cond: c%9 == 0 && IsPowerOfTwo(c/9)
	// result: (SLLconst [int32(Log32(c/9))] (ADDshiftLL <x.Type> x x [3]))
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			if !(c%9 == 0 && IsPowerOfTwo(c/9)) {
				continue
			}
			v.Reset(ssaop.OpARMSLLconst)
			v.AuxInt = Int32ToAuxInt(int32(Log32(c / 9)))
			v0 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
			v0.AuxInt = Int32ToAuxInt(3)
			v0.AddArg2(x, x)
			v.AddArg(v0)
			return true
		}
		break
	}
	// match: (MUL (MOVWconst [c]) (MOVWconst [d]))
	// result: (MOVWconst [c*d])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_0.AuxInt)
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			d := AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.OpARMMOVWconst)
			v.AuxInt = Int32ToAuxInt(c * d)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMMULA(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MULA x (MOVWconst [c]) a)
	// cond: c == -1
	// result: (SUB a x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(c == -1) {
			break
		}
		v.Reset(ssaop.OpARMSUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MULA _ (MOVWconst [0]) a)
	// result: a
	for {
		if v_1.Op != ssaop.OpARMMOVWconst || AuxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		a := v_2
		v.CopyOf(a)
		return true
	}
	// match: (MULA x (MOVWconst [1]) a)
	// result: (ADD x a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst || AuxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		a := v_2
		v.Reset(ssaop.OpARMADD)
		v.AddArg2(x, a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: IsPowerOfTwo(c)
	// result: (ADD (SLLconst <x.Type> [int32(Log32(c))] x) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c)))
		v0.AddArg(x)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: IsPowerOfTwo(c-1) && c >= 3
	// result: (ADD (ADDshiftLL <x.Type> x x [int32(Log32(c-1))]) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(IsPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c - 1)))
		v0.AddArg2(x, x)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: IsPowerOfTwo(c+1) && c >= 7
	// result: (ADD (RSBshiftLL <x.Type> x x [int32(Log32(c+1))]) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(IsPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMRSBshiftLL, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c + 1)))
		v0.AddArg2(x, x)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: c%3 == 0 && IsPowerOfTwo(c/3)
	// result: (ADD (SLLconst <x.Type> [int32(Log32(c/3))] (ADDshiftLL <x.Type> x x [1])) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(c%3 == 0 && IsPowerOfTwo(c/3)) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 3)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(1)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: c%5 == 0 && IsPowerOfTwo(c/5)
	// result: (ADD (SLLconst <x.Type> [int32(Log32(c/5))] (ADDshiftLL <x.Type> x x [2])) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(c%5 == 0 && IsPowerOfTwo(c/5)) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 5)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: c%7 == 0 && IsPowerOfTwo(c/7)
	// result: (ADD (SLLconst <x.Type> [int32(Log32(c/7))] (RSBshiftLL <x.Type> x x [3])) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(c%7 == 0 && IsPowerOfTwo(c/7)) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 7)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMRSBshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA x (MOVWconst [c]) a)
	// cond: c%9 == 0 && IsPowerOfTwo(c/9)
	// result: (ADD (SLLconst <x.Type> [int32(Log32(c/9))] (ADDshiftLL <x.Type> x x [3])) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(c%9 == 0 && IsPowerOfTwo(c/9)) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 9)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: c == -1
	// result: (SUB a x)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(c == -1) {
			break
		}
		v.Reset(ssaop.OpARMSUB)
		v.AddArg2(a, x)
		return true
	}
	// match: (MULA (MOVWconst [0]) _ a)
	// result: a
	for {
		if v_0.Op != ssaop.OpARMMOVWconst || AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		a := v_2
		v.CopyOf(a)
		return true
	}
	// match: (MULA (MOVWconst [1]) x a)
	// result: (ADD x a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst || AuxIntToInt32(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		a := v_2
		v.Reset(ssaop.OpARMADD)
		v.AddArg2(x, a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: IsPowerOfTwo(c)
	// result: (ADD (SLLconst <x.Type> [int32(Log32(c))] x) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c)))
		v0.AddArg(x)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: IsPowerOfTwo(c-1) && c >= 3
	// result: (ADD (ADDshiftLL <x.Type> x x [int32(Log32(c-1))]) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(IsPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c - 1)))
		v0.AddArg2(x, x)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: IsPowerOfTwo(c+1) && c >= 7
	// result: (ADD (RSBshiftLL <x.Type> x x [int32(Log32(c+1))]) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(IsPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMRSBshiftLL, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c + 1)))
		v0.AddArg2(x, x)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: c%3 == 0 && IsPowerOfTwo(c/3)
	// result: (ADD (SLLconst <x.Type> [int32(Log32(c/3))] (ADDshiftLL <x.Type> x x [1])) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(c%3 == 0 && IsPowerOfTwo(c/3)) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 3)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(1)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: c%5 == 0 && IsPowerOfTwo(c/5)
	// result: (ADD (SLLconst <x.Type> [int32(Log32(c/5))] (ADDshiftLL <x.Type> x x [2])) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(c%5 == 0 && IsPowerOfTwo(c/5)) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 5)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: c%7 == 0 && IsPowerOfTwo(c/7)
	// result: (ADD (SLLconst <x.Type> [int32(Log32(c/7))] (RSBshiftLL <x.Type> x x [3])) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(c%7 == 0 && IsPowerOfTwo(c/7)) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 7)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMRSBshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA (MOVWconst [c]) x a)
	// cond: c%9 == 0 && IsPowerOfTwo(c/9)
	// result: (ADD (SLLconst <x.Type> [int32(Log32(c/9))] (ADDshiftLL <x.Type> x x [3])) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(c%9 == 0 && IsPowerOfTwo(c/9)) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 9)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULA (MOVWconst [c]) (MOVWconst [d]) a)
	// result: (ADDconst [c*d] a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(c * d)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMULD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULD (NEGD x) y)
	// cond: buildcfg.GOARM.Version >= 6
	// result: (NMULD x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARMNEGD {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			if !(buildcfg.GOARM.Version >= 6) {
				continue
			}
			v.Reset(ssaop.OpARMNMULD)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMMULF(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MULF (NEGF x) y)
	// cond: buildcfg.GOARM.Version >= 6
	// result: (NMULF x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARMNEGF {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			if !(buildcfg.GOARM.Version >= 6) {
				continue
			}
			v.Reset(ssaop.OpARMNMULF)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMMULS(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (MULS x (MOVWconst [c]) a)
	// cond: c == -1
	// result: (ADD a x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(c == -1) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MULS _ (MOVWconst [0]) a)
	// result: a
	for {
		if v_1.Op != ssaop.OpARMMOVWconst || AuxIntToInt32(v_1.AuxInt) != 0 {
			break
		}
		a := v_2
		v.CopyOf(a)
		return true
	}
	// match: (MULS x (MOVWconst [1]) a)
	// result: (RSB x a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst || AuxIntToInt32(v_1.AuxInt) != 1 {
			break
		}
		a := v_2
		v.Reset(ssaop.OpARMRSB)
		v.AddArg2(x, a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: IsPowerOfTwo(c)
	// result: (RSB (SLLconst <x.Type> [int32(Log32(c))] x) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c)))
		v0.AddArg(x)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: IsPowerOfTwo(c-1) && c >= 3
	// result: (RSB (ADDshiftLL <x.Type> x x [int32(Log32(c-1))]) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(IsPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c - 1)))
		v0.AddArg2(x, x)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: IsPowerOfTwo(c+1) && c >= 7
	// result: (RSB (RSBshiftLL <x.Type> x x [int32(Log32(c+1))]) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(IsPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMRSBshiftLL, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c + 1)))
		v0.AddArg2(x, x)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: c%3 == 0 && IsPowerOfTwo(c/3)
	// result: (RSB (SLLconst <x.Type> [int32(Log32(c/3))] (ADDshiftLL <x.Type> x x [1])) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(c%3 == 0 && IsPowerOfTwo(c/3)) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 3)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(1)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: c%5 == 0 && IsPowerOfTwo(c/5)
	// result: (RSB (SLLconst <x.Type> [int32(Log32(c/5))] (ADDshiftLL <x.Type> x x [2])) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(c%5 == 0 && IsPowerOfTwo(c/5)) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 5)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: c%7 == 0 && IsPowerOfTwo(c/7)
	// result: (RSB (SLLconst <x.Type> [int32(Log32(c/7))] (RSBshiftLL <x.Type> x x [3])) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(c%7 == 0 && IsPowerOfTwo(c/7)) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 7)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMRSBshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS x (MOVWconst [c]) a)
	// cond: c%9 == 0 && IsPowerOfTwo(c/9)
	// result: (RSB (SLLconst <x.Type> [int32(Log32(c/9))] (ADDshiftLL <x.Type> x x [3])) a)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		if !(c%9 == 0 && IsPowerOfTwo(c/9)) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 9)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: c == -1
	// result: (ADD a x)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(c == -1) {
			break
		}
		v.Reset(ssaop.OpARMADD)
		v.AddArg2(a, x)
		return true
	}
	// match: (MULS (MOVWconst [0]) _ a)
	// result: a
	for {
		if v_0.Op != ssaop.OpARMMOVWconst || AuxIntToInt32(v_0.AuxInt) != 0 {
			break
		}
		a := v_2
		v.CopyOf(a)
		return true
	}
	// match: (MULS (MOVWconst [1]) x a)
	// result: (RSB x a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst || AuxIntToInt32(v_0.AuxInt) != 1 {
			break
		}
		x := v_1
		a := v_2
		v.Reset(ssaop.OpARMRSB)
		v.AddArg2(x, a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: IsPowerOfTwo(c)
	// result: (RSB (SLLconst <x.Type> [int32(Log32(c))] x) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c)))
		v0.AddArg(x)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: IsPowerOfTwo(c-1) && c >= 3
	// result: (RSB (ADDshiftLL <x.Type> x x [int32(Log32(c-1))]) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(IsPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c - 1)))
		v0.AddArg2(x, x)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: IsPowerOfTwo(c+1) && c >= 7
	// result: (RSB (RSBshiftLL <x.Type> x x [int32(Log32(c+1))]) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(IsPowerOfTwo(c+1) && c >= 7) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMRSBshiftLL, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c + 1)))
		v0.AddArg2(x, x)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: c%3 == 0 && IsPowerOfTwo(c/3)
	// result: (RSB (SLLconst <x.Type> [int32(Log32(c/3))] (ADDshiftLL <x.Type> x x [1])) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(c%3 == 0 && IsPowerOfTwo(c/3)) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 3)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(1)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: c%5 == 0 && IsPowerOfTwo(c/5)
	// result: (RSB (SLLconst <x.Type> [int32(Log32(c/5))] (ADDshiftLL <x.Type> x x [2])) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(c%5 == 0 && IsPowerOfTwo(c/5)) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 5)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(2)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: c%7 == 0 && IsPowerOfTwo(c/7)
	// result: (RSB (SLLconst <x.Type> [int32(Log32(c/7))] (RSBshiftLL <x.Type> x x [3])) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(c%7 == 0 && IsPowerOfTwo(c/7)) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 7)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMRSBshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS (MOVWconst [c]) x a)
	// cond: c%9 == 0 && IsPowerOfTwo(c/9)
	// result: (RSB (SLLconst <x.Type> [int32(Log32(c/9))] (ADDshiftLL <x.Type> x x [3])) a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		a := v_2
		if !(c%9 == 0 && IsPowerOfTwo(c/9)) {
			break
		}
		v.Reset(ssaop.OpARMRSB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(int32(Log32(c / 9)))
		v1 := b.NewValue0(v.Pos, ssaop.OpARMADDshiftLL, x.Type)
		v1.AuxInt = Int32ToAuxInt(3)
		v1.AddArg2(x, x)
		v0.AddArg(v1)
		v.AddArg2(v0, a)
		return true
	}
	// match: (MULS (MOVWconst [c]) (MOVWconst [d]) a)
	// result: (SUBconst [c*d] a)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_1.AuxInt)
		a := v_2
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(c * d)
		v.AddArg(a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVN(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MVN (MOVWconst [c]))
	// result: (MOVWconst [^c])
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(^c)
		return true
	}
	// match: (MVN (SLLconst [c] x))
	// result: (MVNshiftLL x [c])
	for {
		if v_0.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMMVNshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MVN (SRLconst [c] x))
	// result: (MVNshiftRL x [c])
	for {
		if v_0.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMMVNshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MVN (SRAconst [c] x))
	// result: (MVNshiftRA x [c])
	for {
		if v_0.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMMVNshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (MVN (SLL x y))
	// result: (MVNshiftLLreg x y)
	for {
		if v_0.Op != ssaop.OpARMSLL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMMVNshiftLLreg)
		v.AddArg2(x, y)
		return true
	}
	// match: (MVN (SRL x y))
	// result: (MVNshiftRLreg x y)
	for {
		if v_0.Op != ssaop.OpARMSRL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMMVNshiftRLreg)
		v.AddArg2(x, y)
		return true
	}
	// match: (MVN (SRA x y))
	// result: (MVNshiftRAreg x y)
	for {
		if v_0.Op != ssaop.OpARMSRA {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMMVNshiftRAreg)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVNshiftLL(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MVNshiftLL (MOVWconst [c]) [d])
	// result: (MOVWconst [^(c<<uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(^(c << uint64(d)))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVNshiftLLreg(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MVNshiftLLreg x (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (MVNshiftLL x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMMVNshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVNshiftRA(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MVNshiftRA (MOVWconst [c]) [d])
	// result: (MOVWconst [int32(c)>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(c) >> uint64(d))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVNshiftRAreg(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MVNshiftRAreg x (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (MVNshiftRA x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMMVNshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVNshiftRL(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (MVNshiftRL (MOVWconst [c]) [d])
	// result: (MOVWconst [^int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(^int32(uint32(c) >> uint64(d)))
		return true
	}
	return false
}
func rewriteValueARM_OpARMMVNshiftRLreg(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (MVNshiftRLreg x (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (MVNshiftRL x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMMVNshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMNEGD(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NEGD (MULD x y))
	// cond: buildcfg.GOARM.Version >= 6
	// result: (NMULD x y)
	for {
		if v_0.Op != ssaop.OpARMMULD {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(buildcfg.GOARM.Version >= 6) {
			break
		}
		v.Reset(ssaop.OpARMNMULD)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMNEGF(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NEGF (MULF x y))
	// cond: buildcfg.GOARM.Version >= 6
	// result: (NMULF x y)
	for {
		if v_0.Op != ssaop.OpARMMULF {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		if !(buildcfg.GOARM.Version >= 6) {
			break
		}
		v.Reset(ssaop.OpARMNMULF)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMNMULD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NMULD (NEGD x) y)
	// result: (MULD x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARMNEGD {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.Reset(ssaop.OpARMMULD)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMNMULF(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (NMULF (NEGF x) y)
	// result: (MULF x y)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			if v_0.Op != ssaop.OpARMNEGF {
				continue
			}
			x := v_0.Args[0]
			y := v_1
			v.Reset(ssaop.OpARMMULF)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMNotEqual(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (NotEqual (FlagConstant [fc]))
	// result: (MOVWconst [B2i32(fc.Ne())])
	for {
		if v_0.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(B2i32(fc.Ne()))
		return true
	}
	// match: (NotEqual (InvertFlags x))
	// result: (NotEqual x)
	for {
		if v_0.Op != ssaop.OpARMInvertFlags {
			break
		}
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMNotEqual)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMOR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (OR x (MOVWconst [c]))
	// result: (ORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.OpARMORconst)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (OR x (SLLconst [c] y))
	// result: (ORshiftLL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMORshiftLL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR x (SRLconst [c] y))
	// result: (ORshiftRL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMORshiftRL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR x (SRAconst [c] y))
	// result: (ORshiftRA x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRAconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMORshiftRA)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (OR x (SLL y z))
	// result: (ORshiftLLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMORshiftLLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (OR x (SRL y z))
	// result: (ORshiftRLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMORshiftRLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (OR x (SRA y z))
	// result: (ORshiftRAreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRA {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMORshiftRAreg)
			v.AddArg3(x, y, z)
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
func rewriteValueARM_OpARMORconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (ORconst [0] x)
	// result: x
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (ORconst [c] _)
	// cond: int32(c)==-1
	// result: (MOVWconst [-1])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if !(int32(c) == -1) {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(-1)
		return true
	}
	// match: (ORconst [c] (MOVWconst [d]))
	// result: (MOVWconst [c|d])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(c | d)
		return true
	}
	// match: (ORconst [c] (ORconst [d] x))
	// result: (ORconst [c|d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMORconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMORconst)
		v.AuxInt = Int32ToAuxInt(c | d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMORshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (ORshiftLL (MOVWconst [c]) x [d])
	// result: (ORconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftLL x (MOVWconst [c]) [d])
	// result: (ORconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMORconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL <typ.UInt16> [8] (BFXU <typ.UInt16> [int32(ArmBFAuxInt(8, 8))] x) x)
	// result: (REV16 x)
	for {
		if v.Type != typ.UInt16 || AuxIntToInt32(v.AuxInt) != 8 || v_0.Op != ssaop.OpARMBFXU || v_0.Type != typ.UInt16 || AuxIntToInt32(v_0.AuxInt) != int32(ArmBFAuxInt(8, 8)) {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMREV16)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL <typ.UInt16> [8] (SRLconst <typ.UInt16> [24] (SLLconst [16] x)) x)
	// cond: buildcfg.GOARM.Version>=6
	// result: (REV16 x)
	for {
		if v.Type != typ.UInt16 || AuxIntToInt32(v.AuxInt) != 8 || v_0.Op != ssaop.OpARMSRLconst || v_0.Type != typ.UInt16 || AuxIntToInt32(v_0.AuxInt) != 24 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARMSLLconst || AuxIntToInt32(v_0_0.AuxInt) != 16 {
			break
		}
		x := v_0_0.Args[0]
		if x != v_1 || !(buildcfg.GOARM.Version >= 6) {
			break
		}
		v.Reset(ssaop.OpARMREV16)
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL y:(SLLconst x [c]) x [c])
	// result: y
	for {
		c := AuxIntToInt32(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARMSLLconst || AuxIntToInt32(y.AuxInt) != c {
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
func rewriteValueARM_OpARMORshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ORshiftLLreg (MOVWconst [c]) x y)
	// result: (ORconst [c] (SLL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (ORshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMORshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMORshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ORshiftRA (MOVWconst [c]) x [d])
	// result: (ORconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRA x (MOVWconst [c]) [d])
	// result: (ORconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMORconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRA y:(SRAconst x [c]) x [c])
	// result: y
	for {
		c := AuxIntToInt32(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARMSRAconst || AuxIntToInt32(y.AuxInt) != c {
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
func rewriteValueARM_OpARMORshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ORshiftRAreg (MOVWconst [c]) x y)
	// result: (ORconst [c] (SRA <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (ORshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMORshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMORshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ORshiftRL (MOVWconst [c]) x [d])
	// result: (ORconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRL x (MOVWconst [c]) [d])
	// result: (ORconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMORconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRL y:(SRLconst x [c]) x [c])
	// result: y
	for {
		c := AuxIntToInt32(v.AuxInt)
		y := v_0
		if y.Op != ssaop.OpARMSRLconst || AuxIntToInt32(y.AuxInt) != c {
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
func rewriteValueARM_OpARMORshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (ORshiftRLreg (MOVWconst [c]) x y)
	// result: (ORconst [c] (SRL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (ORshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (ORshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMORshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RSB (MOVWconst [c]) x)
	// result: (SUBconst [c] x)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (RSB x (MOVWconst [c]))
	// result: (RSBconst [c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (RSB x (SLLconst [c] y))
	// result: (RSBshiftLL x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMRSBshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (RSB (SLLconst [c] y) x)
	// result: (SUBshiftLL x y [c])
	for {
		if v_0.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMSUBshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (RSB x (SRLconst [c] y))
	// result: (RSBshiftRL x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMRSBshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (RSB (SRLconst [c] y) x)
	// result: (SUBshiftRL x y [c])
	for {
		if v_0.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMSUBshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (RSB x (SRAconst [c] y))
	// result: (RSBshiftRA x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMRSBshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (RSB (SRAconst [c] y) x)
	// result: (SUBshiftRA x y [c])
	for {
		if v_0.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMSUBshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (RSB x (SLL y z))
	// result: (RSBshiftLLreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMRSBshiftLLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (RSB (SLL y z) x)
	// result: (SUBshiftLLreg x y z)
	for {
		if v_0.Op != ssaop.OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMSUBshiftLLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (RSB x (SRL y z))
	// result: (RSBshiftRLreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMRSBshiftRLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (RSB (SRL y z) x)
	// result: (SUBshiftRLreg x y z)
	for {
		if v_0.Op != ssaop.OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMSUBshiftRLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (RSB x (SRA y z))
	// result: (RSBshiftRAreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMRSBshiftRAreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (RSB (SRA y z) x)
	// result: (SUBshiftRAreg x y z)
	for {
		if v_0.Op != ssaop.OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMSUBshiftRAreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (RSB x x)
	// result: (MOVWconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	// match: (RSB (MUL x y) a)
	// cond: buildcfg.GOARM.Version == 7
	// result: (MULS x y a)
	for {
		if v_0.Op != ssaop.OpARMMUL {
			break
		}
		y := v_0.Args[1]
		x := v_0.Args[0]
		a := v_1
		if !(buildcfg.GOARM.Version == 7) {
			break
		}
		v.Reset(ssaop.OpARMMULS)
		v.AddArg3(x, y, a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBSshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSBSshiftLL (MOVWconst [c]) x [d])
	// result: (SUBSconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMSUBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (RSBSshiftLL x (MOVWconst [c]) [d])
	// result: (RSBSconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMRSBSconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBSshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSBSshiftLLreg (MOVWconst [c]) x y)
	// result: (SUBSconst [c] (SLL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMSUBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (RSBSshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (RSBSshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMRSBSshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBSshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSBSshiftRA (MOVWconst [c]) x [d])
	// result: (SUBSconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMSUBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (RSBSshiftRA x (MOVWconst [c]) [d])
	// result: (RSBSconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMRSBSconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBSshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSBSshiftRAreg (MOVWconst [c]) x y)
	// result: (SUBSconst [c] (SRA <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMSUBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (RSBSshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (RSBSshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMRSBSshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBSshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSBSshiftRL (MOVWconst [c]) x [d])
	// result: (SUBSconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMSUBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (RSBSshiftRL x (MOVWconst [c]) [d])
	// result: (RSBSconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMRSBSconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBSshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSBSshiftRLreg (MOVWconst [c]) x y)
	// result: (SUBSconst [c] (SRL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMSUBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (RSBSshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (RSBSshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMRSBSshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (RSBconst [c] (MOVWconst [d]))
	// result: (MOVWconst [c-d])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(c - d)
		return true
	}
	// match: (RSBconst [c] (RSBconst [d] x))
	// result: (ADDconst [c-d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMRSBconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	// match: (RSBconst [c] (ADDconst [d] x))
	// result: (RSBconst [c-d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c - d)
		v.AddArg(x)
		return true
	}
	// match: (RSBconst [c] (SUBconst [d] x))
	// result: (RSBconst [c+d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSBshiftLL (MOVWconst [c]) x [d])
	// result: (SUBconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (RSBshiftLL x (MOVWconst [c]) [d])
	// result: (RSBconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (RSBshiftLL (SLLconst x [c]) x [c])
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSLLconst || AuxIntToInt32(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSBshiftLLreg (MOVWconst [c]) x y)
	// result: (SUBconst [c] (SLL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (RSBshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (RSBshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMRSBshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSBshiftRA (MOVWconst [c]) x [d])
	// result: (SUBconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (RSBshiftRA x (MOVWconst [c]) [d])
	// result: (RSBconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (RSBshiftRA (SRAconst x [c]) x [c])
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSRAconst || AuxIntToInt32(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSBshiftRAreg (MOVWconst [c]) x y)
	// result: (SUBconst [c] (SRA <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (RSBshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (RSBshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMRSBshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSBshiftRL (MOVWconst [c]) x [d])
	// result: (SUBconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (RSBshiftRL x (MOVWconst [c]) [d])
	// result: (RSBconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (RSBshiftRL (SRLconst x [c]) x [c])
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSRLconst || AuxIntToInt32(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSBshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSBshiftRLreg (MOVWconst [c]) x y)
	// result: (SUBconst [c] (SRL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (RSBshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (RSBshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMRSBshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCconst(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (RSCconst [c] (ADDconst [d] x) flags)
	// result: (RSCconst [c-d] x flags)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		flags := v_1
		v.Reset(ssaop.OpARMRSCconst)
		v.AuxInt = Int32ToAuxInt(c - d)
		v.AddArg2(x, flags)
		return true
	}
	// match: (RSCconst [c] (SUBconst [d] x) flags)
	// result: (RSCconst [c+d] x flags)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		flags := v_1
		v.Reset(ssaop.OpARMRSCconst)
		v.AuxInt = Int32ToAuxInt(c + d)
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCshiftLL(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSCshiftLL (MOVWconst [c]) x [d] flags)
	// result: (SBCconst [c] (SLLconst <x.Type> x [d]) flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMSBCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (RSCshiftLL x (MOVWconst [c]) [d] flags)
	// result: (RSCconst x [c<<uint64(d)] flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		flags := v_2
		v.Reset(ssaop.OpARMRSCconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCshiftLLreg(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSCshiftLLreg (MOVWconst [c]) x y flags)
	// result: (SBCconst [c] (SLL <x.Type> x y) flags)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		flags := v_3
		v.Reset(ssaop.OpARMSBCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (RSCshiftLLreg x y (MOVWconst [c]) flags)
	// cond: 0 <= c && c < 32
	// result: (RSCshiftLL x y [c] flags)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		flags := v_3
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMRSCshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCshiftRA(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSCshiftRA (MOVWconst [c]) x [d] flags)
	// result: (SBCconst [c] (SRAconst <x.Type> x [d]) flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMSBCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (RSCshiftRA x (MOVWconst [c]) [d] flags)
	// result: (RSCconst x [c>>uint64(d)] flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		flags := v_2
		v.Reset(ssaop.OpARMRSCconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCshiftRAreg(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSCshiftRAreg (MOVWconst [c]) x y flags)
	// result: (SBCconst [c] (SRA <x.Type> x y) flags)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		flags := v_3
		v.Reset(ssaop.OpARMSBCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (RSCshiftRAreg x y (MOVWconst [c]) flags)
	// cond: 0 <= c && c < 32
	// result: (RSCshiftRA x y [c] flags)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		flags := v_3
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMRSCshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCshiftRL(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSCshiftRL (MOVWconst [c]) x [d] flags)
	// result: (SBCconst [c] (SRLconst <x.Type> x [d]) flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMSBCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (RSCshiftRL x (MOVWconst [c]) [d] flags)
	// result: (RSCconst x [int32(uint32(c)>>uint64(d))] flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		flags := v_2
		v.Reset(ssaop.OpARMRSCconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMRSCshiftRLreg(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RSCshiftRLreg (MOVWconst [c]) x y flags)
	// result: (SBCconst [c] (SRL <x.Type> x y) flags)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		flags := v_3
		v.Reset(ssaop.OpARMSBCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (RSCshiftRLreg x y (MOVWconst [c]) flags)
	// cond: 0 <= c && c < 32
	// result: (RSCshiftRL x y [c] flags)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		flags := v_3
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMRSCshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBC(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SBC (MOVWconst [c]) x flags)
	// result: (RSCconst [c] x flags)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMRSCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, flags)
		return true
	}
	// match: (SBC x (MOVWconst [c]) flags)
	// result: (SBCconst [c] x flags)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		flags := v_2
		v.Reset(ssaop.OpARMSBCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, flags)
		return true
	}
	// match: (SBC x (SLLconst [c] y) flags)
	// result: (SBCshiftLL x y [c] flags)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		flags := v_2
		v.Reset(ssaop.OpARMSBCshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	// match: (SBC (SLLconst [c] y) x flags)
	// result: (RSCshiftLL x y [c] flags)
	for {
		if v_0.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMRSCshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	// match: (SBC x (SRLconst [c] y) flags)
	// result: (SBCshiftRL x y [c] flags)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		flags := v_2
		v.Reset(ssaop.OpARMSBCshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	// match: (SBC (SRLconst [c] y) x flags)
	// result: (RSCshiftRL x y [c] flags)
	for {
		if v_0.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMRSCshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	// match: (SBC x (SRAconst [c] y) flags)
	// result: (SBCshiftRA x y [c] flags)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		flags := v_2
		v.Reset(ssaop.OpARMSBCshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	// match: (SBC (SRAconst [c] y) x flags)
	// result: (RSCshiftRA x y [c] flags)
	for {
		if v_0.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMRSCshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	// match: (SBC x (SLL y z) flags)
	// result: (SBCshiftLLreg x y z flags)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		flags := v_2
		v.Reset(ssaop.OpARMSBCshiftLLreg)
		v.AddArg4(x, y, z, flags)
		return true
	}
	// match: (SBC (SLL y z) x flags)
	// result: (RSCshiftLLreg x y z flags)
	for {
		if v_0.Op != ssaop.OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMRSCshiftLLreg)
		v.AddArg4(x, y, z, flags)
		return true
	}
	// match: (SBC x (SRL y z) flags)
	// result: (SBCshiftRLreg x y z flags)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		flags := v_2
		v.Reset(ssaop.OpARMSBCshiftRLreg)
		v.AddArg4(x, y, z, flags)
		return true
	}
	// match: (SBC (SRL y z) x flags)
	// result: (RSCshiftRLreg x y z flags)
	for {
		if v_0.Op != ssaop.OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMRSCshiftRLreg)
		v.AddArg4(x, y, z, flags)
		return true
	}
	// match: (SBC x (SRA y z) flags)
	// result: (SBCshiftRAreg x y z flags)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		flags := v_2
		v.Reset(ssaop.OpARMSBCshiftRAreg)
		v.AddArg4(x, y, z, flags)
		return true
	}
	// match: (SBC (SRA y z) x flags)
	// result: (RSCshiftRAreg x y z flags)
	for {
		if v_0.Op != ssaop.OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMRSCshiftRAreg)
		v.AddArg4(x, y, z, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCconst(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SBCconst [c] (ADDconst [d] x) flags)
	// result: (SBCconst [c-d] x flags)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		flags := v_1
		v.Reset(ssaop.OpARMSBCconst)
		v.AuxInt = Int32ToAuxInt(c - d)
		v.AddArg2(x, flags)
		return true
	}
	// match: (SBCconst [c] (SUBconst [d] x) flags)
	// result: (SBCconst [c+d] x flags)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		flags := v_1
		v.Reset(ssaop.OpARMSBCconst)
		v.AuxInt = Int32ToAuxInt(c + d)
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCshiftLL(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SBCshiftLL (MOVWconst [c]) x [d] flags)
	// result: (RSCconst [c] (SLLconst <x.Type> x [d]) flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMRSCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (SBCshiftLL x (MOVWconst [c]) [d] flags)
	// result: (SBCconst x [c<<uint64(d)] flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		flags := v_2
		v.Reset(ssaop.OpARMSBCconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCshiftLLreg(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SBCshiftLLreg (MOVWconst [c]) x y flags)
	// result: (RSCconst [c] (SLL <x.Type> x y) flags)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		flags := v_3
		v.Reset(ssaop.OpARMRSCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (SBCshiftLLreg x y (MOVWconst [c]) flags)
	// cond: 0 <= c && c < 32
	// result: (SBCshiftLL x y [c] flags)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		flags := v_3
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMSBCshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCshiftRA(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SBCshiftRA (MOVWconst [c]) x [d] flags)
	// result: (RSCconst [c] (SRAconst <x.Type> x [d]) flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMRSCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (SBCshiftRA x (MOVWconst [c]) [d] flags)
	// result: (SBCconst x [c>>uint64(d)] flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		flags := v_2
		v.Reset(ssaop.OpARMSBCconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCshiftRAreg(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SBCshiftRAreg (MOVWconst [c]) x y flags)
	// result: (RSCconst [c] (SRA <x.Type> x y) flags)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		flags := v_3
		v.Reset(ssaop.OpARMRSCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (SBCshiftRAreg x y (MOVWconst [c]) flags)
	// cond: 0 <= c && c < 32
	// result: (SBCshiftRA x y [c] flags)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		flags := v_3
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMSBCshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCshiftRL(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SBCshiftRL (MOVWconst [c]) x [d] flags)
	// result: (RSCconst [c] (SRLconst <x.Type> x [d]) flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		flags := v_2
		v.Reset(ssaop.OpARMRSCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (SBCshiftRL x (MOVWconst [c]) [d] flags)
	// result: (SBCconst x [int32(uint32(c)>>uint64(d))] flags)
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		flags := v_2
		v.Reset(ssaop.OpARMSBCconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg2(x, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSBCshiftRLreg(v *ssa.Value) bool {
	v_3 := v.Args[3]
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SBCshiftRLreg (MOVWconst [c]) x y flags)
	// result: (RSCconst [c] (SRL <x.Type> x y) flags)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		flags := v_3
		v.Reset(ssaop.OpARMRSCconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg2(v0, flags)
		return true
	}
	// match: (SBCshiftRLreg x y (MOVWconst [c]) flags)
	// cond: 0 <= c && c < 32
	// result: (SBCshiftRL x y [c] flags)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		flags := v_3
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMSBCshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg3(x, y, flags)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SLL x (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (SLLconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMSLLconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSLLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SLLconst [c] (MOVWconst [d]))
	// result: (MOVWconst [d<<uint64(c)])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(d << uint64(c))
		return true
	}
	return false
}
func rewriteValueARM_OpARMSRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRA x (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (SRAconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMSRAconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSRAcond(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRAcond x _ (FlagConstant [fc]))
	// cond: fc.Uge()
	// result: (SRAconst x [31])
	for {
		x := v_0
		if v_2.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_2.AuxInt)
		if !(fc.Uge()) {
			break
		}
		v.Reset(ssaop.OpARMSRAconst)
		v.AuxInt = Int32ToAuxInt(31)
		v.AddArg(x)
		return true
	}
	// match: (SRAcond x y (FlagConstant [fc]))
	// cond: fc.Ult()
	// result: (SRA x y)
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMFlagConstant {
			break
		}
		fc := AuxIntToFlagConstant(v_2.AuxInt)
		if !(fc.Ult()) {
			break
		}
		v.Reset(ssaop.OpARMSRA)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSRAconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SRAconst [c] (MOVWconst [d]))
	// result: (MOVWconst [d>>uint64(c)])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(d >> uint64(c))
		return true
	}
	// match: (SRAconst (SLLconst x [c]) [d])
	// cond: buildcfg.GOARM.Version==7 && uint64(d)>=uint64(c) && uint64(d)<=31
	// result: (BFX [(d-c)|(32-d)<<8] x)
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(buildcfg.GOARM.Version == 7 && uint64(d) >= uint64(c) && uint64(d) <= 31) {
			break
		}
		v.Reset(ssaop.OpARMBFX)
		v.AuxInt = Int32ToAuxInt((d - c) | (32-d)<<8)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRL x (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (SRLconst x [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMSRLconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSRLconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SRLconst [c] (MOVWconst [d]))
	// result: (MOVWconst [int32(uint32(d)>>uint64(c))])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(d) >> uint64(c)))
		return true
	}
	// match: (SRLconst (SLLconst x [c]) [d])
	// cond: buildcfg.GOARM.Version==7 && uint64(d)>=uint64(c) && uint64(d)<=31
	// result: (BFXU [(d-c)|(32-d)<<8] x)
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		if !(buildcfg.GOARM.Version == 7 && uint64(d) >= uint64(c) && uint64(d) <= 31) {
			break
		}
		v.Reset(ssaop.OpARMBFXU)
		v.AuxInt = Int32ToAuxInt((d - c) | (32-d)<<8)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSRR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SRR x (MOVWconst [c]))
	// result: (SRRconst x [c&31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMSRRconst)
		v.AuxInt = Int32ToAuxInt(c & 31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUB (MOVWconst [c]) x)
	// result: (RSBconst [c] x)
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUB x (MOVWconst [c]))
	// result: (SUBconst [c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUB x (SLLconst [c] y))
	// result: (SUBshiftLL x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMSUBshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUB (SLLconst [c] y) x)
	// result: (RSBshiftLL x y [c])
	for {
		if v_0.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMRSBshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUB x (SRLconst [c] y))
	// result: (SUBshiftRL x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMSUBshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUB (SRLconst [c] y) x)
	// result: (RSBshiftRL x y [c])
	for {
		if v_0.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMRSBshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUB x (SRAconst [c] y))
	// result: (SUBshiftRA x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMSUBshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUB (SRAconst [c] y) x)
	// result: (RSBshiftRA x y [c])
	for {
		if v_0.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMRSBshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUB x (SLL y z))
	// result: (SUBshiftLLreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMSUBshiftLLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUB (SLL y z) x)
	// result: (RSBshiftLLreg x y z)
	for {
		if v_0.Op != ssaop.OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMRSBshiftLLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUB x (SRL y z))
	// result: (SUBshiftRLreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMSUBshiftRLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUB (SRL y z) x)
	// result: (RSBshiftRLreg x y z)
	for {
		if v_0.Op != ssaop.OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMRSBshiftRLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUB x (SRA y z))
	// result: (SUBshiftRAreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMSUBshiftRAreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUB (SRA y z) x)
	// result: (RSBshiftRAreg x y z)
	for {
		if v_0.Op != ssaop.OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMRSBshiftRAreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUB x x)
	// result: (MOVWconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	// match: (SUB a (MUL x y))
	// cond: buildcfg.GOARM.Version == 7
	// result: (MULS x y a)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARMMUL {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(buildcfg.GOARM.Version == 7) {
			break
		}
		v.Reset(ssaop.OpARMMULS)
		v.AddArg3(x, y, a)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBD(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBD a (MULD x y))
	// cond: a.Uses == 1 && buildcfg.GOARM.Version >= 6
	// result: (MULSD a x y)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARMMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Uses == 1 && buildcfg.GOARM.Version >= 6) {
			break
		}
		v.Reset(ssaop.OpARMMULSD)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (SUBD a (NMULD x y))
	// cond: a.Uses == 1 && buildcfg.GOARM.Version >= 6
	// result: (MULAD a x y)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARMNMULD {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Uses == 1 && buildcfg.GOARM.Version >= 6) {
			break
		}
		v.Reset(ssaop.OpARMMULAD)
		v.AddArg3(a, x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBF(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBF a (MULF x y))
	// cond: a.Uses == 1 && buildcfg.GOARM.Version >= 6
	// result: (MULSF a x y)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARMMULF {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Uses == 1 && buildcfg.GOARM.Version >= 6) {
			break
		}
		v.Reset(ssaop.OpARMMULSF)
		v.AddArg3(a, x, y)
		return true
	}
	// match: (SUBF a (NMULF x y))
	// cond: a.Uses == 1 && buildcfg.GOARM.Version >= 6
	// result: (MULAF a x y)
	for {
		a := v_0
		if v_1.Op != ssaop.OpARMNMULF {
			break
		}
		y := v_1.Args[1]
		x := v_1.Args[0]
		if !(a.Uses == 1 && buildcfg.GOARM.Version >= 6) {
			break
		}
		v.Reset(ssaop.OpARMMULAF)
		v.AddArg3(a, x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBS(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (SUBS x (MOVWconst [c]))
	// result: (SUBSconst [c] x)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMSUBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg(x)
		return true
	}
	// match: (SUBS x (SLLconst [c] y))
	// result: (SUBSshiftLL x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMSUBSshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUBS (SLLconst [c] y) x)
	// result: (RSBSshiftLL x y [c])
	for {
		if v_0.Op != ssaop.OpARMSLLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMRSBSshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUBS x (SRLconst [c] y))
	// result: (SUBSshiftRL x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMSUBSshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUBS (SRLconst [c] y) x)
	// result: (RSBSshiftRL x y [c])
	for {
		if v_0.Op != ssaop.OpARMSRLconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMRSBSshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUBS x (SRAconst [c] y))
	// result: (SUBSshiftRA x y [c])
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMSUBSshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUBS (SRAconst [c] y) x)
	// result: (RSBSshiftRA x y [c])
	for {
		if v_0.Op != ssaop.OpARMSRAconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMRSBSshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	// match: (SUBS x (SLL y z))
	// result: (SUBSshiftLLreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSLL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMSUBSshiftLLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUBS (SLL y z) x)
	// result: (RSBSshiftLLreg x y z)
	for {
		if v_0.Op != ssaop.OpARMSLL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMRSBSshiftLLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUBS x (SRL y z))
	// result: (SUBSshiftRLreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRL {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMSUBSshiftRLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUBS (SRL y z) x)
	// result: (RSBSshiftRLreg x y z)
	for {
		if v_0.Op != ssaop.OpARMSRL {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMRSBSshiftRLreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUBS x (SRA y z))
	// result: (SUBSshiftRAreg x y z)
	for {
		x := v_0
		if v_1.Op != ssaop.OpARMSRA {
			break
		}
		z := v_1.Args[1]
		y := v_1.Args[0]
		v.Reset(ssaop.OpARMSUBSshiftRAreg)
		v.AddArg3(x, y, z)
		return true
	}
	// match: (SUBS (SRA y z) x)
	// result: (RSBSshiftRAreg x y z)
	for {
		if v_0.Op != ssaop.OpARMSRA {
			break
		}
		z := v_0.Args[1]
		y := v_0.Args[0]
		x := v_1
		v.Reset(ssaop.OpARMRSBSshiftRAreg)
		v.AddArg3(x, y, z)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBSshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBSshiftLL (MOVWconst [c]) x [d])
	// result: (RSBSconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMRSBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBSshiftLL x (MOVWconst [c]) [d])
	// result: (SUBSconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMSUBSconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBSshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBSshiftLLreg (MOVWconst [c]) x y)
	// result: (RSBSconst [c] (SLL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMRSBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (SUBSshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (SUBSshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMSUBSshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBSshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBSshiftRA (MOVWconst [c]) x [d])
	// result: (RSBSconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMRSBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBSshiftRA x (MOVWconst [c]) [d])
	// result: (SUBSconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMSUBSconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBSshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBSshiftRAreg (MOVWconst [c]) x y)
	// result: (RSBSconst [c] (SRA <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMRSBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (SUBSshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (SUBSshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMSUBSshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBSshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBSshiftRL (MOVWconst [c]) x [d])
	// result: (RSBSconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMRSBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBSshiftRL x (MOVWconst [c]) [d])
	// result: (SUBSconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMSUBSconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBSshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBSshiftRLreg (MOVWconst [c]) x y)
	// result: (RSBSconst [c] (SRL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMRSBSconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (SUBSshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (SUBSshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMSUBSshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (SUBconst [off1] (MOVWaddr [off2] {sym} ptr))
	// result: (MOVWaddr [off2-off1] {sym} ptr)
	for {
		off1 := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWaddr {
			break
		}
		off2 := AuxIntToInt32(v_0.AuxInt)
		sym := AuxToSym(v_0.Aux)
		ptr := v_0.Args[0]
		v.Reset(ssaop.OpARMMOVWaddr)
		v.AuxInt = Int32ToAuxInt(off2 - off1)
		v.Aux = SymToAux(sym)
		v.AddArg(ptr)
		return true
	}
	// match: (SUBconst [0] x)
	// result: x
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (SUBconst [c] x)
	// cond: !isARMImmRot(uint32(c)) && isARMImmRot(uint32(-c))
	// result: (ADDconst [-c] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(!isARMImmRot(uint32(c)) && isARMImmRot(uint32(-c))) {
			break
		}
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] x)
	// cond: buildcfg.GOARM.Version==7 && !isARMImmRot(uint32(c)) && uint32(c)>0xffff && uint32(-c)<=0xffff
	// result: (ADDconst [-c] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		x := v_0
		if !(buildcfg.GOARM.Version == 7 && !isARMImmRot(uint32(c)) && uint32(c) > 0xffff && uint32(-c) <= 0xffff) {
			break
		}
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(-c)
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] (MOVWconst [d]))
	// result: (MOVWconst [d-c])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(d - c)
		return true
	}
	// match: (SUBconst [c] (SUBconst [d] x))
	// result: (ADDconst [-c-d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSUBconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(-c - d)
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] (ADDconst [d] x))
	// result: (ADDconst [-c+d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMADDconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(-c + d)
		v.AddArg(x)
		return true
	}
	// match: (SUBconst [c] (RSBconst [d] x))
	// result: (RSBconst [-c+d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMRSBconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(-c + d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBshiftLL (MOVWconst [c]) x [d])
	// result: (RSBconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBshiftLL x (MOVWconst [c]) [d])
	// result: (SUBconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftLL (SLLconst x [c]) x [c])
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSLLconst || AuxIntToInt32(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBshiftLLreg (MOVWconst [c]) x y)
	// result: (RSBconst [c] (SLL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (SUBshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (SUBshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMSUBshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBshiftRA (MOVWconst [c]) x [d])
	// result: (RSBconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBshiftRA x (MOVWconst [c]) [d])
	// result: (SUBconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftRA (SRAconst x [c]) x [c])
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSRAconst || AuxIntToInt32(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBshiftRAreg (MOVWconst [c]) x y)
	// result: (RSBconst [c] (SRA <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (SUBshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (SUBshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMSUBshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBshiftRL (MOVWconst [c]) x [d])
	// result: (RSBconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBshiftRL x (MOVWconst [c]) [d])
	// result: (SUBconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMSUBconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (SUBshiftRL (SRLconst x [c]) x [c])
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSRLconst || AuxIntToInt32(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMSUBshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (SUBshiftRLreg (MOVWconst [c]) x y)
	// result: (RSBconst [c] (SRL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (SUBshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (SUBshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMSUBshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQ(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (TEQ x (MOVWconst [c]))
	// result: (TEQconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.OpARMTEQconst)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (TEQ x (SLLconst [c] y))
	// result: (TEQshiftLL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMTEQshiftLL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (TEQ x (SRLconst [c] y))
	// result: (TEQshiftRL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMTEQshiftRL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (TEQ x (SRAconst [c] y))
	// result: (TEQshiftRA x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRAconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMTEQshiftRA)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (TEQ x (SLL y z))
	// result: (TEQshiftLLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMTEQshiftLLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (TEQ x (SRL y z))
	// result: (TEQshiftRLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMTEQshiftRLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (TEQ x (SRA y z))
	// result: (TEQshiftRAreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRA {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMTEQshiftRAreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMTEQconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (TEQconst (MOVWconst [x]) [y])
	// result: (FlagConstant [LogicFlags32(x^y)])
	for {
		y := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		x := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMFlagConstant)
		v.AuxInt = FlagConstantToAuxInt(LogicFlags32(x ^ y))
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TEQshiftLL (MOVWconst [c]) x [d])
	// result: (TEQconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMTEQconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TEQshiftLL x (MOVWconst [c]) [d])
	// result: (TEQconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMTEQconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TEQshiftLLreg (MOVWconst [c]) x y)
	// result: (TEQconst [c] (SLL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMTEQconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (TEQshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (TEQshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMTEQshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TEQshiftRA (MOVWconst [c]) x [d])
	// result: (TEQconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMTEQconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TEQshiftRA x (MOVWconst [c]) [d])
	// result: (TEQconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMTEQconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TEQshiftRAreg (MOVWconst [c]) x y)
	// result: (TEQconst [c] (SRA <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMTEQconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (TEQshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (TEQshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMTEQshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TEQshiftRL (MOVWconst [c]) x [d])
	// result: (TEQconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMTEQconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TEQshiftRL x (MOVWconst [c]) [d])
	// result: (TEQconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMTEQconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTEQshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TEQshiftRLreg (MOVWconst [c]) x y)
	// result: (TEQconst [c] (SRL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMTEQconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (TEQshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (TEQshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMTEQshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTST(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (TST x (MOVWconst [c]))
	// result: (TSTconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.OpARMTSTconst)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (TST x (SLLconst [c] y))
	// result: (TSTshiftLL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMTSTshiftLL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (TST x (SRLconst [c] y))
	// result: (TSTshiftRL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMTSTshiftRL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (TST x (SRAconst [c] y))
	// result: (TSTshiftRA x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRAconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMTSTshiftRA)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (TST x (SLL y z))
	// result: (TSTshiftLLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMTSTshiftLLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (TST x (SRL y z))
	// result: (TSTshiftRLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMTSTshiftRLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (TST x (SRA y z))
	// result: (TSTshiftRAreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRA {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMTSTshiftRAreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	return false
}
func rewriteValueARM_OpARMTSTconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (TSTconst (MOVWconst [x]) [y])
	// result: (FlagConstant [LogicFlags32(x&y)])
	for {
		y := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		x := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMFlagConstant)
		v.AuxInt = FlagConstantToAuxInt(LogicFlags32(x & y))
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftLL (MOVWconst [c]) x [d])
	// result: (TSTconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMTSTconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftLL x (MOVWconst [c]) [d])
	// result: (TSTconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMTSTconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftLLreg (MOVWconst [c]) x y)
	// result: (TSTconst [c] (SLL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMTSTconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (TSTshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMTSTshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftRA (MOVWconst [c]) x [d])
	// result: (TSTconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMTSTconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRA x (MOVWconst [c]) [d])
	// result: (TSTconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMTSTconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftRAreg (MOVWconst [c]) x y)
	// result: (TSTconst [c] (SRA <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMTSTconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (TSTshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMTSTshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftRL (MOVWconst [c]) x [d])
	// result: (TSTconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMTSTconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRL x (MOVWconst [c]) [d])
	// result: (TSTconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMTSTconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMTSTshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (TSTshiftRLreg (MOVWconst [c]) x y)
	// result: (TSTconst [c] (SRL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMTSTconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (TSTshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (TSTshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMTSTshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXOR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (XOR x (MOVWconst [c]))
	// result: (XORconst [c] x)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMMOVWconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			v.Reset(ssaop.OpARMXORconst)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg(x)
			return true
		}
		break
	}
	// match: (XOR x (SLLconst [c] y))
	// result: (XORshiftLL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMXORshiftLL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (XOR x (SRLconst [c] y))
	// result: (XORshiftRL x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRLconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMXORshiftRL)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (XOR x (SRAconst [c] y))
	// result: (XORshiftRA x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRAconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMXORshiftRA)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (XOR x (SRRconst [c] y))
	// result: (XORshiftRR x y [c])
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRRconst {
				continue
			}
			c := AuxIntToInt32(v_1.AuxInt)
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMXORshiftRR)
			v.AuxInt = Int32ToAuxInt(c)
			v.AddArg2(x, y)
			return true
		}
		break
	}
	// match: (XOR x (SLL y z))
	// result: (XORshiftLLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSLL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMXORshiftLLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (XOR x (SRL y z))
	// result: (XORshiftRLreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRL {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMXORshiftRLreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (XOR x (SRA y z))
	// result: (XORshiftRAreg x y z)
	for {
		for _i0 := 0; _i0 <= 1; _i0, v_0, v_1 = _i0+1, v_1, v_0 {
			x := v_0
			if v_1.Op != ssaop.OpARMSRA {
				continue
			}
			z := v_1.Args[1]
			y := v_1.Args[0]
			v.Reset(ssaop.OpARMXORshiftRAreg)
			v.AddArg3(x, y, z)
			return true
		}
		break
	}
	// match: (XOR x x)
	// result: (MOVWconst [0])
	for {
		x := v_0
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORconst(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (XORconst [0] x)
	// result: x
	for {
		if AuxIntToInt32(v.AuxInt) != 0 {
			break
		}
		x := v_0
		v.CopyOf(x)
		return true
	}
	// match: (XORconst [c] (MOVWconst [d]))
	// result: (MOVWconst [c^d])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(c ^ d)
		return true
	}
	// match: (XORconst [c] (XORconst [d] x))
	// result: (XORconst [c^d] x)
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMXORconst {
			break
		}
		d := AuxIntToInt32(v_0.AuxInt)
		x := v_0.Args[0]
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(c ^ d)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftLL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (XORshiftLL (MOVWconst [c]) x [d])
	// result: (XORconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftLL x (MOVWconst [c]) [d])
	// result: (XORconst x [c<<uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(c << uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL <typ.UInt16> [8] (BFXU <typ.UInt16> [int32(ArmBFAuxInt(8, 8))] x) x)
	// result: (REV16 x)
	for {
		if v.Type != typ.UInt16 || AuxIntToInt32(v.AuxInt) != 8 || v_0.Op != ssaop.OpARMBFXU || v_0.Type != typ.UInt16 || AuxIntToInt32(v_0.AuxInt) != int32(ArmBFAuxInt(8, 8)) {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMREV16)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL <typ.UInt16> [8] (SRLconst <typ.UInt16> [24] (SLLconst [16] x)) x)
	// cond: buildcfg.GOARM.Version>=6
	// result: (REV16 x)
	for {
		if v.Type != typ.UInt16 || AuxIntToInt32(v.AuxInt) != 8 || v_0.Op != ssaop.OpARMSRLconst || v_0.Type != typ.UInt16 || AuxIntToInt32(v_0.AuxInt) != 24 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARMSLLconst || AuxIntToInt32(v_0_0.AuxInt) != 16 {
			break
		}
		x := v_0_0.Args[0]
		if x != v_1 || !(buildcfg.GOARM.Version >= 6) {
			break
		}
		v.Reset(ssaop.OpARMREV16)
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL (SLLconst x [c]) x [c])
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSLLconst || AuxIntToInt32(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftLLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XORshiftLLreg (MOVWconst [c]) x y)
	// result: (XORconst [c] (SLL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftLLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (XORshiftLL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMXORshiftLL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftRA(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XORshiftRA (MOVWconst [c]) x [d])
	// result: (XORconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRAconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRA x (MOVWconst [c]) [d])
	// result: (XORconst x [c>>uint64(d)])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(c >> uint64(d))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRA (SRAconst x [c]) x [c])
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSRAconst || AuxIntToInt32(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftRAreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XORshiftRAreg (MOVWconst [c]) x y)
	// result: (XORconst [c] (SRA <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRA, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRAreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (XORshiftRA x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMXORshiftRA)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftRL(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XORshiftRL (MOVWconst [c]) x [d])
	// result: (XORconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRL x (MOVWconst [c]) [d])
	// result: (XORconst x [int32(uint32(c)>>uint64(d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) >> uint64(d)))
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRL (SRLconst x [c]) x [c])
	// result: (MOVWconst [0])
	for {
		c := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMSRLconst || AuxIntToInt32(v_0.AuxInt) != c {
			break
		}
		x := v_0.Args[0]
		if x != v_1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftRLreg(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XORshiftRLreg (MOVWconst [c]) x y)
	// result: (XORconst [c] (SRL <x.Type> x y))
	for {
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		y := v_2
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRLreg x y (MOVWconst [c]))
	// cond: 0 <= c && c < 32
	// result: (XORshiftRL x y [c])
	for {
		x := v_0
		y := v_1
		if v_2.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_2.AuxInt)
		if !(0 <= c && c < 32) {
			break
		}
		v.Reset(ssaop.OpARMXORshiftRL)
		v.AuxInt = Int32ToAuxInt(c)
		v.AddArg2(x, y)
		return true
	}
	return false
}
func rewriteValueARM_OpARMXORshiftRR(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (XORshiftRR (MOVWconst [c]) x [d])
	// result: (XORconst [c] (SRRconst <x.Type> x [d]))
	for {
		d := AuxIntToInt32(v.AuxInt)
		if v_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0.AuxInt)
		x := v_1
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(c)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRRconst, x.Type)
		v0.AuxInt = Int32ToAuxInt(d)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (XORshiftRR x (MOVWconst [c]) [d])
	// result: (XORconst x [int32(uint32(c)>>uint64(d)|uint32(c)<<uint64(32-d))])
	for {
		d := AuxIntToInt32(v.AuxInt)
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c)>>uint64(d) | uint32(c)<<uint64(32-d)))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpAddr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Addr {sym} base)
	// result: (MOVWaddr {sym} base)
	for {
		sym := AuxToSym(v.Aux)
		base := v_0
		v.Reset(ssaop.OpARMMOVWaddr)
		v.Aux = SymToAux(sym)
		v.AddArg(base)
		return true
	}
}
func rewriteValueARM_OpAvg32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Avg32u <t> x y)
	// result: (ADD (SRLconst <t> (SUB <t> x y) [1]) y)
	for {
		t := v.Type
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMADD)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, t)
		v0.AuxInt = Int32ToAuxInt(1)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMSUB, t)
		v1.AddArg2(x, y)
		v0.AddArg(v1)
		v.AddArg2(v0, y)
		return true
	}
}
func rewriteValueARM_OpBitLen16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen16 x)
	// result: (BitLen32 (ZeroExt16to32 x))
	for {
		x := v_0
		v.Reset(ssaop.OpBitLen32)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpBitLen32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (BitLen32 <t> x)
	// result: (RSBconst [32] (CLZ <t> x))
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCLZ, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpBitLen8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (BitLen8 x)
	// result: (BitLen32 (ZeroExt8to32 x))
	for {
		x := v_0
		v.Reset(ssaop.OpBitLen32)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpBswap32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Bswap32 <t> x)
	// cond: buildcfg.GOARM.Version==5
	// result: (XOR <t> (SRLconst <t> (BICconst <t> (XOR <t> x (SRRconst <t> [16] x)) [0xff0000]) [8]) (SRRconst <t> x [8]))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOARM.Version == 5) {
			break
		}
		v.Reset(ssaop.OpARMXOR)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRLconst, t)
		v0.AuxInt = Int32ToAuxInt(8)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMBICconst, t)
		v1.AuxInt = Int32ToAuxInt(0xff0000)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMXOR, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpARMSRRconst, t)
		v3.AuxInt = Int32ToAuxInt(16)
		v3.AddArg(x)
		v2.AddArg2(x, v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, ssaop.OpARMSRRconst, t)
		v4.AuxInt = Int32ToAuxInt(8)
		v4.AddArg(x)
		v.AddArg2(v0, v4)
		return true
	}
	// match: (Bswap32 x)
	// cond: buildcfg.GOARM.Version>=6
	// result: (REV x)
	for {
		x := v_0
		if !(buildcfg.GOARM.Version >= 6) {
			break
		}
		v.Reset(ssaop.OpARMREV)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpConst16(v *ssa.Value) bool {
	// match: (Const16 [val])
	// result: (MOVWconst [int32(val)])
	for {
		val := AuxIntToInt16(v.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(val))
		return true
	}
}
func rewriteValueARM_OpConst32(v *ssa.Value) bool {
	// match: (Const32 [val])
	// result: (MOVWconst [int32(val)])
	for {
		val := AuxIntToInt32(v.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(val))
		return true
	}
}
func rewriteValueARM_OpConst32F(v *ssa.Value) bool {
	// match: (Const32F [val])
	// result: (MOVFconst [float64(val)])
	for {
		val := AuxIntToFloat32(v.AuxInt)
		v.Reset(ssaop.OpARMMOVFconst)
		v.AuxInt = Float64ToAuxInt(float64(val))
		return true
	}
}
func rewriteValueARM_OpConst64F(v *ssa.Value) bool {
	// match: (Const64F [val])
	// result: (MOVDconst [float64(val)])
	for {
		val := AuxIntToFloat64(v.AuxInt)
		v.Reset(ssaop.OpARMMOVDconst)
		v.AuxInt = Float64ToAuxInt(float64(val))
		return true
	}
}
func rewriteValueARM_OpConst8(v *ssa.Value) bool {
	// match: (Const8 [val])
	// result: (MOVWconst [int32(val)])
	for {
		val := AuxIntToInt8(v.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(val))
		return true
	}
}
func rewriteValueARM_OpConstBool(v *ssa.Value) bool {
	// match: (ConstBool [t])
	// result: (MOVWconst [B2i32(t)])
	for {
		t := AuxIntToBool(v.AuxInt)
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(B2i32(t))
		return true
	}
}
func rewriteValueARM_OpConstNil(v *ssa.Value) bool {
	// match: (ConstNil)
	// result: (MOVWconst [0])
	for {
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
}
func rewriteValueARM_OpCtz16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz16 <t> x)
	// cond: buildcfg.GOARM.Version<=6
	// result: (RSBconst [32] (CLZ <t> (SUBconst <typ.UInt32> (AND <typ.UInt32> (ORconst <typ.UInt32> [0x10000] x) (RSBconst <typ.UInt32> [0] (ORconst <typ.UInt32> [0x10000] x))) [1])))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOARM.Version <= 6) {
			break
		}
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCLZ, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMSUBconst, typ.UInt32)
		v1.AuxInt = Int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMAND, typ.UInt32)
		v3 := b.NewValue0(v.Pos, ssaop.OpARMORconst, typ.UInt32)
		v3.AuxInt = Int32ToAuxInt(0x10000)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, ssaop.OpARMRSBconst, typ.UInt32)
		v4.AuxInt = Int32ToAuxInt(0)
		v4.AddArg(v3)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Ctz16 <t> x)
	// cond: buildcfg.GOARM.Version==7
	// result: (CLZ <t> (RBIT <typ.UInt32> (ORconst <typ.UInt32> [0x10000] x)))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOARM.Version == 7) {
			break
		}
		v.Reset(ssaop.OpARMCLZ)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpARMRBIT, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMORconst, typ.UInt32)
		v1.AuxInt = Int32ToAuxInt(0x10000)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpCtz32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Ctz32 <t> x)
	// cond: buildcfg.GOARM.Version<=6
	// result: (RSBconst [32] (CLZ <t> (SUBconst <t> (AND <t> x (RSBconst <t> [0] x)) [1])))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOARM.Version <= 6) {
			break
		}
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCLZ, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMSUBconst, t)
		v1.AuxInt = Int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMAND, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpARMRSBconst, t)
		v3.AuxInt = Int32ToAuxInt(0)
		v3.AddArg(x)
		v2.AddArg2(x, v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Ctz32 <t> x)
	// cond: buildcfg.GOARM.Version==7
	// result: (CLZ <t> (RBIT <t> x))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOARM.Version == 7) {
			break
		}
		v.Reset(ssaop.OpARMCLZ)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpARMRBIT, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpCtz8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Ctz8 <t> x)
	// cond: buildcfg.GOARM.Version<=6
	// result: (RSBconst [32] (CLZ <t> (SUBconst <typ.UInt32> (AND <typ.UInt32> (ORconst <typ.UInt32> [0x100] x) (RSBconst <typ.UInt32> [0] (ORconst <typ.UInt32> [0x100] x))) [1])))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOARM.Version <= 6) {
			break
		}
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(32)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCLZ, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMSUBconst, typ.UInt32)
		v1.AuxInt = Int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMAND, typ.UInt32)
		v3 := b.NewValue0(v.Pos, ssaop.OpARMORconst, typ.UInt32)
		v3.AuxInt = Int32ToAuxInt(0x100)
		v3.AddArg(x)
		v4 := b.NewValue0(v.Pos, ssaop.OpARMRSBconst, typ.UInt32)
		v4.AuxInt = Int32ToAuxInt(0)
		v4.AddArg(v3)
		v2.AddArg2(v3, v4)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Ctz8 <t> x)
	// cond: buildcfg.GOARM.Version==7
	// result: (CLZ <t> (RBIT <typ.UInt32> (ORconst <typ.UInt32> [0x100] x)))
	for {
		t := v.Type
		x := v_0
		if !(buildcfg.GOARM.Version == 7) {
			break
		}
		v.Reset(ssaop.OpARMCLZ)
		v.Type = t
		v0 := b.NewValue0(v.Pos, ssaop.OpARMRBIT, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMORconst, typ.UInt32)
		v1.AuxInt = Int32ToAuxInt(0x100)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpDiv16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 x y)
	// result: (Div32 (SignExt16to32 x) (SignExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpDiv32)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM_OpDiv16u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (Div32u (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpDiv32u)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM_OpDiv32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32 x y)
	// result: (SUB (XOR <typ.UInt32> (Select0 <typ.UInt32> (CALLudiv (SUB <typ.UInt32> (XOR x <typ.UInt32> (Signmask x)) (Signmask x)) (SUB <typ.UInt32> (XOR y <typ.UInt32> (Signmask y)) (Signmask y)))) (Signmask (XOR <typ.UInt32> x y))) (Signmask (XOR <typ.UInt32> x y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMXOR, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpSelect0, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMCALLudiv, types.NewTuple(typ.UInt32, typ.UInt32))
		v3 := b.NewValue0(v.Pos, ssaop.OpARMSUB, typ.UInt32)
		v4 := b.NewValue0(v.Pos, ssaop.OpARMXOR, typ.UInt32)
		v5 := b.NewValue0(v.Pos, ssaop.OpSignmask, typ.Int32)
		v5.AddArg(x)
		v4.AddArg2(x, v5)
		v3.AddArg2(v4, v5)
		v6 := b.NewValue0(v.Pos, ssaop.OpARMSUB, typ.UInt32)
		v7 := b.NewValue0(v.Pos, ssaop.OpARMXOR, typ.UInt32)
		v8 := b.NewValue0(v.Pos, ssaop.OpSignmask, typ.Int32)
		v8.AddArg(y)
		v7.AddArg2(y, v8)
		v6.AddArg2(v7, v8)
		v2.AddArg2(v3, v6)
		v1.AddArg(v2)
		v9 := b.NewValue0(v.Pos, ssaop.OpSignmask, typ.Int32)
		v10 := b.NewValue0(v.Pos, ssaop.OpARMXOR, typ.UInt32)
		v10.AddArg2(x, y)
		v9.AddArg(v10)
		v0.AddArg2(v1, v9)
		v.AddArg2(v0, v9)
		return true
	}
}
func rewriteValueARM_OpDiv32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div32u x y)
	// result: (Select0 <typ.UInt32> (CALLudiv x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect0)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCALLudiv, types.NewTuple(typ.UInt32, typ.UInt32))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpDiv8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (Div32 (SignExt8to32 x) (SignExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpDiv32)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM_OpDiv8u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (Div32u (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpDiv32u)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM_OpEq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// result: (Equal (CMP (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpEq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32 x y)
	// result: (Equal (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpEq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq32F x y)
	// result: (Equal (CMPF x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPF, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpEq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Eq64F x y)
	// result: (Equal (CMPD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpEq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// result: (Equal (CMP (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpEqB(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// result: (XORconst [1] (XOR <typ.Bool> x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMXOR, typ.Bool)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpEqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (EqPtr x y)
	// result: (Equal (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpFMA(v *ssa.Value) bool {
	v_2 := v.Args[2]
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (FMA x y z)
	// result: (FMULAD z x y)
	for {
		x := v_0
		y := v_1
		z := v_2
		v.Reset(ssaop.OpARMFMULAD)
		v.AddArg3(z, x, y)
		return true
	}
}
func rewriteValueARM_OpIsInBounds(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsInBounds idx len)
	// result: (LessThanU (CMP idx len))
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.OpARMLessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpIsNonNil(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsNonNil ptr)
	// result: (NotEqual (CMPconst [0] ptr))
	for {
		ptr := v_0
		v.Reset(ssaop.OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = Int32ToAuxInt(0)
		v0.AddArg(ptr)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpIsSliceInBounds(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (IsSliceInBounds idx len)
	// result: (LessEqualU (CMP idx len))
	for {
		idx := v_0
		len := v_1
		v.Reset(ssaop.OpARMLessEqualU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v0.AddArg2(idx, len)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (LessEqual (CMP (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMLessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// result: (LessEqualU (CMP (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMLessEqualU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32 x y)
	// result: (LessEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMLessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32F x y)
	// result: (GreaterEqual (CMPF y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMGreaterEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPF, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq32U x y)
	// result: (LessEqualU (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMLessEqualU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Leq64F x y)
	// result: (GreaterEqual (CMPD y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMGreaterEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPD, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (LessEqual (CMP (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMLessEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLeq8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// result: (LessEqualU (CMP (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMLessEqualU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLess16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (LessThan (CMP (SignExt16to32 x) (SignExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMLessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLess16U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// result: (LessThanU (CMP (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMLessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLess32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32 x y)
	// result: (LessThan (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMLessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLess32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32F x y)
	// result: (GreaterThan (CMPF y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMGreaterThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPF, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLess32U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less32U x y)
	// result: (LessThanU (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMLessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLess64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Less64F x y)
	// result: (GreaterThan (CMPD y x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMGreaterThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPD, types.TypeFlags)
		v0.AddArg2(y, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLess8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (LessThan (CMP (SignExt8to32 x) (SignExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMLessThan)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLess8U(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// result: (LessThanU (CMP (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMLessThanU)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpLoad(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpARMMOVBUload)
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
		v.Reset(ssaop.OpARMMOVBload)
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
		v.Reset(ssaop.OpARMMOVBUload)
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
		v.Reset(ssaop.OpARMMOVHload)
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
		v.Reset(ssaop.OpARMMOVHUload)
		v.AddArg2(ptr, mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (Is32BitInt(t) || IsPtr(t))
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		ptr := v_0
		mem := v_1
		if !(Is32BitInt(t) || IsPtr(t)) {
			break
		}
		v.Reset(ssaop.OpARMMOVWload)
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
		v.Reset(ssaop.OpARMMOVFload)
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
		v.Reset(ssaop.OpARMMOVDload)
		v.AddArg2(ptr, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpLocalAddr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (LocalAddr <t> {sym} base mem)
	// cond: t.Elem().HasPointers()
	// result: (MOVWaddr {sym} (SPanchored base mem))
	for {
		t := v.Type
		sym := AuxToSym(v.Aux)
		base := v_0
		mem := v_1
		if !(t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpARMMOVWaddr)
		v.Aux = SymToAux(sym)
		v0 := b.NewValue0(v.Pos, ssaop.OpSPanchored, typ.Uintptr)
		v0.AddArg2(base, mem)
		v.AddArg(v0)
		return true
	}
	// match: (LocalAddr <t> {sym} base _)
	// cond: !t.Elem().HasPointers()
	// result: (MOVWaddr {sym} base)
	for {
		t := v.Type
		sym := AuxToSym(v.Aux)
		base := v_0
		if !(!t.Elem().HasPointers()) {
			break
		}
		v.Reset(ssaop.OpARMMOVWaddr)
		v.Aux = SymToAux(sym)
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValueARM_OpLsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 x y)
	// result: (CMOVWHSconst (SLL <x.Type> x (ZeroExt16to32 y)) (CMPconst [256] (ZeroExt16to32 y)) [0])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = Int32ToAuxInt(256)
		v2.AddArg(v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueARM_OpLsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh16x32 x y)
	// result: (CMOVWHSconst (SLL <x.Type> x y) (CMPconst [256] y) [0])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = Int32ToAuxInt(256)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM_OpLsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh16x64 x (Const64 [c]))
	// cond: uint64(c) < 16
	// result: (SLLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpARMSLLconst)
		v.AuxInt = Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Lsh16x64 _ (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (Const16 [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.Reset(ssaop.OpConst16)
		v.AuxInt = Int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpLsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 x y)
	// result: (SLL x (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSLL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueARM_OpLsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 x y)
	// result: (CMOVWHSconst (SLL <x.Type> x (ZeroExt16to32 y)) (CMPconst [256] (ZeroExt16to32 y)) [0])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = Int32ToAuxInt(256)
		v2.AddArg(v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueARM_OpLsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh32x32 x y)
	// result: (CMOVWHSconst (SLL <x.Type> x y) (CMPconst [256] y) [0])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = Int32ToAuxInt(256)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM_OpLsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh32x64 x (Const64 [c]))
	// cond: uint64(c) < 32
	// result: (SLLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpARMSLLconst)
		v.AuxInt = Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Lsh32x64 _ (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (Const32 [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.Reset(ssaop.OpConst32)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpLsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 x y)
	// result: (SLL x (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSLL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueARM_OpLsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 x y)
	// result: (CMOVWHSconst (SLL <x.Type> x (ZeroExt16to32 y)) (CMPconst [256] (ZeroExt16to32 y)) [0])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = Int32ToAuxInt(256)
		v2.AddArg(v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueARM_OpLsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Lsh8x32 x y)
	// result: (CMOVWHSconst (SLL <x.Type> x y) (CMPconst [256] y) [0])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLL, x.Type)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = Int32ToAuxInt(256)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM_OpLsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Lsh8x64 x (Const64 [c]))
	// cond: uint64(c) < 8
	// result: (SLLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpARMSLLconst)
		v.AuxInt = Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Lsh8x64 _ (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (Const8 [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.Reset(ssaop.OpConst8)
		v.AuxInt = Int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpLsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 x y)
	// result: (SLL x (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSLL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueARM_OpMod16(v *ssa.Value) bool {
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
func rewriteValueARM_OpMod16u(v *ssa.Value) bool {
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
func rewriteValueARM_OpMod32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32 x y)
	// result: (SUB (XOR <typ.UInt32> (Select1 <typ.UInt32> (CALLudiv (SUB <typ.UInt32> (XOR <typ.UInt32> x (Signmask x)) (Signmask x)) (SUB <typ.UInt32> (XOR <typ.UInt32> y (Signmask y)) (Signmask y)))) (Signmask x)) (Signmask x))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSUB)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMXOR, typ.UInt32)
		v1 := b.NewValue0(v.Pos, ssaop.OpSelect1, typ.UInt32)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMCALLudiv, types.NewTuple(typ.UInt32, typ.UInt32))
		v3 := b.NewValue0(v.Pos, ssaop.OpARMSUB, typ.UInt32)
		v4 := b.NewValue0(v.Pos, ssaop.OpARMXOR, typ.UInt32)
		v5 := b.NewValue0(v.Pos, ssaop.OpSignmask, typ.Int32)
		v5.AddArg(x)
		v4.AddArg2(x, v5)
		v3.AddArg2(v4, v5)
		v6 := b.NewValue0(v.Pos, ssaop.OpARMSUB, typ.UInt32)
		v7 := b.NewValue0(v.Pos, ssaop.OpARMXOR, typ.UInt32)
		v8 := b.NewValue0(v.Pos, ssaop.OpSignmask, typ.Int32)
		v8.AddArg(y)
		v7.AddArg2(y, v8)
		v6.AddArg2(v7, v8)
		v2.AddArg2(v3, v6)
		v1.AddArg(v2)
		v0.AddArg2(v1, v5)
		v.AddArg2(v0, v5)
		return true
	}
}
func rewriteValueARM_OpMod32u(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod32u x y)
	// result: (Select1 <typ.UInt32> (CALLudiv x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpSelect1)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCALLudiv, types.NewTuple(typ.UInt32, typ.UInt32))
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpMod8(v *ssa.Value) bool {
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
func rewriteValueARM_OpMod8u(v *ssa.Value) bool {
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
func rewriteValueARM_OpMove(v *ssa.Value) bool {
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
	// result: (MOVBstore dst (MOVBUload src mem) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVBUload, typ.UInt8)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore dst (MOVHUload src mem) mem)
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
		v.Reset(ssaop.OpARMMOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVHUload, typ.UInt16)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (MOVBstore [1] dst (MOVBUload [1] src mem) (MOVBstore dst (MOVBUload src mem) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVBstore)
		v.AuxInt = Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVBUload, typ.UInt8)
		v0.AuxInt = Int32ToAuxInt(1)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMMOVBstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMMOVBUload, typ.UInt8)
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
		v.Reset(ssaop.OpARMMOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVWload, typ.UInt32)
		v0.AddArg2(src, mem)
		v.AddArg3(dst, v0, mem)
		return true
	}
	// match: (Move [4] {t} dst src mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [2] dst (MOVHUload [2] src mem) (MOVHstore dst (MOVHUload src mem) mem))
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
		v.Reset(ssaop.OpARMMOVHstore)
		v.AuxInt = Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVHUload, typ.UInt16)
		v0.AuxInt = Int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMMOVHstore, types.TypeMem)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMMOVHUload, typ.UInt16)
		v2.AddArg2(src, mem)
		v1.AddArg3(dst, v2, mem)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (MOVBstore [3] dst (MOVBUload [3] src mem) (MOVBstore [2] dst (MOVBUload [2] src mem) (MOVBstore [1] dst (MOVBUload [1] src mem) (MOVBstore dst (MOVBUload src mem) mem))))
	for {
		if AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVBstore)
		v.AuxInt = Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVBUload, typ.UInt8)
		v0.AuxInt = Int32ToAuxInt(3)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMMOVBstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMMOVBUload, typ.UInt8)
		v2.AuxInt = Int32ToAuxInt(2)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpARMMOVBstore, types.TypeMem)
		v3.AuxInt = Int32ToAuxInt(1)
		v4 := b.NewValue0(v.Pos, ssaop.OpARMMOVBUload, typ.UInt8)
		v4.AuxInt = Int32ToAuxInt(1)
		v4.AddArg2(src, mem)
		v5 := b.NewValue0(v.Pos, ssaop.OpARMMOVBstore, types.TypeMem)
		v6 := b.NewValue0(v.Pos, ssaop.OpARMMOVBUload, typ.UInt8)
		v6.AddArg2(src, mem)
		v5.AddArg3(dst, v6, mem)
		v3.AddArg3(dst, v4, v5)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [3] dst src mem)
	// result: (MOVBstore [2] dst (MOVBUload [2] src mem) (MOVBstore [1] dst (MOVBUload [1] src mem) (MOVBstore dst (MOVBUload src mem) mem)))
	for {
		if AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		dst := v_0
		src := v_1
		mem := v_2
		v.Reset(ssaop.OpARMMOVBstore)
		v.AuxInt = Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVBUload, typ.UInt8)
		v0.AuxInt = Int32ToAuxInt(2)
		v0.AddArg2(src, mem)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMMOVBstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMMOVBUload, typ.UInt8)
		v2.AuxInt = Int32ToAuxInt(1)
		v2.AddArg2(src, mem)
		v3 := b.NewValue0(v.Pos, ssaop.OpARMMOVBstore, types.TypeMem)
		v4 := b.NewValue0(v.Pos, ssaop.OpARMMOVBUload, typ.UInt8)
		v4.AddArg2(src, mem)
		v3.AddArg3(dst, v4, mem)
		v1.AddArg3(dst, v2, v3)
		v.AddArg3(dst, v0, v1)
		return true
	}
	// match: (Move [s] {t} dst src mem)
	// cond: s%4 == 0 && s > 4 && s <= 512 && t.Alignment()%4 == 0 && LogLargeCopyValue(v, s)
	// result: (DUFFCOPY [8 * (128 - s/4)] dst src mem)
	for {
		s := AuxIntToInt64(v.AuxInt)
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !(s%4 == 0 && s > 4 && s <= 512 && t.Alignment()%4 == 0 && LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.OpARMDUFFCOPY)
		v.AuxInt = Int64ToAuxInt(8 * (128 - s/4))
		v.AddArg3(dst, src, mem)
		return true
	}
	// match: (Move [s] {t} dst src mem)
	// cond: (s > 512 || t.Alignment()%4 != 0) && LogLargeCopyValue(v, s)
	// result: (LoweredMove [t.Alignment()] dst src (ADDconst <src.Type> src [int32(s-MoveSize(t.Alignment(), config))]) mem)
	for {
		s := AuxIntToInt64(v.AuxInt)
		t := AuxToType(v.Aux)
		dst := v_0
		src := v_1
		mem := v_2
		if !((s > 512 || t.Alignment()%4 != 0) && LogLargeCopyValue(v, s)) {
			break
		}
		v.Reset(ssaop.OpARMLoweredMove)
		v.AuxInt = Int64ToAuxInt(t.Alignment())
		v0 := b.NewValue0(v.Pos, ssaop.OpARMADDconst, src.Type)
		v0.AuxInt = Int32ToAuxInt(int32(s - MoveSize(t.Alignment(), config)))
		v0.AddArg(src)
		v.AddArg4(dst, src, v0, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpNeg16(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Neg16 x)
	// result: (RSBconst [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpNeg32(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Neg32 x)
	// result: (RSBconst [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpNeg8(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Neg8 x)
	// result: (RSBconst [0] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARMRSBconst)
		v.AuxInt = Int32ToAuxInt(0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpNeq16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// result: (NotEqual (CMP (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpNeq32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32 x y)
	// result: (NotEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpNeq32F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq32F x y)
	// result: (NotEqual (CMPF x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPF, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpNeq64F(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Neq64F x y)
	// result: (NotEqual (CMPD x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPD, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpNeq8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// result: (NotEqual (CMP (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpNeqPtr(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (NeqPtr x y)
	// result: (NotEqual (CMP x y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMNotEqual)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMP, types.TypeFlags)
		v0.AddArg2(x, y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpNot(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Not x)
	// result: (XORconst [1] x)
	for {
		x := v_0
		v.Reset(ssaop.OpARMXORconst)
		v.AuxInt = Int32ToAuxInt(1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpOffPtr(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (OffPtr [off] ptr:(SP))
	// result: (MOVWaddr [int32(off)] ptr)
	for {
		off := AuxIntToInt64(v.AuxInt)
		ptr := v_0
		if ptr.Op != ssaop.OpSP {
			break
		}
		v.Reset(ssaop.OpARMMOVWaddr)
		v.AuxInt = Int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// result: (ADDconst [int32(off)] ptr)
	for {
		off := AuxIntToInt64(v.AuxInt)
		ptr := v_0
		v.Reset(ssaop.OpARMADDconst)
		v.AuxInt = Int32ToAuxInt(int32(off))
		v.AddArg(ptr)
		return true
	}
}
func rewriteValueARM_OpRotateLeft16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVWconst [c]))
	// result: (Or16 (Lsh16x32 <t> x (MOVWconst [c&15])) (Rsh16Ux32 <t> x (MOVWconst [-c&15])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpOr16)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh16x32, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v1.AuxInt = Int32ToAuxInt(c & 15)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh16Ux32, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v3.AuxInt = Int32ToAuxInt(-c & 15)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueARM_OpRotateLeft32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (RotateLeft32 x y)
	// result: (SRR x (RSBconst [0] <y.Type> y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRR)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMRSBconst, y.Type)
		v0.AuxInt = Int32ToAuxInt(0)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueARM_OpRotateLeft8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVWconst [c]))
	// result: (Or8 (Lsh8x32 <t> x (MOVWconst [c&7])) (Rsh8Ux32 <t> x (MOVWconst [-c&7])))
	for {
		t := v.Type
		x := v_0
		if v_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_1.AuxInt)
		v.Reset(ssaop.OpOr8)
		v0 := b.NewValue0(v.Pos, ssaop.OpLsh8x32, t)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v1.AuxInt = Int32ToAuxInt(c & 7)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpRsh8Ux32, t)
		v3 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v3.AuxInt = Int32ToAuxInt(-c & 7)
		v2.AddArg2(x, v3)
		v.AddArg2(v0, v2)
		return true
	}
	return false
}
func rewriteValueARM_OpRsh16Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 x y)
	// result: (CMOVWHSconst (SRL <x.Type> (ZeroExt16to32 x) (ZeroExt16to32 y)) (CMPconst [256] (ZeroExt16to32 y)) [0])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v3.AuxInt = Int32ToAuxInt(256)
		v3.AddArg(v2)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueARM_OpRsh16Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 x y)
	// result: (CMOVWHSconst (SRL <x.Type> (ZeroExt16to32 x) y) (CMPconst [256] y) [0])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = Int32ToAuxInt(256)
		v2.AddArg(y)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueARM_OpRsh16Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 x (Const64 [c]))
	// cond: uint64(c) < 16
	// result: (SRLconst (SLLconst <typ.UInt32> x [16]) [int32(c+16)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpARMSRLconst)
		v.AuxInt = Int32ToAuxInt(int32(c + 16))
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (Const16 [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.Reset(ssaop.OpConst16)
		v.AuxInt = Int16ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpRsh16Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 x y)
	// result: (SRL (ZeroExt16to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM_OpRsh16x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 x y)
	// result: (SRAcond (SignExt16to32 x) (ZeroExt16to32 y) (CMPconst [256] (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRAcond)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = Int32ToAuxInt(256)
		v2.AddArg(v1)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueARM_OpRsh16x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 x y)
	// result: (SRAcond (SignExt16to32 x) y (CMPconst [256] y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRAcond)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = Int32ToAuxInt(256)
		v1.AddArg(y)
		v.AddArg3(v0, y, v1)
		return true
	}
}
func rewriteValueARM_OpRsh16x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 x (Const64 [c]))
	// cond: uint64(c) < 16
	// result: (SRAconst (SLLconst <typ.UInt32> x [16]) [int32(c+16)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 16) {
			break
		}
		v.Reset(ssaop.OpARMSRAconst)
		v.AuxInt = Int32ToAuxInt(int32(c + 16))
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x64 x (Const64 [c]))
	// cond: uint64(c) >= 16
	// result: (SRAconst (SLLconst <typ.UInt32> x [16]) [31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 16) {
			break
		}
		v.Reset(ssaop.OpARMSRAconst)
		v.AuxInt = Int32ToAuxInt(31)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpRsh16x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 x y)
	// result: (SRA (SignExt16to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM_OpRsh32Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 x y)
	// result: (CMOVWHSconst (SRL <x.Type> x (ZeroExt16to32 y)) (CMPconst [256] (ZeroExt16to32 y)) [0])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v0.AddArg2(x, v1)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = Int32ToAuxInt(256)
		v2.AddArg(v1)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueARM_OpRsh32Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32Ux32 x y)
	// result: (CMOVWHSconst (SRL <x.Type> x y) (CMPconst [256] y) [0])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v0.AddArg2(x, y)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = Int32ToAuxInt(256)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM_OpRsh32Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh32Ux64 x (Const64 [c]))
	// cond: uint64(c) < 32
	// result: (SRLconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpARMSRLconst)
		v.AuxInt = Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh32Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (Const32 [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.Reset(ssaop.OpConst32)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpRsh32Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 x y)
	// result: (SRL x (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueARM_OpRsh32x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 x y)
	// result: (SRAcond x (ZeroExt16to32 y) (CMPconst [256] (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRAcond)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = Int32ToAuxInt(256)
		v1.AddArg(v0)
		v.AddArg3(x, v0, v1)
		return true
	}
}
func rewriteValueARM_OpRsh32x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	// match: (Rsh32x32 x y)
	// result: (SRAcond x y (CMPconst [256] y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRAcond)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v0.AuxInt = Int32ToAuxInt(256)
		v0.AddArg(y)
		v.AddArg3(x, y, v0)
		return true
	}
}
func rewriteValueARM_OpRsh32x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	// match: (Rsh32x64 x (Const64 [c]))
	// cond: uint64(c) < 32
	// result: (SRAconst x [int32(c)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 32) {
			break
		}
		v.Reset(ssaop.OpARMSRAconst)
		v.AuxInt = Int32ToAuxInt(int32(c))
		v.AddArg(x)
		return true
	}
	// match: (Rsh32x64 x (Const64 [c]))
	// cond: uint64(c) >= 32
	// result: (SRAconst x [31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 32) {
			break
		}
		v.Reset(ssaop.OpARMSRAconst)
		v.AuxInt = Int32ToAuxInt(31)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM_OpRsh32x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 x y)
	// result: (SRA x (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(y)
		v.AddArg2(x, v0)
		return true
	}
}
func rewriteValueARM_OpRsh8Ux16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 x y)
	// result: (CMOVWHSconst (SRL <x.Type> (ZeroExt8to32 x) (ZeroExt16to32 y)) (CMPconst [256] (ZeroExt16to32 y)) [0])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v2.AddArg(y)
		v0.AddArg2(v1, v2)
		v3 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v3.AuxInt = Int32ToAuxInt(256)
		v3.AddArg(v2)
		v.AddArg2(v0, v3)
		return true
	}
}
func rewriteValueARM_OpRsh8Ux32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 x y)
	// result: (CMOVWHSconst (SRL <x.Type> (ZeroExt8to32 x) y) (CMPconst [256] y) [0])
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMCMOVWHSconst)
		v.AuxInt = Int32ToAuxInt(0)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSRL, x.Type)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(x)
		v0.AddArg2(v1, y)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = Int32ToAuxInt(256)
		v2.AddArg(y)
		v.AddArg2(v0, v2)
		return true
	}
}
func rewriteValueARM_OpRsh8Ux64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 x (Const64 [c]))
	// cond: uint64(c) < 8
	// result: (SRLconst (SLLconst <typ.UInt32> x [24]) [int32(c+24)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpARMSRLconst)
		v.AuxInt = Int32ToAuxInt(int32(c + 24))
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(24)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (Const8 [0])
	for {
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.Reset(ssaop.OpConst8)
		v.AuxInt = Int8ToAuxInt(0)
		return true
	}
	return false
}
func rewriteValueARM_OpRsh8Ux8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 x y)
	// result: (SRL (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRL)
		v0 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM_OpRsh8x16(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 x y)
	// result: (SRAcond (SignExt8to32 x) (ZeroExt16to32 y) (CMPconst [256] (ZeroExt16to32 y)))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRAcond)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v2.AuxInt = Int32ToAuxInt(256)
		v2.AddArg(v1)
		v.AddArg3(v0, v1, v2)
		return true
	}
}
func rewriteValueARM_OpRsh8x32(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 x y)
	// result: (SRAcond (SignExt8to32 x) y (CMPconst [256] y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRAcond)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
		v1.AuxInt = Int32ToAuxInt(256)
		v1.AddArg(y)
		v.AddArg3(v0, y, v1)
		return true
	}
}
func rewriteValueARM_OpRsh8x64(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 x (Const64 [c]))
	// cond: uint64(c) < 8
	// result: (SRAconst (SLLconst <typ.UInt32> x [24]) [int32(c+24)])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) < 8) {
			break
		}
		v.Reset(ssaop.OpARMSRAconst)
		v.AuxInt = Int32ToAuxInt(int32(c + 24))
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(24)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8x64 x (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (SRAconst (SLLconst <typ.UInt32> x [24]) [31])
	for {
		x := v_0
		if v_1.Op != ssaop.OpConst64 {
			break
		}
		c := AuxIntToInt64(v_1.AuxInt)
		if !(uint64(c) >= 8) {
			break
		}
		v.Reset(ssaop.OpARMSRAconst)
		v.AuxInt = Int32ToAuxInt(31)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMSLLconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(24)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM_OpRsh8x8(v *ssa.Value) bool {
	v_1 := v.Args[1]
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 x y)
	// result: (SRA (SignExt8to32 x) (ZeroExt8to32 y))
	for {
		x := v_0
		y := v_1
		v.Reset(ssaop.OpARMSRA)
		v0 := b.NewValue0(v.Pos, ssaop.OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, ssaop.OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg2(v0, v1)
		return true
	}
}
func rewriteValueARM_OpSelect0(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Select0 (CALLudiv x (MOVWconst [1])))
	// result: x
	for {
		if v_0.Op != ssaop.OpARMCALLudiv {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpARMMOVWconst || AuxIntToInt32(v_0_1.AuxInt) != 1 {
			break
		}
		v.CopyOf(x)
		return true
	}
	// match: (Select0 (CALLudiv x (MOVWconst [c])))
	// cond: IsPowerOfTwo(c)
	// result: (SRLconst [int32(Log32(c))] x)
	for {
		if v_0.Op != ssaop.OpARMCALLudiv {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0_1.AuxInt)
		if !(IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARMSRLconst)
		v.AuxInt = Int32ToAuxInt(int32(Log32(c)))
		v.AddArg(x)
		return true
	}
	// match: (Select0 (CALLudiv (MOVWconst [c]) (MOVWconst [d])))
	// cond: d != 0
	// result: (MOVWconst [int32(uint32(c)/uint32(d))])
	for {
		if v_0.Op != ssaop.OpARMCALLudiv {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0_0.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) / uint32(d)))
		return true
	}
	return false
}
func rewriteValueARM_OpSelect1(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Select1 (CALLudiv _ (MOVWconst [1])))
	// result: (MOVWconst [0])
	for {
		if v_0.Op != ssaop.OpARMCALLudiv {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpARMMOVWconst || AuxIntToInt32(v_0_1.AuxInt) != 1 {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(0)
		return true
	}
	// match: (Select1 (CALLudiv x (MOVWconst [c])))
	// cond: IsPowerOfTwo(c)
	// result: (ANDconst [c-1] x)
	for {
		if v_0.Op != ssaop.OpARMCALLudiv {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0_1.AuxInt)
		if !(IsPowerOfTwo(c)) {
			break
		}
		v.Reset(ssaop.OpARMANDconst)
		v.AuxInt = Int32ToAuxInt(c - 1)
		v.AddArg(x)
		return true
	}
	// match: (Select1 (CALLudiv (MOVWconst [c]) (MOVWconst [d])))
	// cond: d != 0
	// result: (MOVWconst [int32(uint32(c)%uint32(d))])
	for {
		if v_0.Op != ssaop.OpARMCALLudiv {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != ssaop.OpARMMOVWconst {
			break
		}
		c := AuxIntToInt32(v_0_0.AuxInt)
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != ssaop.OpARMMOVWconst {
			break
		}
		d := AuxIntToInt32(v_0_1.AuxInt)
		if !(d != 0) {
			break
		}
		v.Reset(ssaop.OpARMMOVWconst)
		v.AuxInt = Int32ToAuxInt(int32(uint32(c) % uint32(d)))
		return true
	}
	return false
}
func rewriteValueARM_OpSignmask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	// match: (Signmask x)
	// result: (SRAconst x [31])
	for {
		x := v_0
		v.Reset(ssaop.OpARMSRAconst)
		v.AuxInt = Int32ToAuxInt(31)
		v.AddArg(x)
		return true
	}
}
func rewriteValueARM_OpSlicemask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	// match: (Slicemask <t> x)
	// result: (SRAconst (RSBconst <t> [0] x) [31])
	for {
		t := v.Type
		x := v_0
		v.Reset(ssaop.OpARMSRAconst)
		v.AuxInt = Int32ToAuxInt(31)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMRSBconst, t)
		v0.AuxInt = Int32ToAuxInt(0)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM_OpStore(v *ssa.Value) bool {
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
		v.Reset(ssaop.OpARMMOVBstore)
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
		v.Reset(ssaop.OpARMMOVHstore)
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
		v.Reset(ssaop.OpARMMOVWstore)
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
		v.Reset(ssaop.OpARMMOVFstore)
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
		v.Reset(ssaop.OpARMMOVDstore)
		v.AddArg3(ptr, val, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpZero(v *ssa.Value) bool {
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
	// result: (MOVBstore ptr (MOVWconst [0]) mem)
	for {
		if AuxIntToInt64(v.AuxInt) != 1 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARMMOVBstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore ptr (MOVWconst [0]) mem)
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
		v.Reset(ssaop.OpARMMOVHstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [2] ptr mem)
	// result: (MOVBstore [1] ptr (MOVWconst [0]) (MOVBstore [0] ptr (MOVWconst [0]) mem))
	for {
		if AuxIntToInt64(v.AuxInt) != 2 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARMMOVBstore)
		v.AuxInt = Int32ToAuxInt(1)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMMOVBstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(0)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// cond: t.Alignment()%4 == 0
	// result: (MOVWstore ptr (MOVWconst [0]) mem)
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
		v.Reset(ssaop.OpARMMOVWstore)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [4] {t} ptr mem)
	// cond: t.Alignment()%2 == 0
	// result: (MOVHstore [2] ptr (MOVWconst [0]) (MOVHstore [0] ptr (MOVWconst [0]) mem))
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
		v.Reset(ssaop.OpARMMOVHstore)
		v.AuxInt = Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMMOVHstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(0)
		v1.AddArg3(ptr, v0, mem)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [4] ptr mem)
	// result: (MOVBstore [3] ptr (MOVWconst [0]) (MOVBstore [2] ptr (MOVWconst [0]) (MOVBstore [1] ptr (MOVWconst [0]) (MOVBstore [0] ptr (MOVWconst [0]) mem))))
	for {
		if AuxIntToInt64(v.AuxInt) != 4 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARMMOVBstore)
		v.AuxInt = Int32ToAuxInt(3)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMMOVBstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(2)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMMOVBstore, types.TypeMem)
		v2.AuxInt = Int32ToAuxInt(1)
		v3 := b.NewValue0(v.Pos, ssaop.OpARMMOVBstore, types.TypeMem)
		v3.AuxInt = Int32ToAuxInt(0)
		v3.AddArg3(ptr, v0, mem)
		v2.AddArg3(ptr, v0, v3)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [3] ptr mem)
	// result: (MOVBstore [2] ptr (MOVWconst [0]) (MOVBstore [1] ptr (MOVWconst [0]) (MOVBstore [0] ptr (MOVWconst [0]) mem)))
	for {
		if AuxIntToInt64(v.AuxInt) != 3 {
			break
		}
		ptr := v_0
		mem := v_1
		v.Reset(ssaop.OpARMMOVBstore)
		v.AuxInt = Int32ToAuxInt(2)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(0)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMMOVBstore, types.TypeMem)
		v1.AuxInt = Int32ToAuxInt(1)
		v2 := b.NewValue0(v.Pos, ssaop.OpARMMOVBstore, types.TypeMem)
		v2.AuxInt = Int32ToAuxInt(0)
		v2.AddArg3(ptr, v0, mem)
		v1.AddArg3(ptr, v0, v2)
		v.AddArg3(ptr, v0, v1)
		return true
	}
	// match: (Zero [s] {t} ptr mem)
	// cond: s%4 == 0 && s > 4 && s <= 512 && t.Alignment()%4 == 0
	// result: (DUFFZERO [4 * (128 - s/4)] ptr (MOVWconst [0]) mem)
	for {
		s := AuxIntToInt64(v.AuxInt)
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(s%4 == 0 && s > 4 && s <= 512 && t.Alignment()%4 == 0) {
			break
		}
		v.Reset(ssaop.OpARMDUFFZERO)
		v.AuxInt = Int64ToAuxInt(4 * (128 - s/4))
		v0 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v0.AuxInt = Int32ToAuxInt(0)
		v.AddArg3(ptr, v0, mem)
		return true
	}
	// match: (Zero [s] {t} ptr mem)
	// cond: s > 512 || t.Alignment()%4 != 0
	// result: (LoweredZero [t.Alignment()] ptr (ADDconst <ptr.Type> ptr [int32(s-MoveSize(t.Alignment(), config))]) (MOVWconst [0]) mem)
	for {
		s := AuxIntToInt64(v.AuxInt)
		t := AuxToType(v.Aux)
		ptr := v_0
		mem := v_1
		if !(s > 512 || t.Alignment()%4 != 0) {
			break
		}
		v.Reset(ssaop.OpARMLoweredZero)
		v.AuxInt = Int64ToAuxInt(t.Alignment())
		v0 := b.NewValue0(v.Pos, ssaop.OpARMADDconst, ptr.Type)
		v0.AuxInt = Int32ToAuxInt(int32(s - MoveSize(t.Alignment(), config)))
		v0.AddArg(ptr)
		v1 := b.NewValue0(v.Pos, ssaop.OpARMMOVWconst, typ.UInt32)
		v1.AuxInt = Int32ToAuxInt(0)
		v.AddArg4(ptr, v0, v1, mem)
		return true
	}
	return false
}
func rewriteValueARM_OpZeromask(v *ssa.Value) bool {
	v_0 := v.Args[0]
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Zeromask x)
	// result: (SRAconst (RSBshiftRL <typ.Int32> x x [1]) [31])
	for {
		x := v_0
		v.Reset(ssaop.OpARMSRAconst)
		v.AuxInt = Int32ToAuxInt(31)
		v0 := b.NewValue0(v.Pos, ssaop.OpARMRSBshiftRL, typ.Int32)
		v0.AuxInt = Int32ToAuxInt(1)
		v0.AddArg2(x, x)
		v.AddArg(v0)
		return true
	}
}
func rewriteBlockARM(b *ssa.Block) bool {
	switch b.Kind {
	case block.BlockARMEQ:
		// match: (EQ (FlagConstant [fc]) yes no)
		// cond: fc.Eq()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Eq()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (EQ (FlagConstant [fc]) yes no)
		// cond: !fc.Eq()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Eq()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (EQ (InvertFlags cmp) yes no)
		// result: (EQ cmp yes no)
		for b.Controls[0].Op == ssaop.OpARMInvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARMEQ, cmp)
			return true
		}
		// match: (EQ (CMP x (RSBconst [0] y)))
		// result: (EQ (CMN x y))
		for b.Controls[0].Op == ssaop.OpARMCMP {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.OpARMRSBconst || AuxIntToInt32(v_0_1.AuxInt) != 0 {
				break
			}
			y := v_0_1.Args[0]
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMN x (RSBconst [0] y)))
		// result: (EQ (CMP x y))
		for b.Controls[0].Op == ssaop.OpARMCMN {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != ssaop.OpARMRSBconst || AuxIntToInt32(v_0_1.AuxInt) != 0 {
					continue
				}
				y := v_0_1.Args[0]
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMP, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMEQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPconst [0] l:(SUB x y)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMP x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUB {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMP, types.TypeFlags)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(MULS x y a)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMP a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMMULS {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMP, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARMMUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(SUBshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMPshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADD x y)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADD {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMEQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPconst [0] l:(MULA x y a)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMMULA {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARMMUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ADDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (CMNshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(AND x y)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TST x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMAND {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMEQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPconst [0] l:(ANDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(ANDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TSTshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(XOR x y)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQ x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXOR {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQ, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMEQ, v0)
				return true
			}
			break
		}
		// match: (EQ (CMPconst [0] l:(XORconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
		// match: (EQ (CMPconst [0] l:(XORshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (EQ (TEQshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMEQ, v0)
			return true
		}
	case block.BlockARMGE:
		// match: (GE (FlagConstant [fc]) yes no)
		// cond: fc.Ge()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Ge()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GE (FlagConstant [fc]) yes no)
		// cond: !fc.Ge()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Ge()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (GE (InvertFlags cmp) yes no)
		// result: (LE cmp yes no)
		for b.Controls[0].Op == ssaop.OpARMInvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARMLE, cmp)
			return true
		}
		// match: (GE (CMPconst [0] l:(ADD x y)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADD {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMGEnoov, v0)
				return true
			}
			break
		}
		// match: (GE (CMPconst [0] l:(MULA x y a)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMMULA {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARMMUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (CMNconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (CMNshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (CMNshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (CMNshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (CMNshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (CMNshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ADDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (CMNshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(AND x y)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TST x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMAND {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMGEnoov, v0)
				return true
			}
			break
		}
		// match: (GE (CMPconst [0] l:(ANDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TSTconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TSTshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TSTshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TSTshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TSTshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TSTshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(ANDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TSTshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(XOR x y)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TEQ x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXOR {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQ, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMGEnoov, v0)
				return true
			}
			break
		}
		// match: (GE (CMPconst [0] l:(XORconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TEQconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(XORshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TEQshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(XORshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TEQshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(XORshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TEQshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(XORshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TEQshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(XORshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TEQshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
		// match: (GE (CMPconst [0] l:(XORshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GEnoov (TEQshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGEnoov, v0)
			return true
		}
	case block.BlockARMGEnoov:
		// match: (GEnoov (FlagConstant [fc]) yes no)
		// cond: fc.GeNoov()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.GeNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GEnoov (FlagConstant [fc]) yes no)
		// cond: !fc.GeNoov()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.GeNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockARMGT:
		// match: (GT (FlagConstant [fc]) yes no)
		// cond: fc.Gt()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Gt()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GT (FlagConstant [fc]) yes no)
		// cond: !fc.Gt()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Gt()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (GT (InvertFlags cmp) yes no)
		// result: (LT cmp yes no)
		for b.Controls[0].Op == ssaop.OpARMInvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARMLT, cmp)
			return true
		}
		// match: (GT (CMPconst [0] l:(ADD x y)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADD {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMGTnoov, v0)
				return true
			}
			break
		}
		// match: (GT (CMPconst [0] l:(ADDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (CMNconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (CMNshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (CMNshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (CMNshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (CMNshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (CMNshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(ADDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (CMNshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(MULA x y a)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMMULA {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARMMUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(AND x y)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TST x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMAND {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMGTnoov, v0)
				return true
			}
			break
		}
		// match: (GT (CMPconst [0] l:(ANDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TSTconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TSTshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TSTshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TSTshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TSTshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TSTshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(ANDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TSTshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(XOR x y)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TEQ x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXOR {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQ, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMGTnoov, v0)
				return true
			}
			break
		}
		// match: (GT (CMPconst [0] l:(XORconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TEQconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(XORshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TEQshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(XORshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TEQshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(XORshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TEQshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(XORshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TEQshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(XORshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TEQshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
		// match: (GT (CMPconst [0] l:(XORshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (GTnoov (TEQshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMGTnoov, v0)
			return true
		}
	case block.BlockARMGTnoov:
		// match: (GTnoov (FlagConstant [fc]) yes no)
		// cond: fc.GtNoov()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.GtNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (GTnoov (FlagConstant [fc]) yes no)
		// cond: !fc.GtNoov()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
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
		for b.Controls[0].Op == ssaop.OpARMEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARMEQ, cc)
			return true
		}
		// match: (If (NotEqual cc) yes no)
		// result: (NE cc yes no)
		for b.Controls[0].Op == ssaop.OpARMNotEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARMNE, cc)
			return true
		}
		// match: (If (LessThan cc) yes no)
		// result: (LT cc yes no)
		for b.Controls[0].Op == ssaop.OpARMLessThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARMLT, cc)
			return true
		}
		// match: (If (LessThanU cc) yes no)
		// result: (ULT cc yes no)
		for b.Controls[0].Op == ssaop.OpARMLessThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARMULT, cc)
			return true
		}
		// match: (If (LessEqual cc) yes no)
		// result: (LE cc yes no)
		for b.Controls[0].Op == ssaop.OpARMLessEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARMLE, cc)
			return true
		}
		// match: (If (LessEqualU cc) yes no)
		// result: (ULE cc yes no)
		for b.Controls[0].Op == ssaop.OpARMLessEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARMULE, cc)
			return true
		}
		// match: (If (GreaterThan cc) yes no)
		// result: (GT cc yes no)
		for b.Controls[0].Op == ssaop.OpARMGreaterThan {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARMGT, cc)
			return true
		}
		// match: (If (GreaterThanU cc) yes no)
		// result: (UGT cc yes no)
		for b.Controls[0].Op == ssaop.OpARMGreaterThanU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARMUGT, cc)
			return true
		}
		// match: (If (GreaterEqual cc) yes no)
		// result: (GE cc yes no)
		for b.Controls[0].Op == ssaop.OpARMGreaterEqual {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARMGE, cc)
			return true
		}
		// match: (If (GreaterEqualU cc) yes no)
		// result: (UGE cc yes no)
		for b.Controls[0].Op == ssaop.OpARMGreaterEqualU {
			v_0 := b.Controls[0]
			cc := v_0.Args[0]
			b.ResetWithControl(block.BlockARMUGE, cc)
			return true
		}
		// match: (If cond yes no)
		// result: (NE (CMPconst [0] cond) yes no)
		for {
			cond := b.Controls[0]
			v0 := b.NewValue0(cond.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(0)
			v0.AddArg(cond)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
	case block.BlockARMLE:
		// match: (LE (FlagConstant [fc]) yes no)
		// cond: fc.Le()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Le()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LE (FlagConstant [fc]) yes no)
		// cond: !fc.Le()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Le()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (LE (InvertFlags cmp) yes no)
		// result: (GE cmp yes no)
		for b.Controls[0].Op == ssaop.OpARMInvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARMGE, cmp)
			return true
		}
		// match: (LE (CMPconst [0] l:(ADD x y)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADD {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMLEnoov, v0)
				return true
			}
			break
		}
		// match: (LE (CMPconst [0] l:(MULA x y a)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMMULA {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARMMUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (CMNconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (CMNshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (CMNshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (CMNshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (CMNshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (CMNshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ADDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (CMNshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(AND x y)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TST x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMAND {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMLEnoov, v0)
				return true
			}
			break
		}
		// match: (LE (CMPconst [0] l:(ANDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TSTconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TSTshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TSTshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TSTshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TSTshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TSTshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(ANDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TSTshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(XOR x y)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TEQ x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXOR {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQ, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMLEnoov, v0)
				return true
			}
			break
		}
		// match: (LE (CMPconst [0] l:(XORconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TEQconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(XORshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TEQshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(XORshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TEQshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(XORshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TEQshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(XORshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TEQshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(XORshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TEQshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
		// match: (LE (CMPconst [0] l:(XORshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LEnoov (TEQshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLEnoov, v0)
			return true
		}
	case block.BlockARMLEnoov:
		// match: (LEnoov (FlagConstant [fc]) yes no)
		// cond: fc.LeNoov()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.LeNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LEnoov (FlagConstant [fc]) yes no)
		// cond: !fc.LeNoov()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.LeNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockARMLT:
		// match: (LT (FlagConstant [fc]) yes no)
		// cond: fc.Lt()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Lt()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LT (FlagConstant [fc]) yes no)
		// cond: !fc.Lt()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Lt()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (LT (InvertFlags cmp) yes no)
		// result: (GT cmp yes no)
		for b.Controls[0].Op == ssaop.OpARMInvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARMGT, cmp)
			return true
		}
		// match: (LT (CMPconst [0] l:(ADD x y)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADD {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMLTnoov, v0)
				return true
			}
			break
		}
		// match: (LT (CMPconst [0] l:(MULA x y a)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMMULA {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARMMUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (CMNconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (CMNshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (CMNshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (CMNshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (CMNshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (CMNshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ADDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (CMNshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(AND x y)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TST x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMAND {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMLTnoov, v0)
				return true
			}
			break
		}
		// match: (LT (CMPconst [0] l:(ANDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TSTconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TSTshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TSTshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TSTshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TSTshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TSTshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(ANDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TSTshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(XOR x y)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TEQ x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXOR {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQ, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMLTnoov, v0)
				return true
			}
			break
		}
		// match: (LT (CMPconst [0] l:(XORconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TEQconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(XORshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TEQshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(XORshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TEQshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(XORshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TEQshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(XORshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TEQshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(XORshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TEQshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
		// match: (LT (CMPconst [0] l:(XORshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (LTnoov (TEQshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMLTnoov, v0)
			return true
		}
	case block.BlockARMLTnoov:
		// match: (LTnoov (FlagConstant [fc]) yes no)
		// cond: fc.LtNoov()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.LtNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (LTnoov (FlagConstant [fc]) yes no)
		// cond: !fc.LtNoov()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.LtNoov()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
	case block.BlockARMNE:
		// match: (NE (CMPconst [0] (Equal cc)) yes no)
		// result: (EQ cc yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpARMEqual {
				break
			}
			cc := v_0_0.Args[0]
			b.ResetWithControl(block.BlockARMEQ, cc)
			return true
		}
		// match: (NE (CMPconst [0] (NotEqual cc)) yes no)
		// result: (NE cc yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpARMNotEqual {
				break
			}
			cc := v_0_0.Args[0]
			b.ResetWithControl(block.BlockARMNE, cc)
			return true
		}
		// match: (NE (CMPconst [0] (LessThan cc)) yes no)
		// result: (LT cc yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpARMLessThan {
				break
			}
			cc := v_0_0.Args[0]
			b.ResetWithControl(block.BlockARMLT, cc)
			return true
		}
		// match: (NE (CMPconst [0] (LessThanU cc)) yes no)
		// result: (ULT cc yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpARMLessThanU {
				break
			}
			cc := v_0_0.Args[0]
			b.ResetWithControl(block.BlockARMULT, cc)
			return true
		}
		// match: (NE (CMPconst [0] (LessEqual cc)) yes no)
		// result: (LE cc yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpARMLessEqual {
				break
			}
			cc := v_0_0.Args[0]
			b.ResetWithControl(block.BlockARMLE, cc)
			return true
		}
		// match: (NE (CMPconst [0] (LessEqualU cc)) yes no)
		// result: (ULE cc yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpARMLessEqualU {
				break
			}
			cc := v_0_0.Args[0]
			b.ResetWithControl(block.BlockARMULE, cc)
			return true
		}
		// match: (NE (CMPconst [0] (GreaterThan cc)) yes no)
		// result: (GT cc yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpARMGreaterThan {
				break
			}
			cc := v_0_0.Args[0]
			b.ResetWithControl(block.BlockARMGT, cc)
			return true
		}
		// match: (NE (CMPconst [0] (GreaterThanU cc)) yes no)
		// result: (UGT cc yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpARMGreaterThanU {
				break
			}
			cc := v_0_0.Args[0]
			b.ResetWithControl(block.BlockARMUGT, cc)
			return true
		}
		// match: (NE (CMPconst [0] (GreaterEqual cc)) yes no)
		// result: (GE cc yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpARMGreaterEqual {
				break
			}
			cc := v_0_0.Args[0]
			b.ResetWithControl(block.BlockARMGE, cc)
			return true
		}
		// match: (NE (CMPconst [0] (GreaterEqualU cc)) yes no)
		// result: (UGE cc yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != ssaop.OpARMGreaterEqualU {
				break
			}
			cc := v_0_0.Args[0]
			b.ResetWithControl(block.BlockARMUGE, cc)
			return true
		}
		// match: (NE (FlagConstant [fc]) yes no)
		// cond: fc.Ne()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Ne()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (NE (FlagConstant [fc]) yes no)
		// cond: !fc.Ne()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Ne()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (NE (InvertFlags cmp) yes no)
		// result: (NE cmp yes no)
		for b.Controls[0].Op == ssaop.OpARMInvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARMNE, cmp)
			return true
		}
		// match: (NE (CMP x (RSBconst [0] y)))
		// result: (NE (CMN x y))
		for b.Controls[0].Op == ssaop.OpARMCMP {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			x := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			if v_0_1.Op != ssaop.OpARMRSBconst || AuxIntToInt32(v_0_1.AuxInt) != 0 {
				break
			}
			y := v_0_1.Args[0]
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMN x (RSBconst [0] y)))
		// result: (NE (CMP x y))
		for b.Controls[0].Op == ssaop.OpARMCMN {
			v_0 := b.Controls[0]
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			v_0_1 := v_0.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, v_0_0, v_0_1 = _i0+1, v_0_1, v_0_0 {
				x := v_0_0
				if v_0_1.Op != ssaop.OpARMRSBconst || AuxIntToInt32(v_0_1.AuxInt) != 0 {
					continue
				}
				y := v_0_1.Args[0]
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMP, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMNE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPconst [0] l:(SUB x y)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMP x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUB {
				break
			}
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMP, types.TypeFlags)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(MULS x y a)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMP a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMMULS {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMP, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARMMUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(SUBshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMPshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMSUBshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMPshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ADD x y)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMN x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADD {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMNE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPconst [0] l:(MULA x y a)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMN a (MUL <x.Type> x y)) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMMULA {
				break
			}
			a := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMN, types.TypeFlags)
			v1 := b.NewValue0(v_0.Pos, ssaop.OpARMMUL, x.Type)
			v1.AddArg2(x, y)
			v0.AddArg2(a, v1)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ADDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (CMNshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMADDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMCMNshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(AND x y)) yes no)
		// cond: l.Uses==1
		// result: (NE (TST x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMAND {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTST, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMNE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPconst [0] l:(ANDconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(ANDshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (TSTshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMANDshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTSTshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(XOR x y)) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQ x y) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXOR {
				break
			}
			_ = l.Args[1]
			l_0 := l.Args[0]
			l_1 := l.Args[1]
			for _i0 := 0; _i0 <= 1; _i0, l_0, l_1 = _i0+1, l_1, l_0 {
				x := l_0
				y := l_1
				if !(l.Uses == 1) {
					continue
				}
				v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQ, types.TypeFlags)
				v0.AddArg2(x, y)
				b.ResetWithControl(block.BlockARMNE, v0)
				return true
			}
			break
		}
		// match: (NE (CMPconst [0] l:(XORconst [c] x)) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQconst [c] x) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORconst {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQconst, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg(x)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(XORshiftLL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQshiftLL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftLL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftLL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(XORshiftRL x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQshiftRL x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRL {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRL, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(XORshiftRA x y [c])) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQshiftRA x y [c]) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRA {
				break
			}
			c := AuxIntToInt32(l.AuxInt)
			y := l.Args[1]
			x := l.Args[0]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRA, types.TypeFlags)
			v0.AuxInt = Int32ToAuxInt(c)
			v0.AddArg2(x, y)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(XORshiftLLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQshiftLLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftLLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftLLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(XORshiftRLreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQshiftRLreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRLreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRLreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
		// match: (NE (CMPconst [0] l:(XORshiftRAreg x y z)) yes no)
		// cond: l.Uses==1
		// result: (NE (TEQshiftRAreg x y z) yes no)
		for b.Controls[0].Op == ssaop.OpARMCMPconst {
			v_0 := b.Controls[0]
			if AuxIntToInt32(v_0.AuxInt) != 0 {
				break
			}
			l := v_0.Args[0]
			if l.Op != ssaop.OpARMXORshiftRAreg {
				break
			}
			z := l.Args[2]
			x := l.Args[0]
			y := l.Args[1]
			if !(l.Uses == 1) {
				break
			}
			v0 := b.NewValue0(v_0.Pos, ssaop.OpARMTEQshiftRAreg, types.TypeFlags)
			v0.AddArg3(x, y, z)
			b.ResetWithControl(block.BlockARMNE, v0)
			return true
		}
	case block.BlockARMUGE:
		// match: (UGE (FlagConstant [fc]) yes no)
		// cond: fc.Uge()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Uge()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (UGE (FlagConstant [fc]) yes no)
		// cond: !fc.Uge()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Uge()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (UGE (InvertFlags cmp) yes no)
		// result: (ULE cmp yes no)
		for b.Controls[0].Op == ssaop.OpARMInvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARMULE, cmp)
			return true
		}
	case block.BlockARMUGT:
		// match: (UGT (FlagConstant [fc]) yes no)
		// cond: fc.Ugt()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Ugt()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (UGT (FlagConstant [fc]) yes no)
		// cond: !fc.Ugt()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Ugt()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (UGT (InvertFlags cmp) yes no)
		// result: (ULT cmp yes no)
		for b.Controls[0].Op == ssaop.OpARMInvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARMULT, cmp)
			return true
		}
	case block.BlockARMULE:
		// match: (ULE (FlagConstant [fc]) yes no)
		// cond: fc.Ule()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Ule()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (ULE (FlagConstant [fc]) yes no)
		// cond: !fc.Ule()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Ule()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (ULE (InvertFlags cmp) yes no)
		// result: (UGE cmp yes no)
		for b.Controls[0].Op == ssaop.OpARMInvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARMUGE, cmp)
			return true
		}
	case block.BlockARMULT:
		// match: (ULT (FlagConstant [fc]) yes no)
		// cond: fc.Ult()
		// result: (First yes no)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(fc.Ult()) {
				break
			}
			b.Reset(block.BlockFirst)
			return true
		}
		// match: (ULT (FlagConstant [fc]) yes no)
		// cond: !fc.Ult()
		// result: (First no yes)
		for b.Controls[0].Op == ssaop.OpARMFlagConstant {
			v_0 := b.Controls[0]
			fc := AuxIntToFlagConstant(v_0.AuxInt)
			if !(!fc.Ult()) {
				break
			}
			b.Reset(block.BlockFirst)
			b.SwapSuccessors()
			return true
		}
		// match: (ULT (InvertFlags cmp) yes no)
		// result: (UGT cmp yes no)
		for b.Controls[0].Op == ssaop.OpARMInvertFlags {
			v_0 := b.Controls[0]
			cmp := v_0.Args[0]
			b.ResetWithControl(block.BlockARMUGT, cmp)
			return true
		}
	}
	return false
}
