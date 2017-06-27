// Code generated from gen/AMD64.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

import "math"
import "cmd/internal/obj"
import "cmd/internal/objabi"
import "cmd/compile/internal/types"

var _ = math.MinInt8  // in case not otherwise used
var _ = obj.ANOP      // in case not otherwise used
var _ = objabi.GOROOT // in case not otherwise used
var _ = types.TypeMem // in case not otherwise used

func rewriteValueAMD64(v *Value) bool {
	switch v.Op {
	case OpAMD64ADDL:
		return rewriteValueAMD64_OpAMD64ADDL_0(v) || rewriteValueAMD64_OpAMD64ADDL_10(v)
	case OpAMD64ADDLconst:
		return rewriteValueAMD64_OpAMD64ADDLconst_0(v)
	case OpAMD64ADDQ:
		return rewriteValueAMD64_OpAMD64ADDQ_0(v) || rewriteValueAMD64_OpAMD64ADDQ_10(v) || rewriteValueAMD64_OpAMD64ADDQ_20(v)
	case OpAMD64ADDQconst:
		return rewriteValueAMD64_OpAMD64ADDQconst_0(v)
	case OpAMD64ADDSD:
		return rewriteValueAMD64_OpAMD64ADDSD_0(v)
	case OpAMD64ADDSS:
		return rewriteValueAMD64_OpAMD64ADDSS_0(v)
	case OpAMD64ANDL:
		return rewriteValueAMD64_OpAMD64ANDL_0(v)
	case OpAMD64ANDLconst:
		return rewriteValueAMD64_OpAMD64ANDLconst_0(v)
	case OpAMD64ANDQ:
		return rewriteValueAMD64_OpAMD64ANDQ_0(v)
	case OpAMD64ANDQconst:
		return rewriteValueAMD64_OpAMD64ANDQconst_0(v)
	case OpAMD64BSFQ:
		return rewriteValueAMD64_OpAMD64BSFQ_0(v)
	case OpAMD64BTQconst:
		return rewriteValueAMD64_OpAMD64BTQconst_0(v)
	case OpAMD64CMOVQEQ:
		return rewriteValueAMD64_OpAMD64CMOVQEQ_0(v)
	case OpAMD64CMPB:
		return rewriteValueAMD64_OpAMD64CMPB_0(v)
	case OpAMD64CMPBconst:
		return rewriteValueAMD64_OpAMD64CMPBconst_0(v)
	case OpAMD64CMPL:
		return rewriteValueAMD64_OpAMD64CMPL_0(v)
	case OpAMD64CMPLconst:
		return rewriteValueAMD64_OpAMD64CMPLconst_0(v)
	case OpAMD64CMPQ:
		return rewriteValueAMD64_OpAMD64CMPQ_0(v)
	case OpAMD64CMPQconst:
		return rewriteValueAMD64_OpAMD64CMPQconst_0(v) || rewriteValueAMD64_OpAMD64CMPQconst_10(v)
	case OpAMD64CMPW:
		return rewriteValueAMD64_OpAMD64CMPW_0(v)
	case OpAMD64CMPWconst:
		return rewriteValueAMD64_OpAMD64CMPWconst_0(v)
	case OpAMD64CMPXCHGLlock:
		return rewriteValueAMD64_OpAMD64CMPXCHGLlock_0(v)
	case OpAMD64CMPXCHGQlock:
		return rewriteValueAMD64_OpAMD64CMPXCHGQlock_0(v)
	case OpAMD64LEAL:
		return rewriteValueAMD64_OpAMD64LEAL_0(v)
	case OpAMD64LEAQ:
		return rewriteValueAMD64_OpAMD64LEAQ_0(v)
	case OpAMD64LEAQ1:
		return rewriteValueAMD64_OpAMD64LEAQ1_0(v)
	case OpAMD64LEAQ2:
		return rewriteValueAMD64_OpAMD64LEAQ2_0(v)
	case OpAMD64LEAQ4:
		return rewriteValueAMD64_OpAMD64LEAQ4_0(v)
	case OpAMD64LEAQ8:
		return rewriteValueAMD64_OpAMD64LEAQ8_0(v)
	case OpAMD64MOVBQSX:
		return rewriteValueAMD64_OpAMD64MOVBQSX_0(v)
	case OpAMD64MOVBQSXload:
		return rewriteValueAMD64_OpAMD64MOVBQSXload_0(v)
	case OpAMD64MOVBQZX:
		return rewriteValueAMD64_OpAMD64MOVBQZX_0(v)
	case OpAMD64MOVBload:
		return rewriteValueAMD64_OpAMD64MOVBload_0(v)
	case OpAMD64MOVBloadidx1:
		return rewriteValueAMD64_OpAMD64MOVBloadidx1_0(v)
	case OpAMD64MOVBstore:
		return rewriteValueAMD64_OpAMD64MOVBstore_0(v) || rewriteValueAMD64_OpAMD64MOVBstore_10(v)
	case OpAMD64MOVBstoreconst:
		return rewriteValueAMD64_OpAMD64MOVBstoreconst_0(v)
	case OpAMD64MOVBstoreconstidx1:
		return rewriteValueAMD64_OpAMD64MOVBstoreconstidx1_0(v)
	case OpAMD64MOVBstoreidx1:
		return rewriteValueAMD64_OpAMD64MOVBstoreidx1_0(v)
	case OpAMD64MOVLQSX:
		return rewriteValueAMD64_OpAMD64MOVLQSX_0(v)
	case OpAMD64MOVLQSXload:
		return rewriteValueAMD64_OpAMD64MOVLQSXload_0(v)
	case OpAMD64MOVLQZX:
		return rewriteValueAMD64_OpAMD64MOVLQZX_0(v)
	case OpAMD64MOVLatomicload:
		return rewriteValueAMD64_OpAMD64MOVLatomicload_0(v)
	case OpAMD64MOVLload:
		return rewriteValueAMD64_OpAMD64MOVLload_0(v)
	case OpAMD64MOVLloadidx1:
		return rewriteValueAMD64_OpAMD64MOVLloadidx1_0(v)
	case OpAMD64MOVLloadidx4:
		return rewriteValueAMD64_OpAMD64MOVLloadidx4_0(v)
	case OpAMD64MOVLstore:
		return rewriteValueAMD64_OpAMD64MOVLstore_0(v) || rewriteValueAMD64_OpAMD64MOVLstore_10(v)
	case OpAMD64MOVLstoreconst:
		return rewriteValueAMD64_OpAMD64MOVLstoreconst_0(v)
	case OpAMD64MOVLstoreconstidx1:
		return rewriteValueAMD64_OpAMD64MOVLstoreconstidx1_0(v)
	case OpAMD64MOVLstoreconstidx4:
		return rewriteValueAMD64_OpAMD64MOVLstoreconstidx4_0(v)
	case OpAMD64MOVLstoreidx1:
		return rewriteValueAMD64_OpAMD64MOVLstoreidx1_0(v)
	case OpAMD64MOVLstoreidx4:
		return rewriteValueAMD64_OpAMD64MOVLstoreidx4_0(v)
	case OpAMD64MOVOload:
		return rewriteValueAMD64_OpAMD64MOVOload_0(v)
	case OpAMD64MOVOstore:
		return rewriteValueAMD64_OpAMD64MOVOstore_0(v)
	case OpAMD64MOVQatomicload:
		return rewriteValueAMD64_OpAMD64MOVQatomicload_0(v)
	case OpAMD64MOVQload:
		return rewriteValueAMD64_OpAMD64MOVQload_0(v)
	case OpAMD64MOVQloadidx1:
		return rewriteValueAMD64_OpAMD64MOVQloadidx1_0(v)
	case OpAMD64MOVQloadidx8:
		return rewriteValueAMD64_OpAMD64MOVQloadidx8_0(v)
	case OpAMD64MOVQstore:
		return rewriteValueAMD64_OpAMD64MOVQstore_0(v)
	case OpAMD64MOVQstoreconst:
		return rewriteValueAMD64_OpAMD64MOVQstoreconst_0(v)
	case OpAMD64MOVQstoreconstidx1:
		return rewriteValueAMD64_OpAMD64MOVQstoreconstidx1_0(v)
	case OpAMD64MOVQstoreconstidx8:
		return rewriteValueAMD64_OpAMD64MOVQstoreconstidx8_0(v)
	case OpAMD64MOVQstoreidx1:
		return rewriteValueAMD64_OpAMD64MOVQstoreidx1_0(v)
	case OpAMD64MOVQstoreidx8:
		return rewriteValueAMD64_OpAMD64MOVQstoreidx8_0(v)
	case OpAMD64MOVSDload:
		return rewriteValueAMD64_OpAMD64MOVSDload_0(v)
	case OpAMD64MOVSDloadidx1:
		return rewriteValueAMD64_OpAMD64MOVSDloadidx1_0(v)
	case OpAMD64MOVSDloadidx8:
		return rewriteValueAMD64_OpAMD64MOVSDloadidx8_0(v)
	case OpAMD64MOVSDstore:
		return rewriteValueAMD64_OpAMD64MOVSDstore_0(v)
	case OpAMD64MOVSDstoreidx1:
		return rewriteValueAMD64_OpAMD64MOVSDstoreidx1_0(v)
	case OpAMD64MOVSDstoreidx8:
		return rewriteValueAMD64_OpAMD64MOVSDstoreidx8_0(v)
	case OpAMD64MOVSSload:
		return rewriteValueAMD64_OpAMD64MOVSSload_0(v)
	case OpAMD64MOVSSloadidx1:
		return rewriteValueAMD64_OpAMD64MOVSSloadidx1_0(v)
	case OpAMD64MOVSSloadidx4:
		return rewriteValueAMD64_OpAMD64MOVSSloadidx4_0(v)
	case OpAMD64MOVSSstore:
		return rewriteValueAMD64_OpAMD64MOVSSstore_0(v)
	case OpAMD64MOVSSstoreidx1:
		return rewriteValueAMD64_OpAMD64MOVSSstoreidx1_0(v)
	case OpAMD64MOVSSstoreidx4:
		return rewriteValueAMD64_OpAMD64MOVSSstoreidx4_0(v)
	case OpAMD64MOVWQSX:
		return rewriteValueAMD64_OpAMD64MOVWQSX_0(v)
	case OpAMD64MOVWQSXload:
		return rewriteValueAMD64_OpAMD64MOVWQSXload_0(v)
	case OpAMD64MOVWQZX:
		return rewriteValueAMD64_OpAMD64MOVWQZX_0(v)
	case OpAMD64MOVWload:
		return rewriteValueAMD64_OpAMD64MOVWload_0(v)
	case OpAMD64MOVWloadidx1:
		return rewriteValueAMD64_OpAMD64MOVWloadidx1_0(v)
	case OpAMD64MOVWloadidx2:
		return rewriteValueAMD64_OpAMD64MOVWloadidx2_0(v)
	case OpAMD64MOVWstore:
		return rewriteValueAMD64_OpAMD64MOVWstore_0(v) || rewriteValueAMD64_OpAMD64MOVWstore_10(v)
	case OpAMD64MOVWstoreconst:
		return rewriteValueAMD64_OpAMD64MOVWstoreconst_0(v)
	case OpAMD64MOVWstoreconstidx1:
		return rewriteValueAMD64_OpAMD64MOVWstoreconstidx1_0(v)
	case OpAMD64MOVWstoreconstidx2:
		return rewriteValueAMD64_OpAMD64MOVWstoreconstidx2_0(v)
	case OpAMD64MOVWstoreidx1:
		return rewriteValueAMD64_OpAMD64MOVWstoreidx1_0(v)
	case OpAMD64MOVWstoreidx2:
		return rewriteValueAMD64_OpAMD64MOVWstoreidx2_0(v)
	case OpAMD64MULL:
		return rewriteValueAMD64_OpAMD64MULL_0(v)
	case OpAMD64MULLconst:
		return rewriteValueAMD64_OpAMD64MULLconst_0(v)
	case OpAMD64MULQ:
		return rewriteValueAMD64_OpAMD64MULQ_0(v)
	case OpAMD64MULQconst:
		return rewriteValueAMD64_OpAMD64MULQconst_0(v) || rewriteValueAMD64_OpAMD64MULQconst_10(v) || rewriteValueAMD64_OpAMD64MULQconst_20(v)
	case OpAMD64MULSD:
		return rewriteValueAMD64_OpAMD64MULSD_0(v)
	case OpAMD64MULSS:
		return rewriteValueAMD64_OpAMD64MULSS_0(v)
	case OpAMD64NEGL:
		return rewriteValueAMD64_OpAMD64NEGL_0(v)
	case OpAMD64NEGQ:
		return rewriteValueAMD64_OpAMD64NEGQ_0(v)
	case OpAMD64NOTL:
		return rewriteValueAMD64_OpAMD64NOTL_0(v)
	case OpAMD64NOTQ:
		return rewriteValueAMD64_OpAMD64NOTQ_0(v)
	case OpAMD64ORL:
		return rewriteValueAMD64_OpAMD64ORL_0(v) || rewriteValueAMD64_OpAMD64ORL_10(v) || rewriteValueAMD64_OpAMD64ORL_20(v) || rewriteValueAMD64_OpAMD64ORL_30(v) || rewriteValueAMD64_OpAMD64ORL_40(v) || rewriteValueAMD64_OpAMD64ORL_50(v) || rewriteValueAMD64_OpAMD64ORL_60(v) || rewriteValueAMD64_OpAMD64ORL_70(v) || rewriteValueAMD64_OpAMD64ORL_80(v) || rewriteValueAMD64_OpAMD64ORL_90(v) || rewriteValueAMD64_OpAMD64ORL_100(v) || rewriteValueAMD64_OpAMD64ORL_110(v) || rewriteValueAMD64_OpAMD64ORL_120(v) || rewriteValueAMD64_OpAMD64ORL_130(v)
	case OpAMD64ORLconst:
		return rewriteValueAMD64_OpAMD64ORLconst_0(v)
	case OpAMD64ORQ:
		return rewriteValueAMD64_OpAMD64ORQ_0(v) || rewriteValueAMD64_OpAMD64ORQ_10(v) || rewriteValueAMD64_OpAMD64ORQ_20(v) || rewriteValueAMD64_OpAMD64ORQ_30(v) || rewriteValueAMD64_OpAMD64ORQ_40(v) || rewriteValueAMD64_OpAMD64ORQ_50(v) || rewriteValueAMD64_OpAMD64ORQ_60(v) || rewriteValueAMD64_OpAMD64ORQ_70(v) || rewriteValueAMD64_OpAMD64ORQ_80(v) || rewriteValueAMD64_OpAMD64ORQ_90(v) || rewriteValueAMD64_OpAMD64ORQ_100(v) || rewriteValueAMD64_OpAMD64ORQ_110(v) || rewriteValueAMD64_OpAMD64ORQ_120(v) || rewriteValueAMD64_OpAMD64ORQ_130(v) || rewriteValueAMD64_OpAMD64ORQ_140(v) || rewriteValueAMD64_OpAMD64ORQ_150(v) || rewriteValueAMD64_OpAMD64ORQ_160(v)
	case OpAMD64ORQconst:
		return rewriteValueAMD64_OpAMD64ORQconst_0(v)
	case OpAMD64ROLB:
		return rewriteValueAMD64_OpAMD64ROLB_0(v)
	case OpAMD64ROLBconst:
		return rewriteValueAMD64_OpAMD64ROLBconst_0(v)
	case OpAMD64ROLL:
		return rewriteValueAMD64_OpAMD64ROLL_0(v)
	case OpAMD64ROLLconst:
		return rewriteValueAMD64_OpAMD64ROLLconst_0(v)
	case OpAMD64ROLQ:
		return rewriteValueAMD64_OpAMD64ROLQ_0(v)
	case OpAMD64ROLQconst:
		return rewriteValueAMD64_OpAMD64ROLQconst_0(v)
	case OpAMD64ROLW:
		return rewriteValueAMD64_OpAMD64ROLW_0(v)
	case OpAMD64ROLWconst:
		return rewriteValueAMD64_OpAMD64ROLWconst_0(v)
	case OpAMD64RORB:
		return rewriteValueAMD64_OpAMD64RORB_0(v)
	case OpAMD64RORL:
		return rewriteValueAMD64_OpAMD64RORL_0(v)
	case OpAMD64RORQ:
		return rewriteValueAMD64_OpAMD64RORQ_0(v)
	case OpAMD64RORW:
		return rewriteValueAMD64_OpAMD64RORW_0(v)
	case OpAMD64SARB:
		return rewriteValueAMD64_OpAMD64SARB_0(v)
	case OpAMD64SARBconst:
		return rewriteValueAMD64_OpAMD64SARBconst_0(v)
	case OpAMD64SARL:
		return rewriteValueAMD64_OpAMD64SARL_0(v)
	case OpAMD64SARLconst:
		return rewriteValueAMD64_OpAMD64SARLconst_0(v)
	case OpAMD64SARQ:
		return rewriteValueAMD64_OpAMD64SARQ_0(v)
	case OpAMD64SARQconst:
		return rewriteValueAMD64_OpAMD64SARQconst_0(v)
	case OpAMD64SARW:
		return rewriteValueAMD64_OpAMD64SARW_0(v)
	case OpAMD64SARWconst:
		return rewriteValueAMD64_OpAMD64SARWconst_0(v)
	case OpAMD64SBBLcarrymask:
		return rewriteValueAMD64_OpAMD64SBBLcarrymask_0(v)
	case OpAMD64SBBQcarrymask:
		return rewriteValueAMD64_OpAMD64SBBQcarrymask_0(v)
	case OpAMD64SETA:
		return rewriteValueAMD64_OpAMD64SETA_0(v)
	case OpAMD64SETAE:
		return rewriteValueAMD64_OpAMD64SETAE_0(v)
	case OpAMD64SETB:
		return rewriteValueAMD64_OpAMD64SETB_0(v)
	case OpAMD64SETBE:
		return rewriteValueAMD64_OpAMD64SETBE_0(v)
	case OpAMD64SETEQ:
		return rewriteValueAMD64_OpAMD64SETEQ_0(v) || rewriteValueAMD64_OpAMD64SETEQ_10(v)
	case OpAMD64SETG:
		return rewriteValueAMD64_OpAMD64SETG_0(v)
	case OpAMD64SETGE:
		return rewriteValueAMD64_OpAMD64SETGE_0(v)
	case OpAMD64SETL:
		return rewriteValueAMD64_OpAMD64SETL_0(v)
	case OpAMD64SETLE:
		return rewriteValueAMD64_OpAMD64SETLE_0(v)
	case OpAMD64SETNE:
		return rewriteValueAMD64_OpAMD64SETNE_0(v) || rewriteValueAMD64_OpAMD64SETNE_10(v)
	case OpAMD64SHLL:
		return rewriteValueAMD64_OpAMD64SHLL_0(v)
	case OpAMD64SHLLconst:
		return rewriteValueAMD64_OpAMD64SHLLconst_0(v)
	case OpAMD64SHLQ:
		return rewriteValueAMD64_OpAMD64SHLQ_0(v)
	case OpAMD64SHLQconst:
		return rewriteValueAMD64_OpAMD64SHLQconst_0(v)
	case OpAMD64SHRB:
		return rewriteValueAMD64_OpAMD64SHRB_0(v)
	case OpAMD64SHRBconst:
		return rewriteValueAMD64_OpAMD64SHRBconst_0(v)
	case OpAMD64SHRL:
		return rewriteValueAMD64_OpAMD64SHRL_0(v)
	case OpAMD64SHRLconst:
		return rewriteValueAMD64_OpAMD64SHRLconst_0(v)
	case OpAMD64SHRQ:
		return rewriteValueAMD64_OpAMD64SHRQ_0(v)
	case OpAMD64SHRQconst:
		return rewriteValueAMD64_OpAMD64SHRQconst_0(v)
	case OpAMD64SHRW:
		return rewriteValueAMD64_OpAMD64SHRW_0(v)
	case OpAMD64SHRWconst:
		return rewriteValueAMD64_OpAMD64SHRWconst_0(v)
	case OpAMD64SUBL:
		return rewriteValueAMD64_OpAMD64SUBL_0(v)
	case OpAMD64SUBLconst:
		return rewriteValueAMD64_OpAMD64SUBLconst_0(v)
	case OpAMD64SUBQ:
		return rewriteValueAMD64_OpAMD64SUBQ_0(v)
	case OpAMD64SUBQconst:
		return rewriteValueAMD64_OpAMD64SUBQconst_0(v)
	case OpAMD64SUBSD:
		return rewriteValueAMD64_OpAMD64SUBSD_0(v)
	case OpAMD64SUBSS:
		return rewriteValueAMD64_OpAMD64SUBSS_0(v)
	case OpAMD64TESTB:
		return rewriteValueAMD64_OpAMD64TESTB_0(v)
	case OpAMD64TESTL:
		return rewriteValueAMD64_OpAMD64TESTL_0(v)
	case OpAMD64TESTQ:
		return rewriteValueAMD64_OpAMD64TESTQ_0(v)
	case OpAMD64TESTW:
		return rewriteValueAMD64_OpAMD64TESTW_0(v)
	case OpAMD64XADDLlock:
		return rewriteValueAMD64_OpAMD64XADDLlock_0(v)
	case OpAMD64XADDQlock:
		return rewriteValueAMD64_OpAMD64XADDQlock_0(v)
	case OpAMD64XCHGL:
		return rewriteValueAMD64_OpAMD64XCHGL_0(v)
	case OpAMD64XCHGQ:
		return rewriteValueAMD64_OpAMD64XCHGQ_0(v)
	case OpAMD64XORL:
		return rewriteValueAMD64_OpAMD64XORL_0(v) || rewriteValueAMD64_OpAMD64XORL_10(v)
	case OpAMD64XORLconst:
		return rewriteValueAMD64_OpAMD64XORLconst_0(v) || rewriteValueAMD64_OpAMD64XORLconst_10(v)
	case OpAMD64XORQ:
		return rewriteValueAMD64_OpAMD64XORQ_0(v)
	case OpAMD64XORQconst:
		return rewriteValueAMD64_OpAMD64XORQconst_0(v)
	case OpAdd16:
		return rewriteValueAMD64_OpAdd16_0(v)
	case OpAdd32:
		return rewriteValueAMD64_OpAdd32_0(v)
	case OpAdd32F:
		return rewriteValueAMD64_OpAdd32F_0(v)
	case OpAdd64:
		return rewriteValueAMD64_OpAdd64_0(v)
	case OpAdd64F:
		return rewriteValueAMD64_OpAdd64F_0(v)
	case OpAdd8:
		return rewriteValueAMD64_OpAdd8_0(v)
	case OpAddPtr:
		return rewriteValueAMD64_OpAddPtr_0(v)
	case OpAddr:
		return rewriteValueAMD64_OpAddr_0(v)
	case OpAnd16:
		return rewriteValueAMD64_OpAnd16_0(v)
	case OpAnd32:
		return rewriteValueAMD64_OpAnd32_0(v)
	case OpAnd64:
		return rewriteValueAMD64_OpAnd64_0(v)
	case OpAnd8:
		return rewriteValueAMD64_OpAnd8_0(v)
	case OpAndB:
		return rewriteValueAMD64_OpAndB_0(v)
	case OpAtomicAdd32:
		return rewriteValueAMD64_OpAtomicAdd32_0(v)
	case OpAtomicAdd64:
		return rewriteValueAMD64_OpAtomicAdd64_0(v)
	case OpAtomicAnd8:
		return rewriteValueAMD64_OpAtomicAnd8_0(v)
	case OpAtomicCompareAndSwap32:
		return rewriteValueAMD64_OpAtomicCompareAndSwap32_0(v)
	case OpAtomicCompareAndSwap64:
		return rewriteValueAMD64_OpAtomicCompareAndSwap64_0(v)
	case OpAtomicExchange32:
		return rewriteValueAMD64_OpAtomicExchange32_0(v)
	case OpAtomicExchange64:
		return rewriteValueAMD64_OpAtomicExchange64_0(v)
	case OpAtomicLoad32:
		return rewriteValueAMD64_OpAtomicLoad32_0(v)
	case OpAtomicLoad64:
		return rewriteValueAMD64_OpAtomicLoad64_0(v)
	case OpAtomicLoadPtr:
		return rewriteValueAMD64_OpAtomicLoadPtr_0(v)
	case OpAtomicOr8:
		return rewriteValueAMD64_OpAtomicOr8_0(v)
	case OpAtomicStore32:
		return rewriteValueAMD64_OpAtomicStore32_0(v)
	case OpAtomicStore64:
		return rewriteValueAMD64_OpAtomicStore64_0(v)
	case OpAtomicStorePtrNoWB:
		return rewriteValueAMD64_OpAtomicStorePtrNoWB_0(v)
	case OpAvg64u:
		return rewriteValueAMD64_OpAvg64u_0(v)
	case OpBitLen32:
		return rewriteValueAMD64_OpBitLen32_0(v)
	case OpBitLen64:
		return rewriteValueAMD64_OpBitLen64_0(v)
	case OpBswap32:
		return rewriteValueAMD64_OpBswap32_0(v)
	case OpBswap64:
		return rewriteValueAMD64_OpBswap64_0(v)
	case OpClosureCall:
		return rewriteValueAMD64_OpClosureCall_0(v)
	case OpCom16:
		return rewriteValueAMD64_OpCom16_0(v)
	case OpCom32:
		return rewriteValueAMD64_OpCom32_0(v)
	case OpCom64:
		return rewriteValueAMD64_OpCom64_0(v)
	case OpCom8:
		return rewriteValueAMD64_OpCom8_0(v)
	case OpConst16:
		return rewriteValueAMD64_OpConst16_0(v)
	case OpConst32:
		return rewriteValueAMD64_OpConst32_0(v)
	case OpConst32F:
		return rewriteValueAMD64_OpConst32F_0(v)
	case OpConst64:
		return rewriteValueAMD64_OpConst64_0(v)
	case OpConst64F:
		return rewriteValueAMD64_OpConst64F_0(v)
	case OpConst8:
		return rewriteValueAMD64_OpConst8_0(v)
	case OpConstBool:
		return rewriteValueAMD64_OpConstBool_0(v)
	case OpConstNil:
		return rewriteValueAMD64_OpConstNil_0(v)
	case OpConvert:
		return rewriteValueAMD64_OpConvert_0(v)
	case OpCtz32:
		return rewriteValueAMD64_OpCtz32_0(v)
	case OpCtz64:
		return rewriteValueAMD64_OpCtz64_0(v)
	case OpCvt32Fto32:
		return rewriteValueAMD64_OpCvt32Fto32_0(v)
	case OpCvt32Fto64:
		return rewriteValueAMD64_OpCvt32Fto64_0(v)
	case OpCvt32Fto64F:
		return rewriteValueAMD64_OpCvt32Fto64F_0(v)
	case OpCvt32to32F:
		return rewriteValueAMD64_OpCvt32to32F_0(v)
	case OpCvt32to64F:
		return rewriteValueAMD64_OpCvt32to64F_0(v)
	case OpCvt64Fto32:
		return rewriteValueAMD64_OpCvt64Fto32_0(v)
	case OpCvt64Fto32F:
		return rewriteValueAMD64_OpCvt64Fto32F_0(v)
	case OpCvt64Fto64:
		return rewriteValueAMD64_OpCvt64Fto64_0(v)
	case OpCvt64to32F:
		return rewriteValueAMD64_OpCvt64to32F_0(v)
	case OpCvt64to64F:
		return rewriteValueAMD64_OpCvt64to64F_0(v)
	case OpDiv128u:
		return rewriteValueAMD64_OpDiv128u_0(v)
	case OpDiv16:
		return rewriteValueAMD64_OpDiv16_0(v)
	case OpDiv16u:
		return rewriteValueAMD64_OpDiv16u_0(v)
	case OpDiv32:
		return rewriteValueAMD64_OpDiv32_0(v)
	case OpDiv32F:
		return rewriteValueAMD64_OpDiv32F_0(v)
	case OpDiv32u:
		return rewriteValueAMD64_OpDiv32u_0(v)
	case OpDiv64:
		return rewriteValueAMD64_OpDiv64_0(v)
	case OpDiv64F:
		return rewriteValueAMD64_OpDiv64F_0(v)
	case OpDiv64u:
		return rewriteValueAMD64_OpDiv64u_0(v)
	case OpDiv8:
		return rewriteValueAMD64_OpDiv8_0(v)
	case OpDiv8u:
		return rewriteValueAMD64_OpDiv8u_0(v)
	case OpEq16:
		return rewriteValueAMD64_OpEq16_0(v)
	case OpEq32:
		return rewriteValueAMD64_OpEq32_0(v)
	case OpEq32F:
		return rewriteValueAMD64_OpEq32F_0(v)
	case OpEq64:
		return rewriteValueAMD64_OpEq64_0(v)
	case OpEq64F:
		return rewriteValueAMD64_OpEq64F_0(v)
	case OpEq8:
		return rewriteValueAMD64_OpEq8_0(v)
	case OpEqB:
		return rewriteValueAMD64_OpEqB_0(v)
	case OpEqPtr:
		return rewriteValueAMD64_OpEqPtr_0(v)
	case OpGeq16:
		return rewriteValueAMD64_OpGeq16_0(v)
	case OpGeq16U:
		return rewriteValueAMD64_OpGeq16U_0(v)
	case OpGeq32:
		return rewriteValueAMD64_OpGeq32_0(v)
	case OpGeq32F:
		return rewriteValueAMD64_OpGeq32F_0(v)
	case OpGeq32U:
		return rewriteValueAMD64_OpGeq32U_0(v)
	case OpGeq64:
		return rewriteValueAMD64_OpGeq64_0(v)
	case OpGeq64F:
		return rewriteValueAMD64_OpGeq64F_0(v)
	case OpGeq64U:
		return rewriteValueAMD64_OpGeq64U_0(v)
	case OpGeq8:
		return rewriteValueAMD64_OpGeq8_0(v)
	case OpGeq8U:
		return rewriteValueAMD64_OpGeq8U_0(v)
	case OpGetClosurePtr:
		return rewriteValueAMD64_OpGetClosurePtr_0(v)
	case OpGetG:
		return rewriteValueAMD64_OpGetG_0(v)
	case OpGreater16:
		return rewriteValueAMD64_OpGreater16_0(v)
	case OpGreater16U:
		return rewriteValueAMD64_OpGreater16U_0(v)
	case OpGreater32:
		return rewriteValueAMD64_OpGreater32_0(v)
	case OpGreater32F:
		return rewriteValueAMD64_OpGreater32F_0(v)
	case OpGreater32U:
		return rewriteValueAMD64_OpGreater32U_0(v)
	case OpGreater64:
		return rewriteValueAMD64_OpGreater64_0(v)
	case OpGreater64F:
		return rewriteValueAMD64_OpGreater64F_0(v)
	case OpGreater64U:
		return rewriteValueAMD64_OpGreater64U_0(v)
	case OpGreater8:
		return rewriteValueAMD64_OpGreater8_0(v)
	case OpGreater8U:
		return rewriteValueAMD64_OpGreater8U_0(v)
	case OpHmul32:
		return rewriteValueAMD64_OpHmul32_0(v)
	case OpHmul32u:
		return rewriteValueAMD64_OpHmul32u_0(v)
	case OpHmul64:
		return rewriteValueAMD64_OpHmul64_0(v)
	case OpHmul64u:
		return rewriteValueAMD64_OpHmul64u_0(v)
	case OpInt64Hi:
		return rewriteValueAMD64_OpInt64Hi_0(v)
	case OpInterCall:
		return rewriteValueAMD64_OpInterCall_0(v)
	case OpIsInBounds:
		return rewriteValueAMD64_OpIsInBounds_0(v)
	case OpIsNonNil:
		return rewriteValueAMD64_OpIsNonNil_0(v)
	case OpIsSliceInBounds:
		return rewriteValueAMD64_OpIsSliceInBounds_0(v)
	case OpLeq16:
		return rewriteValueAMD64_OpLeq16_0(v)
	case OpLeq16U:
		return rewriteValueAMD64_OpLeq16U_0(v)
	case OpLeq32:
		return rewriteValueAMD64_OpLeq32_0(v)
	case OpLeq32F:
		return rewriteValueAMD64_OpLeq32F_0(v)
	case OpLeq32U:
		return rewriteValueAMD64_OpLeq32U_0(v)
	case OpLeq64:
		return rewriteValueAMD64_OpLeq64_0(v)
	case OpLeq64F:
		return rewriteValueAMD64_OpLeq64F_0(v)
	case OpLeq64U:
		return rewriteValueAMD64_OpLeq64U_0(v)
	case OpLeq8:
		return rewriteValueAMD64_OpLeq8_0(v)
	case OpLeq8U:
		return rewriteValueAMD64_OpLeq8U_0(v)
	case OpLess16:
		return rewriteValueAMD64_OpLess16_0(v)
	case OpLess16U:
		return rewriteValueAMD64_OpLess16U_0(v)
	case OpLess32:
		return rewriteValueAMD64_OpLess32_0(v)
	case OpLess32F:
		return rewriteValueAMD64_OpLess32F_0(v)
	case OpLess32U:
		return rewriteValueAMD64_OpLess32U_0(v)
	case OpLess64:
		return rewriteValueAMD64_OpLess64_0(v)
	case OpLess64F:
		return rewriteValueAMD64_OpLess64F_0(v)
	case OpLess64U:
		return rewriteValueAMD64_OpLess64U_0(v)
	case OpLess8:
		return rewriteValueAMD64_OpLess8_0(v)
	case OpLess8U:
		return rewriteValueAMD64_OpLess8U_0(v)
	case OpLoad:
		return rewriteValueAMD64_OpLoad_0(v)
	case OpLsh16x16:
		return rewriteValueAMD64_OpLsh16x16_0(v)
	case OpLsh16x32:
		return rewriteValueAMD64_OpLsh16x32_0(v)
	case OpLsh16x64:
		return rewriteValueAMD64_OpLsh16x64_0(v)
	case OpLsh16x8:
		return rewriteValueAMD64_OpLsh16x8_0(v)
	case OpLsh32x16:
		return rewriteValueAMD64_OpLsh32x16_0(v)
	case OpLsh32x32:
		return rewriteValueAMD64_OpLsh32x32_0(v)
	case OpLsh32x64:
		return rewriteValueAMD64_OpLsh32x64_0(v)
	case OpLsh32x8:
		return rewriteValueAMD64_OpLsh32x8_0(v)
	case OpLsh64x16:
		return rewriteValueAMD64_OpLsh64x16_0(v)
	case OpLsh64x32:
		return rewriteValueAMD64_OpLsh64x32_0(v)
	case OpLsh64x64:
		return rewriteValueAMD64_OpLsh64x64_0(v)
	case OpLsh64x8:
		return rewriteValueAMD64_OpLsh64x8_0(v)
	case OpLsh8x16:
		return rewriteValueAMD64_OpLsh8x16_0(v)
	case OpLsh8x32:
		return rewriteValueAMD64_OpLsh8x32_0(v)
	case OpLsh8x64:
		return rewriteValueAMD64_OpLsh8x64_0(v)
	case OpLsh8x8:
		return rewriteValueAMD64_OpLsh8x8_0(v)
	case OpMod16:
		return rewriteValueAMD64_OpMod16_0(v)
	case OpMod16u:
		return rewriteValueAMD64_OpMod16u_0(v)
	case OpMod32:
		return rewriteValueAMD64_OpMod32_0(v)
	case OpMod32u:
		return rewriteValueAMD64_OpMod32u_0(v)
	case OpMod64:
		return rewriteValueAMD64_OpMod64_0(v)
	case OpMod64u:
		return rewriteValueAMD64_OpMod64u_0(v)
	case OpMod8:
		return rewriteValueAMD64_OpMod8_0(v)
	case OpMod8u:
		return rewriteValueAMD64_OpMod8u_0(v)
	case OpMove:
		return rewriteValueAMD64_OpMove_0(v) || rewriteValueAMD64_OpMove_10(v)
	case OpMul16:
		return rewriteValueAMD64_OpMul16_0(v)
	case OpMul32:
		return rewriteValueAMD64_OpMul32_0(v)
	case OpMul32F:
		return rewriteValueAMD64_OpMul32F_0(v)
	case OpMul64:
		return rewriteValueAMD64_OpMul64_0(v)
	case OpMul64F:
		return rewriteValueAMD64_OpMul64F_0(v)
	case OpMul64uhilo:
		return rewriteValueAMD64_OpMul64uhilo_0(v)
	case OpMul8:
		return rewriteValueAMD64_OpMul8_0(v)
	case OpNeg16:
		return rewriteValueAMD64_OpNeg16_0(v)
	case OpNeg32:
		return rewriteValueAMD64_OpNeg32_0(v)
	case OpNeg32F:
		return rewriteValueAMD64_OpNeg32F_0(v)
	case OpNeg64:
		return rewriteValueAMD64_OpNeg64_0(v)
	case OpNeg64F:
		return rewriteValueAMD64_OpNeg64F_0(v)
	case OpNeg8:
		return rewriteValueAMD64_OpNeg8_0(v)
	case OpNeq16:
		return rewriteValueAMD64_OpNeq16_0(v)
	case OpNeq32:
		return rewriteValueAMD64_OpNeq32_0(v)
	case OpNeq32F:
		return rewriteValueAMD64_OpNeq32F_0(v)
	case OpNeq64:
		return rewriteValueAMD64_OpNeq64_0(v)
	case OpNeq64F:
		return rewriteValueAMD64_OpNeq64F_0(v)
	case OpNeq8:
		return rewriteValueAMD64_OpNeq8_0(v)
	case OpNeqB:
		return rewriteValueAMD64_OpNeqB_0(v)
	case OpNeqPtr:
		return rewriteValueAMD64_OpNeqPtr_0(v)
	case OpNilCheck:
		return rewriteValueAMD64_OpNilCheck_0(v)
	case OpNot:
		return rewriteValueAMD64_OpNot_0(v)
	case OpOffPtr:
		return rewriteValueAMD64_OpOffPtr_0(v)
	case OpOr16:
		return rewriteValueAMD64_OpOr16_0(v)
	case OpOr32:
		return rewriteValueAMD64_OpOr32_0(v)
	case OpOr64:
		return rewriteValueAMD64_OpOr64_0(v)
	case OpOr8:
		return rewriteValueAMD64_OpOr8_0(v)
	case OpOrB:
		return rewriteValueAMD64_OpOrB_0(v)
	case OpPopCount16:
		return rewriteValueAMD64_OpPopCount16_0(v)
	case OpPopCount32:
		return rewriteValueAMD64_OpPopCount32_0(v)
	case OpPopCount64:
		return rewriteValueAMD64_OpPopCount64_0(v)
	case OpPopCount8:
		return rewriteValueAMD64_OpPopCount8_0(v)
	case OpRound32F:
		return rewriteValueAMD64_OpRound32F_0(v)
	case OpRound64F:
		return rewriteValueAMD64_OpRound64F_0(v)
	case OpRsh16Ux16:
		return rewriteValueAMD64_OpRsh16Ux16_0(v)
	case OpRsh16Ux32:
		return rewriteValueAMD64_OpRsh16Ux32_0(v)
	case OpRsh16Ux64:
		return rewriteValueAMD64_OpRsh16Ux64_0(v)
	case OpRsh16Ux8:
		return rewriteValueAMD64_OpRsh16Ux8_0(v)
	case OpRsh16x16:
		return rewriteValueAMD64_OpRsh16x16_0(v)
	case OpRsh16x32:
		return rewriteValueAMD64_OpRsh16x32_0(v)
	case OpRsh16x64:
		return rewriteValueAMD64_OpRsh16x64_0(v)
	case OpRsh16x8:
		return rewriteValueAMD64_OpRsh16x8_0(v)
	case OpRsh32Ux16:
		return rewriteValueAMD64_OpRsh32Ux16_0(v)
	case OpRsh32Ux32:
		return rewriteValueAMD64_OpRsh32Ux32_0(v)
	case OpRsh32Ux64:
		return rewriteValueAMD64_OpRsh32Ux64_0(v)
	case OpRsh32Ux8:
		return rewriteValueAMD64_OpRsh32Ux8_0(v)
	case OpRsh32x16:
		return rewriteValueAMD64_OpRsh32x16_0(v)
	case OpRsh32x32:
		return rewriteValueAMD64_OpRsh32x32_0(v)
	case OpRsh32x64:
		return rewriteValueAMD64_OpRsh32x64_0(v)
	case OpRsh32x8:
		return rewriteValueAMD64_OpRsh32x8_0(v)
	case OpRsh64Ux16:
		return rewriteValueAMD64_OpRsh64Ux16_0(v)
	case OpRsh64Ux32:
		return rewriteValueAMD64_OpRsh64Ux32_0(v)
	case OpRsh64Ux64:
		return rewriteValueAMD64_OpRsh64Ux64_0(v)
	case OpRsh64Ux8:
		return rewriteValueAMD64_OpRsh64Ux8_0(v)
	case OpRsh64x16:
		return rewriteValueAMD64_OpRsh64x16_0(v)
	case OpRsh64x32:
		return rewriteValueAMD64_OpRsh64x32_0(v)
	case OpRsh64x64:
		return rewriteValueAMD64_OpRsh64x64_0(v)
	case OpRsh64x8:
		return rewriteValueAMD64_OpRsh64x8_0(v)
	case OpRsh8Ux16:
		return rewriteValueAMD64_OpRsh8Ux16_0(v)
	case OpRsh8Ux32:
		return rewriteValueAMD64_OpRsh8Ux32_0(v)
	case OpRsh8Ux64:
		return rewriteValueAMD64_OpRsh8Ux64_0(v)
	case OpRsh8Ux8:
		return rewriteValueAMD64_OpRsh8Ux8_0(v)
	case OpRsh8x16:
		return rewriteValueAMD64_OpRsh8x16_0(v)
	case OpRsh8x32:
		return rewriteValueAMD64_OpRsh8x32_0(v)
	case OpRsh8x64:
		return rewriteValueAMD64_OpRsh8x64_0(v)
	case OpRsh8x8:
		return rewriteValueAMD64_OpRsh8x8_0(v)
	case OpSelect0:
		return rewriteValueAMD64_OpSelect0_0(v)
	case OpSelect1:
		return rewriteValueAMD64_OpSelect1_0(v)
	case OpSignExt16to32:
		return rewriteValueAMD64_OpSignExt16to32_0(v)
	case OpSignExt16to64:
		return rewriteValueAMD64_OpSignExt16to64_0(v)
	case OpSignExt32to64:
		return rewriteValueAMD64_OpSignExt32to64_0(v)
	case OpSignExt8to16:
		return rewriteValueAMD64_OpSignExt8to16_0(v)
	case OpSignExt8to32:
		return rewriteValueAMD64_OpSignExt8to32_0(v)
	case OpSignExt8to64:
		return rewriteValueAMD64_OpSignExt8to64_0(v)
	case OpSlicemask:
		return rewriteValueAMD64_OpSlicemask_0(v)
	case OpSqrt:
		return rewriteValueAMD64_OpSqrt_0(v)
	case OpStaticCall:
		return rewriteValueAMD64_OpStaticCall_0(v)
	case OpStore:
		return rewriteValueAMD64_OpStore_0(v)
	case OpSub16:
		return rewriteValueAMD64_OpSub16_0(v)
	case OpSub32:
		return rewriteValueAMD64_OpSub32_0(v)
	case OpSub32F:
		return rewriteValueAMD64_OpSub32F_0(v)
	case OpSub64:
		return rewriteValueAMD64_OpSub64_0(v)
	case OpSub64F:
		return rewriteValueAMD64_OpSub64F_0(v)
	case OpSub8:
		return rewriteValueAMD64_OpSub8_0(v)
	case OpSubPtr:
		return rewriteValueAMD64_OpSubPtr_0(v)
	case OpTrunc16to8:
		return rewriteValueAMD64_OpTrunc16to8_0(v)
	case OpTrunc32to16:
		return rewriteValueAMD64_OpTrunc32to16_0(v)
	case OpTrunc32to8:
		return rewriteValueAMD64_OpTrunc32to8_0(v)
	case OpTrunc64to16:
		return rewriteValueAMD64_OpTrunc64to16_0(v)
	case OpTrunc64to32:
		return rewriteValueAMD64_OpTrunc64to32_0(v)
	case OpTrunc64to8:
		return rewriteValueAMD64_OpTrunc64to8_0(v)
	case OpXor16:
		return rewriteValueAMD64_OpXor16_0(v)
	case OpXor32:
		return rewriteValueAMD64_OpXor32_0(v)
	case OpXor64:
		return rewriteValueAMD64_OpXor64_0(v)
	case OpXor8:
		return rewriteValueAMD64_OpXor8_0(v)
	case OpZero:
		return rewriteValueAMD64_OpZero_0(v) || rewriteValueAMD64_OpZero_10(v)
	case OpZeroExt16to32:
		return rewriteValueAMD64_OpZeroExt16to32_0(v)
	case OpZeroExt16to64:
		return rewriteValueAMD64_OpZeroExt16to64_0(v)
	case OpZeroExt32to64:
		return rewriteValueAMD64_OpZeroExt32to64_0(v)
	case OpZeroExt8to16:
		return rewriteValueAMD64_OpZeroExt8to16_0(v)
	case OpZeroExt8to32:
		return rewriteValueAMD64_OpZeroExt8to32_0(v)
	case OpZeroExt8to64:
		return rewriteValueAMD64_OpZeroExt8to64_0(v)
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDL_0(v *Value) bool {
	// match: (ADDL x (MOVLconst [c]))
	// cond:
	// result: (ADDLconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDL (MOVLconst [c]) x)
	// cond:
	// result: (ADDLconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDL (SHLLconst x [c]) (SHRLconst x [d]))
	// cond: d==32-c
	// result: (ROLLconst x [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRLconst {
			break
		}
		d := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 32-c) {
			break
		}
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDL (SHRLconst x [d]) (SHLLconst x [c]))
	// cond: d==32-c
	// result: (ROLLconst x [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRLconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 32-c) {
			break
		}
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDL <t> (SHLLconst x [c]) (SHRWconst x [d]))
	// cond: d==16-c && c < 16 && t.Size() == 2
	// result: (ROLWconst x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRWconst {
			break
		}
		d := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 16-c && c < 16 && t.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDL <t> (SHRWconst x [d]) (SHLLconst x [c]))
	// cond: d==16-c && c < 16 && t.Size() == 2
	// result: (ROLWconst x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRWconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 16-c && c < 16 && t.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDL <t> (SHLLconst x [c]) (SHRBconst x [d]))
	// cond: d==8-c  && c < 8 && t.Size() == 1
	// result: (ROLBconst x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRBconst {
			break
		}
		d := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 8-c && c < 8 && t.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDL <t> (SHRBconst x [d]) (SHLLconst x [c]))
	// cond: d==8-c  && c < 8 && t.Size() == 1
	// result: (ROLBconst x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 8-c && c < 8 && t.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDL x (NEGL y))
	// cond:
	// result: (SUBL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64SUBL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDL (NEGL y) x)
	// cond:
	// result: (SUBL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64NEGL {
			break
		}
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpAMD64SUBL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDL_10(v *Value) bool {
	// match: (ADDL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ADDLmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ADDLmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (ADDL l:(MOVLload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ADDLmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		l := v.Args[0]
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		x := v.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ADDLmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDLconst_0(v *Value) bool {
	// match: (ADDLconst [c] x)
	// cond: int32(c)==0
	// result: x
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ADDLconst [c] (MOVLconst [d]))
	// cond:
	// result: (MOVLconst [int64(int32(c+d))])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int64(int32(c + d))
		return true
	}
	// match: (ADDLconst [c] (ADDLconst [d] x))
	// cond:
	// result: (ADDLconst [int64(int32(c+d))] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = int64(int32(c + d))
		v.AddArg(x)
		return true
	}
	// match: (ADDLconst [c] (LEAL [d] {s} x))
	// cond: is32Bit(c+d)
	// result: (LEAL [c+d] {s} x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		d := v_0.AuxInt
		s := v_0.Aux
		x := v_0.Args[0]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpAMD64LEAL)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDQ_0(v *Value) bool {
	// match: (ADDQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (ADDQconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDQ (MOVQconst [c]) x)
	// cond: is32Bit(c)
	// result: (ADDQconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDQ (SHLQconst x [c]) (SHRQconst x [d]))
	// cond: d==64-c
	// result: (ROLQconst x [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		d := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 64-c) {
			break
		}
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDQ (SHRQconst x [d]) (SHLQconst x [c]))
	// cond: d==64-c
	// result: (ROLQconst x [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 64-c) {
			break
		}
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDQ x (SHLQconst [3] y))
	// cond:
	// result: (LEAQ8 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 3 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64LEAQ8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQ (SHLQconst [3] y) x)
	// cond:
	// result: (LEAQ8 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		if v_0.AuxInt != 3 {
			break
		}
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpAMD64LEAQ8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQ x (SHLQconst [2] y))
	// cond:
	// result: (LEAQ4 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64LEAQ4)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQ (SHLQconst [2] y) x)
	// cond:
	// result: (LEAQ4 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpAMD64LEAQ4)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQ x (SHLQconst [1] y))
	// cond:
	// result: (LEAQ2 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64LEAQ2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQ (SHLQconst [1] y) x)
	// cond:
	// result: (LEAQ2 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpAMD64LEAQ2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDQ_10(v *Value) bool {
	// match: (ADDQ x (ADDQ y y))
	// cond:
	// result: (LEAQ2 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQ {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQ (ADDQ y y) x)
	// cond:
	// result: (LEAQ2 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		if y != v_0.Args[1] {
			break
		}
		x := v.Args[1]
		v.reset(OpAMD64LEAQ2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQ x (ADDQ x y))
	// cond:
	// result: (LEAQ2 y x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQ {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpAMD64LEAQ2)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	// match: (ADDQ x (ADDQ y x))
	// cond:
	// result: (LEAQ2 y x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQ {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	// match: (ADDQ (ADDQ x y) x)
	// cond:
	// result: (LEAQ2 y x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	// match: (ADDQ (ADDQ y x) x)
	// cond:
	// result: (LEAQ2 y x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	// match: (ADDQ (ADDQconst [c] x) y)
	// cond:
	// result: (LEAQ1 [c] x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQ y (ADDQconst [c] x))
	// cond:
	// result: (LEAQ1 [c] x y)
	for {
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		x := v_1.Args[0]
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQ x (LEAQ [c] {s} y))
	// cond: x.Op != OpSB && y.Op != OpSB
	// result: (LEAQ1 [c] {s} x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		c := v_1.AuxInt
		s := v_1.Aux
		y := v_1.Args[0]
		if !(x.Op != OpSB && y.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = c
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQ (LEAQ [c] {s} y) x)
	// cond: x.Op != OpSB && y.Op != OpSB
	// result: (LEAQ1 [c] {s} x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		c := v_0.AuxInt
		s := v_0.Aux
		y := v_0.Args[0]
		x := v.Args[1]
		if !(x.Op != OpSB && y.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = c
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDQ_20(v *Value) bool {
	// match: (ADDQ x (NEGQ y))
	// cond:
	// result: (SUBQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64SUBQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQ (NEGQ y) x)
	// cond:
	// result: (SUBQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64NEGQ {
			break
		}
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpAMD64SUBQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQ x l:(MOVQload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ADDQmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ADDQmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (ADDQ l:(MOVQload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ADDQmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		l := v.Args[0]
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		x := v.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ADDQmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDQconst_0(v *Value) bool {
	// match: (ADDQconst [c] (ADDQ x y))
	// cond:
	// result: (LEAQ1 [c] x y)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQconst [c] (LEAQ [d] {s} x))
	// cond: is32Bit(c+d)
	// result: (LEAQ [c+d] {s} x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		d := v_0.AuxInt
		s := v_0.Aux
		x := v_0.Args[0]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		return true
	}
	// match: (ADDQconst [c] (LEAQ1 [d] {s} x y))
	// cond: is32Bit(c+d)
	// result: (LEAQ1 [c+d] {s} x y)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		d := v_0.AuxInt
		s := v_0.Aux
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQconst [c] (LEAQ2 [d] {s} x y))
	// cond: is32Bit(c+d)
	// result: (LEAQ2 [c+d] {s} x y)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ2 {
			break
		}
		d := v_0.AuxInt
		s := v_0.Aux
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQconst [c] (LEAQ4 [d] {s} x y))
	// cond: is32Bit(c+d)
	// result: (LEAQ4 [c+d] {s} x y)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ4 {
			break
		}
		d := v_0.AuxInt
		s := v_0.Aux
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQconst [c] (LEAQ8 [d] {s} x y))
	// cond: is32Bit(c+d)
	// result: (LEAQ8 [c+d] {s} x y)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ8 {
			break
		}
		d := v_0.AuxInt
		s := v_0.Aux
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADDQconst [0] x)
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
	// match: (ADDQconst [c] (MOVQconst [d]))
	// cond:
	// result: (MOVQconst [c+d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = c + d
		return true
	}
	// match: (ADDQconst [c] (ADDQconst [d] x))
	// cond: is32Bit(c+d)
	// result: (ADDQconst [c+d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = c + d
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDSD_0(v *Value) bool {
	// match: (ADDSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ADDSDmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVSDload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ADDSDmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (ADDSD l:(MOVSDload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ADDSDmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		l := v.Args[0]
		if l.Op != OpAMD64MOVSDload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		x := v.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ADDSDmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ADDSS_0(v *Value) bool {
	// match: (ADDSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ADDSSmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVSSload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ADDSSmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (ADDSS l:(MOVSSload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ADDSSmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		l := v.Args[0]
		if l.Op != OpAMD64MOVSSload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		x := v.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ADDSSmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDL_0(v *Value) bool {
	// match: (ANDL x (MOVLconst [c]))
	// cond:
	// result: (ANDLconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ANDL (MOVLconst [c]) x)
	// cond:
	// result: (ANDLconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ANDL x x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ANDL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ANDLmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ANDLmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (ANDL l:(MOVLload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ANDLmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		l := v.Args[0]
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		x := v.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ANDLmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDLconst_0(v *Value) bool {
	// match: (ANDLconst [c] (ANDLconst [d] x))
	// cond:
	// result: (ANDLconst [c & d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = c & d
		v.AddArg(x)
		return true
	}
	// match: (ANDLconst [0xFF] x)
	// cond:
	// result: (MOVBQZX x)
	for {
		if v.AuxInt != 0xFF {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64MOVBQZX)
		v.AddArg(x)
		return true
	}
	// match: (ANDLconst [0xFFFF] x)
	// cond:
	// result: (MOVWQZX x)
	for {
		if v.AuxInt != 0xFFFF {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64MOVWQZX)
		v.AddArg(x)
		return true
	}
	// match: (ANDLconst [c] _)
	// cond: int32(c)==0
	// result: (MOVLconst [0])
	for {
		c := v.AuxInt
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (ANDLconst [c] x)
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
	// match: (ANDLconst [c] (MOVLconst [d]))
	// cond:
	// result: (MOVLconst [c&d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = c & d
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDQ_0(v *Value) bool {
	// match: (ANDQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (ANDQconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64ANDQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ANDQ (MOVQconst [c]) x)
	// cond: is32Bit(c)
	// result: (ANDQconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64ANDQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ANDQ x x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ANDQ x l:(MOVQload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ANDQmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ANDQmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (ANDQ l:(MOVQload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ANDQmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		l := v.Args[0]
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		x := v.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ANDQmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ANDQconst_0(v *Value) bool {
	// match: (ANDQconst [c] (ANDQconst [d] x))
	// cond:
	// result: (ANDQconst [c & d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64ANDQconst)
		v.AuxInt = c & d
		v.AddArg(x)
		return true
	}
	// match: (ANDQconst [0xFF] x)
	// cond:
	// result: (MOVBQZX x)
	for {
		if v.AuxInt != 0xFF {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64MOVBQZX)
		v.AddArg(x)
		return true
	}
	// match: (ANDQconst [0xFFFF] x)
	// cond:
	// result: (MOVWQZX x)
	for {
		if v.AuxInt != 0xFFFF {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64MOVWQZX)
		v.AddArg(x)
		return true
	}
	// match: (ANDQconst [0xFFFFFFFF] x)
	// cond:
	// result: (MOVLQZX x)
	for {
		if v.AuxInt != 0xFFFFFFFF {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64MOVLQZX)
		v.AddArg(x)
		return true
	}
	// match: (ANDQconst [0] _)
	// cond:
	// result: (MOVQconst [0])
	for {
		if v.AuxInt != 0 {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = 0
		return true
	}
	// match: (ANDQconst [-1] x)
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
	// match: (ANDQconst [c] (MOVQconst [d]))
	// cond:
	// result: (MOVQconst [c&d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = c & d
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BSFQ_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (BSFQ (ORQconst <t> [1<<8] (MOVBQZX x)))
	// cond:
	// result: (BSFQ (ORQconst <t> [1<<8] x))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ORQconst {
			break
		}
		t := v_0.Type
		if v_0.AuxInt != 1<<8 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64MOVBQZX {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpAMD64BSFQ)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQconst, t)
		v0.AuxInt = 1 << 8
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (BSFQ (ORQconst <t> [1<<16] (MOVWQZX x)))
	// cond:
	// result: (BSFQ (ORQconst <t> [1<<16] x))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ORQconst {
			break
		}
		t := v_0.Type
		if v_0.AuxInt != 1<<16 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64MOVWQZX {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpAMD64BSFQ)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQconst, t)
		v0.AuxInt = 1 << 16
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64BTQconst_0(v *Value) bool {
	// match: (BTQconst [c] x)
	// cond: c < 32
	// result: (BTLconst [c] x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(c < 32) {
			break
		}
		v.reset(OpAMD64BTLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMOVQEQ_0(v *Value) bool {
	// match: (CMOVQEQ x _ (Select1 (BSFQ (ORQconst [c] _))))
	// cond: c != 0
	// result: x
	for {
		_ = v.Args[2]
		x := v.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpSelect1 {
			break
		}
		v_2_0 := v_2.Args[0]
		if v_2_0.Op != OpAMD64BSFQ {
			break
		}
		v_2_0_0 := v_2_0.Args[0]
		if v_2_0_0.Op != OpAMD64ORQconst {
			break
		}
		c := v_2_0_0.AuxInt
		if !(c != 0) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPB_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (CMPB x (MOVLconst [c]))
	// cond:
	// result: (CMPBconst x [int64(int8(c))])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64CMPBconst)
		v.AuxInt = int64(int8(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPB (MOVLconst [c]) x)
	// cond:
	// result: (InvertFlags (CMPBconst x [int64(int8(c))]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v0.AuxInt = int64(int8(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPBconst_0(v *Value) bool {
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)==int8(y)
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int8(x) == int8(y)) {
			break
		}
		v.reset(OpAMD64FlagEQ)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)<int8(y) && uint8(x)<uint8(y)
	// result: (FlagLT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int8(x) < int8(y) && uint8(x) < uint8(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)<int8(y) && uint8(x)>uint8(y)
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int8(x) < int8(y) && uint8(x) > uint8(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_UGT)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)>int8(y) && uint8(x)<uint8(y)
	// result: (FlagGT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int8(x) > int8(y) && uint8(x) < uint8(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_ULT)
		return true
	}
	// match: (CMPBconst (MOVLconst [x]) [y])
	// cond: int8(x)>int8(y) && uint8(x)>uint8(y)
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int8(x) > int8(y) && uint8(x) > uint8(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_UGT)
		return true
	}
	// match: (CMPBconst (ANDLconst _ [m]) [n])
	// cond: 0 <= int8(m) && int8(m) < int8(n)
	// result: (FlagLT_ULT)
	for {
		n := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		m := v_0.AuxInt
		if !(0 <= int8(m) && int8(m) < int8(n)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPBconst (ANDL x y) [0])
	// cond:
	// result: (TESTB x y)
	for {
		if v.AuxInt != 0 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpAMD64TESTB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMPBconst (ANDLconst [c] x) [0])
	// cond:
	// result: (TESTBconst [int64(int8(c))] x)
	for {
		if v.AuxInt != 0 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64TESTBconst)
		v.AuxInt = int64(int8(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPBconst x [0])
	// cond:
	// result: (TESTB x x)
	for {
		if v.AuxInt != 0 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64TESTB)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPL_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (CMPL x (MOVLconst [c]))
	// cond:
	// result: (CMPLconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64CMPLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (CMPL (MOVLconst [c]) x)
	// cond:
	// result: (InvertFlags (CMPLconst x [c]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v0.AuxInt = c
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPLconst_0(v *Value) bool {
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: int32(x)==int32(y)
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) == int32(y)) {
			break
		}
		v.reset(OpAMD64FlagEQ)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: int32(x)<int32(y) && uint32(x)<uint32(y)
	// result: (FlagLT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) < int32(y) && uint32(x) < uint32(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: int32(x)<int32(y) && uint32(x)>uint32(y)
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) < int32(y) && uint32(x) > uint32(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_UGT)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: int32(x)>int32(y) && uint32(x)<uint32(y)
	// result: (FlagGT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) > int32(y) && uint32(x) < uint32(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_ULT)
		return true
	}
	// match: (CMPLconst (MOVLconst [x]) [y])
	// cond: int32(x)>int32(y) && uint32(x)>uint32(y)
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int32(x) > int32(y) && uint32(x) > uint32(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_UGT)
		return true
	}
	// match: (CMPLconst (SHRLconst _ [c]) [n])
	// cond: 0 <= n && 0 < c && c <= 32 && (1<<uint64(32-c)) <= uint64(n)
	// result: (FlagLT_ULT)
	for {
		n := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRLconst {
			break
		}
		c := v_0.AuxInt
		if !(0 <= n && 0 < c && c <= 32 && (1<<uint64(32-c)) <= uint64(n)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPLconst (ANDLconst _ [m]) [n])
	// cond: 0 <= int32(m) && int32(m) < int32(n)
	// result: (FlagLT_ULT)
	for {
		n := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		m := v_0.AuxInt
		if !(0 <= int32(m) && int32(m) < int32(n)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPLconst (ANDL x y) [0])
	// cond:
	// result: (TESTL x y)
	for {
		if v.AuxInt != 0 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpAMD64TESTL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMPLconst (ANDLconst [c] x) [0])
	// cond:
	// result: (TESTLconst [c] x)
	for {
		if v.AuxInt != 0 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64TESTLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (CMPLconst x [0])
	// cond:
	// result: (TESTL x x)
	for {
		if v.AuxInt != 0 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64TESTL)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPQ_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (CMPQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (CMPQconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64CMPQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (CMPQ (MOVQconst [c]) x)
	// cond: is32Bit(c)
	// result: (InvertFlags (CMPQconst x [c]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v0.AuxInt = c
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPQconst_0(v *Value) bool {
	// match: (CMPQconst (NEGQ (ADDQconst [-16] (ANDQconst [15] _))) [32])
	// cond:
	// result: (FlagLT_ULT)
	for {
		if v.AuxInt != 32 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_0.AuxInt != -16 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_0_0.AuxInt != 15 {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (NEGQ (ADDQconst [ -8] (ANDQconst [7] _))) [32])
	// cond:
	// result: (FlagLT_ULT)
	for {
		if v.AuxInt != 32 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_0.AuxInt != -8 {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_0_0.AuxInt != 7 {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (MOVQconst [x]) [y])
	// cond: x==y
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := v_0.AuxInt
		if !(x == y) {
			break
		}
		v.reset(OpAMD64FlagEQ)
		return true
	}
	// match: (CMPQconst (MOVQconst [x]) [y])
	// cond: x<y && uint64(x)<uint64(y)
	// result: (FlagLT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := v_0.AuxInt
		if !(x < y && uint64(x) < uint64(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (MOVQconst [x]) [y])
	// cond: x<y && uint64(x)>uint64(y)
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := v_0.AuxInt
		if !(x < y && uint64(x) > uint64(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_UGT)
		return true
	}
	// match: (CMPQconst (MOVQconst [x]) [y])
	// cond: x>y && uint64(x)<uint64(y)
	// result: (FlagGT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := v_0.AuxInt
		if !(x > y && uint64(x) < uint64(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_ULT)
		return true
	}
	// match: (CMPQconst (MOVQconst [x]) [y])
	// cond: x>y && uint64(x)>uint64(y)
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		x := v_0.AuxInt
		if !(x > y && uint64(x) > uint64(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_UGT)
		return true
	}
	// match: (CMPQconst (MOVBQZX _) [c])
	// cond: 0xFF < c
	// result: (FlagLT_ULT)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVBQZX {
			break
		}
		if !(0xFF < c) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (MOVWQZX _) [c])
	// cond: 0xFFFF < c
	// result: (FlagLT_ULT)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVWQZX {
			break
		}
		if !(0xFFFF < c) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (MOVLQZX _) [c])
	// cond: 0xFFFFFFFF < c
	// result: (FlagLT_ULT)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLQZX {
			break
		}
		if !(0xFFFFFFFF < c) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPQconst_10(v *Value) bool {
	// match: (CMPQconst (SHRQconst _ [c]) [n])
	// cond: 0 <= n && 0 < c && c <= 64 && (1<<uint64(64-c)) <= uint64(n)
	// result: (FlagLT_ULT)
	for {
		n := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRQconst {
			break
		}
		c := v_0.AuxInt
		if !(0 <= n && 0 < c && c <= 64 && (1<<uint64(64-c)) <= uint64(n)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (ANDQconst _ [m]) [n])
	// cond: 0 <= m && m < n
	// result: (FlagLT_ULT)
	for {
		n := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDQconst {
			break
		}
		m := v_0.AuxInt
		if !(0 <= m && m < n) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (ANDLconst _ [m]) [n])
	// cond: 0 <= m && m < n
	// result: (FlagLT_ULT)
	for {
		n := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		m := v_0.AuxInt
		if !(0 <= m && m < n) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPQconst (ANDQ x y) [0])
	// cond:
	// result: (TESTQ x y)
	for {
		if v.AuxInt != 0 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpAMD64TESTQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMPQconst (ANDQconst [c] x) [0])
	// cond:
	// result: (TESTQconst [c] x)
	for {
		if v.AuxInt != 0 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDQconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64TESTQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (CMPQconst x [0])
	// cond:
	// result: (TESTQ x x)
	for {
		if v.AuxInt != 0 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64TESTQ)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPW_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (CMPW x (MOVLconst [c]))
	// cond:
	// result: (CMPWconst x [int64(int16(c))])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64CMPWconst)
		v.AuxInt = int64(int16(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPW (MOVLconst [c]) x)
	// cond:
	// result: (InvertFlags (CMPWconst x [int64(int16(c))]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpAMD64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v0.AuxInt = int64(int16(c))
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPWconst_0(v *Value) bool {
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)==int16(y)
	// result: (FlagEQ)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int16(x) == int16(y)) {
			break
		}
		v.reset(OpAMD64FlagEQ)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)<int16(y) && uint16(x)<uint16(y)
	// result: (FlagLT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int16(x) < int16(y) && uint16(x) < uint16(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)<int16(y) && uint16(x)>uint16(y)
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int16(x) < int16(y) && uint16(x) > uint16(y)) {
			break
		}
		v.reset(OpAMD64FlagLT_UGT)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)>int16(y) && uint16(x)<uint16(y)
	// result: (FlagGT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int16(x) > int16(y) && uint16(x) < uint16(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_ULT)
		return true
	}
	// match: (CMPWconst (MOVLconst [x]) [y])
	// cond: int16(x)>int16(y) && uint16(x)>uint16(y)
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		x := v_0.AuxInt
		if !(int16(x) > int16(y) && uint16(x) > uint16(y)) {
			break
		}
		v.reset(OpAMD64FlagGT_UGT)
		return true
	}
	// match: (CMPWconst (ANDLconst _ [m]) [n])
	// cond: 0 <= int16(m) && int16(m) < int16(n)
	// result: (FlagLT_ULT)
	for {
		n := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		m := v_0.AuxInt
		if !(0 <= int16(m) && int16(m) < int16(n)) {
			break
		}
		v.reset(OpAMD64FlagLT_ULT)
		return true
	}
	// match: (CMPWconst (ANDL x y) [0])
	// cond:
	// result: (TESTW x y)
	for {
		if v.AuxInt != 0 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpAMD64TESTW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMPWconst (ANDLconst [c] x) [0])
	// cond:
	// result: (TESTWconst [int64(int16(c))] x)
	for {
		if v.AuxInt != 0 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64TESTWconst)
		v.AuxInt = int64(int16(c))
		v.AddArg(x)
		return true
	}
	// match: (CMPWconst x [0])
	// cond:
	// result: (TESTW x x)
	for {
		if v.AuxInt != 0 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64TESTW)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPXCHGLlock_0(v *Value) bool {
	// match: (CMPXCHGLlock [off1] {sym} (ADDQconst [off2] ptr) old new_ mem)
	// cond: is32Bit(off1+off2)
	// result: (CMPXCHGLlock [off1+off2] {sym} ptr old new_ mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		old := v.Args[1]
		new_ := v.Args[2]
		mem := v.Args[3]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64CMPXCHGLlock)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(old)
		v.AddArg(new_)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64CMPXCHGQlock_0(v *Value) bool {
	// match: (CMPXCHGQlock [off1] {sym} (ADDQconst [off2] ptr) old new_ mem)
	// cond: is32Bit(off1+off2)
	// result: (CMPXCHGQlock [off1+off2] {sym} ptr old new_ mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		old := v.Args[1]
		new_ := v.Args[2]
		mem := v.Args[3]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64CMPXCHGQlock)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(old)
		v.AddArg(new_)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAL_0(v *Value) bool {
	// match: (LEAL [c] {s} (ADDLconst [d] x))
	// cond: is32Bit(c+d)
	// result: (LEAL [c+d] {s} x)
	for {
		c := v.AuxInt
		s := v.Aux
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpAMD64LEAL)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAQ_0(v *Value) bool {
	// match: (LEAQ [c] {s} (ADDQconst [d] x))
	// cond: is32Bit(c+d)
	// result: (LEAQ [c+d] {s} x)
	for {
		c := v.AuxInt
		s := v.Aux
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		return true
	}
	// match: (LEAQ [c] {s} (ADDQ x y))
	// cond: x.Op != OpSB && y.Op != OpSB
	// result: (LEAQ1 [c] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if !(x.Op != OpSB && y.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = c
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ [off1] {sym1} (LEAQ [off2] {sym2} x))
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (LEAQ [off1+off2] {mergeSym(sym1,sym2)} x)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		x := v_0.Args[0]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(x)
		return true
	}
	// match: (LEAQ [off1] {sym1} (LEAQ1 [off2] {sym2} x y))
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (LEAQ1 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ [off1] {sym1} (LEAQ2 [off2] {sym2} x y))
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (LEAQ2 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ2 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ [off1] {sym1} (LEAQ4 [off2] {sym2} x y))
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (LEAQ4 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ4 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ [off1] {sym1} (LEAQ8 [off2] {sym2} x y))
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (LEAQ8 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ8 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAQ1_0(v *Value) bool {
	// match: (LEAQ1 [c] {s} (ADDQconst [d] x) y)
	// cond: is32Bit(c+d)   && x.Op != OpSB
	// result: (LEAQ1 [c+d] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		y := v.Args[1]
		if !(is32Bit(c+d) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ1 [c] {s} y (ADDQconst [d] x))
	// cond: is32Bit(c+d)   && x.Op != OpSB
	// result: (LEAQ1 [c+d] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		x := v_1.Args[0]
		if !(is32Bit(c+d) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ1 [c] {s} x (SHLQconst [1] y))
	// cond:
	// result: (LEAQ2 [c] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64LEAQ2)
		v.AuxInt = c
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ1 [c] {s} (SHLQconst [1] y) x)
	// cond:
	// result: (LEAQ2 [c] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpAMD64LEAQ2)
		v.AuxInt = c
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ1 [c] {s} x (SHLQconst [2] y))
	// cond:
	// result: (LEAQ4 [c] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = c
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ1 [c] {s} (SHLQconst [2] y) x)
	// cond:
	// result: (LEAQ4 [c] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = c
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ1 [c] {s} x (SHLQconst [3] y))
	// cond:
	// result: (LEAQ8 [c] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 3 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = c
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ1 [c] {s} (SHLQconst [3] y) x)
	// cond:
	// result: (LEAQ8 [c] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		if v_0.AuxInt != 3 {
			break
		}
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = c
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ1 [off1] {sym1} (LEAQ [off2] {sym2} x) y)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAQ1 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		x := v_0.Args[0]
		y := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ1 [off1] {sym1} y (LEAQ [off2] {sym2} x))
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAQ1 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		y := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		x := v_1.Args[0]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAQ2_0(v *Value) bool {
	// match: (LEAQ2 [c] {s} (ADDQconst [d] x) y)
	// cond: is32Bit(c+d)   && x.Op != OpSB
	// result: (LEAQ2 [c+d] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		y := v.Args[1]
		if !(is32Bit(c+d) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ2 [c] {s} x (ADDQconst [d] y))
	// cond: is32Bit(c+2*d) && y.Op != OpSB
	// result: (LEAQ2 [c+2*d] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		y := v_1.Args[0]
		if !(is32Bit(c+2*d) && y.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AuxInt = c + 2*d
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ2 [c] {s} x (SHLQconst [1] y))
	// cond:
	// result: (LEAQ4 [c] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = c
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ2 [c] {s} x (SHLQconst [2] y))
	// cond:
	// result: (LEAQ8 [c] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = c
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ2 [off1] {sym1} (LEAQ [off2] {sym2} x) y)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAQ2 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		x := v_0.Args[0]
		y := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAQ4_0(v *Value) bool {
	// match: (LEAQ4 [c] {s} (ADDQconst [d] x) y)
	// cond: is32Bit(c+d)   && x.Op != OpSB
	// result: (LEAQ4 [c+d] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		y := v.Args[1]
		if !(is32Bit(c+d) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ4 [c] {s} x (ADDQconst [d] y))
	// cond: is32Bit(c+4*d) && y.Op != OpSB
	// result: (LEAQ4 [c+4*d] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		y := v_1.Args[0]
		if !(is32Bit(c+4*d) && y.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = c + 4*d
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ4 [c] {s} x (SHLQconst [1] y))
	// cond:
	// result: (LEAQ8 [c] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = c
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ4 [off1] {sym1} (LEAQ [off2] {sym2} x) y)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAQ4 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		x := v_0.Args[0]
		y := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64LEAQ8_0(v *Value) bool {
	// match: (LEAQ8 [c] {s} (ADDQconst [d] x) y)
	// cond: is32Bit(c+d)   && x.Op != OpSB
	// result: (LEAQ8 [c+d] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		y := v.Args[1]
		if !(is32Bit(c+d) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ8 [c] {s} x (ADDQconst [d] y))
	// cond: is32Bit(c+8*d) && y.Op != OpSB
	// result: (LEAQ8 [c+8*d] {s} x y)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		y := v_1.Args[0]
		if !(is32Bit(c+8*d) && y.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = c + 8*d
		v.Aux = s
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (LEAQ8 [off1] {sym1} (LEAQ [off2] {sym2} x) y)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2) && x.Op != OpSB
	// result: (LEAQ8 [off1+off2] {mergeSym(sym1,sym2)} x y)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		x := v_0.Args[0]
		y := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2) && x.Op != OpSB) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBQSX_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MOVBQSX x:(MOVBload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVBload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBQSXload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVBQSX x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVWload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBQSXload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVBQSX x:(MOVLload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVLload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBQSXload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVBQSX x:(MOVQload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVQload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBQSXload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVBQSX (ANDLconst [c] x))
	// cond: c & 0x80 == 0
	// result: (ANDLconst [c & 0x7f] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		if !(c&0x80 == 0) {
			break
		}
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = c & 0x7f
		v.AddArg(x)
		return true
	}
	// match: (MOVBQSX x:(MOVBQSX _))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVBQSX {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBQSXload_0(v *Value) bool {
	// match: (MOVBQSXload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVBQSX x)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVBstore {
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
		v.reset(OpAMD64MOVBQSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVBQSXload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVBQSXload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVBQSXload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBQZX_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MOVBQZX x:(MOVBload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVBload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVBQZX x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVWload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVBQZX x:(MOVLload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVLload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVBQZX x:(MOVQload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVQload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVBQZX x:(MOVBloadidx1 [off] {sym} ptr idx mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVBloadidx1 <v.Type> [off] {sym} ptr idx mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVBloadidx1 {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		mem := x.Args[2]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVBQZX (ANDLconst [c] x))
	// cond:
	// result: (ANDLconst [c & 0xff] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = c & 0xff
		v.AddArg(x)
		return true
	}
	// match: (MOVBQZX x:(MOVBQZX _))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVBQZX {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBload_0(v *Value) bool {
	// match: (MOVBload [off] {sym} ptr (MOVBstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVBQZX x)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVBstore {
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
		v.reset(OpAMD64MOVBQZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVBload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVBload  [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVBload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVBload  [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVBload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (LEAQ1 [off2] {sym2} ptr idx) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVBloadidx1 [off1+off2] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVBloadidx1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBload [off] {sym} (ADDQ ptr idx) mem)
	// cond: ptr.Op != OpSB
	// result: (MOVBloadidx1 [off] {sym} ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64MOVBloadidx1)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVBload  [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVBload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVBload  [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVBload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBloadidx1_0(v *Value) bool {
	// match: (MOVBloadidx1 [c] {sym} (ADDQconst [d] ptr) idx mem)
	// cond:
	// result: (MOVBloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVBloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBloadidx1 [c] {sym} idx (ADDQconst [d] ptr) mem)
	// cond:
	// result: (MOVBloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		idx := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		ptr := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVBloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBloadidx1 [c] {sym} ptr (ADDQconst [d] idx) mem)
	// cond:
	// result: (MOVBloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVBloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBloadidx1 [c] {sym} (ADDQconst [d] idx) ptr mem)
	// cond:
	// result: (MOVBloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		idx := v_0.Args[0]
		ptr := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVBloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBstore_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MOVBstore [off] {sym} ptr (MOVBQSX x) mem)
	// cond:
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVBQSX {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVBQZX x) mem)
	// cond:
	// result: (MOVBstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVBQZX {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVBstore  [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off] {sym} ptr (MOVLconst [c]) mem)
	// cond: validOff(off)
	// result: (MOVBstoreconst [makeValAndOff(int64(int8(c)),off)] {sym} ptr mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		mem := v.Args[2]
		if !(validOff(off)) {
			break
		}
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = makeValAndOff(int64(int8(c)), off)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVBstore  [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (LEAQ1 [off2] {sym2} ptr idx) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVBstoreidx1 [off1+off2] {mergeSym(sym1,sym2)} ptr idx val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVBstoreidx1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off] {sym} (ADDQ ptr idx) val mem)
	// cond: ptr.Op != OpSB
	// result: (MOVBstoreidx1 [off] {sym} ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64MOVBstoreidx1)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} p w x0:(MOVBstore [i-1] {s} p (SHRWconst [8] w) mem))
	// cond: x0.Uses == 1   && clobber(x0)
	// result: (MOVWstore [i-1] {s} p (ROLWconst <w.Type> [8] w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		w := v.Args[1]
		x0 := v.Args[2]
		if x0.Op != OpAMD64MOVBstore {
			break
		}
		if x0.AuxInt != i-1 {
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
		if x0_1.Op != OpAMD64SHRWconst {
			break
		}
		if x0_1.AuxInt != 8 {
			break
		}
		if w != x0_1.Args[0] {
			break
		}
		mem := x0.Args[2]
		if !(x0.Uses == 1 && clobber(x0)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(p)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, w.Type)
		v0.AuxInt = 8
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} p w x2:(MOVBstore [i-1] {s} p (SHRLconst [8] w) x1:(MOVBstore [i-2] {s} p (SHRLconst [16] w) x0:(MOVBstore [i-3] {s} p (SHRLconst [24] w) mem))))
	// cond: x0.Uses == 1   && x1.Uses == 1   && x2.Uses == 1   && clobber(x0)   && clobber(x1)   && clobber(x2)
	// result: (MOVLstore [i-3] {s} p (BSWAPL <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		w := v.Args[1]
		x2 := v.Args[2]
		if x2.Op != OpAMD64MOVBstore {
			break
		}
		if x2.AuxInt != i-1 {
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
		if x2_1.Op != OpAMD64SHRLconst {
			break
		}
		if x2_1.AuxInt != 8 {
			break
		}
		if w != x2_1.Args[0] {
			break
		}
		x1 := x2.Args[2]
		if x1.Op != OpAMD64MOVBstore {
			break
		}
		if x1.AuxInt != i-2 {
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
		if x1_1.Op != OpAMD64SHRLconst {
			break
		}
		if x1_1.AuxInt != 16 {
			break
		}
		if w != x1_1.Args[0] {
			break
		}
		x0 := x1.Args[2]
		if x0.Op != OpAMD64MOVBstore {
			break
		}
		if x0.AuxInt != i-3 {
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
		if x0_1.Op != OpAMD64SHRLconst {
			break
		}
		if x0_1.AuxInt != 24 {
			break
		}
		if w != x0_1.Args[0] {
			break
		}
		mem := x0.Args[2]
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = i - 3
		v.Aux = s
		v.AddArg(p)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} p w x6:(MOVBstore [i-1] {s} p (SHRQconst [8] w) x5:(MOVBstore [i-2] {s} p (SHRQconst [16] w) x4:(MOVBstore [i-3] {s} p (SHRQconst [24] w) x3:(MOVBstore [i-4] {s} p (SHRQconst [32] w) x2:(MOVBstore [i-5] {s} p (SHRQconst [40] w) x1:(MOVBstore [i-6] {s} p (SHRQconst [48] w) x0:(MOVBstore [i-7] {s} p (SHRQconst [56] w) mem))))))))
	// cond: x0.Uses == 1   && x1.Uses == 1   && x2.Uses == 1   && x3.Uses == 1   && x4.Uses == 1   && x5.Uses == 1   && x6.Uses == 1   && clobber(x0)   && clobber(x1)   && clobber(x2)   && clobber(x3)   && clobber(x4)   && clobber(x5)   && clobber(x6)
	// result: (MOVQstore [i-7] {s} p (BSWAPQ <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		w := v.Args[1]
		x6 := v.Args[2]
		if x6.Op != OpAMD64MOVBstore {
			break
		}
		if x6.AuxInt != i-1 {
			break
		}
		if x6.Aux != s {
			break
		}
		_ = x6.Args[2]
		if p != x6.Args[0] {
			break
		}
		x6_1 := x6.Args[1]
		if x6_1.Op != OpAMD64SHRQconst {
			break
		}
		if x6_1.AuxInt != 8 {
			break
		}
		if w != x6_1.Args[0] {
			break
		}
		x5 := x6.Args[2]
		if x5.Op != OpAMD64MOVBstore {
			break
		}
		if x5.AuxInt != i-2 {
			break
		}
		if x5.Aux != s {
			break
		}
		_ = x5.Args[2]
		if p != x5.Args[0] {
			break
		}
		x5_1 := x5.Args[1]
		if x5_1.Op != OpAMD64SHRQconst {
			break
		}
		if x5_1.AuxInt != 16 {
			break
		}
		if w != x5_1.Args[0] {
			break
		}
		x4 := x5.Args[2]
		if x4.Op != OpAMD64MOVBstore {
			break
		}
		if x4.AuxInt != i-3 {
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
		if x4_1.Op != OpAMD64SHRQconst {
			break
		}
		if x4_1.AuxInt != 24 {
			break
		}
		if w != x4_1.Args[0] {
			break
		}
		x3 := x4.Args[2]
		if x3.Op != OpAMD64MOVBstore {
			break
		}
		if x3.AuxInt != i-4 {
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
		if x3_1.Op != OpAMD64SHRQconst {
			break
		}
		if x3_1.AuxInt != 32 {
			break
		}
		if w != x3_1.Args[0] {
			break
		}
		x2 := x3.Args[2]
		if x2.Op != OpAMD64MOVBstore {
			break
		}
		if x2.AuxInt != i-5 {
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
		if x2_1.Op != OpAMD64SHRQconst {
			break
		}
		if x2_1.AuxInt != 40 {
			break
		}
		if w != x2_1.Args[0] {
			break
		}
		x1 := x2.Args[2]
		if x1.Op != OpAMD64MOVBstore {
			break
		}
		if x1.AuxInt != i-6 {
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
		if x1_1.Op != OpAMD64SHRQconst {
			break
		}
		if x1_1.AuxInt != 48 {
			break
		}
		if w != x1_1.Args[0] {
			break
		}
		x0 := x1.Args[2]
		if x0.Op != OpAMD64MOVBstore {
			break
		}
		if x0.AuxInt != i-7 {
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
		if x0_1.Op != OpAMD64SHRQconst {
			break
		}
		if x0_1.AuxInt != 56 {
			break
		}
		if w != x0_1.Args[0] {
			break
		}
		mem := x0.Args[2]
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = i - 7
		v.Aux = s
		v.AddArg(p)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPQ, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBstore_10(v *Value) bool {
	// match: (MOVBstore [i] {s} p (SHRQconst [8] w) x:(MOVBstore [i-1] {s} p w mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVWstore [i-1] {s} p w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		if v_1.AuxInt != 8 {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpAMD64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[2]
		if p != x.Args[0] {
			break
		}
		if w != x.Args[1] {
			break
		}
		mem := x.Args[2]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(p)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [i] {s} p (SHRQconst [j] w) x:(MOVBstore [i-1] {s} p w0:(SHRQconst [j-8] w) mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVWstore [i-1] {s} p w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		j := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpAMD64MOVBstore {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[2]
		if p != x.Args[0] {
			break
		}
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRQconst {
			break
		}
		if w0.AuxInt != j-8 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		mem := x.Args[2]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(p)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVBstore  [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVBstore  [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBstoreconst_0(v *Value) bool {
	// match: (MOVBstoreconst [sc] {s} (ADDQconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd(off)
	// result: (MOVBstoreconst [ValAndOff(sc).add(off)] {s} ptr mem)
	for {
		sc := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = s
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreconst [sc] {sym1} (LEAQ [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)
	// result: (MOVBstoreconst [ValAndOff(sc).add(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreconst [x] {sym1} (LEAQ1 [off] {sym2} ptr idx) mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVBstoreconstidx1 [ValAndOff(x).add(off)] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		x := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVBstoreconstidx1)
		v.AuxInt = ValAndOff(x).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreconst [x] {sym} (ADDQ ptr idx) mem)
	// cond:
	// result: (MOVBstoreconstidx1 [x] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		v.reset(OpAMD64MOVBstoreconstidx1)
		v.AuxInt = x
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreconst [c] {s} p x:(MOVBstoreconst [a] {s} p mem))
	// cond: x.Uses == 1   && ValAndOff(a).Off() + 1 == ValAndOff(c).Off()   && clobber(x)
	// result: (MOVWstoreconst [makeValAndOff(ValAndOff(a).Val()&0xff | ValAndOff(c).Val()<<8, ValAndOff(a).Off())] {s} p mem)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		p := v.Args[0]
		x := v.Args[1]
		if x.Op != OpAMD64MOVBstoreconst {
			break
		}
		a := x.AuxInt
		if x.Aux != s {
			break
		}
		_ = x.Args[1]
		if p != x.Args[0] {
			break
		}
		mem := x.Args[1]
		if !(x.Uses == 1 && ValAndOff(a).Off()+1 == ValAndOff(c).Off() && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = makeValAndOff(ValAndOff(a).Val()&0xff|ValAndOff(c).Val()<<8, ValAndOff(a).Off())
		v.Aux = s
		v.AddArg(p)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreconst [sc] {sym1} (LEAL [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)
	// result: (MOVBstoreconst [ValAndOff(sc).add(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreconst [sc] {s} (ADDLconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd(off)
	// result: (MOVBstoreconst [ValAndOff(sc).add(off)] {s} ptr mem)
	for {
		sc := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		off := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = s
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBstoreconstidx1_0(v *Value) bool {
	// match: (MOVBstoreconstidx1 [x] {sym} (ADDQconst [c] ptr) idx mem)
	// cond:
	// result: (MOVBstoreconstidx1 [ValAndOff(x).add(c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVBstoreconstidx1)
		v.AuxInt = ValAndOff(x).add(c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreconstidx1 [x] {sym} ptr (ADDQconst [c] idx) mem)
	// cond:
	// result: (MOVBstoreconstidx1 [ValAndOff(x).add(c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVBstoreconstidx1)
		v.AuxInt = ValAndOff(x).add(c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreconstidx1 [c] {s} p i x:(MOVBstoreconstidx1 [a] {s} p i mem))
	// cond: x.Uses == 1   && ValAndOff(a).Off() + 1 == ValAndOff(c).Off()   && clobber(x)
	// result: (MOVWstoreconstidx1 [makeValAndOff(ValAndOff(a).Val()&0xff | ValAndOff(c).Val()<<8, ValAndOff(a).Off())] {s} p i mem)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		i := v.Args[1]
		x := v.Args[2]
		if x.Op != OpAMD64MOVBstoreconstidx1 {
			break
		}
		a := x.AuxInt
		if x.Aux != s {
			break
		}
		_ = x.Args[2]
		if p != x.Args[0] {
			break
		}
		if i != x.Args[1] {
			break
		}
		mem := x.Args[2]
		if !(x.Uses == 1 && ValAndOff(a).Off()+1 == ValAndOff(c).Off() && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconstidx1)
		v.AuxInt = makeValAndOff(ValAndOff(a).Val()&0xff|ValAndOff(c).Val()<<8, ValAndOff(a).Off())
		v.Aux = s
		v.AddArg(p)
		v.AddArg(i)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVBstoreidx1_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MOVBstoreidx1 [c] {sym} (ADDQconst [d] ptr) idx val mem)
	// cond:
	// result: (MOVBstoreidx1 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVBstoreidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx1 [c] {sym} ptr (ADDQconst [d] idx) val mem)
	// cond:
	// result: (MOVBstoreidx1 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVBstoreidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx1 [i] {s} p idx w x0:(MOVBstoreidx1 [i-1] {s} p idx (SHRWconst [8] w) mem))
	// cond: x0.Uses == 1   && clobber(x0)
	// result: (MOVWstoreidx1 [i-1] {s} p idx (ROLWconst <w.Type> [8] w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		w := v.Args[2]
		x0 := v.Args[3]
		if x0.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x0.AuxInt != i-1 {
			break
		}
		if x0.Aux != s {
			break
		}
		_ = x0.Args[3]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		x0_2 := x0.Args[2]
		if x0_2.Op != OpAMD64SHRWconst {
			break
		}
		if x0_2.AuxInt != 8 {
			break
		}
		if w != x0_2.Args[0] {
			break
		}
		mem := x0.Args[3]
		if !(x0.Uses == 1 && clobber(x0)) {
			break
		}
		v.reset(OpAMD64MOVWstoreidx1)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(p)
		v.AddArg(idx)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, w.Type)
		v0.AuxInt = 8
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx1 [i] {s} p idx w x2:(MOVBstoreidx1 [i-1] {s} p idx (SHRLconst [8] w) x1:(MOVBstoreidx1 [i-2] {s} p idx (SHRLconst [16] w) x0:(MOVBstoreidx1 [i-3] {s} p idx (SHRLconst [24] w) mem))))
	// cond: x0.Uses == 1   && x1.Uses == 1   && x2.Uses == 1   && clobber(x0)   && clobber(x1)   && clobber(x2)
	// result: (MOVLstoreidx1 [i-3] {s} p idx (BSWAPL <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		w := v.Args[2]
		x2 := v.Args[3]
		if x2.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x2.AuxInt != i-1 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[3]
		if p != x2.Args[0] {
			break
		}
		if idx != x2.Args[1] {
			break
		}
		x2_2 := x2.Args[2]
		if x2_2.Op != OpAMD64SHRLconst {
			break
		}
		if x2_2.AuxInt != 8 {
			break
		}
		if w != x2_2.Args[0] {
			break
		}
		x1 := x2.Args[3]
		if x1.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x1.AuxInt != i-2 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[3]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		x1_2 := x1.Args[2]
		if x1_2.Op != OpAMD64SHRLconst {
			break
		}
		if x1_2.AuxInt != 16 {
			break
		}
		if w != x1_2.Args[0] {
			break
		}
		x0 := x1.Args[3]
		if x0.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x0.AuxInt != i-3 {
			break
		}
		if x0.Aux != s {
			break
		}
		_ = x0.Args[3]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		x0_2 := x0.Args[2]
		if x0_2.Op != OpAMD64SHRLconst {
			break
		}
		if x0_2.AuxInt != 24 {
			break
		}
		if w != x0_2.Args[0] {
			break
		}
		mem := x0.Args[3]
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2)) {
			break
		}
		v.reset(OpAMD64MOVLstoreidx1)
		v.AuxInt = i - 3
		v.Aux = s
		v.AddArg(p)
		v.AddArg(idx)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx1 [i] {s} p idx w x6:(MOVBstoreidx1 [i-1] {s} p idx (SHRQconst [8] w) x5:(MOVBstoreidx1 [i-2] {s} p idx (SHRQconst [16] w) x4:(MOVBstoreidx1 [i-3] {s} p idx (SHRQconst [24] w) x3:(MOVBstoreidx1 [i-4] {s} p idx (SHRQconst [32] w) x2:(MOVBstoreidx1 [i-5] {s} p idx (SHRQconst [40] w) x1:(MOVBstoreidx1 [i-6] {s} p idx (SHRQconst [48] w) x0:(MOVBstoreidx1 [i-7] {s} p idx (SHRQconst [56] w) mem))))))))
	// cond: x0.Uses == 1   && x1.Uses == 1   && x2.Uses == 1   && x3.Uses == 1   && x4.Uses == 1   && x5.Uses == 1   && x6.Uses == 1   && clobber(x0)   && clobber(x1)   && clobber(x2)   && clobber(x3)   && clobber(x4)   && clobber(x5)   && clobber(x6)
	// result: (MOVQstoreidx1 [i-7] {s} p idx (BSWAPQ <w.Type> w) mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		w := v.Args[2]
		x6 := v.Args[3]
		if x6.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x6.AuxInt != i-1 {
			break
		}
		if x6.Aux != s {
			break
		}
		_ = x6.Args[3]
		if p != x6.Args[0] {
			break
		}
		if idx != x6.Args[1] {
			break
		}
		x6_2 := x6.Args[2]
		if x6_2.Op != OpAMD64SHRQconst {
			break
		}
		if x6_2.AuxInt != 8 {
			break
		}
		if w != x6_2.Args[0] {
			break
		}
		x5 := x6.Args[3]
		if x5.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x5.AuxInt != i-2 {
			break
		}
		if x5.Aux != s {
			break
		}
		_ = x5.Args[3]
		if p != x5.Args[0] {
			break
		}
		if idx != x5.Args[1] {
			break
		}
		x5_2 := x5.Args[2]
		if x5_2.Op != OpAMD64SHRQconst {
			break
		}
		if x5_2.AuxInt != 16 {
			break
		}
		if w != x5_2.Args[0] {
			break
		}
		x4 := x5.Args[3]
		if x4.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x4.AuxInt != i-3 {
			break
		}
		if x4.Aux != s {
			break
		}
		_ = x4.Args[3]
		if p != x4.Args[0] {
			break
		}
		if idx != x4.Args[1] {
			break
		}
		x4_2 := x4.Args[2]
		if x4_2.Op != OpAMD64SHRQconst {
			break
		}
		if x4_2.AuxInt != 24 {
			break
		}
		if w != x4_2.Args[0] {
			break
		}
		x3 := x4.Args[3]
		if x3.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x3.AuxInt != i-4 {
			break
		}
		if x3.Aux != s {
			break
		}
		_ = x3.Args[3]
		if p != x3.Args[0] {
			break
		}
		if idx != x3.Args[1] {
			break
		}
		x3_2 := x3.Args[2]
		if x3_2.Op != OpAMD64SHRQconst {
			break
		}
		if x3_2.AuxInt != 32 {
			break
		}
		if w != x3_2.Args[0] {
			break
		}
		x2 := x3.Args[3]
		if x2.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x2.AuxInt != i-5 {
			break
		}
		if x2.Aux != s {
			break
		}
		_ = x2.Args[3]
		if p != x2.Args[0] {
			break
		}
		if idx != x2.Args[1] {
			break
		}
		x2_2 := x2.Args[2]
		if x2_2.Op != OpAMD64SHRQconst {
			break
		}
		if x2_2.AuxInt != 40 {
			break
		}
		if w != x2_2.Args[0] {
			break
		}
		x1 := x2.Args[3]
		if x1.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x1.AuxInt != i-6 {
			break
		}
		if x1.Aux != s {
			break
		}
		_ = x1.Args[3]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		x1_2 := x1.Args[2]
		if x1_2.Op != OpAMD64SHRQconst {
			break
		}
		if x1_2.AuxInt != 48 {
			break
		}
		if w != x1_2.Args[0] {
			break
		}
		x0 := x1.Args[3]
		if x0.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x0.AuxInt != i-7 {
			break
		}
		if x0.Aux != s {
			break
		}
		_ = x0.Args[3]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		x0_2 := x0.Args[2]
		if x0_2.Op != OpAMD64SHRQconst {
			break
		}
		if x0_2.AuxInt != 56 {
			break
		}
		if w != x0_2.Args[0] {
			break
		}
		mem := x0.Args[3]
		if !(x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) && clobber(x5) && clobber(x6)) {
			break
		}
		v.reset(OpAMD64MOVQstoreidx1)
		v.AuxInt = i - 7
		v.Aux = s
		v.AddArg(p)
		v.AddArg(idx)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPQ, w.Type)
		v0.AddArg(w)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx1 [i] {s} p idx (SHRQconst [8] w) x:(MOVBstoreidx1 [i-1] {s} p idx w mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVWstoreidx1 [i-1] {s} p idx w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpAMD64SHRQconst {
			break
		}
		if v_2.AuxInt != 8 {
			break
		}
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[3]
		if p != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		if w != x.Args[2] {
			break
		}
		mem := x.Args[3]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstoreidx1)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(p)
		v.AddArg(idx)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstoreidx1 [i] {s} p idx (SHRQconst [j] w) x:(MOVBstoreidx1 [i-1] {s} p idx w0:(SHRQconst [j-8] w) mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVWstoreidx1 [i-1] {s} p idx w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpAMD64SHRQconst {
			break
		}
		j := v_2.AuxInt
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpAMD64MOVBstoreidx1 {
			break
		}
		if x.AuxInt != i-1 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[3]
		if p != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		w0 := x.Args[2]
		if w0.Op != OpAMD64SHRQconst {
			break
		}
		if w0.AuxInt != j-8 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		mem := x.Args[3]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVWstoreidx1)
		v.AuxInt = i - 1
		v.Aux = s
		v.AddArg(p)
		v.AddArg(idx)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLQSX_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MOVLQSX x:(MOVLload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVLQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVLload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLQSXload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVLQSX x:(MOVQload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVLQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVQload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLQSXload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVLQSX (ANDLconst [c] x))
	// cond: c & 0x80000000 == 0
	// result: (ANDLconst [c & 0x7fffffff] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		if !(c&0x80000000 == 0) {
			break
		}
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = c & 0x7fffffff
		v.AddArg(x)
		return true
	}
	// match: (MOVLQSX x:(MOVLQSX _))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVLQSX {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MOVLQSX x:(MOVWQSX _))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVWQSX {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MOVLQSX x:(MOVBQSX _))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVBQSX {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLQSXload_0(v *Value) bool {
	// match: (MOVLQSXload [off] {sym} ptr (MOVLstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVLQSX x)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLstore {
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
		v.reset(OpAMD64MOVLQSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVLQSXload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVLQSXload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLQSXload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLQZX_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MOVLQZX x:(MOVLload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVLload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVLload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVLQZX x:(MOVQload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVLload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVQload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVLQZX x:(MOVLloadidx1 [off] {sym} ptr idx mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVLloadidx1 <v.Type> [off] {sym} ptr idx mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVLloadidx1 {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		mem := x.Args[2]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVLQZX x:(MOVLloadidx4 [off] {sym} ptr idx mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVLloadidx4 <v.Type> [off] {sym} ptr idx mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVLloadidx4 {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		mem := x.Args[2]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx4, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVLQZX (ANDLconst [c] x))
	// cond:
	// result: (ANDLconst [c] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (MOVLQZX x:(MOVLQZX _))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVLQZX {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MOVLQZX x:(MOVWQZX _))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVWQZX {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MOVLQZX x:(MOVBQZX _))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVBQZX {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLatomicload_0(v *Value) bool {
	// match: (MOVLatomicload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVLatomicload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVLatomicload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLatomicload [off1] {sym1} (LEAQ [off2] {sym2} ptr) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVLatomicload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLatomicload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLload_0(v *Value) bool {
	// match: (MOVLload [off] {sym} ptr (MOVLstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVLQZX x)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLstore {
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
		v.reset(OpAMD64MOVLQZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVLload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVLload  [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVLload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVLload  [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLload [off1] {sym1} (LEAQ1 [off2] {sym2} ptr idx) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVLloadidx1 [off1+off2] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLloadidx1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLload [off1] {sym1} (LEAQ4 [off2] {sym2} ptr idx) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVLloadidx4 [off1+off2] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ4 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLloadidx4)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLload [off] {sym} (ADDQ ptr idx) mem)
	// cond: ptr.Op != OpSB
	// result: (MOVLloadidx1 [off] {sym} ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64MOVLloadidx1)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVLload  [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVLload  [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVLload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLloadidx1_0(v *Value) bool {
	// match: (MOVLloadidx1 [c] {sym} ptr (SHLQconst [2] idx) mem)
	// cond:
	// result: (MOVLloadidx4 [c] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLloadidx4)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLloadidx1 [c] {sym} (SHLQconst [2] idx) ptr mem)
	// cond:
	// result: (MOVLloadidx4 [c] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		if v_0.AuxInt != 2 {
			break
		}
		idx := v_0.Args[0]
		ptr := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLloadidx4)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLloadidx1 [c] {sym} (ADDQconst [d] ptr) idx mem)
	// cond:
	// result: (MOVLloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLloadidx1 [c] {sym} idx (ADDQconst [d] ptr) mem)
	// cond:
	// result: (MOVLloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		idx := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		ptr := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLloadidx1 [c] {sym} ptr (ADDQconst [d] idx) mem)
	// cond:
	// result: (MOVLloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLloadidx1 [c] {sym} (ADDQconst [d] idx) ptr mem)
	// cond:
	// result: (MOVLloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		idx := v_0.Args[0]
		ptr := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLloadidx4_0(v *Value) bool {
	// match: (MOVLloadidx4 [c] {sym} (ADDQconst [d] ptr) idx mem)
	// cond:
	// result: (MOVLloadidx4 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLloadidx4)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLloadidx4 [c] {sym} ptr (ADDQconst [d] idx) mem)
	// cond:
	// result: (MOVLloadidx4 [c+4*d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLloadidx4)
		v.AuxInt = c + 4*d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLstore_0(v *Value) bool {
	// match: (MOVLstore [off] {sym} ptr (MOVLQSX x) mem)
	// cond:
	// result: (MOVLstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLQSX {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstore [off] {sym} ptr (MOVLQZX x) mem)
	// cond:
	// result: (MOVLstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLQZX {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVLstore  [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstore [off] {sym} ptr (MOVLconst [c]) mem)
	// cond: validOff(off)
	// result: (MOVLstoreconst [makeValAndOff(int64(int32(c)),off)] {sym} ptr mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		mem := v.Args[2]
		if !(validOff(off)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = makeValAndOff(int64(int32(c)), off)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVLstore  [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstore [off1] {sym1} (LEAQ1 [off2] {sym2} ptr idx) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVLstoreidx1 [off1+off2] {mergeSym(sym1,sym2)} ptr idx val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLstoreidx1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstore [off1] {sym1} (LEAQ4 [off2] {sym2} ptr idx) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVLstoreidx4 [off1+off2] {mergeSym(sym1,sym2)} ptr idx val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ4 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLstoreidx4)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstore [off] {sym} (ADDQ ptr idx) val mem)
	// cond: ptr.Op != OpSB
	// result: (MOVLstoreidx1 [off] {sym} ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64MOVLstoreidx1)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstore [i] {s} p (SHRQconst [32] w) x:(MOVLstore [i-4] {s} p w mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVQstore [i-4] {s} p w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		if v_1.AuxInt != 32 {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpAMD64MOVLstore {
			break
		}
		if x.AuxInt != i-4 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[2]
		if p != x.Args[0] {
			break
		}
		if w != x.Args[1] {
			break
		}
		mem := x.Args[2]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = i - 4
		v.Aux = s
		v.AddArg(p)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstore [i] {s} p (SHRQconst [j] w) x:(MOVLstore [i-4] {s} p w0:(SHRQconst [j-32] w) mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVQstore [i-4] {s} p w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		j := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpAMD64MOVLstore {
			break
		}
		if x.AuxInt != i-4 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[2]
		if p != x.Args[0] {
			break
		}
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRQconst {
			break
		}
		if w0.AuxInt != j-32 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		mem := x.Args[2]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = i - 4
		v.Aux = s
		v.AddArg(p)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLstore_10(v *Value) bool {
	// match: (MOVLstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVLstore  [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVLstore  [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLstoreconst_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (MOVLstoreconst [sc] {s} (ADDQconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd(off)
	// result: (MOVLstoreconst [ValAndOff(sc).add(off)] {s} ptr mem)
	for {
		sc := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = s
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreconst [sc] {sym1} (LEAQ [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)
	// result: (MOVLstoreconst [ValAndOff(sc).add(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreconst [x] {sym1} (LEAQ1 [off] {sym2} ptr idx) mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVLstoreconstidx1 [ValAndOff(x).add(off)] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		x := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconstidx1)
		v.AuxInt = ValAndOff(x).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreconst [x] {sym1} (LEAQ4 [off] {sym2} ptr idx) mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVLstoreconstidx4 [ValAndOff(x).add(off)] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		x := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ4 {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconstidx4)
		v.AuxInt = ValAndOff(x).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreconst [x] {sym} (ADDQ ptr idx) mem)
	// cond:
	// result: (MOVLstoreconstidx1 [x] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		v.reset(OpAMD64MOVLstoreconstidx1)
		v.AuxInt = x
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreconst [c] {s} p x:(MOVLstoreconst [a] {s} p mem))
	// cond: x.Uses == 1   && ValAndOff(a).Off() + 4 == ValAndOff(c).Off()   && clobber(x)
	// result: (MOVQstore [ValAndOff(a).Off()] {s} p (MOVQconst [ValAndOff(a).Val()&0xffffffff | ValAndOff(c).Val()<<32]) mem)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		p := v.Args[0]
		x := v.Args[1]
		if x.Op != OpAMD64MOVLstoreconst {
			break
		}
		a := x.AuxInt
		if x.Aux != s {
			break
		}
		_ = x.Args[1]
		if p != x.Args[0] {
			break
		}
		mem := x.Args[1]
		if !(x.Uses == 1 && ValAndOff(a).Off()+4 == ValAndOff(c).Off() && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = ValAndOff(a).Off()
		v.Aux = s
		v.AddArg(p)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = ValAndOff(a).Val()&0xffffffff | ValAndOff(c).Val()<<32
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreconst [sc] {sym1} (LEAL [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)
	// result: (MOVLstoreconst [ValAndOff(sc).add(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreconst [sc] {s} (ADDLconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd(off)
	// result: (MOVLstoreconst [ValAndOff(sc).add(off)] {s} ptr mem)
	for {
		sc := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		off := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = s
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLstoreconstidx1_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (MOVLstoreconstidx1 [c] {sym} ptr (SHLQconst [2] idx) mem)
	// cond:
	// result: (MOVLstoreconstidx4 [c] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLstoreconstidx4)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreconstidx1 [x] {sym} (ADDQconst [c] ptr) idx mem)
	// cond:
	// result: (MOVLstoreconstidx1 [ValAndOff(x).add(c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLstoreconstidx1)
		v.AuxInt = ValAndOff(x).add(c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreconstidx1 [x] {sym} ptr (ADDQconst [c] idx) mem)
	// cond:
	// result: (MOVLstoreconstidx1 [ValAndOff(x).add(c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLstoreconstidx1)
		v.AuxInt = ValAndOff(x).add(c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreconstidx1 [c] {s} p i x:(MOVLstoreconstidx1 [a] {s} p i mem))
	// cond: x.Uses == 1   && ValAndOff(a).Off() + 4 == ValAndOff(c).Off()   && clobber(x)
	// result: (MOVQstoreidx1 [ValAndOff(a).Off()] {s} p i (MOVQconst [ValAndOff(a).Val()&0xffffffff | ValAndOff(c).Val()<<32]) mem)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		i := v.Args[1]
		x := v.Args[2]
		if x.Op != OpAMD64MOVLstoreconstidx1 {
			break
		}
		a := x.AuxInt
		if x.Aux != s {
			break
		}
		_ = x.Args[2]
		if p != x.Args[0] {
			break
		}
		if i != x.Args[1] {
			break
		}
		mem := x.Args[2]
		if !(x.Uses == 1 && ValAndOff(a).Off()+4 == ValAndOff(c).Off() && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstoreidx1)
		v.AuxInt = ValAndOff(a).Off()
		v.Aux = s
		v.AddArg(p)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = ValAndOff(a).Val()&0xffffffff | ValAndOff(c).Val()<<32
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLstoreconstidx4_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (MOVLstoreconstidx4 [x] {sym} (ADDQconst [c] ptr) idx mem)
	// cond:
	// result: (MOVLstoreconstidx4 [ValAndOff(x).add(c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLstoreconstidx4)
		v.AuxInt = ValAndOff(x).add(c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreconstidx4 [x] {sym} ptr (ADDQconst [c] idx) mem)
	// cond:
	// result: (MOVLstoreconstidx4 [ValAndOff(x).add(4*c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLstoreconstidx4)
		v.AuxInt = ValAndOff(x).add(4 * c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreconstidx4 [c] {s} p i x:(MOVLstoreconstidx4 [a] {s} p i mem))
	// cond: x.Uses == 1   && ValAndOff(a).Off() + 4 == ValAndOff(c).Off()   && clobber(x)
	// result: (MOVQstoreidx1 [ValAndOff(a).Off()] {s} p (SHLQconst <i.Type> [2] i) (MOVQconst [ValAndOff(a).Val()&0xffffffff | ValAndOff(c).Val()<<32]) mem)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		i := v.Args[1]
		x := v.Args[2]
		if x.Op != OpAMD64MOVLstoreconstidx4 {
			break
		}
		a := x.AuxInt
		if x.Aux != s {
			break
		}
		_ = x.Args[2]
		if p != x.Args[0] {
			break
		}
		if i != x.Args[1] {
			break
		}
		mem := x.Args[2]
		if !(x.Uses == 1 && ValAndOff(a).Off()+4 == ValAndOff(c).Off() && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstoreidx1)
		v.AuxInt = ValAndOff(a).Off()
		v.Aux = s
		v.AddArg(p)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, i.Type)
		v0.AuxInt = 2
		v0.AddArg(i)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v1.AuxInt = ValAndOff(a).Val()&0xffffffff | ValAndOff(c).Val()<<32
		v.AddArg(v1)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLstoreidx1_0(v *Value) bool {
	// match: (MOVLstoreidx1 [c] {sym} ptr (SHLQconst [2] idx) val mem)
	// cond:
	// result: (MOVLstoreidx4 [c] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVLstoreidx4)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreidx1 [c] {sym} (ADDQconst [d] ptr) idx val mem)
	// cond:
	// result: (MOVLstoreidx1 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVLstoreidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreidx1 [c] {sym} ptr (ADDQconst [d] idx) val mem)
	// cond:
	// result: (MOVLstoreidx1 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVLstoreidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreidx1 [i] {s} p idx (SHRQconst [32] w) x:(MOVLstoreidx1 [i-4] {s} p idx w mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVQstoreidx1 [i-4] {s} p idx w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpAMD64SHRQconst {
			break
		}
		if v_2.AuxInt != 32 {
			break
		}
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpAMD64MOVLstoreidx1 {
			break
		}
		if x.AuxInt != i-4 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[3]
		if p != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		if w != x.Args[2] {
			break
		}
		mem := x.Args[3]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstoreidx1)
		v.AuxInt = i - 4
		v.Aux = s
		v.AddArg(p)
		v.AddArg(idx)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreidx1 [i] {s} p idx (SHRQconst [j] w) x:(MOVLstoreidx1 [i-4] {s} p idx w0:(SHRQconst [j-32] w) mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVQstoreidx1 [i-4] {s} p idx w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpAMD64SHRQconst {
			break
		}
		j := v_2.AuxInt
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpAMD64MOVLstoreidx1 {
			break
		}
		if x.AuxInt != i-4 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[3]
		if p != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		w0 := x.Args[2]
		if w0.Op != OpAMD64SHRQconst {
			break
		}
		if w0.AuxInt != j-32 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		mem := x.Args[3]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstoreidx1)
		v.AuxInt = i - 4
		v.Aux = s
		v.AddArg(p)
		v.AddArg(idx)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVLstoreidx4_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MOVLstoreidx4 [c] {sym} (ADDQconst [d] ptr) idx val mem)
	// cond:
	// result: (MOVLstoreidx4 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVLstoreidx4)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreidx4 [c] {sym} ptr (ADDQconst [d] idx) val mem)
	// cond:
	// result: (MOVLstoreidx4 [c+4*d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVLstoreidx4)
		v.AuxInt = c + 4*d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreidx4 [i] {s} p idx (SHRQconst [32] w) x:(MOVLstoreidx4 [i-4] {s} p idx w mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVQstoreidx1 [i-4] {s} p (SHLQconst <idx.Type> [2] idx) w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpAMD64SHRQconst {
			break
		}
		if v_2.AuxInt != 32 {
			break
		}
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpAMD64MOVLstoreidx4 {
			break
		}
		if x.AuxInt != i-4 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[3]
		if p != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		if w != x.Args[2] {
			break
		}
		mem := x.Args[3]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstoreidx1)
		v.AuxInt = i - 4
		v.Aux = s
		v.AddArg(p)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, idx.Type)
		v0.AuxInt = 2
		v0.AddArg(idx)
		v.AddArg(v0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVLstoreidx4 [i] {s} p idx (SHRQconst [j] w) x:(MOVLstoreidx4 [i-4] {s} p idx w0:(SHRQconst [j-32] w) mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVQstoreidx1 [i-4] {s} p (SHLQconst <idx.Type> [2] idx) w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpAMD64SHRQconst {
			break
		}
		j := v_2.AuxInt
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpAMD64MOVLstoreidx4 {
			break
		}
		if x.AuxInt != i-4 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[3]
		if p != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		w0 := x.Args[2]
		if w0.Op != OpAMD64SHRQconst {
			break
		}
		if w0.AuxInt != j-32 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		mem := x.Args[3]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVQstoreidx1)
		v.AuxInt = i - 4
		v.Aux = s
		v.AddArg(p)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, idx.Type)
		v0.AuxInt = 2
		v0.AddArg(idx)
		v.AddArg(v0)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVOload_0(v *Value) bool {
	// match: (MOVOload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVOload  [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVOload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVOload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVOload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVOload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVOstore_0(v *Value) bool {
	// match: (MOVOstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVOstore  [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVOstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVOstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVOstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVOstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQatomicload_0(v *Value) bool {
	// match: (MOVQatomicload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVQatomicload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVQatomicload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQatomicload [off1] {sym1} (LEAQ [off2] {sym2} ptr) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVQatomicload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQatomicload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQload_0(v *Value) bool {
	// match: (MOVQload [off] {sym} ptr (MOVQstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: x
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQstore {
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
	// match: (MOVQload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVQload  [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVQload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVQload  [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQload [off1] {sym1} (LEAQ1 [off2] {sym2} ptr idx) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVQloadidx1 [off1+off2] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQloadidx1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQload [off1] {sym1} (LEAQ8 [off2] {sym2} ptr idx) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVQloadidx8 [off1+off2] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ8 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQloadidx8)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQload [off] {sym} (ADDQ ptr idx) mem)
	// cond: ptr.Op != OpSB
	// result: (MOVQloadidx1 [off] {sym} ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64MOVQloadidx1)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVQload  [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVQload  [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVQload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQloadidx1_0(v *Value) bool {
	// match: (MOVQloadidx1 [c] {sym} ptr (SHLQconst [3] idx) mem)
	// cond:
	// result: (MOVQloadidx8 [c] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 3 {
			break
		}
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQloadidx8)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQloadidx1 [c] {sym} (SHLQconst [3] idx) ptr mem)
	// cond:
	// result: (MOVQloadidx8 [c] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		if v_0.AuxInt != 3 {
			break
		}
		idx := v_0.Args[0]
		ptr := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQloadidx8)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQloadidx1 [c] {sym} (ADDQconst [d] ptr) idx mem)
	// cond:
	// result: (MOVQloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQloadidx1 [c] {sym} idx (ADDQconst [d] ptr) mem)
	// cond:
	// result: (MOVQloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		idx := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		ptr := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQloadidx1 [c] {sym} ptr (ADDQconst [d] idx) mem)
	// cond:
	// result: (MOVQloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQloadidx1 [c] {sym} (ADDQconst [d] idx) ptr mem)
	// cond:
	// result: (MOVQloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		idx := v_0.Args[0]
		ptr := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQloadidx8_0(v *Value) bool {
	// match: (MOVQloadidx8 [c] {sym} (ADDQconst [d] ptr) idx mem)
	// cond:
	// result: (MOVQloadidx8 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQloadidx8)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQloadidx8 [c] {sym} ptr (ADDQconst [d] idx) mem)
	// cond:
	// result: (MOVQloadidx8 [c+8*d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQloadidx8)
		v.AuxInt = c + 8*d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQstore_0(v *Value) bool {
	// match: (MOVQstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVQstore  [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstore [off] {sym} ptr (MOVQconst [c]) mem)
	// cond: validValAndOff(c,off)
	// result: (MOVQstoreconst [makeValAndOff(c,off)] {sym} ptr mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		mem := v.Args[2]
		if !(validValAndOff(c, off)) {
			break
		}
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = makeValAndOff(c, off)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVQstore  [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstore [off1] {sym1} (LEAQ1 [off2] {sym2} ptr idx) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVQstoreidx1 [off1+off2] {mergeSym(sym1,sym2)} ptr idx val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQstoreidx1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstore [off1] {sym1} (LEAQ8 [off2] {sym2} ptr idx) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVQstoreidx8 [off1+off2] {mergeSym(sym1,sym2)} ptr idx val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ8 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQstoreidx8)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstore [off] {sym} (ADDQ ptr idx) val mem)
	// cond: ptr.Op != OpSB
	// result: (MOVQstoreidx1 [off] {sym} ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64MOVQstoreidx1)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVQstore  [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVQstore  [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQstoreconst_0(v *Value) bool {
	// match: (MOVQstoreconst [sc] {s} (ADDQconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd(off)
	// result: (MOVQstoreconst [ValAndOff(sc).add(off)] {s} ptr mem)
	for {
		sc := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = s
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstoreconst [sc] {sym1} (LEAQ [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)
	// result: (MOVQstoreconst [ValAndOff(sc).add(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstoreconst [x] {sym1} (LEAQ1 [off] {sym2} ptr idx) mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVQstoreconstidx1 [ValAndOff(x).add(off)] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		x := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQstoreconstidx1)
		v.AuxInt = ValAndOff(x).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstoreconst [x] {sym1} (LEAQ8 [off] {sym2} ptr idx) mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVQstoreconstidx8 [ValAndOff(x).add(off)] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		x := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ8 {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVQstoreconstidx8)
		v.AuxInt = ValAndOff(x).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstoreconst [x] {sym} (ADDQ ptr idx) mem)
	// cond:
	// result: (MOVQstoreconstidx1 [x] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		v.reset(OpAMD64MOVQstoreconstidx1)
		v.AuxInt = x
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstoreconst [sc] {sym1} (LEAL [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)
	// result: (MOVQstoreconst [ValAndOff(sc).add(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstoreconst [sc] {s} (ADDLconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd(off)
	// result: (MOVQstoreconst [ValAndOff(sc).add(off)] {s} ptr mem)
	for {
		sc := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		off := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = s
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQstoreconstidx1_0(v *Value) bool {
	// match: (MOVQstoreconstidx1 [c] {sym} ptr (SHLQconst [3] idx) mem)
	// cond:
	// result: (MOVQstoreconstidx8 [c] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 3 {
			break
		}
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQstoreconstidx8)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstoreconstidx1 [x] {sym} (ADDQconst [c] ptr) idx mem)
	// cond:
	// result: (MOVQstoreconstidx1 [ValAndOff(x).add(c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQstoreconstidx1)
		v.AuxInt = ValAndOff(x).add(c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstoreconstidx1 [x] {sym} ptr (ADDQconst [c] idx) mem)
	// cond:
	// result: (MOVQstoreconstidx1 [ValAndOff(x).add(c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQstoreconstidx1)
		v.AuxInt = ValAndOff(x).add(c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQstoreconstidx8_0(v *Value) bool {
	// match: (MOVQstoreconstidx8 [x] {sym} (ADDQconst [c] ptr) idx mem)
	// cond:
	// result: (MOVQstoreconstidx8 [ValAndOff(x).add(c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQstoreconstidx8)
		v.AuxInt = ValAndOff(x).add(c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstoreconstidx8 [x] {sym} ptr (ADDQconst [c] idx) mem)
	// cond:
	// result: (MOVQstoreconstidx8 [ValAndOff(x).add(8*c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQstoreconstidx8)
		v.AuxInt = ValAndOff(x).add(8 * c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQstoreidx1_0(v *Value) bool {
	// match: (MOVQstoreidx1 [c] {sym} ptr (SHLQconst [3] idx) val mem)
	// cond:
	// result: (MOVQstoreidx8 [c] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 3 {
			break
		}
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVQstoreidx8)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstoreidx1 [c] {sym} (ADDQconst [d] ptr) idx val mem)
	// cond:
	// result: (MOVQstoreidx1 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVQstoreidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstoreidx1 [c] {sym} ptr (ADDQconst [d] idx) val mem)
	// cond:
	// result: (MOVQstoreidx1 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVQstoreidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVQstoreidx8_0(v *Value) bool {
	// match: (MOVQstoreidx8 [c] {sym} (ADDQconst [d] ptr) idx val mem)
	// cond:
	// result: (MOVQstoreidx8 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVQstoreidx8)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVQstoreidx8 [c] {sym} ptr (ADDQconst [d] idx) val mem)
	// cond:
	// result: (MOVQstoreidx8 [c+8*d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVQstoreidx8)
		v.AuxInt = c + 8*d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSDload_0(v *Value) bool {
	// match: (MOVSDload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVSDload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVSDload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVSDload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSDload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDload [off1] {sym1} (LEAQ1 [off2] {sym2} ptr idx) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVSDloadidx1 [off1+off2] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSDloadidx1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDload [off1] {sym1} (LEAQ8 [off2] {sym2} ptr idx) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVSDloadidx8 [off1+off2] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ8 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSDloadidx8)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDload [off] {sym} (ADDQ ptr idx) mem)
	// cond: ptr.Op != OpSB
	// result: (MOVSDloadidx1 [off] {sym} ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64MOVSDloadidx1)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSDloadidx1_0(v *Value) bool {
	// match: (MOVSDloadidx1 [c] {sym} ptr (SHLQconst [3] idx) mem)
	// cond:
	// result: (MOVSDloadidx8 [c] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 3 {
			break
		}
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVSDloadidx8)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDloadidx1 [c] {sym} (ADDQconst [d] ptr) idx mem)
	// cond:
	// result: (MOVSDloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVSDloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDloadidx1 [c] {sym} ptr (ADDQconst [d] idx) mem)
	// cond:
	// result: (MOVSDloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVSDloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSDloadidx8_0(v *Value) bool {
	// match: (MOVSDloadidx8 [c] {sym} (ADDQconst [d] ptr) idx mem)
	// cond:
	// result: (MOVSDloadidx8 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVSDloadidx8)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDloadidx8 [c] {sym} ptr (ADDQconst [d] idx) mem)
	// cond:
	// result: (MOVSDloadidx8 [c+8*d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVSDloadidx8)
		v.AuxInt = c + 8*d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSDstore_0(v *Value) bool {
	// match: (MOVSDstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVSDstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVSDstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVSDstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSDstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDstore [off1] {sym1} (LEAQ1 [off2] {sym2} ptr idx) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVSDstoreidx1 [off1+off2] {mergeSym(sym1,sym2)} ptr idx val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSDstoreidx1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDstore [off1] {sym1} (LEAQ8 [off2] {sym2} ptr idx) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVSDstoreidx8 [off1+off2] {mergeSym(sym1,sym2)} ptr idx val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ8 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSDstoreidx8)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDstore [off] {sym} (ADDQ ptr idx) val mem)
	// cond: ptr.Op != OpSB
	// result: (MOVSDstoreidx1 [off] {sym} ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64MOVSDstoreidx1)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSDstoreidx1_0(v *Value) bool {
	// match: (MOVSDstoreidx1 [c] {sym} ptr (SHLQconst [3] idx) val mem)
	// cond:
	// result: (MOVSDstoreidx8 [c] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 3 {
			break
		}
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVSDstoreidx8)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDstoreidx1 [c] {sym} (ADDQconst [d] ptr) idx val mem)
	// cond:
	// result: (MOVSDstoreidx1 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVSDstoreidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDstoreidx1 [c] {sym} ptr (ADDQconst [d] idx) val mem)
	// cond:
	// result: (MOVSDstoreidx1 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVSDstoreidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSDstoreidx8_0(v *Value) bool {
	// match: (MOVSDstoreidx8 [c] {sym} (ADDQconst [d] ptr) idx val mem)
	// cond:
	// result: (MOVSDstoreidx8 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVSDstoreidx8)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSDstoreidx8 [c] {sym} ptr (ADDQconst [d] idx) val mem)
	// cond:
	// result: (MOVSDstoreidx8 [c+8*d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVSDstoreidx8)
		v.AuxInt = c + 8*d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSSload_0(v *Value) bool {
	// match: (MOVSSload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVSSload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVSSload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVSSload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSSload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSload [off1] {sym1} (LEAQ1 [off2] {sym2} ptr idx) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVSSloadidx1 [off1+off2] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSSloadidx1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSload [off1] {sym1} (LEAQ4 [off2] {sym2} ptr idx) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVSSloadidx4 [off1+off2] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ4 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSSloadidx4)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSload [off] {sym} (ADDQ ptr idx) mem)
	// cond: ptr.Op != OpSB
	// result: (MOVSSloadidx1 [off] {sym} ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64MOVSSloadidx1)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSSloadidx1_0(v *Value) bool {
	// match: (MOVSSloadidx1 [c] {sym} ptr (SHLQconst [2] idx) mem)
	// cond:
	// result: (MOVSSloadidx4 [c] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVSSloadidx4)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSloadidx1 [c] {sym} (ADDQconst [d] ptr) idx mem)
	// cond:
	// result: (MOVSSloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVSSloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSloadidx1 [c] {sym} ptr (ADDQconst [d] idx) mem)
	// cond:
	// result: (MOVSSloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVSSloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSSloadidx4_0(v *Value) bool {
	// match: (MOVSSloadidx4 [c] {sym} (ADDQconst [d] ptr) idx mem)
	// cond:
	// result: (MOVSSloadidx4 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVSSloadidx4)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSloadidx4 [c] {sym} ptr (ADDQconst [d] idx) mem)
	// cond:
	// result: (MOVSSloadidx4 [c+4*d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVSSloadidx4)
		v.AuxInt = c + 4*d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSSstore_0(v *Value) bool {
	// match: (MOVSSstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVSSstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVSSstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVSSstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSSstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSstore [off1] {sym1} (LEAQ1 [off2] {sym2} ptr idx) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVSSstoreidx1 [off1+off2] {mergeSym(sym1,sym2)} ptr idx val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSSstoreidx1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSstore [off1] {sym1} (LEAQ4 [off2] {sym2} ptr idx) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVSSstoreidx4 [off1+off2] {mergeSym(sym1,sym2)} ptr idx val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ4 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVSSstoreidx4)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSstore [off] {sym} (ADDQ ptr idx) val mem)
	// cond: ptr.Op != OpSB
	// result: (MOVSSstoreidx1 [off] {sym} ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64MOVSSstoreidx1)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSSstoreidx1_0(v *Value) bool {
	// match: (MOVSSstoreidx1 [c] {sym} ptr (SHLQconst [2] idx) val mem)
	// cond:
	// result: (MOVSSstoreidx4 [c] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 2 {
			break
		}
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVSSstoreidx4)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSstoreidx1 [c] {sym} (ADDQconst [d] ptr) idx val mem)
	// cond:
	// result: (MOVSSstoreidx1 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVSSstoreidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSstoreidx1 [c] {sym} ptr (ADDQconst [d] idx) val mem)
	// cond:
	// result: (MOVSSstoreidx1 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVSSstoreidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVSSstoreidx4_0(v *Value) bool {
	// match: (MOVSSstoreidx4 [c] {sym} (ADDQconst [d] ptr) idx val mem)
	// cond:
	// result: (MOVSSstoreidx4 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVSSstoreidx4)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVSSstoreidx4 [c] {sym} ptr (ADDQconst [d] idx) val mem)
	// cond:
	// result: (MOVSSstoreidx4 [c+4*d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVSSstoreidx4)
		v.AuxInt = c + 4*d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWQSX_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MOVWQSX x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVWload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWQSXload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVWQSX x:(MOVLload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVLload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWQSXload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVWQSX x:(MOVQload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWQSXload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVQload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWQSXload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVWQSX (ANDLconst [c] x))
	// cond: c & 0x8000 == 0
	// result: (ANDLconst [c & 0x7fff] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		if !(c&0x8000 == 0) {
			break
		}
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = c & 0x7fff
		v.AddArg(x)
		return true
	}
	// match: (MOVWQSX x:(MOVWQSX _))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVWQSX {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MOVWQSX x:(MOVBQSX _))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVBQSX {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWQSXload_0(v *Value) bool {
	// match: (MOVWQSXload [off] {sym} ptr (MOVWstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVWQSX x)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVWstore {
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
		v.reset(OpAMD64MOVWQSX)
		v.AddArg(x)
		return true
	}
	// match: (MOVWQSXload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWQSXload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWQSXload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWQZX_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MOVWQZX x:(MOVWload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVWload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVWQZX x:(MOVLload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVLload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVWQZX x:(MOVQload [off] {sym} ptr mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWload <v.Type> [off] {sym} ptr mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVQload {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVWQZX x:(MOVWloadidx1 [off] {sym} ptr idx mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWloadidx1 <v.Type> [off] {sym} ptr idx mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVWloadidx1 {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		mem := x.Args[2]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVWQZX x:(MOVWloadidx2 [off] {sym} ptr idx mem))
	// cond: x.Uses == 1 && clobber(x)
	// result: @x.Block (MOVWloadidx2 <v.Type> [off] {sym} ptr idx mem)
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVWloadidx2 {
			break
		}
		off := x.AuxInt
		sym := x.Aux
		_ = x.Args[2]
		ptr := x.Args[0]
		idx := x.Args[1]
		mem := x.Args[2]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx2, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = off
		v0.Aux = sym
		v0.AddArg(ptr)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (MOVWQZX (ANDLconst [c] x))
	// cond:
	// result: (ANDLconst [c & 0xffff] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64ANDLconst)
		v.AuxInt = c & 0xffff
		v.AddArg(x)
		return true
	}
	// match: (MOVWQZX x:(MOVWQZX _))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVWQZX {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MOVWQZX x:(MOVBQZX _))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpAMD64MOVBQZX {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWload_0(v *Value) bool {
	// match: (MOVWload [off] {sym} ptr (MOVWstore [off2] {sym2} ptr2 x _))
	// cond: sym == sym2 && off == off2 && isSamePtr(ptr, ptr2)
	// result: (MOVWQZX x)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVWstore {
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
		v.reset(OpAMD64MOVWQZX)
		v.AddArg(x)
		return true
	}
	// match: (MOVWload [off1] {sym} (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVWload  [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVWload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (LEAQ [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWload  [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (LEAQ1 [off2] {sym2} ptr idx) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWloadidx1 [off1+off2] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWloadidx1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (LEAQ2 [off2] {sym2} ptr idx) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWloadidx2 [off1+off2] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ2 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWloadidx2)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off] {sym} (ADDQ ptr idx) mem)
	// cond: ptr.Op != OpSB
	// result: (MOVWloadidx1 [off] {sym} ptr idx mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64MOVWloadidx1)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off1] {sym1} (LEAL [off2] {sym2} base) mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVWload  [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off1] {sym} (ADDLconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVWload  [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVWload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWloadidx1_0(v *Value) bool {
	// match: (MOVWloadidx1 [c] {sym} ptr (SHLQconst [1] idx) mem)
	// cond:
	// result: (MOVWloadidx2 [c] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWloadidx2)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx1 [c] {sym} (SHLQconst [1] idx) ptr mem)
	// cond:
	// result: (MOVWloadidx2 [c] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		idx := v_0.Args[0]
		ptr := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWloadidx2)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx1 [c] {sym} (ADDQconst [d] ptr) idx mem)
	// cond:
	// result: (MOVWloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx1 [c] {sym} idx (ADDQconst [d] ptr) mem)
	// cond:
	// result: (MOVWloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		idx := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		ptr := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx1 [c] {sym} ptr (ADDQconst [d] idx) mem)
	// cond:
	// result: (MOVWloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx1 [c] {sym} (ADDQconst [d] idx) ptr mem)
	// cond:
	// result: (MOVWloadidx1 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		idx := v_0.Args[0]
		ptr := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWloadidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWloadidx2_0(v *Value) bool {
	// match: (MOVWloadidx2 [c] {sym} (ADDQconst [d] ptr) idx mem)
	// cond:
	// result: (MOVWloadidx2 [c+d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWloadidx2)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWloadidx2 [c] {sym} ptr (ADDQconst [d] idx) mem)
	// cond:
	// result: (MOVWloadidx2 [c+2*d] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWloadidx2)
		v.AuxInt = c + 2*d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWstore_0(v *Value) bool {
	// match: (MOVWstore [off] {sym} ptr (MOVWQSX x) mem)
	// cond:
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVWQSX {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVWQZX x) mem)
	// cond:
	// result: (MOVWstore [off] {sym} ptr x mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVWQZX {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDQconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVWstore  [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off] {sym} ptr (MOVLconst [c]) mem)
	// cond: validOff(off)
	// result: (MOVWstoreconst [makeValAndOff(int64(int16(c)),off)] {sym} ptr mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		mem := v.Args[2]
		if !(validOff(off)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = makeValAndOff(int64(int16(c)), off)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (LEAQ [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWstore  [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (LEAQ1 [off2] {sym2} ptr idx) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWstoreidx1 [off1+off2] {mergeSym(sym1,sym2)} ptr idx val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWstoreidx1)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off1] {sym1} (LEAQ2 [off2] {sym2} ptr idx) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWstoreidx2 [off1+off2] {mergeSym(sym1,sym2)} ptr idx val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ2 {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWstoreidx2)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off] {sym} (ADDQ ptr idx) val mem)
	// cond: ptr.Op != OpSB
	// result: (MOVWstoreidx1 [off] {sym} ptr idx val mem)
	for {
		off := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		val := v.Args[1]
		mem := v.Args[2]
		if !(ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64MOVWstoreidx1)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [i] {s} p (SHRQconst [16] w) x:(MOVWstore [i-2] {s} p w mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVLstore [i-2] {s} p w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		if v_1.AuxInt != 16 {
			break
		}
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpAMD64MOVWstore {
			break
		}
		if x.AuxInt != i-2 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[2]
		if p != x.Args[0] {
			break
		}
		if w != x.Args[1] {
			break
		}
		mem := x.Args[2]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = i - 2
		v.Aux = s
		v.AddArg(p)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [i] {s} p (SHRQconst [j] w) x:(MOVWstore [i-2] {s} p w0:(SHRQconst [j-16] w) mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVLstore [i-2] {s} p w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		j := v_1.AuxInt
		w := v_1.Args[0]
		x := v.Args[2]
		if x.Op != OpAMD64MOVWstore {
			break
		}
		if x.AuxInt != i-2 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[2]
		if p != x.Args[0] {
			break
		}
		w0 := x.Args[1]
		if w0.Op != OpAMD64SHRQconst {
			break
		}
		if w0.AuxInt != j-16 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		mem := x.Args[2]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = i - 2
		v.Aux = s
		v.AddArg(p)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWstore_10(v *Value) bool {
	// match: (MOVWstore [off1] {sym1} (LEAL [off2] {sym2} base) val mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVWstore  [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDLconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVWstore  [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWstoreconst_0(v *Value) bool {
	// match: (MOVWstoreconst [sc] {s} (ADDQconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd(off)
	// result: (MOVWstoreconst [ValAndOff(sc).add(off)] {s} ptr mem)
	for {
		sc := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		off := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = s
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreconst [sc] {sym1} (LEAQ [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)
	// result: (MOVWstoreconst [ValAndOff(sc).add(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreconst [x] {sym1} (LEAQ1 [off] {sym2} ptr idx) mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVWstoreconstidx1 [ValAndOff(x).add(off)] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		x := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ1 {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconstidx1)
		v.AuxInt = ValAndOff(x).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreconst [x] {sym1} (LEAQ2 [off] {sym2} ptr idx) mem)
	// cond: canMergeSym(sym1, sym2)
	// result: (MOVWstoreconstidx2 [ValAndOff(x).add(off)] {mergeSym(sym1,sym2)} ptr idx mem)
	for {
		x := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAQ2 {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconstidx2)
		v.AuxInt = ValAndOff(x).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreconst [x] {sym} (ADDQ ptr idx) mem)
	// cond:
	// result: (MOVWstoreconstidx1 [x] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQ {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		idx := v_0.Args[1]
		mem := v.Args[1]
		v.reset(OpAMD64MOVWstoreconstidx1)
		v.AuxInt = x
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreconst [c] {s} p x:(MOVWstoreconst [a] {s} p mem))
	// cond: x.Uses == 1   && ValAndOff(a).Off() + 2 == ValAndOff(c).Off()   && clobber(x)
	// result: (MOVLstoreconst [makeValAndOff(ValAndOff(a).Val()&0xffff | ValAndOff(c).Val()<<16, ValAndOff(a).Off())] {s} p mem)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		p := v.Args[0]
		x := v.Args[1]
		if x.Op != OpAMD64MOVWstoreconst {
			break
		}
		a := x.AuxInt
		if x.Aux != s {
			break
		}
		_ = x.Args[1]
		if p != x.Args[0] {
			break
		}
		mem := x.Args[1]
		if !(x.Uses == 1 && ValAndOff(a).Off()+2 == ValAndOff(c).Off() && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = makeValAndOff(ValAndOff(a).Val()&0xffff|ValAndOff(c).Val()<<16, ValAndOff(a).Off())
		v.Aux = s
		v.AddArg(p)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreconst [sc] {sym1} (LEAL [off] {sym2} ptr) mem)
	// cond: canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)
	// result: (MOVWstoreconst [ValAndOff(sc).add(off)] {mergeSym(sym1, sym2)} ptr mem)
	for {
		sc := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64LEAL {
			break
		}
		off := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(canMergeSym(sym1, sym2) && ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreconst [sc] {s} (ADDLconst [off] ptr) mem)
	// cond: ValAndOff(sc).canAdd(off)
	// result: (MOVWstoreconst [ValAndOff(sc).add(off)] {s} ptr mem)
	for {
		sc := v.AuxInt
		s := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDLconst {
			break
		}
		off := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
		if !(ValAndOff(sc).canAdd(off)) {
			break
		}
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = ValAndOff(sc).add(off)
		v.Aux = s
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWstoreconstidx1_0(v *Value) bool {
	// match: (MOVWstoreconstidx1 [c] {sym} ptr (SHLQconst [1] idx) mem)
	// cond:
	// result: (MOVWstoreconstidx2 [c] {sym} ptr idx mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWstoreconstidx2)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreconstidx1 [x] {sym} (ADDQconst [c] ptr) idx mem)
	// cond:
	// result: (MOVWstoreconstidx1 [ValAndOff(x).add(c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWstoreconstidx1)
		v.AuxInt = ValAndOff(x).add(c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreconstidx1 [x] {sym} ptr (ADDQconst [c] idx) mem)
	// cond:
	// result: (MOVWstoreconstidx1 [ValAndOff(x).add(c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWstoreconstidx1)
		v.AuxInt = ValAndOff(x).add(c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreconstidx1 [c] {s} p i x:(MOVWstoreconstidx1 [a] {s} p i mem))
	// cond: x.Uses == 1   && ValAndOff(a).Off() + 2 == ValAndOff(c).Off()   && clobber(x)
	// result: (MOVLstoreconstidx1 [makeValAndOff(ValAndOff(a).Val()&0xffff | ValAndOff(c).Val()<<16, ValAndOff(a).Off())] {s} p i mem)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		i := v.Args[1]
		x := v.Args[2]
		if x.Op != OpAMD64MOVWstoreconstidx1 {
			break
		}
		a := x.AuxInt
		if x.Aux != s {
			break
		}
		_ = x.Args[2]
		if p != x.Args[0] {
			break
		}
		if i != x.Args[1] {
			break
		}
		mem := x.Args[2]
		if !(x.Uses == 1 && ValAndOff(a).Off()+2 == ValAndOff(c).Off() && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconstidx1)
		v.AuxInt = makeValAndOff(ValAndOff(a).Val()&0xffff|ValAndOff(c).Val()<<16, ValAndOff(a).Off())
		v.Aux = s
		v.AddArg(p)
		v.AddArg(i)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWstoreconstidx2_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MOVWstoreconstidx2 [x] {sym} (ADDQconst [c] ptr) idx mem)
	// cond:
	// result: (MOVWstoreconstidx2 [ValAndOff(x).add(c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWstoreconstidx2)
		v.AuxInt = ValAndOff(x).add(c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreconstidx2 [x] {sym} ptr (ADDQconst [c] idx) mem)
	// cond:
	// result: (MOVWstoreconstidx2 [ValAndOff(x).add(2*c)] {sym} ptr idx mem)
	for {
		x := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		idx := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWstoreconstidx2)
		v.AuxInt = ValAndOff(x).add(2 * c)
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreconstidx2 [c] {s} p i x:(MOVWstoreconstidx2 [a] {s} p i mem))
	// cond: x.Uses == 1   && ValAndOff(a).Off() + 2 == ValAndOff(c).Off()   && clobber(x)
	// result: (MOVLstoreconstidx1 [makeValAndOff(ValAndOff(a).Val()&0xffff | ValAndOff(c).Val()<<16, ValAndOff(a).Off())] {s} p (SHLQconst <i.Type> [1] i) mem)
	for {
		c := v.AuxInt
		s := v.Aux
		_ = v.Args[2]
		p := v.Args[0]
		i := v.Args[1]
		x := v.Args[2]
		if x.Op != OpAMD64MOVWstoreconstidx2 {
			break
		}
		a := x.AuxInt
		if x.Aux != s {
			break
		}
		_ = x.Args[2]
		if p != x.Args[0] {
			break
		}
		if i != x.Args[1] {
			break
		}
		mem := x.Args[2]
		if !(x.Uses == 1 && ValAndOff(a).Off()+2 == ValAndOff(c).Off() && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstoreconstidx1)
		v.AuxInt = makeValAndOff(ValAndOff(a).Val()&0xffff|ValAndOff(c).Val()<<16, ValAndOff(a).Off())
		v.Aux = s
		v.AddArg(p)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, i.Type)
		v0.AuxInt = 1
		v0.AddArg(i)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWstoreidx1_0(v *Value) bool {
	// match: (MOVWstoreidx1 [c] {sym} ptr (SHLQconst [1] idx) val mem)
	// cond:
	// result: (MOVWstoreidx2 [c] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVWstoreidx2)
		v.AuxInt = c
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx1 [c] {sym} (ADDQconst [d] ptr) idx val mem)
	// cond:
	// result: (MOVWstoreidx1 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVWstoreidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx1 [c] {sym} ptr (ADDQconst [d] idx) val mem)
	// cond:
	// result: (MOVWstoreidx1 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVWstoreidx1)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx1 [i] {s} p idx (SHRQconst [16] w) x:(MOVWstoreidx1 [i-2] {s} p idx w mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVLstoreidx1 [i-2] {s} p idx w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpAMD64SHRQconst {
			break
		}
		if v_2.AuxInt != 16 {
			break
		}
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpAMD64MOVWstoreidx1 {
			break
		}
		if x.AuxInt != i-2 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[3]
		if p != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		if w != x.Args[2] {
			break
		}
		mem := x.Args[3]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstoreidx1)
		v.AuxInt = i - 2
		v.Aux = s
		v.AddArg(p)
		v.AddArg(idx)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx1 [i] {s} p idx (SHRQconst [j] w) x:(MOVWstoreidx1 [i-2] {s} p idx w0:(SHRQconst [j-16] w) mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVLstoreidx1 [i-2] {s} p idx w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpAMD64SHRQconst {
			break
		}
		j := v_2.AuxInt
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpAMD64MOVWstoreidx1 {
			break
		}
		if x.AuxInt != i-2 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[3]
		if p != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		w0 := x.Args[2]
		if w0.Op != OpAMD64SHRQconst {
			break
		}
		if w0.AuxInt != j-16 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		mem := x.Args[3]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstoreidx1)
		v.AuxInt = i - 2
		v.Aux = s
		v.AddArg(p)
		v.AddArg(idx)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MOVWstoreidx2_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MOVWstoreidx2 [c] {sym} (ADDQconst [d] ptr) idx val mem)
	// cond:
	// result: (MOVWstoreidx2 [c+d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		d := v_0.AuxInt
		ptr := v_0.Args[0]
		idx := v.Args[1]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVWstoreidx2)
		v.AuxInt = c + d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx2 [c] {sym} ptr (ADDQconst [d] idx) val mem)
	// cond:
	// result: (MOVWstoreidx2 [c+2*d] {sym} ptr idx val mem)
	for {
		c := v.AuxInt
		sym := v.Aux
		_ = v.Args[3]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		d := v_1.AuxInt
		idx := v_1.Args[0]
		val := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64MOVWstoreidx2)
		v.AuxInt = c + 2*d
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(idx)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx2 [i] {s} p idx (SHRQconst [16] w) x:(MOVWstoreidx2 [i-2] {s} p idx w mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVLstoreidx1 [i-2] {s} p (SHLQconst <idx.Type> [1] idx) w mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpAMD64SHRQconst {
			break
		}
		if v_2.AuxInt != 16 {
			break
		}
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpAMD64MOVWstoreidx2 {
			break
		}
		if x.AuxInt != i-2 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[3]
		if p != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		if w != x.Args[2] {
			break
		}
		mem := x.Args[3]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstoreidx1)
		v.AuxInt = i - 2
		v.Aux = s
		v.AddArg(p)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, idx.Type)
		v0.AuxInt = 1
		v0.AddArg(idx)
		v.AddArg(v0)
		v.AddArg(w)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstoreidx2 [i] {s} p idx (SHRQconst [j] w) x:(MOVWstoreidx2 [i-2] {s} p idx w0:(SHRQconst [j-16] w) mem))
	// cond: x.Uses == 1   && clobber(x)
	// result: (MOVLstoreidx1 [i-2] {s} p (SHLQconst <idx.Type> [1] idx) w0 mem)
	for {
		i := v.AuxInt
		s := v.Aux
		_ = v.Args[3]
		p := v.Args[0]
		idx := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpAMD64SHRQconst {
			break
		}
		j := v_2.AuxInt
		w := v_2.Args[0]
		x := v.Args[3]
		if x.Op != OpAMD64MOVWstoreidx2 {
			break
		}
		if x.AuxInt != i-2 {
			break
		}
		if x.Aux != s {
			break
		}
		_ = x.Args[3]
		if p != x.Args[0] {
			break
		}
		if idx != x.Args[1] {
			break
		}
		w0 := x.Args[2]
		if w0.Op != OpAMD64SHRQconst {
			break
		}
		if w0.AuxInt != j-16 {
			break
		}
		if w != w0.Args[0] {
			break
		}
		mem := x.Args[3]
		if !(x.Uses == 1 && clobber(x)) {
			break
		}
		v.reset(OpAMD64MOVLstoreidx1)
		v.AuxInt = i - 2
		v.Aux = s
		v.AddArg(p)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, idx.Type)
		v0.AuxInt = 1
		v0.AddArg(idx)
		v.AddArg(v0)
		v.AddArg(w0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULL_0(v *Value) bool {
	// match: (MULL x (MOVLconst [c]))
	// cond:
	// result: (MULLconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64MULLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (MULL (MOVLconst [c]) x)
	// cond:
	// result: (MULLconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpAMD64MULLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULLconst_0(v *Value) bool {
	// match: (MULLconst [c] (MULLconst [d] x))
	// cond:
	// result: (MULLconst [int64(int32(c * d))] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MULLconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64MULLconst)
		v.AuxInt = int64(int32(c * d))
		v.AddArg(x)
		return true
	}
	// match: (MULLconst [c] (MOVLconst [d]))
	// cond:
	// result: (MOVLconst [int64(int32(c*d))])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int64(int32(c * d))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULQ_0(v *Value) bool {
	// match: (MULQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (MULQconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64MULQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (MULQ (MOVQconst [c]) x)
	// cond: is32Bit(c)
	// result: (MULQconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64MULQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULQconst_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MULQconst [c] (MULQconst [d] x))
	// cond: is32Bit(c*d)
	// result: (MULQconst [c * d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MULQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		if !(is32Bit(c * d)) {
			break
		}
		v.reset(OpAMD64MULQconst)
		v.AuxInt = c * d
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [-1] x)
	// cond:
	// result: (NEGQ x)
	for {
		if v.AuxInt != -1 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64NEGQ)
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [0] _)
	// cond:
	// result: (MOVQconst [0])
	for {
		if v.AuxInt != 0 {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = 0
		return true
	}
	// match: (MULQconst [1] x)
	// cond:
	// result: x
	for {
		if v.AuxInt != 1 {
			break
		}
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [3] x)
	// cond:
	// result: (LEAQ2 x x)
	for {
		if v.AuxInt != 3 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64LEAQ2)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [5] x)
	// cond:
	// result: (LEAQ4 x x)
	for {
		if v.AuxInt != 5 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64LEAQ4)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [7] x)
	// cond:
	// result: (LEAQ8 (NEGQ <v.Type> x) x)
	for {
		if v.AuxInt != 7 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64LEAQ8)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, v.Type)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [9] x)
	// cond:
	// result: (LEAQ8 x x)
	for {
		if v.AuxInt != 9 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64LEAQ8)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [11] x)
	// cond:
	// result: (LEAQ2 x (LEAQ4 <v.Type> x x))
	for {
		if v.AuxInt != 11 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64LEAQ2)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ4, v.Type)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [13] x)
	// cond:
	// result: (LEAQ4 x (LEAQ2 <v.Type> x x))
	for {
		if v.AuxInt != 13 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64LEAQ4)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ2, v.Type)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULQconst_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MULQconst [21] x)
	// cond:
	// result: (LEAQ4 x (LEAQ4 <v.Type> x x))
	for {
		if v.AuxInt != 21 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64LEAQ4)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ4, v.Type)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [25] x)
	// cond:
	// result: (LEAQ8 x (LEAQ2 <v.Type> x x))
	for {
		if v.AuxInt != 25 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64LEAQ8)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ2, v.Type)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [37] x)
	// cond:
	// result: (LEAQ4 x (LEAQ8 <v.Type> x x))
	for {
		if v.AuxInt != 37 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64LEAQ4)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ8, v.Type)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [41] x)
	// cond:
	// result: (LEAQ8 x (LEAQ4 <v.Type> x x))
	for {
		if v.AuxInt != 41 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64LEAQ8)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ4, v.Type)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [73] x)
	// cond:
	// result: (LEAQ8 x (LEAQ8 <v.Type> x x))
	for {
		if v.AuxInt != 73 {
			break
		}
		x := v.Args[0]
		v.reset(OpAMD64LEAQ8)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ8, v.Type)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: isPowerOfTwo(c)
	// result: (SHLQconst [log2(c)] x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = log2(c)
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: isPowerOfTwo(c+1) && c >= 15
	// result: (SUBQ (SHLQconst <v.Type> [log2(c+1)] x) x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(isPowerOfTwo(c+1) && c >= 15) {
			break
		}
		v.reset(OpAMD64SUBQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v0.AuxInt = log2(c + 1)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: isPowerOfTwo(c-1) && c >= 17
	// result: (LEAQ1 (SHLQconst <v.Type> [log2(c-1)] x) x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(isPowerOfTwo(c-1) && c >= 17) {
			break
		}
		v.reset(OpAMD64LEAQ1)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v0.AuxInt = log2(c - 1)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: isPowerOfTwo(c-2) && c >= 34
	// result: (LEAQ2 (SHLQconst <v.Type> [log2(c-2)] x) x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(isPowerOfTwo(c-2) && c >= 34) {
			break
		}
		v.reset(OpAMD64LEAQ2)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v0.AuxInt = log2(c - 2)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: isPowerOfTwo(c-4) && c >= 68
	// result: (LEAQ4 (SHLQconst <v.Type> [log2(c-4)] x) x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(isPowerOfTwo(c-4) && c >= 68) {
			break
		}
		v.reset(OpAMD64LEAQ4)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v0.AuxInt = log2(c - 4)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULQconst_20(v *Value) bool {
	b := v.Block
	_ = b
	// match: (MULQconst [c] x)
	// cond: isPowerOfTwo(c-8) && c >= 136
	// result: (LEAQ8 (SHLQconst <v.Type> [log2(c-8)] x) x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(isPowerOfTwo(c-8) && c >= 136) {
			break
		}
		v.reset(OpAMD64LEAQ8)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v0.AuxInt = log2(c - 8)
		v0.AddArg(x)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: c%3 == 0 && isPowerOfTwo(c/3)
	// result: (SHLQconst [log2(c/3)] (LEAQ2 <v.Type> x x))
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(c%3 == 0 && isPowerOfTwo(c/3)) {
			break
		}
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = log2(c / 3)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ2, v.Type)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: c%5 == 0 && isPowerOfTwo(c/5)
	// result: (SHLQconst [log2(c/5)] (LEAQ4 <v.Type> x x))
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(c%5 == 0 && isPowerOfTwo(c/5)) {
			break
		}
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = log2(c / 5)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ4, v.Type)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [c] x)
	// cond: c%9 == 0 && isPowerOfTwo(c/9)
	// result: (SHLQconst [log2(c/9)] (LEAQ8 <v.Type> x x))
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(c%9 == 0 && isPowerOfTwo(c/9)) {
			break
		}
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = log2(c / 9)
		v0 := b.NewValue0(v.Pos, OpAMD64LEAQ8, v.Type)
		v0.AddArg(x)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (MULQconst [c] (MOVQconst [d]))
	// cond:
	// result: (MOVQconst [c*d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = c * d
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULSD_0(v *Value) bool {
	// match: (MULSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (MULSDmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVSDload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64MULSDmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MULSD l:(MOVSDload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (MULSDmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		l := v.Args[0]
		if l.Op != OpAMD64MOVSDload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		x := v.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64MULSDmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64MULSS_0(v *Value) bool {
	// match: (MULSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (MULSSmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVSSload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64MULSSmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (MULSS l:(MOVSSload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (MULSSmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		l := v.Args[0]
		if l.Op != OpAMD64MOVSSload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		x := v.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64MULSSmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64NEGL_0(v *Value) bool {
	// match: (NEGL (MOVLconst [c]))
	// cond:
	// result: (MOVLconst [int64(int32(-c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = int64(int32(-c))
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64NEGQ_0(v *Value) bool {
	// match: (NEGQ (MOVQconst [c]))
	// cond:
	// result: (MOVQconst [-c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = -c
		return true
	}
	// match: (NEGQ (ADDQconst [c] (NEGQ x)))
	// cond: c != -(1<<31)
	// result: (ADDQconst [-c] x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_0.AuxInt
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64NEGQ {
			break
		}
		x := v_0_0.Args[0]
		if !(c != -(1 << 31)) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = -c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64NOTL_0(v *Value) bool {
	// match: (NOTL (MOVLconst [c]))
	// cond:
	// result: (MOVLconst [^c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = ^c
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64NOTQ_0(v *Value) bool {
	// match: (NOTQ (MOVQconst [c]))
	// cond:
	// result: (MOVQconst [^c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = ^c
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_0(v *Value) bool {
	// match: (ORL x (MOVLconst [c]))
	// cond:
	// result: (ORLconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ORLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORL (MOVLconst [c]) x)
	// cond:
	// result: (ORLconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpAMD64ORLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORL (SHLLconst x [c]) (SHRLconst x [d]))
	// cond: d==32-c
	// result: (ROLLconst x [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRLconst {
			break
		}
		d := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 32-c) {
			break
		}
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORL (SHRLconst x [d]) (SHLLconst x [c]))
	// cond: d==32-c
	// result: (ROLLconst x [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRLconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 32-c) {
			break
		}
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORL <t> (SHLLconst x [c]) (SHRWconst x [d]))
	// cond: d==16-c && c < 16 && t.Size() == 2
	// result: (ROLWconst x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRWconst {
			break
		}
		d := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 16-c && c < 16 && t.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORL <t> (SHRWconst x [d]) (SHLLconst x [c]))
	// cond: d==16-c && c < 16 && t.Size() == 2
	// result: (ROLWconst x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRWconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 16-c && c < 16 && t.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORL <t> (SHLLconst x [c]) (SHRBconst x [d]))
	// cond: d==8-c  && c < 8 && t.Size() == 1
	// result: (ROLBconst x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRBconst {
			break
		}
		d := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 8-c && c < 8 && t.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORL <t> (SHRBconst x [d]) (SHLLconst x [c]))
	// cond: d==8-c  && c < 8 && t.Size() == 1
	// result: (ROLBconst x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 8-c && c < 8 && t.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORL (SHLL x y) (ANDL (SHRL x (NEGQ y)) (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [31]) [-32])) [32]))))
	// cond:
	// result: (ROLL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SHRL {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpAMD64NEGQ {
			break
		}
		if y != v_1_0_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_1_1_0.AuxInt != 32 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_1_1_0_0_0 := v_1_1_0_0.Args[0]
		if v_1_1_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_1_0_0_0.AuxInt != -32 {
			break
		}
		v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
		if v_1_1_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1_0_0_0_0.AuxInt != 31 {
			break
		}
		if y != v_1_1_0_0_0_0.Args[0] {
			break
		}
		v.reset(OpAMD64ROLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x y) (ANDL (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [31]) [-32])) [32])) (SHRL x (NEGQ y))))
	// cond:
	// result: (ROLL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_1_0_0.AuxInt != 32 {
			break
		}
		v_1_0_0_0 := v_1_0_0.Args[0]
		if v_1_0_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_1_0_0_0_0 := v_1_0_0_0.Args[0]
		if v_1_0_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_0_0_0_0.AuxInt != -32 {
			break
		}
		v_1_0_0_0_0_0 := v_1_0_0_0_0.Args[0]
		if v_1_0_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_0_0_0_0_0.AuxInt != 31 {
			break
		}
		if y != v_1_0_0_0_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SHRL {
			break
		}
		_ = v_1_1.Args[1]
		if x != v_1_1.Args[0] {
			break
		}
		v_1_1_1 := v_1_1.Args[1]
		if v_1_1_1.Op != OpAMD64NEGQ {
			break
		}
		if y != v_1_1_1.Args[0] {
			break
		}
		v.reset(OpAMD64ROLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_10(v *Value) bool {
	// match: (ORL (ANDL (SHRL x (NEGQ y)) (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [31]) [-32])) [32]))) (SHLL x y))
	// cond:
	// result: (ROLL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHRL {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_0_0_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_0_1_0.AuxInt != 32 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_1_0_0_0 := v_0_1_0_0.Args[0]
		if v_0_1_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_1_0_0_0.AuxInt != -32 {
			break
		}
		v_0_1_0_0_0_0 := v_0_1_0_0_0.Args[0]
		if v_0_1_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1_0_0_0_0.AuxInt != 31 {
			break
		}
		if y != v_0_1_0_0_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64ROLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [31]) [-32])) [32])) (SHRL x (NEGQ y))) (SHLL x y))
	// cond:
	// result: (ROLL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_0_0_0.AuxInt != 32 {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_0_0_0_0.AuxInt != -32 {
			break
		}
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_0_0_0_0_0.AuxInt != 31 {
			break
		}
		y := v_0_0_0_0_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHRL {
			break
		}
		_ = v_0_1.Args[1]
		x := v_0_1.Args[0]
		v_0_1_1 := v_0_1.Args[1]
		if v_0_1_1.Op != OpAMD64NEGQ {
			break
		}
		if y != v_0_1_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64ROLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x y) (ANDL (SHRL x (NEGL y)) (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [31]) [-32])) [32]))))
	// cond:
	// result: (ROLL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SHRL {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpAMD64NEGL {
			break
		}
		if y != v_1_0_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_1_1_0.AuxInt != 32 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64NEGL {
			break
		}
		v_1_1_0_0_0 := v_1_1_0_0.Args[0]
		if v_1_1_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_1_0_0_0.AuxInt != -32 {
			break
		}
		v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
		if v_1_1_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1_0_0_0_0.AuxInt != 31 {
			break
		}
		if y != v_1_1_0_0_0_0.Args[0] {
			break
		}
		v.reset(OpAMD64ROLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x y) (ANDL (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [31]) [-32])) [32])) (SHRL x (NEGL y))))
	// cond:
	// result: (ROLL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_1_0_0.AuxInt != 32 {
			break
		}
		v_1_0_0_0 := v_1_0_0.Args[0]
		if v_1_0_0_0.Op != OpAMD64NEGL {
			break
		}
		v_1_0_0_0_0 := v_1_0_0_0.Args[0]
		if v_1_0_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_0_0_0_0.AuxInt != -32 {
			break
		}
		v_1_0_0_0_0_0 := v_1_0_0_0_0.Args[0]
		if v_1_0_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_0_0_0_0_0.AuxInt != 31 {
			break
		}
		if y != v_1_0_0_0_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SHRL {
			break
		}
		_ = v_1_1.Args[1]
		if x != v_1_1.Args[0] {
			break
		}
		v_1_1_1 := v_1_1.Args[1]
		if v_1_1_1.Op != OpAMD64NEGL {
			break
		}
		if y != v_1_1_1.Args[0] {
			break
		}
		v.reset(OpAMD64ROLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SHRL x (NEGL y)) (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [31]) [-32])) [32]))) (SHLL x y))
	// cond:
	// result: (ROLL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHRL {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpAMD64NEGL {
			break
		}
		y := v_0_0_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_0_1_0.AuxInt != 32 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64NEGL {
			break
		}
		v_0_1_0_0_0 := v_0_1_0_0.Args[0]
		if v_0_1_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_1_0_0_0.AuxInt != -32 {
			break
		}
		v_0_1_0_0_0_0 := v_0_1_0_0_0.Args[0]
		if v_0_1_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1_0_0_0_0.AuxInt != 31 {
			break
		}
		if y != v_0_1_0_0_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64ROLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [31]) [-32])) [32])) (SHRL x (NEGL y))) (SHLL x y))
	// cond:
	// result: (ROLL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_0_0_0.AuxInt != 32 {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAMD64NEGL {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_0_0_0_0.AuxInt != -32 {
			break
		}
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_0_0_0_0_0.AuxInt != 31 {
			break
		}
		y := v_0_0_0_0_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHRL {
			break
		}
		_ = v_0_1.Args[1]
		x := v_0_1.Args[0]
		v_0_1_1 := v_0_1.Args[1]
		if v_0_1_1.Op != OpAMD64NEGL {
			break
		}
		if y != v_0_1_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64ROLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHRL x y) (ANDL (SHLL x (NEGQ y)) (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [31]) [-32])) [32]))))
	// cond:
	// result: (RORL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpAMD64NEGQ {
			break
		}
		if y != v_1_0_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_1_1_0.AuxInt != 32 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_1_1_0_0_0 := v_1_1_0_0.Args[0]
		if v_1_1_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_1_0_0_0.AuxInt != -32 {
			break
		}
		v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
		if v_1_1_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1_0_0_0_0.AuxInt != 31 {
			break
		}
		if y != v_1_1_0_0_0_0.Args[0] {
			break
		}
		v.reset(OpAMD64RORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHRL x y) (ANDL (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [31]) [-32])) [32])) (SHLL x (NEGQ y))))
	// cond:
	// result: (RORL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_1_0_0.AuxInt != 32 {
			break
		}
		v_1_0_0_0 := v_1_0_0.Args[0]
		if v_1_0_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_1_0_0_0_0 := v_1_0_0_0.Args[0]
		if v_1_0_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_0_0_0_0.AuxInt != -32 {
			break
		}
		v_1_0_0_0_0_0 := v_1_0_0_0_0.Args[0]
		if v_1_0_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_0_0_0_0_0.AuxInt != 31 {
			break
		}
		if y != v_1_0_0_0_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1_1.Args[1]
		if x != v_1_1.Args[0] {
			break
		}
		v_1_1_1 := v_1_1.Args[1]
		if v_1_1_1.Op != OpAMD64NEGQ {
			break
		}
		if y != v_1_1_1.Args[0] {
			break
		}
		v.reset(OpAMD64RORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SHLL x (NEGQ y)) (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [31]) [-32])) [32]))) (SHRL x y))
	// cond:
	// result: (RORL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_0_0_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_0_1_0.AuxInt != 32 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_1_0_0_0 := v_0_1_0_0.Args[0]
		if v_0_1_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_1_0_0_0.AuxInt != -32 {
			break
		}
		v_0_1_0_0_0_0 := v_0_1_0_0_0.Args[0]
		if v_0_1_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1_0_0_0_0.AuxInt != 31 {
			break
		}
		if y != v_0_1_0_0_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64RORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [31]) [-32])) [32])) (SHLL x (NEGQ y))) (SHRL x y))
	// cond:
	// result: (RORL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_0_0_0.AuxInt != 32 {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_0_0_0_0.AuxInt != -32 {
			break
		}
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_0_0_0_0_0.AuxInt != 31 {
			break
		}
		y := v_0_0_0_0_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_0_1.Args[1]
		x := v_0_1.Args[0]
		v_0_1_1 := v_0_1.Args[1]
		if v_0_1_1.Op != OpAMD64NEGQ {
			break
		}
		if y != v_0_1_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64RORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_20(v *Value) bool {
	// match: (ORL (SHRL x y) (ANDL (SHLL x (NEGL y)) (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [31]) [-32])) [32]))))
	// cond:
	// result: (RORL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpAMD64NEGL {
			break
		}
		if y != v_1_0_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_1_1_0.AuxInt != 32 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64NEGL {
			break
		}
		v_1_1_0_0_0 := v_1_1_0_0.Args[0]
		if v_1_1_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_1_0_0_0.AuxInt != -32 {
			break
		}
		v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
		if v_1_1_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1_0_0_0_0.AuxInt != 31 {
			break
		}
		if y != v_1_1_0_0_0_0.Args[0] {
			break
		}
		v.reset(OpAMD64RORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHRL x y) (ANDL (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [31]) [-32])) [32])) (SHLL x (NEGL y))))
	// cond:
	// result: (RORL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_1_0_0.AuxInt != 32 {
			break
		}
		v_1_0_0_0 := v_1_0_0.Args[0]
		if v_1_0_0_0.Op != OpAMD64NEGL {
			break
		}
		v_1_0_0_0_0 := v_1_0_0_0.Args[0]
		if v_1_0_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_0_0_0_0.AuxInt != -32 {
			break
		}
		v_1_0_0_0_0_0 := v_1_0_0_0_0.Args[0]
		if v_1_0_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_0_0_0_0_0.AuxInt != 31 {
			break
		}
		if y != v_1_0_0_0_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1_1.Args[1]
		if x != v_1_1.Args[0] {
			break
		}
		v_1_1_1 := v_1_1.Args[1]
		if v_1_1_1.Op != OpAMD64NEGL {
			break
		}
		if y != v_1_1_1.Args[0] {
			break
		}
		v.reset(OpAMD64RORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SHLL x (NEGL y)) (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [31]) [-32])) [32]))) (SHRL x y))
	// cond:
	// result: (RORL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpAMD64NEGL {
			break
		}
		y := v_0_0_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_0_1_0.AuxInt != 32 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64NEGL {
			break
		}
		v_0_1_0_0_0 := v_0_1_0_0.Args[0]
		if v_0_1_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_1_0_0_0.AuxInt != -32 {
			break
		}
		v_0_1_0_0_0_0 := v_0_1_0_0_0.Args[0]
		if v_0_1_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1_0_0_0_0.AuxInt != 31 {
			break
		}
		if y != v_0_1_0_0_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64RORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [31]) [-32])) [32])) (SHLL x (NEGL y))) (SHRL x y))
	// cond:
	// result: (RORL x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_0_0_0.AuxInt != 32 {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAMD64NEGL {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_0_0_0_0.AuxInt != -32 {
			break
		}
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_0_0_0_0_0.AuxInt != 31 {
			break
		}
		y := v_0_0_0_0_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_0_1.Args[1]
		x := v_0_1.Args[0]
		v_0_1_1 := v_0_1.Args[1]
		if v_0_1_1.Op != OpAMD64NEGL {
			break
		}
		if y != v_0_1_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64RORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x (ANDQconst y [15])) (ANDL (SHRW x (NEGQ (ADDQconst (ANDQconst y [15]) [-16]))) (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [15]) [-16])) [16]))))
	// cond: v.Type.Size() == 2
	// result: (ROLW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1.AuxInt != 15 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SHRW {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpAMD64NEGQ {
			break
		}
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_0_1_0.AuxInt != -16 {
			break
		}
		v_1_0_1_0_0 := v_1_0_1_0.Args[0]
		if v_1_0_1_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_0_1_0_0.AuxInt != 15 {
			break
		}
		if y != v_1_0_1_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_1_1_0.AuxInt != 16 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_1_1_0_0_0 := v_1_1_0_0.Args[0]
		if v_1_1_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_1_0_0_0.AuxInt != -16 {
			break
		}
		v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
		if v_1_1_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1_0_0_0_0.AuxInt != 15 {
			break
		}
		if y != v_1_1_0_0_0_0.Args[0] {
			break
		}
		if !(v.Type.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x (ANDQconst y [15])) (ANDL (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [15]) [-16])) [16])) (SHRW x (NEGQ (ADDQconst (ANDQconst y [15]) [-16])))))
	// cond: v.Type.Size() == 2
	// result: (ROLW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1.AuxInt != 15 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_1_0_0.AuxInt != 16 {
			break
		}
		v_1_0_0_0 := v_1_0_0.Args[0]
		if v_1_0_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_1_0_0_0_0 := v_1_0_0_0.Args[0]
		if v_1_0_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_0_0_0_0.AuxInt != -16 {
			break
		}
		v_1_0_0_0_0_0 := v_1_0_0_0_0.Args[0]
		if v_1_0_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_0_0_0_0_0.AuxInt != 15 {
			break
		}
		if y != v_1_0_0_0_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SHRW {
			break
		}
		_ = v_1_1.Args[1]
		if x != v_1_1.Args[0] {
			break
		}
		v_1_1_1 := v_1_1.Args[1]
		if v_1_1_1.Op != OpAMD64NEGQ {
			break
		}
		v_1_1_1_0 := v_1_1_1.Args[0]
		if v_1_1_1_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_1_1_0.AuxInt != -16 {
			break
		}
		v_1_1_1_0_0 := v_1_1_1_0.Args[0]
		if v_1_1_1_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1_1_0_0.AuxInt != 15 {
			break
		}
		if y != v_1_1_1_0_0.Args[0] {
			break
		}
		if !(v.Type.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SHRW x (NEGQ (ADDQconst (ANDQconst y [15]) [-16]))) (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [15]) [-16])) [16]))) (SHLL x (ANDQconst y [15])))
	// cond: v.Type.Size() == 2
	// result: (ROLW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHRW {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpAMD64NEGQ {
			break
		}
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_0_1_0.AuxInt != -16 {
			break
		}
		v_0_0_1_0_0 := v_0_0_1_0.Args[0]
		if v_0_0_1_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_0_1_0_0.AuxInt != 15 {
			break
		}
		y := v_0_0_1_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_0_1_0.AuxInt != 16 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_1_0_0_0 := v_0_1_0_0.Args[0]
		if v_0_1_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_1_0_0_0.AuxInt != -16 {
			break
		}
		v_0_1_0_0_0_0 := v_0_1_0_0_0.Args[0]
		if v_0_1_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1_0_0_0_0.AuxInt != 15 {
			break
		}
		if y != v_0_1_0_0_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1.AuxInt != 15 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(v.Type.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [15]) [-16])) [16])) (SHRW x (NEGQ (ADDQconst (ANDQconst y [15]) [-16])))) (SHLL x (ANDQconst y [15])))
	// cond: v.Type.Size() == 2
	// result: (ROLW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_0_0_0.AuxInt != 16 {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_0_0_0_0.AuxInt != -16 {
			break
		}
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_0_0_0_0_0.AuxInt != 15 {
			break
		}
		y := v_0_0_0_0_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHRW {
			break
		}
		_ = v_0_1.Args[1]
		x := v_0_1.Args[0]
		v_0_1_1 := v_0_1.Args[1]
		if v_0_1_1.Op != OpAMD64NEGQ {
			break
		}
		v_0_1_1_0 := v_0_1_1.Args[0]
		if v_0_1_1_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_1_1_0.AuxInt != -16 {
			break
		}
		v_0_1_1_0_0 := v_0_1_1_0.Args[0]
		if v_0_1_1_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1_1_0_0.AuxInt != 15 {
			break
		}
		if y != v_0_1_1_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1.AuxInt != 15 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(v.Type.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x (ANDLconst y [15])) (ANDL (SHRW x (NEGL (ADDLconst (ANDLconst y [15]) [-16]))) (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [15]) [-16])) [16]))))
	// cond: v.Type.Size() == 2
	// result: (ROLW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1.AuxInt != 15 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SHRW {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpAMD64NEGL {
			break
		}
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_0_1_0.AuxInt != -16 {
			break
		}
		v_1_0_1_0_0 := v_1_0_1_0.Args[0]
		if v_1_0_1_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_0_1_0_0.AuxInt != 15 {
			break
		}
		if y != v_1_0_1_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_1_1_0.AuxInt != 16 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64NEGL {
			break
		}
		v_1_1_0_0_0 := v_1_1_0_0.Args[0]
		if v_1_1_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_1_0_0_0.AuxInt != -16 {
			break
		}
		v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
		if v_1_1_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1_0_0_0_0.AuxInt != 15 {
			break
		}
		if y != v_1_1_0_0_0_0.Args[0] {
			break
		}
		if !(v.Type.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x (ANDLconst y [15])) (ANDL (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [15]) [-16])) [16])) (SHRW x (NEGL (ADDLconst (ANDLconst y [15]) [-16])))))
	// cond: v.Type.Size() == 2
	// result: (ROLW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1.AuxInt != 15 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_1_0_0.AuxInt != 16 {
			break
		}
		v_1_0_0_0 := v_1_0_0.Args[0]
		if v_1_0_0_0.Op != OpAMD64NEGL {
			break
		}
		v_1_0_0_0_0 := v_1_0_0_0.Args[0]
		if v_1_0_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_0_0_0_0.AuxInt != -16 {
			break
		}
		v_1_0_0_0_0_0 := v_1_0_0_0_0.Args[0]
		if v_1_0_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_0_0_0_0_0.AuxInt != 15 {
			break
		}
		if y != v_1_0_0_0_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SHRW {
			break
		}
		_ = v_1_1.Args[1]
		if x != v_1_1.Args[0] {
			break
		}
		v_1_1_1 := v_1_1.Args[1]
		if v_1_1_1.Op != OpAMD64NEGL {
			break
		}
		v_1_1_1_0 := v_1_1_1.Args[0]
		if v_1_1_1_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_1_1_0.AuxInt != -16 {
			break
		}
		v_1_1_1_0_0 := v_1_1_1_0.Args[0]
		if v_1_1_1_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1_1_0_0.AuxInt != 15 {
			break
		}
		if y != v_1_1_1_0_0.Args[0] {
			break
		}
		if !(v.Type.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_30(v *Value) bool {
	// match: (ORL (ANDL (SHRW x (NEGL (ADDLconst (ANDLconst y [15]) [-16]))) (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [15]) [-16])) [16]))) (SHLL x (ANDLconst y [15])))
	// cond: v.Type.Size() == 2
	// result: (ROLW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHRW {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpAMD64NEGL {
			break
		}
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_0_1_0.AuxInt != -16 {
			break
		}
		v_0_0_1_0_0 := v_0_0_1_0.Args[0]
		if v_0_0_1_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_0_1_0_0.AuxInt != 15 {
			break
		}
		y := v_0_0_1_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_0_1_0.AuxInt != 16 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64NEGL {
			break
		}
		v_0_1_0_0_0 := v_0_1_0_0.Args[0]
		if v_0_1_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_1_0_0_0.AuxInt != -16 {
			break
		}
		v_0_1_0_0_0_0 := v_0_1_0_0_0.Args[0]
		if v_0_1_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1_0_0_0_0.AuxInt != 15 {
			break
		}
		if y != v_0_1_0_0_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1.AuxInt != 15 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(v.Type.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [15]) [-16])) [16])) (SHRW x (NEGL (ADDLconst (ANDLconst y [15]) [-16])))) (SHLL x (ANDLconst y [15])))
	// cond: v.Type.Size() == 2
	// result: (ROLW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_0_0_0.AuxInt != 16 {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAMD64NEGL {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_0_0_0_0.AuxInt != -16 {
			break
		}
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_0_0_0_0_0.AuxInt != 15 {
			break
		}
		y := v_0_0_0_0_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHRW {
			break
		}
		_ = v_0_1.Args[1]
		x := v_0_1.Args[0]
		v_0_1_1 := v_0_1.Args[1]
		if v_0_1_1.Op != OpAMD64NEGL {
			break
		}
		v_0_1_1_0 := v_0_1_1.Args[0]
		if v_0_1_1_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_1_1_0.AuxInt != -16 {
			break
		}
		v_0_1_1_0_0 := v_0_1_1_0.Args[0]
		if v_0_1_1_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1_1_0_0.AuxInt != 15 {
			break
		}
		if y != v_0_1_1_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1.AuxInt != 15 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(v.Type.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHRW x (ANDQconst y [15])) (SHLL x (NEGQ (ADDQconst (ANDQconst y [15]) [-16]))))
	// cond: v.Type.Size() == 2
	// result: (RORW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRW {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1.AuxInt != 15 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64NEGQ {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_1_0.AuxInt != -16 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1_0_0.AuxInt != 15 {
			break
		}
		if y != v_1_1_0_0.Args[0] {
			break
		}
		if !(v.Type.Size() == 2) {
			break
		}
		v.reset(OpAMD64RORW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x (NEGQ (ADDQconst (ANDQconst y [15]) [-16]))) (SHRW x (ANDQconst y [15])))
	// cond: v.Type.Size() == 2
	// result: (RORW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64NEGQ {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_1_0.AuxInt != -16 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1_0_0.AuxInt != 15 {
			break
		}
		y := v_0_1_0_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRW {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1.AuxInt != 15 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(v.Type.Size() == 2) {
			break
		}
		v.reset(OpAMD64RORW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHRW x (ANDLconst y [15])) (SHLL x (NEGL (ADDLconst (ANDLconst y [15]) [-16]))))
	// cond: v.Type.Size() == 2
	// result: (RORW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRW {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1.AuxInt != 15 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64NEGL {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_1_0.AuxInt != -16 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1_0_0.AuxInt != 15 {
			break
		}
		if y != v_1_1_0_0.Args[0] {
			break
		}
		if !(v.Type.Size() == 2) {
			break
		}
		v.reset(OpAMD64RORW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x (NEGL (ADDLconst (ANDLconst y [15]) [-16]))) (SHRW x (ANDLconst y [15])))
	// cond: v.Type.Size() == 2
	// result: (RORW x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64NEGL {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_1_0.AuxInt != -16 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1_0_0.AuxInt != 15 {
			break
		}
		y := v_0_1_0_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRW {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1.AuxInt != 15 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(v.Type.Size() == 2) {
			break
		}
		v.reset(OpAMD64RORW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x (ANDQconst y [ 7])) (ANDL (SHRB x (NEGQ (ADDQconst (ANDQconst y [ 7]) [ -8]))) (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [ 7]) [ -8])) [ 8]))))
	// cond: v.Type.Size() == 1
	// result: (ROLB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1.AuxInt != 7 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SHRB {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpAMD64NEGQ {
			break
		}
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_0_1_0.AuxInt != -8 {
			break
		}
		v_1_0_1_0_0 := v_1_0_1_0.Args[0]
		if v_1_0_1_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_0_1_0_0.AuxInt != 7 {
			break
		}
		if y != v_1_0_1_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_1_1_0.AuxInt != 8 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_1_1_0_0_0 := v_1_1_0_0.Args[0]
		if v_1_1_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_1_0_0_0.AuxInt != -8 {
			break
		}
		v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
		if v_1_1_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1_0_0_0_0.AuxInt != 7 {
			break
		}
		if y != v_1_1_0_0_0_0.Args[0] {
			break
		}
		if !(v.Type.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x (ANDQconst y [ 7])) (ANDL (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [ 7]) [ -8])) [ 8])) (SHRB x (NEGQ (ADDQconst (ANDQconst y [ 7]) [ -8])))))
	// cond: v.Type.Size() == 1
	// result: (ROLB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1.AuxInt != 7 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_1_0_0.AuxInt != 8 {
			break
		}
		v_1_0_0_0 := v_1_0_0.Args[0]
		if v_1_0_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_1_0_0_0_0 := v_1_0_0_0.Args[0]
		if v_1_0_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_0_0_0_0.AuxInt != -8 {
			break
		}
		v_1_0_0_0_0_0 := v_1_0_0_0_0.Args[0]
		if v_1_0_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_0_0_0_0_0.AuxInt != 7 {
			break
		}
		if y != v_1_0_0_0_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SHRB {
			break
		}
		_ = v_1_1.Args[1]
		if x != v_1_1.Args[0] {
			break
		}
		v_1_1_1 := v_1_1.Args[1]
		if v_1_1_1.Op != OpAMD64NEGQ {
			break
		}
		v_1_1_1_0 := v_1_1_1.Args[0]
		if v_1_1_1_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_1_1_0.AuxInt != -8 {
			break
		}
		v_1_1_1_0_0 := v_1_1_1_0.Args[0]
		if v_1_1_1_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1_1_0_0.AuxInt != 7 {
			break
		}
		if y != v_1_1_1_0_0.Args[0] {
			break
		}
		if !(v.Type.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SHRB x (NEGQ (ADDQconst (ANDQconst y [ 7]) [ -8]))) (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [ 7]) [ -8])) [ 8]))) (SHLL x (ANDQconst y [ 7])))
	// cond: v.Type.Size() == 1
	// result: (ROLB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHRB {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpAMD64NEGQ {
			break
		}
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_0_1_0.AuxInt != -8 {
			break
		}
		v_0_0_1_0_0 := v_0_0_1_0.Args[0]
		if v_0_0_1_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_0_1_0_0.AuxInt != 7 {
			break
		}
		y := v_0_0_1_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_0_1_0.AuxInt != 8 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_1_0_0_0 := v_0_1_0_0.Args[0]
		if v_0_1_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_1_0_0_0.AuxInt != -8 {
			break
		}
		v_0_1_0_0_0_0 := v_0_1_0_0_0.Args[0]
		if v_0_1_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1_0_0_0_0.AuxInt != 7 {
			break
		}
		if y != v_0_1_0_0_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1.AuxInt != 7 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(v.Type.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SBBLcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [ 7]) [ -8])) [ 8])) (SHRB x (NEGQ (ADDQconst (ANDQconst y [ 7]) [ -8])))) (SHLL x (ANDQconst y [ 7])))
	// cond: v.Type.Size() == 1
	// result: (ROLB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_0_0_0.AuxInt != 8 {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_0_0_0_0.AuxInt != -8 {
			break
		}
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_0_0_0_0_0.AuxInt != 7 {
			break
		}
		y := v_0_0_0_0_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHRB {
			break
		}
		_ = v_0_1.Args[1]
		x := v_0_1.Args[0]
		v_0_1_1 := v_0_1.Args[1]
		if v_0_1_1.Op != OpAMD64NEGQ {
			break
		}
		v_0_1_1_0 := v_0_1_1.Args[0]
		if v_0_1_1_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_1_1_0.AuxInt != -8 {
			break
		}
		v_0_1_1_0_0 := v_0_1_1_0.Args[0]
		if v_0_1_1_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1_1_0_0.AuxInt != 7 {
			break
		}
		if y != v_0_1_1_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1.AuxInt != 7 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(v.Type.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_40(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORL (SHLL x (ANDLconst y [ 7])) (ANDL (SHRB x (NEGL (ADDLconst (ANDLconst y [ 7]) [ -8]))) (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [ 7]) [ -8])) [ 8]))))
	// cond: v.Type.Size() == 1
	// result: (ROLB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1.AuxInt != 7 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SHRB {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpAMD64NEGL {
			break
		}
		v_1_0_1_0 := v_1_0_1.Args[0]
		if v_1_0_1_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_0_1_0.AuxInt != -8 {
			break
		}
		v_1_0_1_0_0 := v_1_0_1_0.Args[0]
		if v_1_0_1_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_0_1_0_0.AuxInt != 7 {
			break
		}
		if y != v_1_0_1_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_1_1_0.AuxInt != 8 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64NEGL {
			break
		}
		v_1_1_0_0_0 := v_1_1_0_0.Args[0]
		if v_1_1_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_1_0_0_0.AuxInt != -8 {
			break
		}
		v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
		if v_1_1_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1_0_0_0_0.AuxInt != 7 {
			break
		}
		if y != v_1_1_0_0_0_0.Args[0] {
			break
		}
		if !(v.Type.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x (ANDLconst y [ 7])) (ANDL (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [ 7]) [ -8])) [ 8])) (SHRB x (NEGL (ADDLconst (ANDLconst y [ 7]) [ -8])))))
	// cond: v.Type.Size() == 1
	// result: (ROLB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1.AuxInt != 7 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDL {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_1_0_0.AuxInt != 8 {
			break
		}
		v_1_0_0_0 := v_1_0_0.Args[0]
		if v_1_0_0_0.Op != OpAMD64NEGL {
			break
		}
		v_1_0_0_0_0 := v_1_0_0_0.Args[0]
		if v_1_0_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_0_0_0_0.AuxInt != -8 {
			break
		}
		v_1_0_0_0_0_0 := v_1_0_0_0_0.Args[0]
		if v_1_0_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_0_0_0_0_0.AuxInt != 7 {
			break
		}
		if y != v_1_0_0_0_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SHRB {
			break
		}
		_ = v_1_1.Args[1]
		if x != v_1_1.Args[0] {
			break
		}
		v_1_1_1 := v_1_1.Args[1]
		if v_1_1_1.Op != OpAMD64NEGL {
			break
		}
		v_1_1_1_0 := v_1_1_1.Args[0]
		if v_1_1_1_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_1_1_0.AuxInt != -8 {
			break
		}
		v_1_1_1_0_0 := v_1_1_1_0.Args[0]
		if v_1_1_1_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1_1_0_0.AuxInt != 7 {
			break
		}
		if y != v_1_1_1_0_0.Args[0] {
			break
		}
		if !(v.Type.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SHRB x (NEGL (ADDLconst (ANDLconst y [ 7]) [ -8]))) (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [ 7]) [ -8])) [ 8]))) (SHLL x (ANDLconst y [ 7])))
	// cond: v.Type.Size() == 1
	// result: (ROLB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHRB {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpAMD64NEGL {
			break
		}
		v_0_0_1_0 := v_0_0_1.Args[0]
		if v_0_0_1_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_0_1_0.AuxInt != -8 {
			break
		}
		v_0_0_1_0_0 := v_0_0_1_0.Args[0]
		if v_0_0_1_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_0_1_0_0.AuxInt != 7 {
			break
		}
		y := v_0_0_1_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_0_1_0.AuxInt != 8 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64NEGL {
			break
		}
		v_0_1_0_0_0 := v_0_1_0_0.Args[0]
		if v_0_1_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_1_0_0_0.AuxInt != -8 {
			break
		}
		v_0_1_0_0_0_0 := v_0_1_0_0_0.Args[0]
		if v_0_1_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1_0_0_0_0.AuxInt != 7 {
			break
		}
		if y != v_0_1_0_0_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1.AuxInt != 7 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(v.Type.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (ANDL (SBBLcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [ 7]) [ -8])) [ 8])) (SHRB x (NEGL (ADDLconst (ANDLconst y [ 7]) [ -8])))) (SHLL x (ANDLconst y [ 7])))
	// cond: v.Type.Size() == 1
	// result: (ROLB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SBBLcarrymask {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_0_0_0.AuxInt != 8 {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAMD64NEGL {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_0_0_0_0.AuxInt != -8 {
			break
		}
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_0_0_0_0_0.AuxInt != 7 {
			break
		}
		y := v_0_0_0_0_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHRB {
			break
		}
		_ = v_0_1.Args[1]
		x := v_0_1.Args[0]
		v_0_1_1 := v_0_1.Args[1]
		if v_0_1_1.Op != OpAMD64NEGL {
			break
		}
		v_0_1_1_0 := v_0_1_1.Args[0]
		if v_0_1_1_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_1_1_0.AuxInt != -8 {
			break
		}
		v_0_1_1_0_0 := v_0_1_1_0.Args[0]
		if v_0_1_1_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1_1_0_0.AuxInt != 7 {
			break
		}
		if y != v_0_1_1_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1.AuxInt != 7 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(v.Type.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHRB x (ANDQconst y [ 7])) (SHLL x (NEGQ (ADDQconst (ANDQconst y [ 7]) [ -8]))))
	// cond: v.Type.Size() == 1
	// result: (RORB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRB {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1.AuxInt != 7 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64NEGQ {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_1_0.AuxInt != -8 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1_0_0.AuxInt != 7 {
			break
		}
		if y != v_1_1_0_0.Args[0] {
			break
		}
		if !(v.Type.Size() == 1) {
			break
		}
		v.reset(OpAMD64RORB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x (NEGQ (ADDQconst (ANDQconst y [ 7]) [ -8]))) (SHRB x (ANDQconst y [ 7])))
	// cond: v.Type.Size() == 1
	// result: (RORB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64NEGQ {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_1_0.AuxInt != -8 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1_0_0.AuxInt != 7 {
			break
		}
		y := v_0_1_0_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRB {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1.AuxInt != 7 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(v.Type.Size() == 1) {
			break
		}
		v.reset(OpAMD64RORB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHRB x (ANDLconst y [ 7])) (SHLL x (NEGL (ADDLconst (ANDLconst y [ 7]) [ -8]))))
	// cond: v.Type.Size() == 1
	// result: (RORB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRB {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1.AuxInt != 7 {
			break
		}
		y := v_0_1.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64NEGL {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_1_0.AuxInt != -8 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1_0_0.AuxInt != 7 {
			break
		}
		if y != v_1_1_0_0.Args[0] {
			break
		}
		if !(v.Type.Size() == 1) {
			break
		}
		v.reset(OpAMD64RORB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL (SHLL x (NEGL (ADDLconst (ANDLconst y [ 7]) [ -8]))) (SHRB x (ANDLconst y [ 7])))
	// cond: v.Type.Size() == 1
	// result: (RORB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64NEGL {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_1_0.AuxInt != -8 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1_0_0.AuxInt != 7 {
			break
		}
		y := v_0_1_0_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRB {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1.AuxInt != 7 {
			break
		}
		if y != v_1_1.Args[0] {
			break
		}
		if !(v.Type.Size() == 1) {
			break
		}
		v.reset(OpAMD64RORB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORL x x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ORL x0:(MOVBload [i0] {s} p mem) sh:(SHLLconst [8] x1:(MOVBload [i1] {s} p mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWload [i0] {s} p mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVBload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_50(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORL sh:(SHLLconst [8] x1:(MOVBload [i1] {s} p mem)) x0:(MOVBload [i0] {s} p mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWload [i0] {s} p mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL x0:(MOVWload [i0] {s} p mem) sh:(SHLLconst [16] x1:(MOVWload [i1] {s} p mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLload [i0] {s} p mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVWload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL sh:(SHLLconst [16] x1:(MOVWload [i1] {s} p mem)) x0:(MOVWload [i0] {s} p mem))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLload [i0] {s} p mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL s1:(SHLLconst [j1] x1:(MOVBload [i1] {s} p mem)) or:(ORL s0:(SHLLconst [j0] x0:(MOVBload [i0] {s} p mem)) y))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWload [i0] {s} p mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
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
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s1:(SHLLconst [j1] x1:(MOVBload [i1] {s} p mem)) or:(ORL y s0:(SHLLconst [j0] x0:(MOVBload [i0] {s} p mem))))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWload [i0] {s} p mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL s0:(SHLLconst [j0] x0:(MOVBload [i0] {s} p mem)) y) s1:(SHLLconst [j1] x1:(MOVBload [i1] {s} p mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWload [i0] {s} p mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL y s0:(SHLLconst [j0] x0:(MOVBload [i0] {s} p mem))) s1:(SHLLconst [j1] x1:(MOVBload [i1] {s} p mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWload [i0] {s} p mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL x0:(MOVBloadidx1 [i0] {s} p idx mem) sh:(SHLLconst [8] x1:(MOVBloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL x0:(MOVBloadidx1 [i0] {s} idx p mem) sh:(SHLLconst [8] x1:(MOVBloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL x0:(MOVBloadidx1 [i0] {s} p idx mem) sh:(SHLLconst [8] x1:(MOVBloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_60(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORL x0:(MOVBloadidx1 [i0] {s} idx p mem) sh:(SHLLconst [8] x1:(MOVBloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL sh:(SHLLconst [8] x1:(MOVBloadidx1 [i1] {s} p idx mem)) x0:(MOVBloadidx1 [i0] {s} p idx mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL sh:(SHLLconst [8] x1:(MOVBloadidx1 [i1] {s} idx p mem)) x0:(MOVBloadidx1 [i0] {s} p idx mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL sh:(SHLLconst [8] x1:(MOVBloadidx1 [i1] {s} p idx mem)) x0:(MOVBloadidx1 [i0] {s} idx p mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL sh:(SHLLconst [8] x1:(MOVBloadidx1 [i1] {s} idx p mem)) x0:(MOVBloadidx1 [i0] {s} idx p mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL x0:(MOVWloadidx1 [i0] {s} p idx mem) sh:(SHLLconst [16] x1:(MOVWloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL x0:(MOVWloadidx1 [i0] {s} idx p mem) sh:(SHLLconst [16] x1:(MOVWloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL x0:(MOVWloadidx1 [i0] {s} p idx mem) sh:(SHLLconst [16] x1:(MOVWloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL x0:(MOVWloadidx1 [i0] {s} idx p mem) sh:(SHLLconst [16] x1:(MOVWloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL sh:(SHLLconst [16] x1:(MOVWloadidx1 [i1] {s} p idx mem)) x0:(MOVWloadidx1 [i0] {s} p idx mem))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_70(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORL sh:(SHLLconst [16] x1:(MOVWloadidx1 [i1] {s} idx p mem)) x0:(MOVWloadidx1 [i0] {s} p idx mem))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL sh:(SHLLconst [16] x1:(MOVWloadidx1 [i1] {s} p idx mem)) x0:(MOVWloadidx1 [i0] {s} idx p mem))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL sh:(SHLLconst [16] x1:(MOVWloadidx1 [i1] {s} idx p mem)) x0:(MOVWloadidx1 [i0] {s} idx p mem))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) or:(ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) y))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) or:(ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) y))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) or:(ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) y))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) or:(ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) y))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) or:(ORL y s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) or:(ORL y s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) or:(ORL y s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_80(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) or:(ORL y s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) y) s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) y) s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL y s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem))) s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL y s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem))) s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) y) s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) y) s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL y s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem))) s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL y s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem))) s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL x1:(MOVBload [i1] {s} p mem) sh:(SHLLconst [8] x0:(MOVBload [i0] {s} p mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWload [i0] {s} p mem))
	for {
		_ = v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpAMD64MOVBload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_90(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORL sh:(SHLLconst [8] x0:(MOVBload [i0] {s} p mem)) x1:(MOVBload [i1] {s} p mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWload [i0] {s} p mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		x1 := v.Args[1]
		if x1.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL r1:(ROLWconst [8] x1:(MOVWload [i1] {s} p mem)) sh:(SHLLconst [16] r0:(ROLWconst [8] x0:(MOVWload [i0] {s} p mem))))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLload [i0] {s} p mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL sh:(SHLLconst [16] r0:(ROLWconst [8] x0:(MOVWload [i0] {s} p mem))) r1:(ROLWconst [8] x1:(MOVWload [i1] {s} p mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLload [i0] {s} p mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		r1 := v.Args[1]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL s0:(SHLLconst [j0] x0:(MOVBload [i0] {s} p mem)) or:(ORL s1:(SHLLconst [j1] x1:(MOVBload [i1] {s} p mem)) y))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWload [i0] {s} p mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
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
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s0:(SHLLconst [j0] x0:(MOVBload [i0] {s} p mem)) or:(ORL y s1:(SHLLconst [j1] x1:(MOVBload [i1] {s} p mem))))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWload [i0] {s} p mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL s1:(SHLLconst [j1] x1:(MOVBload [i1] {s} p mem)) y) s0:(SHLLconst [j0] x0:(MOVBload [i0] {s} p mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWload [i0] {s} p mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL y s1:(SHLLconst [j1] x1:(MOVBload [i1] {s} p mem))) s0:(SHLLconst [j0] x0:(MOVBload [i0] {s} p mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWload [i0] {s} p mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL x1:(MOVBloadidx1 [i1] {s} p idx mem) sh:(SHLLconst [8] x0:(MOVBloadidx1 [i0] {s} p idx mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL x1:(MOVBloadidx1 [i1] {s} idx p mem) sh:(SHLLconst [8] x0:(MOVBloadidx1 [i0] {s} p idx mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL x1:(MOVBloadidx1 [i1] {s} p idx mem) sh:(SHLLconst [8] x0:(MOVBloadidx1 [i0] {s} idx p mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_100(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORL x1:(MOVBloadidx1 [i1] {s} idx p mem) sh:(SHLLconst [8] x0:(MOVBloadidx1 [i0] {s} idx p mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL sh:(SHLLconst [8] x0:(MOVBloadidx1 [i0] {s} p idx mem)) x1:(MOVBloadidx1 [i1] {s} p idx mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		x1 := v.Args[1]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL sh:(SHLLconst [8] x0:(MOVBloadidx1 [i0] {s} idx p mem)) x1:(MOVBloadidx1 [i1] {s} p idx mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		x1 := v.Args[1]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL sh:(SHLLconst [8] x0:(MOVBloadidx1 [i0] {s} p idx mem)) x1:(MOVBloadidx1 [i1] {s} idx p mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		x1 := v.Args[1]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL sh:(SHLLconst [8] x0:(MOVBloadidx1 [i0] {s} idx p mem)) x1:(MOVBloadidx1 [i1] {s} idx p mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		x1 := v.Args[1]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem)) sh:(SHLLconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem)) sh:(SHLLconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem)) sh:(SHLLconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem)) sh:(SHLLconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL sh:(SHLLconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))) r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		r1 := v.Args[1]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_110(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORL sh:(SHLLconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))) r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		r1 := v.Args[1]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL sh:(SHLLconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))) r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		r1 := v.Args[1]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL sh:(SHLLconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))) r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLLconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		r1 := v.Args[1]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) or:(ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) y))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) or:(ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) y))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) or:(ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) y))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) or:(ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) y))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) or:(ORL y s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem))))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) or:(ORL y s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem))))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) or:(ORL y s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem))))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_120(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORL s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) or:(ORL y s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem))))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) y) s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) y) s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL y s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem))) s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL y s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem))) s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) y) s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) y) s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL y s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem))) s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL or:(ORL y s1:(SHLLconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem))) s0:(SHLLconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORL <v.Type> (SHLLconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORL {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLLconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLLconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLLconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ORLmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ORLmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORL_130(v *Value) bool {
	// match: (ORL l:(MOVLload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ORLmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		l := v.Args[0]
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		x := v.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ORLmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORLconst_0(v *Value) bool {
	// match: (ORLconst [c] x)
	// cond: int32(c)==0
	// result: x
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ORLconst [c] _)
	// cond: int32(c)==-1
	// result: (MOVLconst [-1])
	for {
		c := v.AuxInt
		if !(int32(c) == -1) {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = -1
		return true
	}
	// match: (ORLconst [c] (MOVLconst [d]))
	// cond:
	// result: (MOVLconst [c|d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = c | d
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_0(v *Value) bool {
	// match: (ORQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (ORQconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64ORQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORQ (MOVQconst [c]) x)
	// cond: is32Bit(c)
	// result: (ORQconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64ORQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORQ (SHLQconst x [c]) (SHRQconst x [d]))
	// cond: d==64-c
	// result: (ROLQconst x [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		d := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 64-c) {
			break
		}
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORQ (SHRQconst x [d]) (SHLQconst x [c]))
	// cond: d==64-c
	// result: (ROLQconst x [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 64-c) {
			break
		}
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORQ (SHLQ x y) (ANDQ (SHRQ x (NEGQ y)) (SBBQcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [63]) [-64])) [64]))))
	// cond:
	// result: (ROLQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SHRQ {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpAMD64NEGQ {
			break
		}
		if y != v_1_0_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_1_1_0.AuxInt != 64 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_1_1_0_0_0 := v_1_1_0_0.Args[0]
		if v_1_1_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_1_0_0_0.AuxInt != -64 {
			break
		}
		v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
		if v_1_1_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1_0_0_0_0.AuxInt != 63 {
			break
		}
		if y != v_1_1_0_0_0_0.Args[0] {
			break
		}
		v.reset(OpAMD64ROLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (SHLQ x y) (ANDQ (SBBQcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [63]) [-64])) [64])) (SHRQ x (NEGQ y))))
	// cond:
	// result: (ROLQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_1_0_0.AuxInt != 64 {
			break
		}
		v_1_0_0_0 := v_1_0_0.Args[0]
		if v_1_0_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_1_0_0_0_0 := v_1_0_0_0.Args[0]
		if v_1_0_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_0_0_0_0.AuxInt != -64 {
			break
		}
		v_1_0_0_0_0_0 := v_1_0_0_0_0.Args[0]
		if v_1_0_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_0_0_0_0_0.AuxInt != 63 {
			break
		}
		if y != v_1_0_0_0_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SHRQ {
			break
		}
		_ = v_1_1.Args[1]
		if x != v_1_1.Args[0] {
			break
		}
		v_1_1_1 := v_1_1.Args[1]
		if v_1_1_1.Op != OpAMD64NEGQ {
			break
		}
		if y != v_1_1_1.Args[0] {
			break
		}
		v.reset(OpAMD64ROLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (ANDQ (SHRQ x (NEGQ y)) (SBBQcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [63]) [-64])) [64]))) (SHLQ x y))
	// cond:
	// result: (ROLQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHRQ {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_0_0_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_0_1_0.AuxInt != 64 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_1_0_0_0 := v_0_1_0_0.Args[0]
		if v_0_1_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_1_0_0_0.AuxInt != -64 {
			break
		}
		v_0_1_0_0_0_0 := v_0_1_0_0_0.Args[0]
		if v_0_1_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1_0_0_0_0.AuxInt != 63 {
			break
		}
		if y != v_0_1_0_0_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQ {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64ROLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (ANDQ (SBBQcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [63]) [-64])) [64])) (SHRQ x (NEGQ y))) (SHLQ x y))
	// cond:
	// result: (ROLQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_0_0_0.AuxInt != 64 {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_0_0_0_0.AuxInt != -64 {
			break
		}
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_0_0_0_0_0.AuxInt != 63 {
			break
		}
		y := v_0_0_0_0_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHRQ {
			break
		}
		_ = v_0_1.Args[1]
		x := v_0_1.Args[0]
		v_0_1_1 := v_0_1.Args[1]
		if v_0_1_1.Op != OpAMD64NEGQ {
			break
		}
		if y != v_0_1_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQ {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64ROLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (SHLQ x y) (ANDQ (SHRQ x (NEGL y)) (SBBQcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [63]) [-64])) [64]))))
	// cond:
	// result: (ROLQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SHRQ {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpAMD64NEGL {
			break
		}
		if y != v_1_0_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_1_1_0.AuxInt != 64 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64NEGL {
			break
		}
		v_1_1_0_0_0 := v_1_1_0_0.Args[0]
		if v_1_1_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_1_0_0_0.AuxInt != -64 {
			break
		}
		v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
		if v_1_1_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1_0_0_0_0.AuxInt != 63 {
			break
		}
		if y != v_1_1_0_0_0_0.Args[0] {
			break
		}
		v.reset(OpAMD64ROLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (SHLQ x y) (ANDQ (SBBQcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [63]) [-64])) [64])) (SHRQ x (NEGL y))))
	// cond:
	// result: (ROLQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_1_0_0.AuxInt != 64 {
			break
		}
		v_1_0_0_0 := v_1_0_0.Args[0]
		if v_1_0_0_0.Op != OpAMD64NEGL {
			break
		}
		v_1_0_0_0_0 := v_1_0_0_0.Args[0]
		if v_1_0_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_0_0_0_0.AuxInt != -64 {
			break
		}
		v_1_0_0_0_0_0 := v_1_0_0_0_0.Args[0]
		if v_1_0_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_0_0_0_0_0.AuxInt != 63 {
			break
		}
		if y != v_1_0_0_0_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SHRQ {
			break
		}
		_ = v_1_1.Args[1]
		if x != v_1_1.Args[0] {
			break
		}
		v_1_1_1 := v_1_1.Args[1]
		if v_1_1_1.Op != OpAMD64NEGL {
			break
		}
		if y != v_1_1_1.Args[0] {
			break
		}
		v.reset(OpAMD64ROLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_10(v *Value) bool {
	// match: (ORQ (ANDQ (SHRQ x (NEGL y)) (SBBQcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [63]) [-64])) [64]))) (SHLQ x y))
	// cond:
	// result: (ROLQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHRQ {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpAMD64NEGL {
			break
		}
		y := v_0_0_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_0_1_0.AuxInt != 64 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64NEGL {
			break
		}
		v_0_1_0_0_0 := v_0_1_0_0.Args[0]
		if v_0_1_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_1_0_0_0.AuxInt != -64 {
			break
		}
		v_0_1_0_0_0_0 := v_0_1_0_0_0.Args[0]
		if v_0_1_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1_0_0_0_0.AuxInt != 63 {
			break
		}
		if y != v_0_1_0_0_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQ {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64ROLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (ANDQ (SBBQcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [63]) [-64])) [64])) (SHRQ x (NEGL y))) (SHLQ x y))
	// cond:
	// result: (ROLQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_0_0_0.AuxInt != 64 {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAMD64NEGL {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_0_0_0_0.AuxInt != -64 {
			break
		}
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_0_0_0_0_0.AuxInt != 63 {
			break
		}
		y := v_0_0_0_0_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHRQ {
			break
		}
		_ = v_0_1.Args[1]
		x := v_0_1.Args[0]
		v_0_1_1 := v_0_1.Args[1]
		if v_0_1_1.Op != OpAMD64NEGL {
			break
		}
		if y != v_0_1_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQ {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64ROLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (SHRQ x y) (ANDQ (SHLQ x (NEGQ y)) (SBBQcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [63]) [-64])) [64]))))
	// cond:
	// result: (RORQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SHLQ {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpAMD64NEGQ {
			break
		}
		if y != v_1_0_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_1_1_0.AuxInt != 64 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_1_1_0_0_0 := v_1_1_0_0.Args[0]
		if v_1_1_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_1_0_0_0.AuxInt != -64 {
			break
		}
		v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
		if v_1_1_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_1_0_0_0_0.AuxInt != 63 {
			break
		}
		if y != v_1_1_0_0_0_0.Args[0] {
			break
		}
		v.reset(OpAMD64RORQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (SHRQ x y) (ANDQ (SBBQcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [63]) [-64])) [64])) (SHLQ x (NEGQ y))))
	// cond:
	// result: (RORQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_1_0_0.AuxInt != 64 {
			break
		}
		v_1_0_0_0 := v_1_0_0.Args[0]
		if v_1_0_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_1_0_0_0_0 := v_1_0_0_0.Args[0]
		if v_1_0_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_1_0_0_0_0.AuxInt != -64 {
			break
		}
		v_1_0_0_0_0_0 := v_1_0_0_0_0.Args[0]
		if v_1_0_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_1_0_0_0_0_0.AuxInt != 63 {
			break
		}
		if y != v_1_0_0_0_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SHLQ {
			break
		}
		_ = v_1_1.Args[1]
		if x != v_1_1.Args[0] {
			break
		}
		v_1_1_1 := v_1_1.Args[1]
		if v_1_1_1.Op != OpAMD64NEGQ {
			break
		}
		if y != v_1_1_1.Args[0] {
			break
		}
		v.reset(OpAMD64RORQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (ANDQ (SHLQ x (NEGQ y)) (SBBQcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [63]) [-64])) [64]))) (SHRQ x y))
	// cond:
	// result: (RORQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHLQ {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_0_0_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_0_1_0.AuxInt != 64 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_1_0_0_0 := v_0_1_0_0.Args[0]
		if v_0_1_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_1_0_0_0.AuxInt != -64 {
			break
		}
		v_0_1_0_0_0_0 := v_0_1_0_0_0.Args[0]
		if v_0_1_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_1_0_0_0_0.AuxInt != 63 {
			break
		}
		if y != v_0_1_0_0_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQ {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64RORQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (ANDQ (SBBQcarrymask (CMPQconst (NEGQ (ADDQconst (ANDQconst y [63]) [-64])) [64])) (SHLQ x (NEGQ y))) (SHRQ x y))
	// cond:
	// result: (RORQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64CMPQconst {
			break
		}
		if v_0_0_0.AuxInt != 64 {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAMD64NEGQ {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAMD64ADDQconst {
			break
		}
		if v_0_0_0_0_0.AuxInt != -64 {
			break
		}
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpAMD64ANDQconst {
			break
		}
		if v_0_0_0_0_0_0.AuxInt != 63 {
			break
		}
		y := v_0_0_0_0_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHLQ {
			break
		}
		_ = v_0_1.Args[1]
		x := v_0_1.Args[0]
		v_0_1_1 := v_0_1.Args[1]
		if v_0_1_1.Op != OpAMD64NEGQ {
			break
		}
		if y != v_0_1_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQ {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64RORQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (SHRQ x y) (ANDQ (SHLQ x (NEGL y)) (SBBQcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [63]) [-64])) [64]))))
	// cond:
	// result: (RORQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SHLQ {
			break
		}
		_ = v_1_0.Args[1]
		if x != v_1_0.Args[0] {
			break
		}
		v_1_0_1 := v_1_0.Args[1]
		if v_1_0_1.Op != OpAMD64NEGL {
			break
		}
		if y != v_1_0_1.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_1_1_0 := v_1_1.Args[0]
		if v_1_1_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_1_1_0.AuxInt != 64 {
			break
		}
		v_1_1_0_0 := v_1_1_0.Args[0]
		if v_1_1_0_0.Op != OpAMD64NEGL {
			break
		}
		v_1_1_0_0_0 := v_1_1_0_0.Args[0]
		if v_1_1_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_1_0_0_0.AuxInt != -64 {
			break
		}
		v_1_1_0_0_0_0 := v_1_1_0_0_0.Args[0]
		if v_1_1_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_1_0_0_0_0.AuxInt != 63 {
			break
		}
		if y != v_1_1_0_0_0_0.Args[0] {
			break
		}
		v.reset(OpAMD64RORQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (SHRQ x y) (ANDQ (SBBQcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [63]) [-64])) [64])) (SHLQ x (NEGL y))))
	// cond:
	// result: (RORQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQ {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_1_0_0 := v_1_0.Args[0]
		if v_1_0_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_1_0_0.AuxInt != 64 {
			break
		}
		v_1_0_0_0 := v_1_0_0.Args[0]
		if v_1_0_0_0.Op != OpAMD64NEGL {
			break
		}
		v_1_0_0_0_0 := v_1_0_0_0.Args[0]
		if v_1_0_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_1_0_0_0_0.AuxInt != -64 {
			break
		}
		v_1_0_0_0_0_0 := v_1_0_0_0_0.Args[0]
		if v_1_0_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_1_0_0_0_0_0.AuxInt != 63 {
			break
		}
		if y != v_1_0_0_0_0_0.Args[0] {
			break
		}
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpAMD64SHLQ {
			break
		}
		_ = v_1_1.Args[1]
		if x != v_1_1.Args[0] {
			break
		}
		v_1_1_1 := v_1_1.Args[1]
		if v_1_1_1.Op != OpAMD64NEGL {
			break
		}
		if y != v_1_1_1.Args[0] {
			break
		}
		v.reset(OpAMD64RORQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (ANDQ (SHLQ x (NEGL y)) (SBBQcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [63]) [-64])) [64]))) (SHRQ x y))
	// cond:
	// result: (RORQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHLQ {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpAMD64NEGL {
			break
		}
		y := v_0_0_1.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_0_1_0.AuxInt != 64 {
			break
		}
		v_0_1_0_0 := v_0_1_0.Args[0]
		if v_0_1_0_0.Op != OpAMD64NEGL {
			break
		}
		v_0_1_0_0_0 := v_0_1_0_0.Args[0]
		if v_0_1_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_1_0_0_0.AuxInt != -64 {
			break
		}
		v_0_1_0_0_0_0 := v_0_1_0_0_0.Args[0]
		if v_0_1_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_1_0_0_0_0.AuxInt != 63 {
			break
		}
		if y != v_0_1_0_0_0_0.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQ {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64RORQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ORQ (ANDQ (SBBQcarrymask (CMPLconst (NEGL (ADDLconst (ANDLconst y [63]) [-64])) [64])) (SHLQ x (NEGL y))) (SHRQ x y))
	// cond:
	// result: (RORQ x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ANDQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SBBQcarrymask {
			break
		}
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64CMPLconst {
			break
		}
		if v_0_0_0.AuxInt != 64 {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpAMD64NEGL {
			break
		}
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAMD64ADDLconst {
			break
		}
		if v_0_0_0_0_0.AuxInt != -64 {
			break
		}
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpAMD64ANDLconst {
			break
		}
		if v_0_0_0_0_0_0.AuxInt != 63 {
			break
		}
		y := v_0_0_0_0_0_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHLQ {
			break
		}
		_ = v_0_1.Args[1]
		x := v_0_1.Args[0]
		v_0_1_1 := v_0_1.Args[1]
		if v_0_1_1.Op != OpAMD64NEGL {
			break
		}
		if y != v_0_1_1.Args[0] {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQ {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		if y != v_1.Args[1] {
			break
		}
		v.reset(OpAMD64RORQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_20(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ x x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ORQ x0:(MOVBload [i0] {s} p mem) sh:(SHLQconst [8] x1:(MOVBload [i1] {s} p mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWload [i0] {s} p mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVBload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [8] x1:(MOVBload [i1] {s} p mem)) x0:(MOVBload [i0] {s} p mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWload [i0] {s} p mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVWload [i0] {s} p mem) sh:(SHLQconst [16] x1:(MOVWload [i1] {s} p mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLload [i0] {s} p mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVWload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [16] x1:(MOVWload [i1] {s} p mem)) x0:(MOVWload [i0] {s} p mem))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLload [i0] {s} p mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVLload [i0] {s} p mem) sh:(SHLQconst [32] x1:(MOVLload [i1] {s} p mem)))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVQload [i0] {s} p mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVLload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVLload {
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
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [32] x1:(MOVLload [i1] {s} p mem)) x0:(MOVLload [i0] {s} p mem))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVQload [i0] {s} p mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVLload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVLload {
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
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVBload [i1] {s} p mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVBload [i0] {s} p mem)) y))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWload [i0] {s} p mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
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
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVBload [i1] {s} p mem)) or:(ORQ y s0:(SHLQconst [j0] x0:(MOVBload [i0] {s} p mem))))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWload [i0] {s} p mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s0:(SHLQconst [j0] x0:(MOVBload [i0] {s} p mem)) y) s1:(SHLQconst [j1] x1:(MOVBload [i1] {s} p mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWload [i0] {s} p mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_30(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ or:(ORQ y s0:(SHLQconst [j0] x0:(MOVBload [i0] {s} p mem))) s1:(SHLQconst [j1] x1:(MOVBload [i1] {s} p mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWload [i0] {s} p mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVWload [i1] {s} p mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVWload [i0] {s} p mem)) y))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLload [i0] {s} p mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWload {
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
		y := or.Args[1]
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVWload [i1] {s} p mem)) or:(ORQ y s0:(SHLQconst [j0] x0:(MOVWload [i0] {s} p mem))))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLload [i0] {s} p mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s0:(SHLQconst [j0] x0:(MOVWload [i0] {s} p mem)) y) s1:(SHLQconst [j1] x1:(MOVWload [i1] {s} p mem)))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLload [i0] {s} p mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s0:(SHLQconst [j0] x0:(MOVWload [i0] {s} p mem))) s1:(SHLQconst [j1] x1:(MOVWload [i1] {s} p mem)))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLload [i0] {s} p mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ x0:(MOVBloadidx1 [i0] {s} p idx mem) sh:(SHLQconst [8] x1:(MOVBloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVBloadidx1 [i0] {s} idx p mem) sh:(SHLQconst [8] x1:(MOVBloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVBloadidx1 [i0] {s} p idx mem) sh:(SHLQconst [8] x1:(MOVBloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVBloadidx1 [i0] {s} idx p mem) sh:(SHLQconst [8] x1:(MOVBloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [8] x1:(MOVBloadidx1 [i1] {s} p idx mem)) x0:(MOVBloadidx1 [i0] {s} p idx mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_40(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ sh:(SHLQconst [8] x1:(MOVBloadidx1 [i1] {s} idx p mem)) x0:(MOVBloadidx1 [i0] {s} p idx mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [8] x1:(MOVBloadidx1 [i1] {s} p idx mem)) x0:(MOVBloadidx1 [i0] {s} idx p mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [8] x1:(MOVBloadidx1 [i1] {s} idx p mem)) x0:(MOVBloadidx1 [i0] {s} idx p mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVWloadidx1 <v.Type> [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVWloadidx1 [i0] {s} p idx mem) sh:(SHLQconst [16] x1:(MOVWloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVWloadidx1 [i0] {s} idx p mem) sh:(SHLQconst [16] x1:(MOVWloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVWloadidx1 [i0] {s} p idx mem) sh:(SHLQconst [16] x1:(MOVWloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVWloadidx1 [i0] {s} idx p mem) sh:(SHLQconst [16] x1:(MOVWloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [16] x1:(MOVWloadidx1 [i1] {s} p idx mem)) x0:(MOVWloadidx1 [i0] {s} p idx mem))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [16] x1:(MOVWloadidx1 [i1] {s} idx p mem)) x0:(MOVWloadidx1 [i0] {s} p idx mem))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [16] x1:(MOVWloadidx1 [i1] {s} p idx mem)) x0:(MOVWloadidx1 [i0] {s} idx p mem))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_50(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ sh:(SHLQconst [16] x1:(MOVWloadidx1 [i1] {s} idx p mem)) x0:(MOVWloadidx1 [i0] {s} idx p mem))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVLloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVLloadidx1 [i0] {s} p idx mem) sh:(SHLQconst [32] x1:(MOVLloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVQloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVLloadidx1 [i0] {s} idx p mem) sh:(SHLQconst [32] x1:(MOVLloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVQloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVLloadidx1 [i0] {s} p idx mem) sh:(SHLQconst [32] x1:(MOVLloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVQloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ x0:(MOVLloadidx1 [i0] {s} idx p mem) sh:(SHLQconst [32] x1:(MOVLloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVQloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		x0 := v.Args[0]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [32] x1:(MOVLloadidx1 [i1] {s} p idx mem)) x0:(MOVLloadidx1 [i0] {s} p idx mem))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVQloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [32] x1:(MOVLloadidx1 [i1] {s} idx p mem)) x0:(MOVLloadidx1 [i0] {s} p idx mem))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVQloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [32] x1:(MOVLloadidx1 [i1] {s} p idx mem)) x0:(MOVLloadidx1 [i0] {s} idx p mem))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVQloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ sh:(SHLQconst [32] x1:(MOVLloadidx1 [i1] {s} idx p mem)) x0:(MOVLloadidx1 [i0] {s} idx p mem))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (MOVQloadidx1 [i0] {s} p idx mem)
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		x1 := sh.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		x0 := v.Args[1]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = i0
		v0.Aux = s
		v0.AddArg(p)
		v0.AddArg(idx)
		v0.AddArg(mem)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) y))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_60(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) y))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) y))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) y))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) or:(ORQ y s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) or:(ORQ y s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) or:(ORQ y s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) or:(ORQ y s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) y) s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) y) s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem))) s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_70(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ or:(ORQ y s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem))) s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) y) s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) y) s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem))) s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem))) s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0+8   && j0 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVWloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0+8 && j0%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} p idx mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} p idx mem)) y))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} idx p mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} p idx mem)) y))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} p idx mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} idx p mem)) y))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} idx p mem)) or:(ORQ s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} idx p mem)) y))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} p idx mem)) or:(ORQ y s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_80(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} idx p mem)) or:(ORQ y s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} p idx mem)) or:(ORQ y s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} idx p mem)) or:(ORQ y s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		s1 := v.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} p idx mem)) y) s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} idx p mem)) y) s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} p idx mem))) s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} idx p mem))) s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} p idx mem)) y) s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} idx p mem)) y) s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s0 := or.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		y := or.Args[1]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} p idx mem))) s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_90(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ or:(ORQ y s0:(SHLQconst [j0] x0:(MOVWloadidx1 [i0] {s} idx p mem))) s1:(SHLQconst [j1] x1:(MOVWloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+2   && j1 == j0+16   && j0 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j0] (MOVLloadidx1 [i0] {s} p idx mem)) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s0 := or.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		s1 := v.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0+16 && j0%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j0
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v2.AuxInt = i0
		v2.Aux = s
		v2.AddArg(p)
		v2.AddArg(idx)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ x1:(MOVBload [i1] {s} p mem) sh:(SHLQconst [8] x0:(MOVBload [i0] {s} p mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWload [i0] {s} p mem))
	for {
		_ = v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpAMD64MOVBload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [8] x0:(MOVBload [i0] {s} p mem)) x1:(MOVBload [i1] {s} p mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWload [i0] {s} p mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		x1 := v.Args[1]
		if x1.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ r1:(ROLWconst [8] x1:(MOVWload [i1] {s} p mem)) sh:(SHLQconst [16] r0:(ROLWconst [8] x0:(MOVWload [i0] {s} p mem))))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLload [i0] {s} p mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [16] r0:(ROLWconst [8] x0:(MOVWload [i0] {s} p mem))) r1:(ROLWconst [8] x1:(MOVWload [i1] {s} p mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLload [i0] {s} p mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		r1 := v.Args[1]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ r1:(BSWAPL x1:(MOVLload [i1] {s} p mem)) sh:(SHLQconst [32] r0:(BSWAPL x0:(MOVLload [i0] {s} p mem))))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPQ <v.Type> (MOVQload [i0] {s} p mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64BSWAPL {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVLload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64BSWAPL {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVLload {
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
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [32] r0:(BSWAPL x0:(MOVLload [i0] {s} p mem))) r1:(BSWAPL x1:(MOVLload [i1] {s} p mem)))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPQ <v.Type> (MOVQload [i0] {s} p mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64BSWAPL {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVLload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		r1 := v.Args[1]
		if r1.Op != OpAMD64BSWAPL {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVLload {
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
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] x0:(MOVBload [i0] {s} p mem)) or:(ORQ s1:(SHLQconst [j1] x1:(MOVBload [i1] {s} p mem)) y))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWload [i0] {s} p mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
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
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] x0:(MOVBload [i0] {s} p mem)) or:(ORQ y s1:(SHLQconst [j1] x1:(MOVBload [i1] {s} p mem))))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWload [i0] {s} p mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s1:(SHLQconst [j1] x1:(MOVBload [i1] {s} p mem)) y) s0:(SHLQconst [j0] x0:(MOVBload [i0] {s} p mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWload [i0] {s} p mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_100(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ or:(ORQ y s1:(SHLQconst [j1] x1:(MOVBload [i1] {s} p mem))) s0:(SHLQconst [j0] x0:(MOVBload [i0] {s} p mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWload [i0] {s} p mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBload {
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
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWload [i0] {s} p mem))) or:(ORQ s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWload [i1] {s} p mem))) y))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLload [i0] {s} p mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWload {
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
		y := or.Args[1]
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWload [i0] {s} p mem))) or:(ORQ y s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWload [i1] {s} p mem)))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLload [i0] {s} p mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWload {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWload [i1] {s} p mem))) y) s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWload [i0] {s} p mem))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLload [i0] {s} p mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWload [i1] {s} p mem)))) s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWload [i0] {s} p mem))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLload [i0] {s} p mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWload {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[1]
		p := x1.Args[0]
		mem := x1.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWload {
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
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ x1:(MOVBloadidx1 [i1] {s} p idx mem) sh:(SHLQconst [8] x0:(MOVBloadidx1 [i0] {s} p idx mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ x1:(MOVBloadidx1 [i1] {s} idx p mem) sh:(SHLQconst [8] x0:(MOVBloadidx1 [i0] {s} p idx mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ x1:(MOVBloadidx1 [i1] {s} p idx mem) sh:(SHLQconst [8] x0:(MOVBloadidx1 [i0] {s} idx p mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ x1:(MOVBloadidx1 [i1] {s} idx p mem) sh:(SHLQconst [8] x0:(MOVBloadidx1 [i0] {s} idx p mem)))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		x1 := v.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [8] x0:(MOVBloadidx1 [i0] {s} p idx mem)) x1:(MOVBloadidx1 [i1] {s} p idx mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		x1 := v.Args[1]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_110(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ sh:(SHLQconst [8] x0:(MOVBloadidx1 [i0] {s} idx p mem)) x1:(MOVBloadidx1 [i1] {s} p idx mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		x1 := v.Args[1]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [8] x0:(MOVBloadidx1 [i0] {s} p idx mem)) x1:(MOVBloadidx1 [i1] {s} idx p mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		x1 := v.Args[1]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [8] x0:(MOVBloadidx1 [i0] {s} idx p mem)) x1:(MOVBloadidx1 [i1] {s} idx p mem))
	// cond: i1 == i0+1   && x0.Uses == 1   && x1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (ROLWconst <v.Type> [8] (MOVWloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 8 {
			break
		}
		x0 := sh.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		x1 := v.Args[1]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && x0.Uses == 1 && x1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ROLWconst, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.AuxInt = 8
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem)) sh:(SHLQconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem)) sh:(SHLQconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem)) sh:(SHLQconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem)) sh:(SHLQconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))) r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		r1 := v.Args[1]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))) r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		r1 := v.Args[1]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))) r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		r1 := v.Args[1]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_120(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ sh:(SHLQconst [16] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))) r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+2   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPL <v.Type> (MOVLloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 16 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		r1 := v.Args[1]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPL, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ r1:(BSWAPL x1:(MOVLloadidx1 [i1] {s} p idx mem)) sh:(SHLQconst [32] r0:(BSWAPL x0:(MOVLloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPQ <v.Type> (MOVQloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64BSWAPL {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64BSWAPL {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ r1:(BSWAPL x1:(MOVLloadidx1 [i1] {s} idx p mem)) sh:(SHLQconst [32] r0:(BSWAPL x0:(MOVLloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPQ <v.Type> (MOVQloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64BSWAPL {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64BSWAPL {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ r1:(BSWAPL x1:(MOVLloadidx1 [i1] {s} p idx mem)) sh:(SHLQconst [32] r0:(BSWAPL x0:(MOVLloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPQ <v.Type> (MOVQloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64BSWAPL {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64BSWAPL {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ r1:(BSWAPL x1:(MOVLloadidx1 [i1] {s} idx p mem)) sh:(SHLQconst [32] r0:(BSWAPL x0:(MOVLloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPQ <v.Type> (MOVQloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		r1 := v.Args[0]
		if r1.Op != OpAMD64BSWAPL {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		sh := v.Args[1]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64BSWAPL {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [32] r0:(BSWAPL x0:(MOVLloadidx1 [i0] {s} p idx mem))) r1:(BSWAPL x1:(MOVLloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPQ <v.Type> (MOVQloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64BSWAPL {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		r1 := v.Args[1]
		if r1.Op != OpAMD64BSWAPL {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [32] r0:(BSWAPL x0:(MOVLloadidx1 [i0] {s} idx p mem))) r1:(BSWAPL x1:(MOVLloadidx1 [i1] {s} p idx mem)))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPQ <v.Type> (MOVQloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64BSWAPL {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		r1 := v.Args[1]
		if r1.Op != OpAMD64BSWAPL {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [32] r0:(BSWAPL x0:(MOVLloadidx1 [i0] {s} p idx mem))) r1:(BSWAPL x1:(MOVLloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPQ <v.Type> (MOVQloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64BSWAPL {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		r1 := v.Args[1]
		if r1.Op != OpAMD64BSWAPL {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ sh:(SHLQconst [32] r0:(BSWAPL x0:(MOVLloadidx1 [i0] {s} idx p mem))) r1:(BSWAPL x1:(MOVLloadidx1 [i1] {s} idx p mem)))
	// cond: i1 == i0+4   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && sh.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(sh)
	// result: @mergePoint(b,x0,x1) (BSWAPQ <v.Type> (MOVQloadidx1 [i0] {s} p idx mem))
	for {
		_ = v.Args[1]
		sh := v.Args[0]
		if sh.Op != OpAMD64SHLQconst {
			break
		}
		if sh.AuxInt != 32 {
			break
		}
		r0 := sh.Args[0]
		if r0.Op != OpAMD64BSWAPL {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		r1 := v.Args[1]
		if r1.Op != OpAMD64BSWAPL {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVLloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+4 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && sh.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(sh)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64BSWAPQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQloadidx1, typ.UInt64)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(idx)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) or:(ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) y))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_130(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) or:(ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) y))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) or:(ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) y))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) or:(ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) y))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) or:(ORQ y s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem))))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) or:(ORQ y s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem))))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)) or:(ORQ y s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem))))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)) or:(ORQ y s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem))))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) y) s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) y) s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem))) s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_140(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ or:(ORQ y s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem))) s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} p idx mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem)) y) s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem)) y) s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} p idx mem))) s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s1:(SHLQconst [j1] x1:(MOVBloadidx1 [i1] {s} idx p mem))) s0:(SHLQconst [j0] x0:(MOVBloadidx1 [i0] {s} idx p mem)))
	// cond: i1 == i0+1   && j1 == j0-8   && j1 % 16 == 0   && x0.Uses == 1   && x1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (ROLWconst <typ.UInt16> [8] (MOVWloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		x1 := s1.Args[0]
		if x1.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		x0 := s0.Args[0]
		if x0.Op != OpAMD64MOVBloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+1 && j1 == j0-8 && j1%16 == 0 && x0.Uses == 1 && x1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64ROLWconst, typ.UInt16)
		v2.AuxInt = 8
		v3 := b.NewValue0(v.Pos, OpAMD64MOVWloadidx1, typ.UInt16)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))) or:(ORQ s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem))) y))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))) or:(ORQ s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem))) y))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))) or:(ORQ s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem))) y))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))) or:(ORQ s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem))) y))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		y := or.Args[1]
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))) or:(ORQ y s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem)))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_150(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))) or:(ORQ y s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem)))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if p != x1.Args[0] {
			break
		}
		if idx != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))) or:(ORQ y s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem)))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		p := x0.Args[0]
		idx := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))) or:(ORQ y s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem)))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		s0 := v.Args[0]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[2]
		idx := x0.Args[0]
		p := x0.Args[1]
		mem := x0.Args[2]
		or := v.Args[1]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		if x1.Aux != s {
			break
		}
		_ = x1.Args[2]
		if idx != x1.Args[0] {
			break
		}
		if p != x1.Args[1] {
			break
		}
		if mem != x1.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem))) y) s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem))) y) s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem)))) s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem)))) s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} p idx mem))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if p != x0.Args[0] {
			break
		}
		if idx != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem))) y) s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem))) y) s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		s1 := or.Args[0]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		y := or.Args[1]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ or:(ORQ y s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} p idx mem)))) s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		p := x1.Args[0]
		idx := x1.Args[1]
		mem := x1.Args[2]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQ_160(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ORQ or:(ORQ y s1:(SHLQconst [j1] r1:(ROLWconst [8] x1:(MOVWloadidx1 [i1] {s} idx p mem)))) s0:(SHLQconst [j0] r0:(ROLWconst [8] x0:(MOVWloadidx1 [i0] {s} idx p mem))))
	// cond: i1 == i0+2   && j1 == j0-16   && j1 % 32 == 0   && x0.Uses == 1   && x1.Uses == 1   && r0.Uses == 1   && r1.Uses == 1   && s0.Uses == 1   && s1.Uses == 1   && or.Uses == 1   && mergePoint(b,x0,x1) != nil   && clobber(x0)   && clobber(x1)   && clobber(r0)   && clobber(r1)   && clobber(s0)   && clobber(s1)   && clobber(or)
	// result: @mergePoint(b,x0,x1) (ORQ <v.Type> (SHLQconst <v.Type> [j1] (BSWAPL <typ.UInt32> (MOVLloadidx1 [i0] {s} p idx mem))) y)
	for {
		_ = v.Args[1]
		or := v.Args[0]
		if or.Op != OpAMD64ORQ {
			break
		}
		_ = or.Args[1]
		y := or.Args[0]
		s1 := or.Args[1]
		if s1.Op != OpAMD64SHLQconst {
			break
		}
		j1 := s1.AuxInt
		r1 := s1.Args[0]
		if r1.Op != OpAMD64ROLWconst {
			break
		}
		if r1.AuxInt != 8 {
			break
		}
		x1 := r1.Args[0]
		if x1.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i1 := x1.AuxInt
		s := x1.Aux
		_ = x1.Args[2]
		idx := x1.Args[0]
		p := x1.Args[1]
		mem := x1.Args[2]
		s0 := v.Args[1]
		if s0.Op != OpAMD64SHLQconst {
			break
		}
		j0 := s0.AuxInt
		r0 := s0.Args[0]
		if r0.Op != OpAMD64ROLWconst {
			break
		}
		if r0.AuxInt != 8 {
			break
		}
		x0 := r0.Args[0]
		if x0.Op != OpAMD64MOVWloadidx1 {
			break
		}
		i0 := x0.AuxInt
		if x0.Aux != s {
			break
		}
		_ = x0.Args[2]
		if idx != x0.Args[0] {
			break
		}
		if p != x0.Args[1] {
			break
		}
		if mem != x0.Args[2] {
			break
		}
		if !(i1 == i0+2 && j1 == j0-16 && j1%32 == 0 && x0.Uses == 1 && x1.Uses == 1 && r0.Uses == 1 && r1.Uses == 1 && s0.Uses == 1 && s1.Uses == 1 && or.Uses == 1 && mergePoint(b, x0, x1) != nil && clobber(x0) && clobber(x1) && clobber(r0) && clobber(r1) && clobber(s0) && clobber(s1) && clobber(or)) {
			break
		}
		b = mergePoint(b, x0, x1)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SHLQconst, v.Type)
		v1.AuxInt = j1
		v2 := b.NewValue0(v.Pos, OpAMD64BSWAPL, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVLloadidx1, typ.UInt32)
		v3.AuxInt = i0
		v3.Aux = s
		v3.AddArg(p)
		v3.AddArg(idx)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v0.AddArg(y)
		return true
	}
	// match: (ORQ x l:(MOVQload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ORQmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ORQmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (ORQ l:(MOVQload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (ORQmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		l := v.Args[0]
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		x := v.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64ORQmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ORQconst_0(v *Value) bool {
	// match: (ORQconst [0] x)
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
	// match: (ORQconst [-1] _)
	// cond:
	// result: (MOVQconst [-1])
	for {
		if v.AuxInt != -1 {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = -1
		return true
	}
	// match: (ORQconst [c] (MOVQconst [d]))
	// cond:
	// result: (MOVQconst [c|d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = c | d
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ROLB_0(v *Value) bool {
	// match: (ROLB x (NEGQ y))
	// cond:
	// result: (RORB x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ROLB x (NEGL y))
	// cond:
	// result: (RORB x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ROLB x (MOVQconst [c]))
	// cond:
	// result: (ROLBconst [c&7 ] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = c & 7
		v.AddArg(x)
		return true
	}
	// match: (ROLB x (MOVLconst [c]))
	// cond:
	// result: (ROLBconst [c&7 ] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = c & 7
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ROLBconst_0(v *Value) bool {
	// match: (ROLBconst [c] (ROLBconst [d] x))
	// cond:
	// result: (ROLBconst [(c+d)& 7] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ROLBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = (c + d) & 7
		v.AddArg(x)
		return true
	}
	// match: (ROLBconst x [0])
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
	return false
}
func rewriteValueAMD64_OpAMD64ROLL_0(v *Value) bool {
	// match: (ROLL x (NEGQ y))
	// cond:
	// result: (RORL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ROLL x (NEGL y))
	// cond:
	// result: (RORL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ROLL x (MOVQconst [c]))
	// cond:
	// result: (ROLLconst [c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	// match: (ROLL x (MOVLconst [c]))
	// cond:
	// result: (ROLLconst [c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ROLLconst_0(v *Value) bool {
	// match: (ROLLconst [c] (ROLLconst [d] x))
	// cond:
	// result: (ROLLconst [(c+d)&31] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ROLLconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = (c + d) & 31
		v.AddArg(x)
		return true
	}
	// match: (ROLLconst x [0])
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
	return false
}
func rewriteValueAMD64_OpAMD64ROLQ_0(v *Value) bool {
	// match: (ROLQ x (NEGQ y))
	// cond:
	// result: (RORQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ROLQ x (NEGL y))
	// cond:
	// result: (RORQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ROLQ x (MOVQconst [c]))
	// cond:
	// result: (ROLQconst [c&63] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = c & 63
		v.AddArg(x)
		return true
	}
	// match: (ROLQ x (MOVLconst [c]))
	// cond:
	// result: (ROLQconst [c&63] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = c & 63
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ROLQconst_0(v *Value) bool {
	// match: (ROLQconst [c] (ROLQconst [d] x))
	// cond:
	// result: (ROLQconst [(c+d)&63] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ROLQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = (c + d) & 63
		v.AddArg(x)
		return true
	}
	// match: (ROLQconst x [0])
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
	return false
}
func rewriteValueAMD64_OpAMD64ROLW_0(v *Value) bool {
	// match: (ROLW x (NEGQ y))
	// cond:
	// result: (RORW x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ROLW x (NEGL y))
	// cond:
	// result: (RORW x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64RORW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ROLW x (MOVQconst [c]))
	// cond:
	// result: (ROLWconst [c&15] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = c & 15
		v.AddArg(x)
		return true
	}
	// match: (ROLW x (MOVLconst [c]))
	// cond:
	// result: (ROLWconst [c&15] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = c & 15
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64ROLWconst_0(v *Value) bool {
	// match: (ROLWconst [c] (ROLWconst [d] x))
	// cond:
	// result: (ROLWconst [(c+d)&15] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64ROLWconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = (c + d) & 15
		v.AddArg(x)
		return true
	}
	// match: (ROLWconst x [0])
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
	return false
}
func rewriteValueAMD64_OpAMD64RORB_0(v *Value) bool {
	// match: (RORB x (NEGQ y))
	// cond:
	// result: (ROLB x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RORB x (NEGL y))
	// cond:
	// result: (ROLB x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RORB x (MOVQconst [c]))
	// cond:
	// result: (ROLBconst [(-c)&7 ] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = (-c) & 7
		v.AddArg(x)
		return true
	}
	// match: (RORB x (MOVLconst [c]))
	// cond:
	// result: (ROLBconst [(-c)&7 ] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = (-c) & 7
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64RORL_0(v *Value) bool {
	// match: (RORL x (NEGQ y))
	// cond:
	// result: (ROLL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RORL x (NEGL y))
	// cond:
	// result: (ROLL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RORL x (MOVQconst [c]))
	// cond:
	// result: (ROLLconst [(-c)&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = (-c) & 31
		v.AddArg(x)
		return true
	}
	// match: (RORL x (MOVLconst [c]))
	// cond:
	// result: (ROLLconst [(-c)&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = (-c) & 31
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64RORQ_0(v *Value) bool {
	// match: (RORQ x (NEGQ y))
	// cond:
	// result: (ROLQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RORQ x (NEGL y))
	// cond:
	// result: (ROLQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RORQ x (MOVQconst [c]))
	// cond:
	// result: (ROLQconst [(-c)&63] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = (-c) & 63
		v.AddArg(x)
		return true
	}
	// match: (RORQ x (MOVLconst [c]))
	// cond:
	// result: (ROLQconst [(-c)&63] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = (-c) & 63
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64RORW_0(v *Value) bool {
	// match: (RORW x (NEGQ y))
	// cond:
	// result: (ROLW x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RORW x (NEGL y))
	// cond:
	// result: (ROLW x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		y := v_1.Args[0]
		v.reset(OpAMD64ROLW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (RORW x (MOVQconst [c]))
	// cond:
	// result: (ROLWconst [(-c)&15] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = (-c) & 15
		v.AddArg(x)
		return true
	}
	// match: (RORW x (MOVLconst [c]))
	// cond:
	// result: (ROLWconst [(-c)&15] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = (-c) & 15
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARB_0(v *Value) bool {
	// match: (SARB x (MOVQconst [c]))
	// cond:
	// result: (SARBconst [min(c&31,7)] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SARBconst)
		v.AuxInt = min(c&31, 7)
		v.AddArg(x)
		return true
	}
	// match: (SARB x (MOVLconst [c]))
	// cond:
	// result: (SARBconst [min(c&31,7)] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SARBconst)
		v.AuxInt = min(c&31, 7)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARBconst_0(v *Value) bool {
	// match: (SARBconst x [0])
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
	// match: (SARBconst [c] (MOVQconst [d]))
	// cond:
	// result: (MOVQconst [d>>uint64(c)])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = d >> uint64(c)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARL_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (SARL x (MOVQconst [c]))
	// cond:
	// result: (SARLconst [c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SARLconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	// match: (SARL x (MOVLconst [c]))
	// cond:
	// result: (SARLconst [c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SARLconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	// match: (SARL x (ADDQconst [c] y))
	// cond: c & 31 == 0
	// result: (SARL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SARL x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 31 == 0
	// result: (SARL x (NEGQ <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SARL x (ANDQconst [c] y))
	// cond: c & 31 == 31
	// result: (SARL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SARL x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 31 == 31
	// result: (SARL x (NEGQ <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDQconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SARL x (ADDLconst [c] y))
	// cond: c & 31 == 0
	// result: (SARL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SARL x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 31 == 0
	// result: (SARL x (NEGL <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDLconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SARL x (ANDLconst [c] y))
	// cond: c & 31 == 31
	// result: (SARL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SARL x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 31 == 31
	// result: (SARL x (NEGL <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SARL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARLconst_0(v *Value) bool {
	// match: (SARLconst x [0])
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
	// match: (SARLconst [c] (MOVQconst [d]))
	// cond:
	// result: (MOVQconst [d>>uint64(c)])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = d >> uint64(c)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARQ_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (SARQ x (MOVQconst [c]))
	// cond:
	// result: (SARQconst [c&63] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SARQconst)
		v.AuxInt = c & 63
		v.AddArg(x)
		return true
	}
	// match: (SARQ x (MOVLconst [c]))
	// cond:
	// result: (SARQconst [c&63] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SARQconst)
		v.AuxInt = c & 63
		v.AddArg(x)
		return true
	}
	// match: (SARQ x (ADDQconst [c] y))
	// cond: c & 63 == 0
	// result: (SARQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SARQ x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 63 == 0
	// result: (SARQ x (NEGQ <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SARQ x (ANDQconst [c] y))
	// cond: c & 63 == 63
	// result: (SARQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SARQ x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 63 == 63
	// result: (SARQ x (NEGQ <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDQconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SARQ x (ADDLconst [c] y))
	// cond: c & 63 == 0
	// result: (SARQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SARQ x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 63 == 0
	// result: (SARQ x (NEGL <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDLconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SARQ x (ANDLconst [c] y))
	// cond: c & 63 == 63
	// result: (SARQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SARQ x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 63 == 63
	// result: (SARQ x (NEGL <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SARQ)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARQconst_0(v *Value) bool {
	// match: (SARQconst x [0])
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
	// match: (SARQconst [c] (MOVQconst [d]))
	// cond:
	// result: (MOVQconst [d>>uint64(c)])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = d >> uint64(c)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARW_0(v *Value) bool {
	// match: (SARW x (MOVQconst [c]))
	// cond:
	// result: (SARWconst [min(c&31,15)] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SARWconst)
		v.AuxInt = min(c&31, 15)
		v.AddArg(x)
		return true
	}
	// match: (SARW x (MOVLconst [c]))
	// cond:
	// result: (SARWconst [min(c&31,15)] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SARWconst)
		v.AuxInt = min(c&31, 15)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SARWconst_0(v *Value) bool {
	// match: (SARWconst x [0])
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
	// match: (SARWconst [c] (MOVQconst [d]))
	// cond:
	// result: (MOVQconst [d>>uint64(c)])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = d >> uint64(c)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SBBLcarrymask_0(v *Value) bool {
	// match: (SBBLcarrymask (FlagEQ))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SBBLcarrymask (FlagLT_ULT))
	// cond:
	// result: (MOVLconst [-1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = -1
		return true
	}
	// match: (SBBLcarrymask (FlagLT_UGT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SBBLcarrymask (FlagGT_ULT))
	// cond:
	// result: (MOVLconst [-1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = -1
		return true
	}
	// match: (SBBLcarrymask (FlagGT_UGT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SBBQcarrymask_0(v *Value) bool {
	// match: (SBBQcarrymask (FlagEQ))
	// cond:
	// result: (MOVQconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = 0
		return true
	}
	// match: (SBBQcarrymask (FlagLT_ULT))
	// cond:
	// result: (MOVQconst [-1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = -1
		return true
	}
	// match: (SBBQcarrymask (FlagLT_UGT))
	// cond:
	// result: (MOVQconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = 0
		return true
	}
	// match: (SBBQcarrymask (FlagGT_ULT))
	// cond:
	// result: (MOVQconst [-1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = -1
		return true
	}
	// match: (SBBQcarrymask (FlagGT_UGT))
	// cond:
	// result: (MOVQconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETA_0(v *Value) bool {
	// match: (SETA (InvertFlags x))
	// cond:
	// result: (SETB x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETB)
		v.AddArg(x)
		return true
	}
	// match: (SETA (FlagEQ))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETA (FlagLT_ULT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETA (FlagLT_UGT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETA (FlagGT_ULT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETA (FlagGT_UGT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETAE_0(v *Value) bool {
	// match: (SETAE (InvertFlags x))
	// cond:
	// result: (SETBE x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETBE)
		v.AddArg(x)
		return true
	}
	// match: (SETAE (FlagEQ))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETAE (FlagLT_ULT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETAE (FlagLT_UGT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETAE (FlagGT_ULT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETAE (FlagGT_UGT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETB_0(v *Value) bool {
	// match: (SETB (InvertFlags x))
	// cond:
	// result: (SETA x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETA)
		v.AddArg(x)
		return true
	}
	// match: (SETB (FlagEQ))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETB (FlagLT_ULT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETB (FlagLT_UGT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETB (FlagGT_ULT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETB (FlagGT_UGT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETBE_0(v *Value) bool {
	// match: (SETBE (InvertFlags x))
	// cond:
	// result: (SETAE x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETAE)
		v.AddArg(x)
		return true
	}
	// match: (SETBE (FlagEQ))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETBE (FlagLT_ULT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETBE (FlagLT_UGT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETBE (FlagGT_ULT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETBE (FlagGT_UGT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETEQ_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (SETEQ (TESTL (SHLL (MOVLconst [1]) x) y))
	// cond: !config.nacl
	// result: (SETAE (BTL x y))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64MOVLconst {
			break
		}
		if v_0_0_0.AuxInt != 1 {
			break
		}
		x := v_0_0.Args[1]
		y := v_0.Args[1]
		if !(!config.nacl) {
			break
		}
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64BTL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (TESTL y (SHLL (MOVLconst [1]) x)))
	// cond: !config.nacl
	// result: (SETAE (BTL x y))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_0_1.Args[1]
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64MOVLconst {
			break
		}
		if v_0_1_0.AuxInt != 1 {
			break
		}
		x := v_0_1.Args[1]
		if !(!config.nacl) {
			break
		}
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64BTL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (TESTQ (SHLQ (MOVQconst [1]) x) y))
	// cond: !config.nacl
	// result: (SETAE (BTQ x y))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHLQ {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64MOVQconst {
			break
		}
		if v_0_0_0.AuxInt != 1 {
			break
		}
		x := v_0_0.Args[1]
		y := v_0.Args[1]
		if !(!config.nacl) {
			break
		}
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (TESTQ y (SHLQ (MOVQconst [1]) x)))
	// cond: !config.nacl
	// result: (SETAE (BTQ x y))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHLQ {
			break
		}
		_ = v_0_1.Args[1]
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64MOVQconst {
			break
		}
		if v_0_1_0.AuxInt != 1 {
			break
		}
		x := v_0_1.Args[1]
		if !(!config.nacl) {
			break
		}
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (TESTLconst [c] x))
	// cond: isPowerOfTwo(c) && log2(c) < 32 && !config.nacl
	// result: (SETAE (BTLconst [log2(c)] x))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		if !(isPowerOfTwo(c) && log2(c) < 32 && !config.nacl) {
			break
		}
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (TESTQconst [c] x))
	// cond: isPowerOfTwo(c) && log2(c) < 64 && !config.nacl
	// result: (SETAE (BTQconst [log2(c)] x))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTQconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		if !(isPowerOfTwo(c) && log2(c) < 64 && !config.nacl) {
			break
		}
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (TESTQ (MOVQconst [c]) x))
	// cond: isPowerOfTwo(c) && log2(c) < 64 && !config.nacl
	// result: (SETAE (BTQconst [log2(c)] x))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0_0.AuxInt
		x := v_0.Args[1]
		if !(isPowerOfTwo(c) && log2(c) < 64 && !config.nacl) {
			break
		}
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (TESTQ x (MOVQconst [c])))
	// cond: isPowerOfTwo(c) && log2(c) < 64 && !config.nacl
	// result: (SETAE (BTQconst [log2(c)] x))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0_1.AuxInt
		if !(isPowerOfTwo(c) && log2(c) < 64 && !config.nacl) {
			break
		}
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETEQ (InvertFlags x))
	// cond:
	// result: (SETEQ x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETEQ)
		v.AddArg(x)
		return true
	}
	// match: (SETEQ (FlagEQ))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETEQ_10(v *Value) bool {
	// match: (SETEQ (FlagLT_ULT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETEQ (FlagLT_UGT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETEQ (FlagGT_ULT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETEQ (FlagGT_UGT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETG_0(v *Value) bool {
	// match: (SETG (InvertFlags x))
	// cond:
	// result: (SETL x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETL)
		v.AddArg(x)
		return true
	}
	// match: (SETG (FlagEQ))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETG (FlagLT_ULT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETG (FlagLT_UGT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETG (FlagGT_ULT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETG (FlagGT_UGT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETGE_0(v *Value) bool {
	// match: (SETGE (InvertFlags x))
	// cond:
	// result: (SETLE x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETLE)
		v.AddArg(x)
		return true
	}
	// match: (SETGE (FlagEQ))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETGE (FlagLT_ULT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETGE (FlagLT_UGT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETGE (FlagGT_ULT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETGE (FlagGT_UGT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETL_0(v *Value) bool {
	// match: (SETL (InvertFlags x))
	// cond:
	// result: (SETG x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETG)
		v.AddArg(x)
		return true
	}
	// match: (SETL (FlagEQ))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETL (FlagLT_ULT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETL (FlagLT_UGT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETL (FlagGT_ULT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETL (FlagGT_UGT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETLE_0(v *Value) bool {
	// match: (SETLE (InvertFlags x))
	// cond:
	// result: (SETGE x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETGE)
		v.AddArg(x)
		return true
	}
	// match: (SETLE (FlagEQ))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETLE (FlagLT_ULT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETLE (FlagLT_UGT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETLE (FlagGT_ULT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SETLE (FlagGT_UGT))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETNE_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (SETNE (TESTL (SHLL (MOVLconst [1]) x) y))
	// cond: !config.nacl
	// result: (SETB  (BTL x y))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHLL {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64MOVLconst {
			break
		}
		if v_0_0_0.AuxInt != 1 {
			break
		}
		x := v_0_0.Args[1]
		y := v_0.Args[1]
		if !(!config.nacl) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64BTL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (TESTL y (SHLL (MOVLconst [1]) x)))
	// cond: !config.nacl
	// result: (SETB  (BTL x y))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTL {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHLL {
			break
		}
		_ = v_0_1.Args[1]
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64MOVLconst {
			break
		}
		if v_0_1_0.AuxInt != 1 {
			break
		}
		x := v_0_1.Args[1]
		if !(!config.nacl) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64BTL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (TESTQ (SHLQ (MOVQconst [1]) x) y))
	// cond: !config.nacl
	// result: (SETB  (BTQ x y))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64SHLQ {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpAMD64MOVQconst {
			break
		}
		if v_0_0_0.AuxInt != 1 {
			break
		}
		x := v_0_0.Args[1]
		y := v_0.Args[1]
		if !(!config.nacl) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (TESTQ y (SHLQ (MOVQconst [1]) x)))
	// cond: !config.nacl
	// result: (SETB  (BTQ x y))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64SHLQ {
			break
		}
		_ = v_0_1.Args[1]
		v_0_1_0 := v_0_1.Args[0]
		if v_0_1_0.Op != OpAMD64MOVQconst {
			break
		}
		if v_0_1_0.AuxInt != 1 {
			break
		}
		x := v_0_1.Args[1]
		if !(!config.nacl) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (TESTLconst [c] x))
	// cond: isPowerOfTwo(c) && log2(c) < 32 && !config.nacl
	// result: (SETB  (BTLconst [log2(c)] x))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		if !(isPowerOfTwo(c) && log2(c) < 32 && !config.nacl) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (TESTQconst [c] x))
	// cond: isPowerOfTwo(c) && log2(c) < 64 && !config.nacl
	// result: (SETB  (BTQconst [log2(c)] x))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTQconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		if !(isPowerOfTwo(c) && log2(c) < 64 && !config.nacl) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (TESTQ (MOVQconst [c]) x))
	// cond: isPowerOfTwo(c) && log2(c) < 64 && !config.nacl
	// result: (SETB  (BTQconst [log2(c)] x))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0_0.AuxInt
		x := v_0.Args[1]
		if !(isPowerOfTwo(c) && log2(c) < 64 && !config.nacl) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (TESTQ x (MOVQconst [c])))
	// cond: isPowerOfTwo(c) && log2(c) < 64 && !config.nacl
	// result: (SETB  (BTQconst [log2(c)] x))
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64TESTQ {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0_1.AuxInt
		if !(isPowerOfTwo(c) && log2(c) < 64 && !config.nacl) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
		v0.AuxInt = log2(c)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SETNE (InvertFlags x))
	// cond:
	// result: (SETNE x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64InvertFlags {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETNE)
		v.AddArg(x)
		return true
	}
	// match: (SETNE (FlagEQ))
	// cond:
	// result: (MOVLconst [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagEQ {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SETNE_10(v *Value) bool {
	// match: (SETNE (FlagLT_ULT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETNE (FlagLT_UGT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagLT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETNE (FlagGT_ULT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_ULT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	// match: (SETNE (FlagGT_UGT))
	// cond:
	// result: (MOVLconst [1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64FlagGT_UGT {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 1
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHLL_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (SHLL x (MOVQconst [c]))
	// cond:
	// result: (SHLLconst [c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SHLLconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	// match: (SHLL x (MOVLconst [c]))
	// cond:
	// result: (SHLLconst [c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SHLLconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	// match: (SHLL x (ADDQconst [c] y))
	// cond: c & 31 == 0
	// result: (SHLL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHLL x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 31 == 0
	// result: (SHLL x (NEGQ <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SHLL x (ANDQconst [c] y))
	// cond: c & 31 == 31
	// result: (SHLL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHLL x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 31 == 31
	// result: (SHLL x (NEGQ <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDQconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SHLL x (ADDLconst [c] y))
	// cond: c & 31 == 0
	// result: (SHLL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHLL x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 31 == 0
	// result: (SHLL x (NEGL <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDLconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SHLL x (ANDLconst [c] y))
	// cond: c & 31 == 31
	// result: (SHLL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHLL x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 31 == 31
	// result: (SHLL x (NEGL <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHLL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHLLconst_0(v *Value) bool {
	// match: (SHLLconst x [0])
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
	return false
}
func rewriteValueAMD64_OpAMD64SHLQ_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (SHLQ x (MOVQconst [c]))
	// cond:
	// result: (SHLQconst [c&63] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = c & 63
		v.AddArg(x)
		return true
	}
	// match: (SHLQ x (MOVLconst [c]))
	// cond:
	// result: (SHLQconst [c&63] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SHLQconst)
		v.AuxInt = c & 63
		v.AddArg(x)
		return true
	}
	// match: (SHLQ x (ADDQconst [c] y))
	// cond: c & 63 == 0
	// result: (SHLQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHLQ x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 63 == 0
	// result: (SHLQ x (NEGQ <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SHLQ x (ANDQconst [c] y))
	// cond: c & 63 == 63
	// result: (SHLQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHLQ x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 63 == 63
	// result: (SHLQ x (NEGQ <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDQconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SHLQ x (ADDLconst [c] y))
	// cond: c & 63 == 0
	// result: (SHLQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHLQ x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 63 == 0
	// result: (SHLQ x (NEGL <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDLconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SHLQ x (ANDLconst [c] y))
	// cond: c & 63 == 63
	// result: (SHLQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHLQ x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 63 == 63
	// result: (SHLQ x (NEGL <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHLQ)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHLQconst_0(v *Value) bool {
	// match: (SHLQconst x [0])
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
	return false
}
func rewriteValueAMD64_OpAMD64SHRB_0(v *Value) bool {
	// match: (SHRB x (MOVQconst [c]))
	// cond: c&31 < 8
	// result: (SHRBconst [c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		if !(c&31 < 8) {
			break
		}
		v.reset(OpAMD64SHRBconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	// match: (SHRB x (MOVLconst [c]))
	// cond: c&31 < 8
	// result: (SHRBconst [c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		if !(c&31 < 8) {
			break
		}
		v.reset(OpAMD64SHRBconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	// match: (SHRB _ (MOVQconst [c]))
	// cond: c&31 >= 8
	// result: (MOVLconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		if !(c&31 >= 8) {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SHRB _ (MOVLconst [c]))
	// cond: c&31 >= 8
	// result: (MOVLconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		if !(c&31 >= 8) {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHRBconst_0(v *Value) bool {
	// match: (SHRBconst x [0])
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
	return false
}
func rewriteValueAMD64_OpAMD64SHRL_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (SHRL x (MOVQconst [c]))
	// cond:
	// result: (SHRLconst [c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SHRLconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	// match: (SHRL x (MOVLconst [c]))
	// cond:
	// result: (SHRLconst [c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SHRLconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	// match: (SHRL x (ADDQconst [c] y))
	// cond: c & 31 == 0
	// result: (SHRL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHRL x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 31 == 0
	// result: (SHRL x (NEGQ <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SHRL x (ANDQconst [c] y))
	// cond: c & 31 == 31
	// result: (SHRL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHRL x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 31 == 31
	// result: (SHRL x (NEGQ <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDQconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SHRL x (ADDLconst [c] y))
	// cond: c & 31 == 0
	// result: (SHRL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHRL x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 31 == 0
	// result: (SHRL x (NEGL <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDLconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&31 == 0) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SHRL x (ANDLconst [c] y))
	// cond: c & 31 == 31
	// result: (SHRL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHRL x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 31 == 31
	// result: (SHRL x (NEGL <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&31 == 31) {
			break
		}
		v.reset(OpAMD64SHRL)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHRLconst_0(v *Value) bool {
	// match: (SHRLconst x [0])
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
	return false
}
func rewriteValueAMD64_OpAMD64SHRQ_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (SHRQ x (MOVQconst [c]))
	// cond:
	// result: (SHRQconst [c&63] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SHRQconst)
		v.AuxInt = c & 63
		v.AddArg(x)
		return true
	}
	// match: (SHRQ x (MOVLconst [c]))
	// cond:
	// result: (SHRQconst [c&63] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SHRQconst)
		v.AuxInt = c & 63
		v.AddArg(x)
		return true
	}
	// match: (SHRQ x (ADDQconst [c] y))
	// cond: c & 63 == 0
	// result: (SHRQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHRQ x (NEGQ <t> (ADDQconst [c] y)))
	// cond: c & 63 == 0
	// result: (SHRQ x (NEGQ <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDQconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SHRQ x (ANDQconst [c] y))
	// cond: c & 63 == 63
	// result: (SHRQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDQconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHRQ x (NEGQ <t> (ANDQconst [c] y)))
	// cond: c & 63 == 63
	// result: (SHRQ x (NEGQ <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGQ {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDQconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SHRQ x (ADDLconst [c] y))
	// cond: c & 63 == 0
	// result: (SHRQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHRQ x (NEGL <t> (ADDLconst [c] y)))
	// cond: c & 63 == 0
	// result: (SHRQ x (NEGL <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ADDLconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&63 == 0) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (SHRQ x (ANDLconst [c] y))
	// cond: c & 63 == 63
	// result: (SHRQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ANDLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SHRQ x (NEGL <t> (ANDLconst [c] y)))
	// cond: c & 63 == 63
	// result: (SHRQ x (NEGL <t> y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64NEGL {
			break
		}
		t := v_1.Type
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpAMD64ANDLconst {
			break
		}
		c := v_1_0.AuxInt
		y := v_1_0.Args[0]
		if !(c&63 == 63) {
			break
		}
		v.reset(OpAMD64SHRQ)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64NEGL, t)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHRQconst_0(v *Value) bool {
	// match: (SHRQconst x [0])
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
	return false
}
func rewriteValueAMD64_OpAMD64SHRW_0(v *Value) bool {
	// match: (SHRW x (MOVQconst [c]))
	// cond: c&31 < 16
	// result: (SHRWconst [c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		if !(c&31 < 16) {
			break
		}
		v.reset(OpAMD64SHRWconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	// match: (SHRW x (MOVLconst [c]))
	// cond: c&31 < 16
	// result: (SHRWconst [c&31] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		if !(c&31 < 16) {
			break
		}
		v.reset(OpAMD64SHRWconst)
		v.AuxInt = c & 31
		v.AddArg(x)
		return true
	}
	// match: (SHRW _ (MOVQconst [c]))
	// cond: c&31 >= 16
	// result: (MOVLconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		if !(c&31 >= 16) {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SHRW _ (MOVLconst [c]))
	// cond: c&31 >= 16
	// result: (MOVLconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		if !(c&31 >= 16) {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SHRWconst_0(v *Value) bool {
	// match: (SHRWconst x [0])
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
	return false
}
func rewriteValueAMD64_OpAMD64SUBL_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (SUBL x (MOVLconst [c]))
	// cond:
	// result: (SUBLconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64SUBLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (SUBL (MOVLconst [c]) x)
	// cond:
	// result: (NEGL (SUBLconst <v.Type> x [c]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpAMD64NEGL)
		v0 := b.NewValue0(v.Pos, OpAMD64SUBLconst, v.Type)
		v0.AuxInt = c
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBL x x)
	// cond:
	// result: (MOVLconst [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (SUBL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (SUBLmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SUBLmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBLconst_0(v *Value) bool {
	// match: (SUBLconst [c] x)
	// cond: int32(c) == 0
	// result: x
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (SUBLconst [c] x)
	// cond:
	// result: (ADDLconst [int64(int32(-c))] x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = int64(int32(-c))
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpAMD64SUBQ_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (SUBQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (SUBQconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64SUBQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (SUBQ (MOVQconst [c]) x)
	// cond: is32Bit(c)
	// result: (NEGQ (SUBQconst <v.Type> x [c]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64NEGQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SUBQconst, v.Type)
		v0.AuxInt = c
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (SUBQ x x)
	// cond:
	// result: (MOVQconst [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = 0
		return true
	}
	// match: (SUBQ x l:(MOVQload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (SUBQmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SUBQmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBQconst_0(v *Value) bool {
	// match: (SUBQconst [0] x)
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
	// match: (SUBQconst [c] x)
	// cond: c != -(1<<31)
	// result: (ADDQconst [-c] x)
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(c != -(1 << 31)) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = -c
		v.AddArg(x)
		return true
	}
	// match: (SUBQconst (MOVQconst [d]) [c])
	// cond:
	// result: (MOVQconst [d-c])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = d - c
		return true
	}
	// match: (SUBQconst (SUBQconst x [d]) [c])
	// cond: is32Bit(-c-d)
	// result: (ADDQconst [-c-d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SUBQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		if !(is32Bit(-c - d)) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = -c - d
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBSD_0(v *Value) bool {
	// match: (SUBSD x l:(MOVSDload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (SUBSDmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVSDload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SUBSDmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64SUBSS_0(v *Value) bool {
	// match: (SUBSS x l:(MOVSSload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (SUBSSmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVSSload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64SUBSSmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64TESTB_0(v *Value) bool {
	// match: (TESTB (MOVLconst [c]) x)
	// cond:
	// result: (TESTBconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpAMD64TESTBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (TESTB x (MOVLconst [c]))
	// cond:
	// result: (TESTBconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64TESTBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64TESTL_0(v *Value) bool {
	// match: (TESTL (MOVLconst [c]) x)
	// cond:
	// result: (TESTLconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpAMD64TESTLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (TESTL x (MOVLconst [c]))
	// cond:
	// result: (TESTLconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64TESTLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64TESTQ_0(v *Value) bool {
	// match: (TESTQ (MOVQconst [c]) x)
	// cond: is32Bit(c)
	// result: (TESTQconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64TESTQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (TESTQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (TESTQconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64TESTQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64TESTW_0(v *Value) bool {
	// match: (TESTW (MOVLconst [c]) x)
	// cond:
	// result: (TESTWconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpAMD64TESTWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (TESTW x (MOVLconst [c]))
	// cond:
	// result: (TESTWconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64TESTWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XADDLlock_0(v *Value) bool {
	// match: (XADDLlock [off1] {sym} val (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (XADDLlock [off1+off2] {sym} val ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		val := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_1.AuxInt
		ptr := v_1.Args[0]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64XADDLlock)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(val)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XADDQlock_0(v *Value) bool {
	// match: (XADDQlock [off1] {sym} val (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (XADDQlock [off1+off2] {sym} val ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		val := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_1.AuxInt
		ptr := v_1.Args[0]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64XADDQlock)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(val)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XCHGL_0(v *Value) bool {
	// match: (XCHGL [off1] {sym} val (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (XCHGL [off1+off2] {sym} val ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		val := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_1.AuxInt
		ptr := v_1.Args[0]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64XCHGL)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(val)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (XCHGL [off1] {sym1} val (LEAQ [off2] {sym2} ptr) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2) && ptr.Op != OpSB
	// result: (XCHGL [off1+off2] {mergeSym(sym1,sym2)} val ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		val := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		ptr := v_1.Args[0]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2) && ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64XCHGL)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(val)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XCHGQ_0(v *Value) bool {
	// match: (XCHGQ [off1] {sym} val (ADDQconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2)
	// result: (XCHGQ [off1+off2] {sym} val ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		val := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64ADDQconst {
			break
		}
		off2 := v_1.AuxInt
		ptr := v_1.Args[0]
		mem := v.Args[2]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpAMD64XCHGQ)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(val)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (XCHGQ [off1] {sym1} val (LEAQ [off2] {sym2} ptr) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2) && ptr.Op != OpSB
	// result: (XCHGQ [off1+off2] {mergeSym(sym1,sym2)} val ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		val := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64LEAQ {
			break
		}
		off2 := v_1.AuxInt
		sym2 := v_1.Aux
		ptr := v_1.Args[0]
		mem := v.Args[2]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2) && ptr.Op != OpSB) {
			break
		}
		v.reset(OpAMD64XCHGQ)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(val)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORL_0(v *Value) bool {
	// match: (XORL x (MOVLconst [c]))
	// cond:
	// result: (XORLconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVLconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAMD64XORLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORL (MOVLconst [c]) x)
	// cond:
	// result: (XORLconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpAMD64XORLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORL (SHLLconst x [c]) (SHRLconst x [d]))
	// cond: d==32-c
	// result: (ROLLconst x [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRLconst {
			break
		}
		d := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 32-c) {
			break
		}
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORL (SHRLconst x [d]) (SHLLconst x [c]))
	// cond: d==32-c
	// result: (ROLLconst x [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRLconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 32-c) {
			break
		}
		v.reset(OpAMD64ROLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORL <t> (SHLLconst x [c]) (SHRWconst x [d]))
	// cond: d==16-c && c < 16 && t.Size() == 2
	// result: (ROLWconst x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRWconst {
			break
		}
		d := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 16-c && c < 16 && t.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORL <t> (SHRWconst x [d]) (SHLLconst x [c]))
	// cond: d==16-c && c < 16 && t.Size() == 2
	// result: (ROLWconst x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRWconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 16-c && c < 16 && t.Size() == 2) {
			break
		}
		v.reset(OpAMD64ROLWconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORL <t> (SHLLconst x [c]) (SHRBconst x [d]))
	// cond: d==8-c  && c < 8 && t.Size() == 1
	// result: (ROLBconst x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLLconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRBconst {
			break
		}
		d := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 8-c && c < 8 && t.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORL <t> (SHRBconst x [d]) (SHLLconst x [c]))
	// cond: d==8-c  && c < 8 && t.Size() == 1
	// result: (ROLBconst x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRBconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLLconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 8-c && c < 8 && t.Size() == 1) {
			break
		}
		v.reset(OpAMD64ROLBconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORL x x)
	// cond:
	// result: (MOVLconst [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	// match: (XORL x l:(MOVLload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (XORLmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64XORLmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORL_10(v *Value) bool {
	// match: (XORL l:(MOVLload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (XORLmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		l := v.Args[0]
		if l.Op != OpAMD64MOVLload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		x := v.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64XORLmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORLconst_0(v *Value) bool {
	// match: (XORLconst [1] (SETNE x))
	// cond:
	// result: (SETEQ x)
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SETNE {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETEQ)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETEQ x))
	// cond:
	// result: (SETNE x)
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SETEQ {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETNE)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETL x))
	// cond:
	// result: (SETGE x)
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SETL {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETGE)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETGE x))
	// cond:
	// result: (SETL  x)
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SETGE {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETL)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETLE x))
	// cond:
	// result: (SETG  x)
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SETLE {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETG)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETG x))
	// cond:
	// result: (SETLE x)
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SETG {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETLE)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETB x))
	// cond:
	// result: (SETAE x)
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SETB {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETAE)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETAE x))
	// cond:
	// result: (SETB  x)
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SETAE {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETB)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETBE x))
	// cond:
	// result: (SETA  x)
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SETBE {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETA)
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [1] (SETA x))
	// cond:
	// result: (SETBE x)
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SETA {
			break
		}
		x := v_0.Args[0]
		v.reset(OpAMD64SETBE)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORLconst_10(v *Value) bool {
	// match: (XORLconst [c] (XORLconst [d] x))
	// cond:
	// result: (XORLconst [c ^ d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64XORLconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64XORLconst)
		v.AuxInt = c ^ d
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [c] x)
	// cond: int32(c)==0
	// result: x
	for {
		c := v.AuxInt
		x := v.Args[0]
		if !(int32(c) == 0) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (XORLconst [c] (MOVLconst [d]))
	// cond:
	// result: (MOVLconst [c^d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVLconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = c ^ d
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORQ_0(v *Value) bool {
	// match: (XORQ x (MOVQconst [c]))
	// cond: is32Bit(c)
	// result: (XORQconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64MOVQconst {
			break
		}
		c := v_1.AuxInt
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64XORQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORQ (MOVQconst [c]) x)
	// cond: is32Bit(c)
	// result: (XORQconst [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(is32Bit(c)) {
			break
		}
		v.reset(OpAMD64XORQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORQ (SHLQconst x [c]) (SHRQconst x [d]))
	// cond: d==64-c
	// result: (ROLQconst x [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHLQconst {
			break
		}
		c := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHRQconst {
			break
		}
		d := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 64-c) {
			break
		}
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORQ (SHRQconst x [d]) (SHLQconst x [c]))
	// cond: d==64-c
	// result: (ROLQconst x [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64SHRQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAMD64SHLQconst {
			break
		}
		c := v_1.AuxInt
		if x != v_1.Args[0] {
			break
		}
		if !(d == 64-c) {
			break
		}
		v.reset(OpAMD64ROLQconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORQ x x)
	// cond:
	// result: (MOVQconst [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = 0
		return true
	}
	// match: (XORQ x l:(MOVQload [off] {sym} ptr mem))
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (XORQmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		l := v.Args[1]
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64XORQmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (XORQ l:(MOVQload [off] {sym} ptr mem) x)
	// cond: canMergeLoad(v, l, x) && clobber(l)
	// result: (XORQmem x [off] {sym} ptr mem)
	for {
		_ = v.Args[1]
		l := v.Args[0]
		if l.Op != OpAMD64MOVQload {
			break
		}
		off := l.AuxInt
		sym := l.Aux
		_ = l.Args[1]
		ptr := l.Args[0]
		mem := l.Args[1]
		x := v.Args[1]
		if !(canMergeLoad(v, l, x) && clobber(l)) {
			break
		}
		v.reset(OpAMD64XORQmem)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(x)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAMD64XORQconst_0(v *Value) bool {
	// match: (XORQconst [c] (XORQconst [d] x))
	// cond:
	// result: (XORQconst [c ^ d] x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64XORQconst {
			break
		}
		d := v_0.AuxInt
		x := v_0.Args[0]
		v.reset(OpAMD64XORQconst)
		v.AuxInt = c ^ d
		v.AddArg(x)
		return true
	}
	// match: (XORQconst [0] x)
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
	// match: (XORQconst [c] (MOVQconst [d]))
	// cond:
	// result: (MOVQconst [c^d])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64MOVQconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = c ^ d
		return true
	}
	return false
}
func rewriteValueAMD64_OpAdd16_0(v *Value) bool {
	// match: (Add16 x y)
	// cond:
	// result: (ADDL  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ADDL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpAdd32_0(v *Value) bool {
	// match: (Add32 x y)
	// cond:
	// result: (ADDL  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ADDL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpAdd32F_0(v *Value) bool {
	// match: (Add32F x y)
	// cond:
	// result: (ADDSS x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ADDSS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpAdd64_0(v *Value) bool {
	// match: (Add64 x y)
	// cond:
	// result: (ADDQ  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ADDQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpAdd64F_0(v *Value) bool {
	// match: (Add64F x y)
	// cond:
	// result: (ADDSD x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ADDSD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpAdd8_0(v *Value) bool {
	// match: (Add8 x y)
	// cond:
	// result: (ADDL  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ADDL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpAddPtr_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (AddPtr x y)
	// cond: config.PtrSize == 8
	// result: (ADDQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64ADDQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AddPtr x y)
	// cond: config.PtrSize == 4
	// result: (ADDL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64ADDL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAddr_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (Addr {sym} base)
	// cond: config.PtrSize == 8
	// result: (LEAQ {sym} base)
	for {
		sym := v.Aux
		base := v.Args[0]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64LEAQ)
		v.Aux = sym
		v.AddArg(base)
		return true
	}
	// match: (Addr {sym} base)
	// cond: config.PtrSize == 4
	// result: (LEAL {sym} base)
	for {
		sym := v.Aux
		base := v.Args[0]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64LEAL)
		v.Aux = sym
		v.AddArg(base)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAnd16_0(v *Value) bool {
	// match: (And16 x y)
	// cond:
	// result: (ANDL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpAnd32_0(v *Value) bool {
	// match: (And32 x y)
	// cond:
	// result: (ANDL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpAnd64_0(v *Value) bool {
	// match: (And64 x y)
	// cond:
	// result: (ANDQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpAnd8_0(v *Value) bool {
	// match: (And8 x y)
	// cond:
	// result: (ANDL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpAndB_0(v *Value) bool {
	// match: (AndB x y)
	// cond:
	// result: (ANDL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpAtomicAdd32_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (AtomicAdd32 ptr val mem)
	// cond:
	// result: (AddTupleFirst32 val (XADDLlock val ptr mem))
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64AddTupleFirst32)
		v.AddArg(val)
		v0 := b.NewValue0(v.Pos, OpAMD64XADDLlock, types.NewTuple(typ.UInt32, types.TypeMem))
		v0.AddArg(val)
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpAtomicAdd64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (AtomicAdd64 ptr val mem)
	// cond:
	// result: (AddTupleFirst64 val (XADDQlock val ptr mem))
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64AddTupleFirst64)
		v.AddArg(val)
		v0 := b.NewValue0(v.Pos, OpAMD64XADDQlock, types.NewTuple(typ.UInt64, types.TypeMem))
		v0.AddArg(val)
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpAtomicAnd8_0(v *Value) bool {
	// match: (AtomicAnd8 ptr val mem)
	// cond:
	// result: (ANDBlock ptr val mem)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64ANDBlock)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicCompareAndSwap32_0(v *Value) bool {
	// match: (AtomicCompareAndSwap32 ptr old new_ mem)
	// cond:
	// result: (CMPXCHGLlock ptr old new_ mem)
	for {
		_ = v.Args[3]
		ptr := v.Args[0]
		old := v.Args[1]
		new_ := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64CMPXCHGLlock)
		v.AddArg(ptr)
		v.AddArg(old)
		v.AddArg(new_)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicCompareAndSwap64_0(v *Value) bool {
	// match: (AtomicCompareAndSwap64 ptr old new_ mem)
	// cond:
	// result: (CMPXCHGQlock ptr old new_ mem)
	for {
		_ = v.Args[3]
		ptr := v.Args[0]
		old := v.Args[1]
		new_ := v.Args[2]
		mem := v.Args[3]
		v.reset(OpAMD64CMPXCHGQlock)
		v.AddArg(ptr)
		v.AddArg(old)
		v.AddArg(new_)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicExchange32_0(v *Value) bool {
	// match: (AtomicExchange32 ptr val mem)
	// cond:
	// result: (XCHGL val ptr mem)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64XCHGL)
		v.AddArg(val)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicExchange64_0(v *Value) bool {
	// match: (AtomicExchange64 ptr val mem)
	// cond:
	// result: (XCHGQ val ptr mem)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64XCHGQ)
		v.AddArg(val)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicLoad32_0(v *Value) bool {
	// match: (AtomicLoad32 ptr mem)
	// cond:
	// result: (MOVLatomicload ptr mem)
	for {
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVLatomicload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicLoad64_0(v *Value) bool {
	// match: (AtomicLoad64 ptr mem)
	// cond:
	// result: (MOVQatomicload ptr mem)
	for {
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVQatomicload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicLoadPtr_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (AtomicLoadPtr ptr mem)
	// cond: config.PtrSize == 8
	// result: (MOVQatomicload ptr mem)
	for {
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64MOVQatomicload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (AtomicLoadPtr ptr mem)
	// cond: config.PtrSize == 4
	// result: (MOVLatomicload ptr mem)
	for {
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64MOVLatomicload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAtomicOr8_0(v *Value) bool {
	// match: (AtomicOr8 ptr val mem)
	// cond:
	// result: (ORBlock ptr val mem)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64ORBlock)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpAtomicStore32_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (AtomicStore32 ptr val mem)
	// cond:
	// result: (Select1 (XCHGL <types.NewTuple(typ.UInt32,types.TypeMem)> val ptr mem))
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64XCHGL, types.NewTuple(typ.UInt32, types.TypeMem))
		v0.AddArg(val)
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpAtomicStore64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (AtomicStore64 ptr val mem)
	// cond:
	// result: (Select1 (XCHGQ <types.NewTuple(typ.UInt64,types.TypeMem)> val ptr mem))
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64XCHGQ, types.NewTuple(typ.UInt64, types.TypeMem))
		v0.AddArg(val)
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpAtomicStorePtrNoWB_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (AtomicStorePtrNoWB ptr val mem)
	// cond: config.PtrSize == 8
	// result: (Select1 (XCHGQ <types.NewTuple(typ.BytePtr,types.TypeMem)> val ptr mem))
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64XCHGQ, types.NewTuple(typ.BytePtr, types.TypeMem))
		v0.AddArg(val)
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
	// match: (AtomicStorePtrNoWB ptr val mem)
	// cond: config.PtrSize == 4
	// result: (Select1 (XCHGL <types.NewTuple(typ.BytePtr,types.TypeMem)> val ptr mem))
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64XCHGL, types.NewTuple(typ.BytePtr, types.TypeMem))
		v0.AddArg(val)
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpAvg64u_0(v *Value) bool {
	// match: (Avg64u x y)
	// cond:
	// result: (AVGQU x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64AVGQU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpBitLen32_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (BitLen32 x)
	// cond:
	// result: (BitLen64 (MOVLQZX <typ.UInt64> x))
	for {
		x := v.Args[0]
		v.reset(OpBitLen64)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLQZX, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpBitLen64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (BitLen64 <t> x)
	// cond:
	// result: (ADDQconst [1] (CMOVQEQ <t> (Select0 <t> (BSRQ x)) (MOVQconst <t> [-1]) (Select1 <types.TypeFlags> (BSRQ x))))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = 1
		v0 := b.NewValue0(v.Pos, OpAMD64CMOVQEQ, t)
		v1 := b.NewValue0(v.Pos, OpSelect0, t)
		v2 := b.NewValue0(v.Pos, OpAMD64BSRQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v2.AddArg(x)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVQconst, t)
		v3.AuxInt = -1
		v0.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v5 := b.NewValue0(v.Pos, OpAMD64BSRQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v5.AddArg(x)
		v4.AddArg(v5)
		v0.AddArg(v4)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpBswap32_0(v *Value) bool {
	// match: (Bswap32 x)
	// cond:
	// result: (BSWAPL x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64BSWAPL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpBswap64_0(v *Value) bool {
	// match: (Bswap64 x)
	// cond:
	// result: (BSWAPQ x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64BSWAPQ)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpClosureCall_0(v *Value) bool {
	// match: (ClosureCall [argwid] entry closure mem)
	// cond:
	// result: (CALLclosure [argwid] entry closure mem)
	for {
		argwid := v.AuxInt
		_ = v.Args[2]
		entry := v.Args[0]
		closure := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64CALLclosure)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(closure)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpCom16_0(v *Value) bool {
	// match: (Com16 x)
	// cond:
	// result: (NOTL x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64NOTL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCom32_0(v *Value) bool {
	// match: (Com32 x)
	// cond:
	// result: (NOTL x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64NOTL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCom64_0(v *Value) bool {
	// match: (Com64 x)
	// cond:
	// result: (NOTQ x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64NOTQ)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCom8_0(v *Value) bool {
	// match: (Com8 x)
	// cond:
	// result: (NOTL x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64NOTL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpConst16_0(v *Value) bool {
	// match: (Const16 [val])
	// cond:
	// result: (MOVLconst [val])
	for {
		val := v.AuxInt
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueAMD64_OpConst32_0(v *Value) bool {
	// match: (Const32 [val])
	// cond:
	// result: (MOVLconst [val])
	for {
		val := v.AuxInt
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueAMD64_OpConst32F_0(v *Value) bool {
	// match: (Const32F [val])
	// cond:
	// result: (MOVSSconst [val])
	for {
		val := v.AuxInt
		v.reset(OpAMD64MOVSSconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueAMD64_OpConst64_0(v *Value) bool {
	// match: (Const64 [val])
	// cond:
	// result: (MOVQconst [val])
	for {
		val := v.AuxInt
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueAMD64_OpConst64F_0(v *Value) bool {
	// match: (Const64F [val])
	// cond:
	// result: (MOVSDconst [val])
	for {
		val := v.AuxInt
		v.reset(OpAMD64MOVSDconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueAMD64_OpConst8_0(v *Value) bool {
	// match: (Const8 [val])
	// cond:
	// result: (MOVLconst [val])
	for {
		val := v.AuxInt
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueAMD64_OpConstBool_0(v *Value) bool {
	// match: (ConstBool [b])
	// cond:
	// result: (MOVLconst [b])
	for {
		b := v.AuxInt
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = b
		return true
	}
}
func rewriteValueAMD64_OpConstNil_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (ConstNil)
	// cond: config.PtrSize == 8
	// result: (MOVQconst [0])
	for {
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64MOVQconst)
		v.AuxInt = 0
		return true
	}
	// match: (ConstNil)
	// cond: config.PtrSize == 4
	// result: (MOVLconst [0])
	for {
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64MOVLconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueAMD64_OpConvert_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (Convert <t> x mem)
	// cond: config.PtrSize == 8
	// result: (MOVQconvert <t> x mem)
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		mem := v.Args[1]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64MOVQconvert)
		v.Type = t
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	// match: (Convert <t> x mem)
	// cond: config.PtrSize == 4
	// result: (MOVLconvert <t> x mem)
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		mem := v.Args[1]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64MOVLconvert)
		v.Type = t
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpCtz32_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Ctz32 x)
	// cond:
	// result: (Select0 (BSFQ (ORQ <typ.UInt64> (MOVQconst [1<<32]) x)))
	for {
		x := v.Args[0]
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64BSFQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v1 := b.NewValue0(v.Pos, OpAMD64ORQ, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v2.AuxInt = 1 << 32
		v1.AddArg(v2)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpCtz64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Ctz64 <t> x)
	// cond:
	// result: (CMOVQEQ (Select0 <t> (BSFQ x)) (MOVQconst <t> [64]) (Select1 <types.TypeFlags> (BSFQ x)))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpAMD64CMOVQEQ)
		v0 := b.NewValue0(v.Pos, OpSelect0, t)
		v1 := b.NewValue0(v.Pos, OpAMD64BSFQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQconst, t)
		v2.AuxInt = 64
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpSelect1, types.TypeFlags)
		v4 := b.NewValue0(v.Pos, OpAMD64BSFQ, types.NewTuple(typ.UInt64, types.TypeFlags))
		v4.AddArg(x)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueAMD64_OpCvt32Fto32_0(v *Value) bool {
	// match: (Cvt32Fto32 x)
	// cond:
	// result: (CVTTSS2SL x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64CVTTSS2SL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCvt32Fto64_0(v *Value) bool {
	// match: (Cvt32Fto64 x)
	// cond:
	// result: (CVTTSS2SQ x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64CVTTSS2SQ)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCvt32Fto64F_0(v *Value) bool {
	// match: (Cvt32Fto64F x)
	// cond:
	// result: (CVTSS2SD x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64CVTSS2SD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCvt32to32F_0(v *Value) bool {
	// match: (Cvt32to32F x)
	// cond:
	// result: (CVTSL2SS x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64CVTSL2SS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCvt32to64F_0(v *Value) bool {
	// match: (Cvt32to64F x)
	// cond:
	// result: (CVTSL2SD x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64CVTSL2SD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCvt64Fto32_0(v *Value) bool {
	// match: (Cvt64Fto32 x)
	// cond:
	// result: (CVTTSD2SL x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64CVTTSD2SL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCvt64Fto32F_0(v *Value) bool {
	// match: (Cvt64Fto32F x)
	// cond:
	// result: (CVTSD2SS x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64CVTSD2SS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCvt64Fto64_0(v *Value) bool {
	// match: (Cvt64Fto64 x)
	// cond:
	// result: (CVTTSD2SQ x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64CVTTSD2SQ)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCvt64to32F_0(v *Value) bool {
	// match: (Cvt64to32F x)
	// cond:
	// result: (CVTSQ2SS x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64CVTSQ2SS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpCvt64to64F_0(v *Value) bool {
	// match: (Cvt64to64F x)
	// cond:
	// result: (CVTSQ2SD x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64CVTSQ2SD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpDiv128u_0(v *Value) bool {
	// match: (Div128u xhi xlo y)
	// cond:
	// result: (DIVQU2 xhi xlo y)
	for {
		_ = v.Args[2]
		xhi := v.Args[0]
		xlo := v.Args[1]
		y := v.Args[2]
		v.reset(OpAMD64DIVQU2)
		v.AddArg(xhi)
		v.AddArg(xlo)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpDiv16_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div16 x y)
	// cond:
	// result: (Select0 (DIVW  x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVW, types.NewTuple(typ.Int16, typ.Int16))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv16u_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div16u x y)
	// cond:
	// result: (Select0 (DIVWU x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVWU, types.NewTuple(typ.UInt16, typ.UInt16))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv32_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div32 x y)
	// cond:
	// result: (Select0 (DIVL  x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVL, types.NewTuple(typ.Int32, typ.Int32))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv32F_0(v *Value) bool {
	// match: (Div32F x y)
	// cond:
	// result: (DIVSS x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64DIVSS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpDiv32u_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div32u x y)
	// cond:
	// result: (Select0 (DIVLU x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVLU, types.NewTuple(typ.UInt32, typ.UInt32))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div64 x y)
	// cond:
	// result: (Select0 (DIVQ  x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVQ, types.NewTuple(typ.Int64, typ.Int64))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv64F_0(v *Value) bool {
	// match: (Div64F x y)
	// cond:
	// result: (DIVSD x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64DIVSD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpDiv64u_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div64u x y)
	// cond:
	// result: (Select0 (DIVQU x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVQU, types.NewTuple(typ.UInt64, typ.UInt64))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv8_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div8 x y)
	// cond:
	// result: (Select0 (DIVW  (SignExt8to16 x) (SignExt8to16 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVW, types.NewTuple(typ.Int16, typ.Int16))
		v1 := b.NewValue0(v.Pos, OpSignExt8to16, typ.Int16)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSignExt8to16, typ.Int16)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpDiv8u_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div8u x y)
	// cond:
	// result: (Select0 (DIVWU (ZeroExt8to16 x) (ZeroExt8to16 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect0)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVWU, types.NewTuple(typ.UInt16, typ.UInt16))
		v1 := b.NewValue0(v.Pos, OpZeroExt8to16, typ.UInt16)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to16, typ.UInt16)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEq16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Eq16 x y)
	// cond:
	// result: (SETEQ (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEq32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Eq32 x y)
	// cond:
	// result: (SETEQ (CMPL x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEq32F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Eq32F x y)
	// cond:
	// result: (SETEQF (UCOMISS x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETEQF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISS, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEq64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Eq64 x y)
	// cond:
	// result: (SETEQ (CMPQ x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEq64F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Eq64F x y)
	// cond:
	// result: (SETEQF (UCOMISD x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETEQF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEq8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Eq8 x y)
	// cond:
	// result: (SETEQ (CMPB x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqB_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (EqB x y)
	// cond:
	// result: (SETEQ (CMPB x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpEqPtr_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (EqPtr x y)
	// cond: config.PtrSize == 8
	// result: (SETEQ (CMPQ x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (EqPtr x y)
	// cond: config.PtrSize == 4
	// result: (SETEQ (CMPL x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64SETEQ)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpGeq16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Geq16 x y)
	// cond:
	// result: (SETGE (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETGE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGeq16U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Geq16U x y)
	// cond:
	// result: (SETAE (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGeq32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Geq32 x y)
	// cond:
	// result: (SETGE (CMPL x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETGE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGeq32F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Geq32F x y)
	// cond:
	// result: (SETGEF (UCOMISS x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETGEF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISS, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGeq32U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Geq32U x y)
	// cond:
	// result: (SETAE (CMPL x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGeq64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Geq64 x y)
	// cond:
	// result: (SETGE (CMPQ x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETGE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGeq64F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Geq64F x y)
	// cond:
	// result: (SETGEF (UCOMISD x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETGEF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGeq64U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Geq64U x y)
	// cond:
	// result: (SETAE (CMPQ x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGeq8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Geq8 x y)
	// cond:
	// result: (SETGE (CMPB x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETGE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGeq8U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Geq8U x y)
	// cond:
	// result: (SETAE (CMPB x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETAE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGetClosurePtr_0(v *Value) bool {
	// match: (GetClosurePtr)
	// cond:
	// result: (LoweredGetClosurePtr)
	for {
		v.reset(OpAMD64LoweredGetClosurePtr)
		return true
	}
}
func rewriteValueAMD64_OpGetG_0(v *Value) bool {
	// match: (GetG mem)
	// cond:
	// result: (LoweredGetG mem)
	for {
		mem := v.Args[0]
		v.reset(OpAMD64LoweredGetG)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpGreater16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Greater16 x y)
	// cond:
	// result: (SETG (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETG)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreater16U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Greater16U x y)
	// cond:
	// result: (SETA (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETA)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreater32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Greater32 x y)
	// cond:
	// result: (SETG (CMPL x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETG)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreater32F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Greater32F x y)
	// cond:
	// result: (SETGF (UCOMISS x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETGF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISS, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreater32U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Greater32U x y)
	// cond:
	// result: (SETA (CMPL x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETA)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreater64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Greater64 x y)
	// cond:
	// result: (SETG (CMPQ x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETG)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreater64F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Greater64F x y)
	// cond:
	// result: (SETGF (UCOMISD x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETGF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreater64U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Greater64U x y)
	// cond:
	// result: (SETA (CMPQ x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETA)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreater8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Greater8 x y)
	// cond:
	// result: (SETG (CMPB x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETG)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpGreater8U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Greater8U x y)
	// cond:
	// result: (SETA (CMPB x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETA)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpHmul32_0(v *Value) bool {
	// match: (Hmul32 x y)
	// cond:
	// result: (HMULL  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64HMULL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpHmul32u_0(v *Value) bool {
	// match: (Hmul32u x y)
	// cond:
	// result: (HMULLU x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64HMULLU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpHmul64_0(v *Value) bool {
	// match: (Hmul64 x y)
	// cond:
	// result: (HMULQ  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64HMULQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpHmul64u_0(v *Value) bool {
	// match: (Hmul64u x y)
	// cond:
	// result: (HMULQU x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64HMULQU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpInt64Hi_0(v *Value) bool {
	// match: (Int64Hi x)
	// cond:
	// result: (SHRQconst [32] x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64SHRQconst)
		v.AuxInt = 32
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpInterCall_0(v *Value) bool {
	// match: (InterCall [argwid] entry mem)
	// cond:
	// result: (CALLinter [argwid] entry mem)
	for {
		argwid := v.AuxInt
		_ = v.Args[1]
		entry := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64CALLinter)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpIsInBounds_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (IsInBounds idx len)
	// cond: config.PtrSize == 8
	// result: (SETB (CMPQ idx len))
	for {
		_ = v.Args[1]
		idx := v.Args[0]
		len := v.Args[1]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(idx)
		v0.AddArg(len)
		v.AddArg(v0)
		return true
	}
	// match: (IsInBounds idx len)
	// cond: config.PtrSize == 4
	// result: (SETB (CMPL idx len))
	for {
		_ = v.Args[1]
		idx := v.Args[0]
		len := v.Args[1]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(idx)
		v0.AddArg(len)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpIsNonNil_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (IsNonNil p)
	// cond: config.PtrSize == 8
	// result: (SETNE (TESTQ p p))
	for {
		p := v.Args[0]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64TESTQ, types.TypeFlags)
		v0.AddArg(p)
		v0.AddArg(p)
		v.AddArg(v0)
		return true
	}
	// match: (IsNonNil p)
	// cond: config.PtrSize == 4
	// result: (SETNE (TESTL p p))
	for {
		p := v.Args[0]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64TESTL, types.TypeFlags)
		v0.AddArg(p)
		v0.AddArg(p)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpIsSliceInBounds_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (IsSliceInBounds idx len)
	// cond: config.PtrSize == 8
	// result: (SETBE (CMPQ idx len))
	for {
		_ = v.Args[1]
		idx := v.Args[0]
		len := v.Args[1]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(idx)
		v0.AddArg(len)
		v.AddArg(v0)
		return true
	}
	// match: (IsSliceInBounds idx len)
	// cond: config.PtrSize == 4
	// result: (SETBE (CMPL idx len))
	for {
		_ = v.Args[1]
		idx := v.Args[0]
		len := v.Args[1]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(idx)
		v0.AddArg(len)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLeq16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Leq16 x y)
	// cond:
	// result: (SETLE (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq16U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Leq16U x y)
	// cond:
	// result: (SETBE (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Leq32 x y)
	// cond:
	// result: (SETLE (CMPL x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq32F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Leq32F x y)
	// cond:
	// result: (SETGEF (UCOMISS y x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETGEF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISS, types.TypeFlags)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq32U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Leq32U x y)
	// cond:
	// result: (SETBE (CMPL x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Leq64 x y)
	// cond:
	// result: (SETLE (CMPQ x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq64F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Leq64F x y)
	// cond:
	// result: (SETGEF (UCOMISD y x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETGEF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISD, types.TypeFlags)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq64U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Leq64U x y)
	// cond:
	// result: (SETBE (CMPQ x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Leq8 x y)
	// cond:
	// result: (SETLE (CMPB x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETLE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLeq8U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Leq8U x y)
	// cond:
	// result: (SETBE (CMPB x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETBE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Less16 x y)
	// cond:
	// result: (SETL (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETL)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess16U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Less16U x y)
	// cond:
	// result: (SETB (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Less32 x y)
	// cond:
	// result: (SETL (CMPL x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETL)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess32F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Less32F x y)
	// cond:
	// result: (SETGF (UCOMISS y x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETGF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISS, types.TypeFlags)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess32U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Less32U x y)
	// cond:
	// result: (SETB (CMPL x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Less64 x y)
	// cond:
	// result: (SETL (CMPQ x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETL)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess64F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Less64F x y)
	// cond:
	// result: (SETGF (UCOMISD y x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETGF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISD, types.TypeFlags)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess64U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Less64U x y)
	// cond:
	// result: (SETB (CMPQ x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Less8 x y)
	// cond:
	// result: (SETL (CMPB x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETL)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLess8U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Less8U x y)
	// cond:
	// result: (SETB (CMPB x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETB)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpLoad_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (Load <t> ptr mem)
	// cond: (is64BitInt(t) || isPtr(t) && config.PtrSize == 8)
	// result: (MOVQload ptr mem)
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(is64BitInt(t) || isPtr(t) && config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64MOVQload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (is32BitInt(t) || isPtr(t) && config.PtrSize == 4)
	// result: (MOVLload ptr mem)
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(is32BitInt(t) || isPtr(t) && config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64MOVLload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is16BitInt(t)
	// result: (MOVWload ptr mem)
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(is16BitInt(t)) {
			break
		}
		v.reset(OpAMD64MOVWload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: (t.IsBoolean() || is8BitInt(t))
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(t.IsBoolean() || is8BitInt(t)) {
			break
		}
		v.reset(OpAMD64MOVBload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is32BitFloat(t)
	// result: (MOVSSload ptr mem)
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(is32BitFloat(t)) {
			break
		}
		v.reset(OpAMD64MOVSSload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is64BitFloat(t)
	// result: (MOVSDload ptr mem)
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(is64BitFloat(t)) {
			break
		}
		v.reset(OpAMD64MOVSDload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpLsh16x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh16x16 <t> x y)
	// cond:
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPWconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh16x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh16x32 <t> x y)
	// cond:
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPLconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh16x64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh16x64 <t> x y)
	// cond:
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPQconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh16x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh16x8 <t> x y)
	// cond:
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPBconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh32x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh32x16 <t> x y)
	// cond:
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPWconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh32x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh32x32 <t> x y)
	// cond:
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPLconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh32x64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh32x64 <t> x y)
	// cond:
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPQconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh32x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh32x8 <t> x y)
	// cond:
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPBconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh64x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh64x16 <t> x y)
	// cond:
	// result: (ANDQ (SHLQ <t> x y) (SBBQcarrymask <t> (CMPWconst y [64])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQ, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh64x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh64x32 <t> x y)
	// cond:
	// result: (ANDQ (SHLQ <t> x y) (SBBQcarrymask <t> (CMPLconst y [64])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQ, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh64x64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh64x64 <t> x y)
	// cond:
	// result: (ANDQ (SHLQ <t> x y) (SBBQcarrymask <t> (CMPQconst y [64])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQ, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh64x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh64x8 <t> x y)
	// cond:
	// result: (ANDQ (SHLQ <t> x y) (SBBQcarrymask <t> (CMPBconst y [64])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLQ, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh8x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh8x16 <t> x y)
	// cond:
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPWconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh8x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh8x32 <t> x y)
	// cond:
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPLconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh8x64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh8x64 <t> x y)
	// cond:
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPQconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpLsh8x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh8x8 <t> x y)
	// cond:
	// result: (ANDL (SHLL <t> x y) (SBBLcarrymask <t> (CMPBconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpMod16_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mod16 x y)
	// cond:
	// result: (Select1 (DIVW  x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVW, types.NewTuple(typ.Int16, typ.Int16))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod16u_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mod16u x y)
	// cond:
	// result: (Select1 (DIVWU x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVWU, types.NewTuple(typ.UInt16, typ.UInt16))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod32_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mod32 x y)
	// cond:
	// result: (Select1 (DIVL  x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVL, types.NewTuple(typ.Int32, typ.Int32))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod32u_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mod32u x y)
	// cond:
	// result: (Select1 (DIVLU x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVLU, types.NewTuple(typ.UInt32, typ.UInt32))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mod64 x y)
	// cond:
	// result: (Select1 (DIVQ  x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVQ, types.NewTuple(typ.Int64, typ.Int64))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod64u_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mod64u x y)
	// cond:
	// result: (Select1 (DIVQU x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVQU, types.NewTuple(typ.UInt64, typ.UInt64))
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod8_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mod8 x y)
	// cond:
	// result: (Select1 (DIVW  (SignExt8to16 x) (SignExt8to16 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVW, types.NewTuple(typ.Int16, typ.Int16))
		v1 := b.NewValue0(v.Pos, OpSignExt8to16, typ.Int16)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSignExt8to16, typ.Int16)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMod8u_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mod8u x y)
	// cond:
	// result: (Select1 (DIVWU (ZeroExt8to16 x) (ZeroExt8to16 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpSelect1)
		v0 := b.NewValue0(v.Pos, OpAMD64DIVWU, types.NewTuple(typ.UInt16, typ.UInt16))
		v1 := b.NewValue0(v.Pos, OpZeroExt8to16, typ.UInt16)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt8to16, typ.UInt16)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpMove_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Move [0] _ _ mem)
	// cond:
	// result: mem
	for {
		if v.AuxInt != 0 {
			break
		}
		_ = v.Args[2]
		mem := v.Args[2]
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	// match: (Move [1] dst src mem)
	// cond:
	// result: (MOVBstore dst (MOVBload src mem) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVBstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, typ.UInt8)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// cond:
	// result: (MOVWstore dst (MOVWload src mem) mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [4] dst src mem)
	// cond:
	// result: (MOVLstore dst (MOVLload src mem) mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [8] dst src mem)
	// cond:
	// result: (MOVQstore dst (MOVQload src mem) mem)
	for {
		if v.AuxInt != 8 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVQstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [16] dst src mem)
	// cond:
	// result: (MOVOstore dst (MOVOload src mem) mem)
	for {
		if v.AuxInt != 16 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVOstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVOload, types.TypeInt128)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [3] dst src mem)
	// cond:
	// result: (MOVBstore [2] dst (MOVBload [2] src mem) 		(MOVWstore dst (MOVWload src mem) mem))
	for {
		if v.AuxInt != 3 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = 2
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, typ.UInt8)
		v0.AuxInt = 2
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVWstore, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [5] dst src mem)
	// cond:
	// result: (MOVBstore [4] dst (MOVBload [4] src mem) 		(MOVLstore dst (MOVLload src mem) mem))
	for {
		if v.AuxInt != 5 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVBstore)
		v.AuxInt = 4
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBload, typ.UInt8)
		v0.AuxInt = 4
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLstore, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [6] dst src mem)
	// cond:
	// result: (MOVWstore [4] dst (MOVWload [4] src mem) 		(MOVLstore dst (MOVLload src mem) mem))
	for {
		if v.AuxInt != 6 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVWstore)
		v.AuxInt = 4
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWload, typ.UInt16)
		v0.AuxInt = 4
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLstore, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [7] dst src mem)
	// cond:
	// result: (MOVLstore [3] dst (MOVLload [3] src mem) 		(MOVLstore dst (MOVLload src mem) mem))
	for {
		if v.AuxInt != 7 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		v.reset(OpAMD64MOVLstore)
		v.AuxInt = 3
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v0.AuxInt = 3
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVLstore, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVLload, typ.UInt32)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueAMD64_OpMove_10(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Move [s] dst src mem)
	// cond: s > 8 && s < 16
	// result: (MOVQstore [s-8] dst (MOVQload [s-8] src mem) 		(MOVQstore dst (MOVQload src mem) mem))
	for {
		s := v.AuxInt
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		if !(s > 8 && s < 16) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AuxInt = s - 8
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v0.AuxInt = s - 8
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v1.AddArg(dst)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v2.AddArg(src)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 16 && s%16 != 0 && s%16 <= 8
	// result: (Move [s-s%16] 		(OffPtr <dst.Type> dst [s%16]) 		(OffPtr <src.Type> src [s%16]) 		(MOVQstore dst (MOVQload src mem) mem))
	for {
		s := v.AuxInt
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		if !(s > 16 && s%16 != 0 && s%16 <= 8) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = s - s%16
		v0 := b.NewValue0(v.Pos, OpOffPtr, dst.Type)
		v0.AuxInt = s % 16
		v0.AddArg(dst)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, src.Type)
		v1.AuxInt = s % 16
		v1.AddArg(src)
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVQload, typ.UInt64)
		v3.AddArg(src)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v2.AddArg(mem)
		v.AddArg(v2)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 16 && s%16 != 0 && s%16 > 8
	// result: (Move [s-s%16] 		(OffPtr <dst.Type> dst [s%16]) 		(OffPtr <src.Type> src [s%16]) 		(MOVOstore dst (MOVOload src mem) mem))
	for {
		s := v.AuxInt
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		if !(s > 16 && s%16 != 0 && s%16 > 8) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = s - s%16
		v0 := b.NewValue0(v.Pos, OpOffPtr, dst.Type)
		v0.AuxInt = s % 16
		v0.AddArg(dst)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, src.Type)
		v1.AuxInt = s % 16
		v1.AddArg(src)
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVOstore, types.TypeMem)
		v2.AddArg(dst)
		v3 := b.NewValue0(v.Pos, OpAMD64MOVOload, types.TypeInt128)
		v3.AddArg(src)
		v3.AddArg(mem)
		v2.AddArg(v3)
		v2.AddArg(mem)
		v.AddArg(v2)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s >= 32 && s <= 16*64 && s%16 == 0 	&& !config.noDuffDevice
	// result: (DUFFCOPY [14*(64-s/16)] dst src mem)
	for {
		s := v.AuxInt
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		if !(s >= 32 && s <= 16*64 && s%16 == 0 && !config.noDuffDevice) {
			break
		}
		v.reset(OpAMD64DUFFCOPY)
		v.AuxInt = 14 * (64 - s/16)
		v.AddArg(dst)
		v.AddArg(src)
		v.AddArg(mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: (s > 16*64 || config.noDuffDevice) && s%8 == 0
	// result: (REPMOVSQ dst src (MOVQconst [s/8]) mem)
	for {
		s := v.AuxInt
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		if !((s > 16*64 || config.noDuffDevice) && s%8 == 0) {
			break
		}
		v.reset(OpAMD64REPMOVSQ)
		v.AddArg(dst)
		v.AddArg(src)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = s / 8
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpMul16_0(v *Value) bool {
	// match: (Mul16 x y)
	// cond:
	// result: (MULL  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64MULL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpMul32_0(v *Value) bool {
	// match: (Mul32 x y)
	// cond:
	// result: (MULL  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64MULL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpMul32F_0(v *Value) bool {
	// match: (Mul32F x y)
	// cond:
	// result: (MULSS x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64MULSS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpMul64_0(v *Value) bool {
	// match: (Mul64 x y)
	// cond:
	// result: (MULQ  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64MULQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpMul64F_0(v *Value) bool {
	// match: (Mul64F x y)
	// cond:
	// result: (MULSD x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64MULSD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpMul64uhilo_0(v *Value) bool {
	// match: (Mul64uhilo x y)
	// cond:
	// result: (MULQU2 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64MULQU2)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpMul8_0(v *Value) bool {
	// match: (Mul8 x y)
	// cond:
	// result: (MULL  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64MULL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpNeg16_0(v *Value) bool {
	// match: (Neg16 x)
	// cond:
	// result: (NEGL x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64NEGL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpNeg32_0(v *Value) bool {
	// match: (Neg32 x)
	// cond:
	// result: (NEGL x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64NEGL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpNeg32F_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Neg32F x)
	// cond:
	// result: (PXOR x (MOVSSconst <typ.Float32> [f2i(math.Copysign(0, -1))]))
	for {
		x := v.Args[0]
		v.reset(OpAMD64PXOR)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVSSconst, typ.Float32)
		v0.AuxInt = f2i(math.Copysign(0, -1))
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeg64_0(v *Value) bool {
	// match: (Neg64 x)
	// cond:
	// result: (NEGQ x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64NEGQ)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpNeg64F_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Neg64F x)
	// cond:
	// result: (PXOR x (MOVSDconst <typ.Float64> [f2i(math.Copysign(0, -1))]))
	for {
		x := v.Args[0]
		v.reset(OpAMD64PXOR)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVSDconst, typ.Float64)
		v0.AuxInt = f2i(math.Copysign(0, -1))
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeg8_0(v *Value) bool {
	// match: (Neg8 x)
	// cond:
	// result: (NEGL x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64NEGL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpNeq16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Neq16 x y)
	// cond:
	// result: (SETNE (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPW, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeq32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Neq32 x y)
	// cond:
	// result: (SETNE (CMPL x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeq32F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Neq32F x y)
	// cond:
	// result: (SETNEF (UCOMISS x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETNEF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISS, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeq64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Neq64 x y)
	// cond:
	// result: (SETNE (CMPQ x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeq64F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Neq64F x y)
	// cond:
	// result: (SETNEF (UCOMISD x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETNEF)
		v0 := b.NewValue0(v.Pos, OpAMD64UCOMISD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeq8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Neq8 x y)
	// cond:
	// result: (SETNE (CMPB x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeqB_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (NeqB x y)
	// cond:
	// result: (SETNE (CMPB x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPB, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpNeqPtr_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (NeqPtr x y)
	// cond: config.PtrSize == 8
	// result: (SETNE (CMPQ x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPQ, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	// match: (NeqPtr x y)
	// cond: config.PtrSize == 4
	// result: (SETNE (CMPL x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64SETNE)
		v0 := b.NewValue0(v.Pos, OpAMD64CMPL, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpNilCheck_0(v *Value) bool {
	// match: (NilCheck ptr mem)
	// cond:
	// result: (LoweredNilCheck ptr mem)
	for {
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64LoweredNilCheck)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpNot_0(v *Value) bool {
	// match: (Not x)
	// cond:
	// result: (XORLconst [1] x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64XORLconst)
		v.AuxInt = 1
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpOffPtr_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (OffPtr [off] ptr)
	// cond: config.PtrSize == 8 && is32Bit(off)
	// result: (ADDQconst [off] ptr)
	for {
		off := v.AuxInt
		ptr := v.Args[0]
		if !(config.PtrSize == 8 && is32Bit(off)) {
			break
		}
		v.reset(OpAMD64ADDQconst)
		v.AuxInt = off
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// cond: config.PtrSize == 8
	// result: (ADDQ (MOVQconst [off]) ptr)
	for {
		off := v.AuxInt
		ptr := v.Args[0]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64ADDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = off
		v.AddArg(v0)
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// cond: config.PtrSize == 4
	// result: (ADDLconst [off] ptr)
	for {
		off := v.AuxInt
		ptr := v.Args[0]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64ADDLconst)
		v.AuxInt = off
		v.AddArg(ptr)
		return true
	}
	return false
}
func rewriteValueAMD64_OpOr16_0(v *Value) bool {
	// match: (Or16 x y)
	// cond:
	// result: (ORL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpOr32_0(v *Value) bool {
	// match: (Or32 x y)
	// cond:
	// result: (ORL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpOr64_0(v *Value) bool {
	// match: (Or64 x y)
	// cond:
	// result: (ORQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ORQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpOr8_0(v *Value) bool {
	// match: (Or8 x y)
	// cond:
	// result: (ORL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpOrB_0(v *Value) bool {
	// match: (OrB x y)
	// cond:
	// result: (ORL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpPopCount16_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (PopCount16 x)
	// cond:
	// result: (POPCNTL (MOVWQZX <typ.UInt32> x))
	for {
		x := v.Args[0]
		v.reset(OpAMD64POPCNTL)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWQZX, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpPopCount32_0(v *Value) bool {
	// match: (PopCount32 x)
	// cond:
	// result: (POPCNTL x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64POPCNTL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpPopCount64_0(v *Value) bool {
	// match: (PopCount64 x)
	// cond:
	// result: (POPCNTQ x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64POPCNTQ)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpPopCount8_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (PopCount8 x)
	// cond:
	// result: (POPCNTL (MOVBQZX <typ.UInt32> x))
	for {
		x := v.Args[0]
		v.reset(OpAMD64POPCNTL)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVBQZX, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRound32F_0(v *Value) bool {
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
func rewriteValueAMD64_OpRound64F_0(v *Value) bool {
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
func rewriteValueAMD64_OpRsh16Ux16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16Ux16 <t> x y)
	// cond:
	// result: (ANDL (SHRW <t> x y) (SBBLcarrymask <t> (CMPWconst y [16])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRW, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = 16
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh16Ux32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16Ux32 <t> x y)
	// cond:
	// result: (ANDL (SHRW <t> x y) (SBBLcarrymask <t> (CMPLconst y [16])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRW, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = 16
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh16Ux64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16Ux64 <t> x y)
	// cond:
	// result: (ANDL (SHRW <t> x y) (SBBLcarrymask <t> (CMPQconst y [16])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRW, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = 16
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh16Ux8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16Ux8 <t> x y)
	// cond:
	// result: (ANDL (SHRW <t> x y) (SBBLcarrymask <t> (CMPBconst y [16])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRW, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = 16
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh16x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16x16 <t> x y)
	// cond:
	// result: (SARW <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPWconst y [16])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARW)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v3.AuxInt = 16
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh16x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16x32 <t> x y)
	// cond:
	// result: (SARW <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPLconst y [16])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARW)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v3.AuxInt = 16
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh16x64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16x64 <t> x y)
	// cond:
	// result: (SARW <t> x (ORQ <y.Type> y (NOTQ <y.Type> (SBBQcarrymask <y.Type> (CMPQconst y [16])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARW)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTQ, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v3.AuxInt = 16
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh16x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16x8 <t> x y)
	// cond:
	// result: (SARW <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPBconst y [16])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARW)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v3.AuxInt = 16
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh32Ux16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32Ux16 <t> x y)
	// cond:
	// result: (ANDL (SHRL <t> x y) (SBBLcarrymask <t> (CMPWconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh32Ux32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32Ux32 <t> x y)
	// cond:
	// result: (ANDL (SHRL <t> x y) (SBBLcarrymask <t> (CMPLconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh32Ux64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32Ux64 <t> x y)
	// cond:
	// result: (ANDL (SHRL <t> x y) (SBBLcarrymask <t> (CMPQconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh32Ux8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32Ux8 <t> x y)
	// cond:
	// result: (ANDL (SHRL <t> x y) (SBBLcarrymask <t> (CMPBconst y [32])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = 32
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh32x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32x16 <t> x y)
	// cond:
	// result: (SARL <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPWconst y [32])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARL)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v3.AuxInt = 32
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh32x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32x32 <t> x y)
	// cond:
	// result: (SARL <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPLconst y [32])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARL)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v3.AuxInt = 32
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh32x64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32x64 <t> x y)
	// cond:
	// result: (SARL <t> x (ORQ <y.Type> y (NOTQ <y.Type> (SBBQcarrymask <y.Type> (CMPQconst y [32])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARL)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTQ, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v3.AuxInt = 32
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh32x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32x8 <t> x y)
	// cond:
	// result: (SARL <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPBconst y [32])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARL)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v3.AuxInt = 32
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh64Ux16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64Ux16 <t> x y)
	// cond:
	// result: (ANDQ (SHRQ <t> x y) (SBBQcarrymask <t> (CMPWconst y [64])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRQ, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh64Ux32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64Ux32 <t> x y)
	// cond:
	// result: (ANDQ (SHRQ <t> x y) (SBBQcarrymask <t> (CMPLconst y [64])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRQ, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh64Ux64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64Ux64 <t> x y)
	// cond:
	// result: (ANDQ (SHRQ <t> x y) (SBBQcarrymask <t> (CMPQconst y [64])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRQ, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh64Ux8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64Ux8 <t> x y)
	// cond:
	// result: (ANDQ (SHRQ <t> x y) (SBBQcarrymask <t> (CMPBconst y [64])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDQ)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRQ, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh64x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64x16 <t> x y)
	// cond:
	// result: (SARQ <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPWconst y [64])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARQ)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v3.AuxInt = 64
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh64x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64x32 <t> x y)
	// cond:
	// result: (SARQ <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPLconst y [64])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARQ)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v3.AuxInt = 64
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh64x64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64x64 <t> x y)
	// cond:
	// result: (SARQ <t> x (ORQ <y.Type> y (NOTQ <y.Type> (SBBQcarrymask <y.Type> (CMPQconst y [64])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARQ)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTQ, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v3.AuxInt = 64
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh64x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64x8 <t> x y)
	// cond:
	// result: (SARQ <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPBconst y [64])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARQ)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v3.AuxInt = 64
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh8Ux16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8Ux16 <t> x y)
	// cond:
	// result: (ANDL (SHRB <t> x y) (SBBLcarrymask <t> (CMPWconst y [8])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRB, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v2.AuxInt = 8
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh8Ux32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8Ux32 <t> x y)
	// cond:
	// result: (ANDL (SHRB <t> x y) (SBBLcarrymask <t> (CMPLconst y [8])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRB, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v2.AuxInt = 8
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh8Ux64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8Ux64 <t> x y)
	// cond:
	// result: (ANDL (SHRB <t> x y) (SBBLcarrymask <t> (CMPQconst y [8])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRB, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v2.AuxInt = 8
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh8Ux8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8Ux8 <t> x y)
	// cond:
	// result: (ANDL (SHRB <t> x y) (SBBLcarrymask <t> (CMPBconst y [8])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64ANDL)
		v0 := b.NewValue0(v.Pos, OpAMD64SHRB, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, t)
		v2 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v2.AuxInt = 8
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueAMD64_OpRsh8x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8x16 <t> x y)
	// cond:
	// result: (SARB <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPWconst y [8])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARB)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPWconst, types.TypeFlags)
		v3.AuxInt = 8
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh8x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8x32 <t> x y)
	// cond:
	// result: (SARB <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPLconst y [8])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARB)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPLconst, types.TypeFlags)
		v3.AuxInt = 8
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh8x64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8x64 <t> x y)
	// cond:
	// result: (SARB <t> x (ORQ <y.Type> y (NOTQ <y.Type> (SBBQcarrymask <y.Type> (CMPQconst y [8])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARB)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORQ, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTQ, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBQcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPQconst, types.TypeFlags)
		v3.AuxInt = 8
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpRsh8x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8x8 <t> x y)
	// cond:
	// result: (SARB <t> x (ORL <y.Type> y (NOTL <y.Type> (SBBLcarrymask <y.Type> (CMPBconst y [8])))))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SARB)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpAMD64ORL, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpAMD64NOTL, y.Type)
		v2 := b.NewValue0(v.Pos, OpAMD64SBBLcarrymask, y.Type)
		v3 := b.NewValue0(v.Pos, OpAMD64CMPBconst, types.TypeFlags)
		v3.AuxInt = 8
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpSelect0_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Select0 <t> (AddTupleFirst32 val tuple))
	// cond:
	// result: (ADDL val (Select0 <t> tuple))
	for {
		t := v.Type
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64AddTupleFirst32 {
			break
		}
		_ = v_0.Args[1]
		val := v_0.Args[0]
		tuple := v_0.Args[1]
		v.reset(OpAMD64ADDL)
		v.AddArg(val)
		v0 := b.NewValue0(v.Pos, OpSelect0, t)
		v0.AddArg(tuple)
		v.AddArg(v0)
		return true
	}
	// match: (Select0 <t> (AddTupleFirst64 val tuple))
	// cond:
	// result: (ADDQ val (Select0 <t> tuple))
	for {
		t := v.Type
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64AddTupleFirst64 {
			break
		}
		_ = v_0.Args[1]
		val := v_0.Args[0]
		tuple := v_0.Args[1]
		v.reset(OpAMD64ADDQ)
		v.AddArg(val)
		v0 := b.NewValue0(v.Pos, OpSelect0, t)
		v0.AddArg(tuple)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueAMD64_OpSelect1_0(v *Value) bool {
	// match: (Select1 (AddTupleFirst32 _ tuple))
	// cond:
	// result: (Select1 tuple)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64AddTupleFirst32 {
			break
		}
		_ = v_0.Args[1]
		tuple := v_0.Args[1]
		v.reset(OpSelect1)
		v.AddArg(tuple)
		return true
	}
	// match: (Select1 (AddTupleFirst64 _ tuple))
	// cond:
	// result: (Select1 tuple)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAMD64AddTupleFirst64 {
			break
		}
		_ = v_0.Args[1]
		tuple := v_0.Args[1]
		v.reset(OpSelect1)
		v.AddArg(tuple)
		return true
	}
	return false
}
func rewriteValueAMD64_OpSignExt16to32_0(v *Value) bool {
	// match: (SignExt16to32 x)
	// cond:
	// result: (MOVWQSX x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64MOVWQSX)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpSignExt16to64_0(v *Value) bool {
	// match: (SignExt16to64 x)
	// cond:
	// result: (MOVWQSX x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64MOVWQSX)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpSignExt32to64_0(v *Value) bool {
	// match: (SignExt32to64 x)
	// cond:
	// result: (MOVLQSX x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64MOVLQSX)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpSignExt8to16_0(v *Value) bool {
	// match: (SignExt8to16 x)
	// cond:
	// result: (MOVBQSX x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64MOVBQSX)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpSignExt8to32_0(v *Value) bool {
	// match: (SignExt8to32 x)
	// cond:
	// result: (MOVBQSX x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64MOVBQSX)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpSignExt8to64_0(v *Value) bool {
	// match: (SignExt8to64 x)
	// cond:
	// result: (MOVBQSX x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64MOVBQSX)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpSlicemask_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Slicemask <t> x)
	// cond:
	// result: (SARQconst (NEGQ <t> x) [63])
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpAMD64SARQconst)
		v.AuxInt = 63
		v0 := b.NewValue0(v.Pos, OpAMD64NEGQ, t)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueAMD64_OpSqrt_0(v *Value) bool {
	// match: (Sqrt x)
	// cond:
	// result: (SQRTSD x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64SQRTSD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpStaticCall_0(v *Value) bool {
	// match: (StaticCall [argwid] {target} mem)
	// cond:
	// result: (CALLstatic [argwid] {target} mem)
	for {
		argwid := v.AuxInt
		target := v.Aux
		mem := v.Args[0]
		v.reset(OpAMD64CALLstatic)
		v.AuxInt = argwid
		v.Aux = target
		v.AddArg(mem)
		return true
	}
}
func rewriteValueAMD64_OpStore_0(v *Value) bool {
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 8 && is64BitFloat(val.Type)
	// result: (MOVSDstore ptr val mem)
	for {
		t := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(t.(*types.Type).Size() == 8 && is64BitFloat(val.Type)) {
			break
		}
		v.reset(OpAMD64MOVSDstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 4 && is32BitFloat(val.Type)
	// result: (MOVSSstore ptr val mem)
	for {
		t := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(t.(*types.Type).Size() == 4 && is32BitFloat(val.Type)) {
			break
		}
		v.reset(OpAMD64MOVSSstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 8
	// result: (MOVQstore ptr val mem)
	for {
		t := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(t.(*types.Type).Size() == 8) {
			break
		}
		v.reset(OpAMD64MOVQstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 4
	// result: (MOVLstore ptr val mem)
	for {
		t := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(t.(*types.Type).Size() == 4) {
			break
		}
		v.reset(OpAMD64MOVLstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 2
	// result: (MOVWstore ptr val mem)
	for {
		t := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(t.(*types.Type).Size() == 2) {
			break
		}
		v.reset(OpAMD64MOVWstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 1
	// result: (MOVBstore ptr val mem)
	for {
		t := v.Aux
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		if !(t.(*types.Type).Size() == 1) {
			break
		}
		v.reset(OpAMD64MOVBstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpSub16_0(v *Value) bool {
	// match: (Sub16 x y)
	// cond:
	// result: (SUBL  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SUBL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpSub32_0(v *Value) bool {
	// match: (Sub32 x y)
	// cond:
	// result: (SUBL  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SUBL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpSub32F_0(v *Value) bool {
	// match: (Sub32F x y)
	// cond:
	// result: (SUBSS x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SUBSS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpSub64_0(v *Value) bool {
	// match: (Sub64 x y)
	// cond:
	// result: (SUBQ  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SUBQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpSub64F_0(v *Value) bool {
	// match: (Sub64F x y)
	// cond:
	// result: (SUBSD x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SUBSD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpSub8_0(v *Value) bool {
	// match: (Sub8 x y)
	// cond:
	// result: (SUBL  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64SUBL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpSubPtr_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (SubPtr x y)
	// cond: config.PtrSize == 8
	// result: (SUBQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAMD64SUBQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (SubPtr x y)
	// cond: config.PtrSize == 4
	// result: (SUBL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAMD64SUBL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueAMD64_OpTrunc16to8_0(v *Value) bool {
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
func rewriteValueAMD64_OpTrunc32to16_0(v *Value) bool {
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
func rewriteValueAMD64_OpTrunc32to8_0(v *Value) bool {
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
func rewriteValueAMD64_OpTrunc64to16_0(v *Value) bool {
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
func rewriteValueAMD64_OpTrunc64to32_0(v *Value) bool {
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
func rewriteValueAMD64_OpTrunc64to8_0(v *Value) bool {
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
func rewriteValueAMD64_OpXor16_0(v *Value) bool {
	// match: (Xor16 x y)
	// cond:
	// result: (XORL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64XORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpXor32_0(v *Value) bool {
	// match: (Xor32 x y)
	// cond:
	// result: (XORL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64XORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpXor64_0(v *Value) bool {
	// match: (Xor64 x y)
	// cond:
	// result: (XORQ x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64XORQ)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpXor8_0(v *Value) bool {
	// match: (Xor8 x y)
	// cond:
	// result: (XORL x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpAMD64XORL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueAMD64_OpZero_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Zero [0] _ mem)
	// cond:
	// result: mem
	for {
		if v.AuxInt != 0 {
			break
		}
		_ = v.Args[1]
		mem := v.Args[1]
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	// match: (Zero [1] destptr mem)
	// cond:
	// result: (MOVBstoreconst [0] destptr mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = 0
		v.AddArg(destptr)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [2] destptr mem)
	// cond:
	// result: (MOVWstoreconst [0] destptr mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = 0
		v.AddArg(destptr)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [4] destptr mem)
	// cond:
	// result: (MOVLstoreconst [0] destptr mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = 0
		v.AddArg(destptr)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [8] destptr mem)
	// cond:
	// result: (MOVQstoreconst [0] destptr mem)
	for {
		if v.AuxInt != 8 {
			break
		}
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = 0
		v.AddArg(destptr)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [3] destptr mem)
	// cond:
	// result: (MOVBstoreconst [makeValAndOff(0,2)] destptr 		(MOVWstoreconst [0] destptr mem))
	for {
		if v.AuxInt != 3 {
			break
		}
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = makeValAndOff(0, 2)
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVWstoreconst, types.TypeMem)
		v0.AuxInt = 0
		v0.AddArg(destptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
	// match: (Zero [5] destptr mem)
	// cond:
	// result: (MOVBstoreconst [makeValAndOff(0,4)] destptr 		(MOVLstoreconst [0] destptr mem))
	for {
		if v.AuxInt != 5 {
			break
		}
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVBstoreconst)
		v.AuxInt = makeValAndOff(0, 4)
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLstoreconst, types.TypeMem)
		v0.AuxInt = 0
		v0.AddArg(destptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
	// match: (Zero [6] destptr mem)
	// cond:
	// result: (MOVWstoreconst [makeValAndOff(0,4)] destptr 		(MOVLstoreconst [0] destptr mem))
	for {
		if v.AuxInt != 6 {
			break
		}
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVWstoreconst)
		v.AuxInt = makeValAndOff(0, 4)
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLstoreconst, types.TypeMem)
		v0.AuxInt = 0
		v0.AddArg(destptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
	// match: (Zero [7] destptr mem)
	// cond:
	// result: (MOVLstoreconst [makeValAndOff(0,3)] destptr 		(MOVLstoreconst [0] destptr mem))
	for {
		if v.AuxInt != 7 {
			break
		}
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVLstoreconst)
		v.AuxInt = makeValAndOff(0, 3)
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVLstoreconst, types.TypeMem)
		v0.AuxInt = 0
		v0.AddArg(destptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s%8 != 0 && s > 8
	// result: (Zero [s-s%8] (OffPtr <destptr.Type> destptr [s%8]) 		(MOVQstoreconst [0] destptr mem))
	for {
		s := v.AuxInt
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		if !(s%8 != 0 && s > 8) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = s - s%8
		v0 := b.NewValue0(v.Pos, OpOffPtr, destptr.Type)
		v0.AuxInt = s % 8
		v0.AddArg(destptr)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v1.AuxInt = 0
		v1.AddArg(destptr)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueAMD64_OpZero_10(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Zero [16] destptr mem)
	// cond:
	// result: (MOVQstoreconst [makeValAndOff(0,8)] destptr 		(MOVQstoreconst [0] destptr mem))
	for {
		if v.AuxInt != 16 {
			break
		}
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = makeValAndOff(0, 8)
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v0.AuxInt = 0
		v0.AddArg(destptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
	// match: (Zero [24] destptr mem)
	// cond:
	// result: (MOVQstoreconst [makeValAndOff(0,16)] destptr 		(MOVQstoreconst [makeValAndOff(0,8)] destptr 			(MOVQstoreconst [0] destptr mem)))
	for {
		if v.AuxInt != 24 {
			break
		}
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = makeValAndOff(0, 16)
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v0.AuxInt = makeValAndOff(0, 8)
		v0.AddArg(destptr)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v1.AuxInt = 0
		v1.AddArg(destptr)
		v1.AddArg(mem)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Zero [32] destptr mem)
	// cond:
	// result: (MOVQstoreconst [makeValAndOff(0,24)] destptr 		(MOVQstoreconst [makeValAndOff(0,16)] destptr 			(MOVQstoreconst [makeValAndOff(0,8)] destptr 				(MOVQstoreconst [0] destptr mem))))
	for {
		if v.AuxInt != 32 {
			break
		}
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpAMD64MOVQstoreconst)
		v.AuxInt = makeValAndOff(0, 24)
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v0.AuxInt = makeValAndOff(0, 16)
		v0.AddArg(destptr)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v1.AuxInt = makeValAndOff(0, 8)
		v1.AddArg(destptr)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQstoreconst, types.TypeMem)
		v2.AuxInt = 0
		v2.AddArg(destptr)
		v2.AddArg(mem)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s <= 1024 && s%8 == 0 && s%16 != 0 	&& !config.noDuffDevice
	// result: (Zero [s-8] (OffPtr <destptr.Type> [8] destptr) (MOVQstore destptr (MOVQconst [0]) mem))
	for {
		s := v.AuxInt
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		if !(s <= 1024 && s%8 == 0 && s%16 != 0 && !config.noDuffDevice) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = s - 8
		v0 := b.NewValue0(v.Pos, OpOffPtr, destptr.Type)
		v0.AuxInt = 8
		v0.AddArg(destptr)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQstore, types.TypeMem)
		v1.AddArg(destptr)
		v2 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v2.AuxInt = 0
		v1.AddArg(v2)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: s <= 1024 && s%16 == 0 && !config.noDuffDevice
	// result: (DUFFZERO [s] destptr (MOVOconst [0]) mem)
	for {
		s := v.AuxInt
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		if !(s <= 1024 && s%16 == 0 && !config.noDuffDevice) {
			break
		}
		v.reset(OpAMD64DUFFZERO)
		v.AuxInt = s
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVOconst, types.TypeInt128)
		v0.AuxInt = 0
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [s] destptr mem)
	// cond: (s > 1024 || (config.noDuffDevice && s > 32)) 	&& s%8 == 0
	// result: (REPSTOSQ destptr (MOVQconst [s/8]) (MOVQconst [0]) mem)
	for {
		s := v.AuxInt
		_ = v.Args[1]
		destptr := v.Args[0]
		mem := v.Args[1]
		if !((s > 1024 || (config.noDuffDevice && s > 32)) && s%8 == 0) {
			break
		}
		v.reset(OpAMD64REPSTOSQ)
		v.AddArg(destptr)
		v0 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v0.AuxInt = s / 8
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpAMD64MOVQconst, typ.UInt64)
		v1.AuxInt = 0
		v.AddArg(v1)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueAMD64_OpZeroExt16to32_0(v *Value) bool {
	// match: (ZeroExt16to32 x)
	// cond:
	// result: (MOVWQZX x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64MOVWQZX)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpZeroExt16to64_0(v *Value) bool {
	// match: (ZeroExt16to64 x)
	// cond:
	// result: (MOVWQZX x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64MOVWQZX)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpZeroExt32to64_0(v *Value) bool {
	// match: (ZeroExt32to64 x)
	// cond:
	// result: (MOVLQZX x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64MOVLQZX)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpZeroExt8to16_0(v *Value) bool {
	// match: (ZeroExt8to16 x)
	// cond:
	// result: (MOVBQZX x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64MOVBQZX)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpZeroExt8to32_0(v *Value) bool {
	// match: (ZeroExt8to32 x)
	// cond:
	// result: (MOVBQZX x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64MOVBQZX)
		v.AddArg(x)
		return true
	}
}
func rewriteValueAMD64_OpZeroExt8to64_0(v *Value) bool {
	// match: (ZeroExt8to64 x)
	// cond:
	// result: (MOVBQZX x)
	for {
		x := v.Args[0]
		v.reset(OpAMD64MOVBQZX)
		v.AddArg(x)
		return true
	}
}
func rewriteBlockAMD64(b *Block) bool {
	config := b.Func.Config
	_ = config
	fe := b.Func.fe
	_ = fe
	typ := &config.Types
	_ = typ
	switch b.Kind {
	case BlockAMD64EQ:
		// match: (EQ (TESTL (SHLL (MOVLconst [1]) x) y))
		// cond: !config.nacl
		// result: (UGE (BTL x y))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTL {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SHLL {
				break
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64MOVLconst {
				break
			}
			if v_0_0.AuxInt != 1 {
				break
			}
			x := v_0.Args[1]
			y := v.Args[1]
			if !(!config.nacl) {
				break
			}
			b.Kind = BlockAMD64UGE
			v0 := b.NewValue0(v.Pos, OpAMD64BTL, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			return true
		}
		// match: (EQ (TESTL y (SHLL (MOVLconst [1]) x)))
		// cond: !config.nacl
		// result: (UGE (BTL x y))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTL {
				break
			}
			_ = v.Args[1]
			y := v.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SHLL {
				break
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpAMD64MOVLconst {
				break
			}
			if v_1_0.AuxInt != 1 {
				break
			}
			x := v_1.Args[1]
			if !(!config.nacl) {
				break
			}
			b.Kind = BlockAMD64UGE
			v0 := b.NewValue0(v.Pos, OpAMD64BTL, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			return true
		}
		// match: (EQ (TESTQ (SHLQ (MOVQconst [1]) x) y))
		// cond: !config.nacl
		// result: (UGE (BTQ x y))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTQ {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SHLQ {
				break
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64MOVQconst {
				break
			}
			if v_0_0.AuxInt != 1 {
				break
			}
			x := v_0.Args[1]
			y := v.Args[1]
			if !(!config.nacl) {
				break
			}
			b.Kind = BlockAMD64UGE
			v0 := b.NewValue0(v.Pos, OpAMD64BTQ, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			return true
		}
		// match: (EQ (TESTQ y (SHLQ (MOVQconst [1]) x)))
		// cond: !config.nacl
		// result: (UGE (BTQ x y))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTQ {
				break
			}
			_ = v.Args[1]
			y := v.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SHLQ {
				break
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpAMD64MOVQconst {
				break
			}
			if v_1_0.AuxInt != 1 {
				break
			}
			x := v_1.Args[1]
			if !(!config.nacl) {
				break
			}
			b.Kind = BlockAMD64UGE
			v0 := b.NewValue0(v.Pos, OpAMD64BTQ, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			return true
		}
		// match: (EQ (TESTLconst [c] x))
		// cond: isPowerOfTwo(c) && log2(c) < 32 && !config.nacl
		// result: (UGE (BTLconst [log2(c)] x))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTLconst {
				break
			}
			c := v.AuxInt
			x := v.Args[0]
			if !(isPowerOfTwo(c) && log2(c) < 32 && !config.nacl) {
				break
			}
			b.Kind = BlockAMD64UGE
			v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = log2(c)
			v0.AddArg(x)
			b.SetControl(v0)
			return true
		}
		// match: (EQ (TESTQconst [c] x))
		// cond: isPowerOfTwo(c) && log2(c) < 64 && !config.nacl
		// result: (UGE (BTQconst [log2(c)] x))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTQconst {
				break
			}
			c := v.AuxInt
			x := v.Args[0]
			if !(isPowerOfTwo(c) && log2(c) < 64 && !config.nacl) {
				break
			}
			b.Kind = BlockAMD64UGE
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = log2(c)
			v0.AddArg(x)
			b.SetControl(v0)
			return true
		}
		// match: (EQ (TESTQ (MOVQconst [c]) x))
		// cond: isPowerOfTwo(c) && log2(c) < 64 && !config.nacl
		// result: (UGE (BTQconst [log2(c)] x))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTQ {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64MOVQconst {
				break
			}
			c := v_0.AuxInt
			x := v.Args[1]
			if !(isPowerOfTwo(c) && log2(c) < 64 && !config.nacl) {
				break
			}
			b.Kind = BlockAMD64UGE
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = log2(c)
			v0.AddArg(x)
			b.SetControl(v0)
			return true
		}
		// match: (EQ (TESTQ x (MOVQconst [c])))
		// cond: isPowerOfTwo(c) && log2(c) < 64 && !config.nacl
		// result: (UGE (BTQconst [log2(c)] x))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTQ {
				break
			}
			_ = v.Args[1]
			x := v.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64MOVQconst {
				break
			}
			c := v_1.AuxInt
			if !(isPowerOfTwo(c) && log2(c) < 64 && !config.nacl) {
				break
			}
			b.Kind = BlockAMD64UGE
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = log2(c)
			v0.AddArg(x)
			b.SetControl(v0)
			return true
		}
		// match: (EQ (InvertFlags cmp) yes no)
		// cond:
		// result: (EQ cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64EQ
			b.SetControl(cmp)
			return true
		}
		// match: (EQ (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagEQ {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (EQ (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
	case BlockAMD64GE:
		// match: (GE (InvertFlags cmp) yes no)
		// cond:
		// result: (LE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64LE
			b.SetControl(cmp)
			return true
		}
		// match: (GE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagEQ {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (GE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (GE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (GE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (GE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
	case BlockAMD64GT:
		// match: (GT (InvertFlags cmp) yes no)
		// cond:
		// result: (LT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64LT
			b.SetControl(cmp)
			return true
		}
		// match: (GT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagEQ {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (GT (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (GT (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
	case BlockIf:
		// match: (If (SETL cmp) yes no)
		// cond:
		// result: (LT  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETL {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64LT
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETLE cmp) yes no)
		// cond:
		// result: (LE  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETLE {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64LE
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETG cmp) yes no)
		// cond:
		// result: (GT  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETG {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64GT
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETGE cmp) yes no)
		// cond:
		// result: (GE  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETGE {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64GE
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETEQ cmp) yes no)
		// cond:
		// result: (EQ  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETEQ {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64EQ
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETNE cmp) yes no)
		// cond:
		// result: (NE  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETNE {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64NE
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETB cmp) yes no)
		// cond:
		// result: (ULT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETB {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64ULT
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETBE cmp) yes no)
		// cond:
		// result: (ULE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETBE {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64ULE
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETA cmp) yes no)
		// cond:
		// result: (UGT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETA {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64UGT
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETAE cmp) yes no)
		// cond:
		// result: (UGE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETAE {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64UGE
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETGF cmp) yes no)
		// cond:
		// result: (UGT  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETGF {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64UGT
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETGEF cmp) yes no)
		// cond:
		// result: (UGE  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETGEF {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64UGE
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETEQF cmp) yes no)
		// cond:
		// result: (EQF  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETEQF {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64EQF
			b.SetControl(cmp)
			return true
		}
		// match: (If (SETNEF cmp) yes no)
		// cond:
		// result: (NEF  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64SETNEF {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64NEF
			b.SetControl(cmp)
			return true
		}
		// match: (If cond yes no)
		// cond:
		// result: (NE (TESTB cond cond) yes no)
		for {
			v := b.Control
			_ = v
			cond := b.Control
			b.Kind = BlockAMD64NE
			v0 := b.NewValue0(v.Pos, OpAMD64TESTB, types.TypeFlags)
			v0.AddArg(cond)
			v0.AddArg(cond)
			b.SetControl(v0)
			return true
		}
	case BlockAMD64LE:
		// match: (LE (InvertFlags cmp) yes no)
		// cond:
		// result: (GE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64GE
			b.SetControl(cmp)
			return true
		}
		// match: (LE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagEQ {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (LE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (LE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (LE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (LE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
	case BlockAMD64LT:
		// match: (LT (InvertFlags cmp) yes no)
		// cond:
		// result: (GT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64GT
			b.SetControl(cmp)
			return true
		}
		// match: (LT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagEQ {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (LT (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (LT (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (LT (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (LT (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
	case BlockAMD64NE:
		// match: (NE (TESTB (SETL cmp) (SETL cmp)) yes no)
		// cond:
		// result: (LT  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETL {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETL {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64LT
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETL cmp) (SETL cmp)) yes no)
		// cond:
		// result: (LT  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETL {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETL {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64LT
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETLE cmp) (SETLE cmp)) yes no)
		// cond:
		// result: (LE  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETLE {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETLE {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64LE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETLE cmp) (SETLE cmp)) yes no)
		// cond:
		// result: (LE  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETLE {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETLE {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64LE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETG cmp) (SETG cmp)) yes no)
		// cond:
		// result: (GT  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETG {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETG {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64GT
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETG cmp) (SETG cmp)) yes no)
		// cond:
		// result: (GT  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETG {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETG {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64GT
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETGE cmp) (SETGE cmp)) yes no)
		// cond:
		// result: (GE  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETGE {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETGE {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64GE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETGE cmp) (SETGE cmp)) yes no)
		// cond:
		// result: (GE  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETGE {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETGE {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64GE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETEQ cmp) (SETEQ cmp)) yes no)
		// cond:
		// result: (EQ  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETEQ {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETEQ {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64EQ
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETEQ cmp) (SETEQ cmp)) yes no)
		// cond:
		// result: (EQ  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETEQ {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETEQ {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64EQ
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETNE cmp) (SETNE cmp)) yes no)
		// cond:
		// result: (NE  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETNE {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETNE {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64NE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETNE cmp) (SETNE cmp)) yes no)
		// cond:
		// result: (NE  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETNE {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETNE {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64NE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETB cmp) (SETB cmp)) yes no)
		// cond:
		// result: (ULT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETB {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETB {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64ULT
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETB cmp) (SETB cmp)) yes no)
		// cond:
		// result: (ULT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETB {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETB {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64ULT
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETBE cmp) (SETBE cmp)) yes no)
		// cond:
		// result: (ULE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETBE {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETBE {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64ULE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETBE cmp) (SETBE cmp)) yes no)
		// cond:
		// result: (ULE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETBE {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETBE {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64ULE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETA cmp) (SETA cmp)) yes no)
		// cond:
		// result: (UGT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETA {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETA {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64UGT
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETA cmp) (SETA cmp)) yes no)
		// cond:
		// result: (UGT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETA {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETA {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64UGT
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETAE cmp) (SETAE cmp)) yes no)
		// cond:
		// result: (UGE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETAE {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETAE {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64UGE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETAE cmp) (SETAE cmp)) yes no)
		// cond:
		// result: (UGE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETAE {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETAE {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64UGE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTL (SHLL (MOVLconst [1]) x) y))
		// cond: !config.nacl
		// result: (ULT (BTL x y))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTL {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SHLL {
				break
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64MOVLconst {
				break
			}
			if v_0_0.AuxInt != 1 {
				break
			}
			x := v_0.Args[1]
			y := v.Args[1]
			if !(!config.nacl) {
				break
			}
			b.Kind = BlockAMD64ULT
			v0 := b.NewValue0(v.Pos, OpAMD64BTL, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			return true
		}
		// match: (NE (TESTL y (SHLL (MOVLconst [1]) x)))
		// cond: !config.nacl
		// result: (ULT (BTL x y))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTL {
				break
			}
			_ = v.Args[1]
			y := v.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SHLL {
				break
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpAMD64MOVLconst {
				break
			}
			if v_1_0.AuxInt != 1 {
				break
			}
			x := v_1.Args[1]
			if !(!config.nacl) {
				break
			}
			b.Kind = BlockAMD64ULT
			v0 := b.NewValue0(v.Pos, OpAMD64BTL, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			return true
		}
		// match: (NE (TESTQ (SHLQ (MOVQconst [1]) x) y))
		// cond: !config.nacl
		// result: (ULT (BTQ x y))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTQ {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SHLQ {
				break
			}
			_ = v_0.Args[1]
			v_0_0 := v_0.Args[0]
			if v_0_0.Op != OpAMD64MOVQconst {
				break
			}
			if v_0_0.AuxInt != 1 {
				break
			}
			x := v_0.Args[1]
			y := v.Args[1]
			if !(!config.nacl) {
				break
			}
			b.Kind = BlockAMD64ULT
			v0 := b.NewValue0(v.Pos, OpAMD64BTQ, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			return true
		}
		// match: (NE (TESTQ y (SHLQ (MOVQconst [1]) x)))
		// cond: !config.nacl
		// result: (ULT (BTQ x y))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTQ {
				break
			}
			_ = v.Args[1]
			y := v.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SHLQ {
				break
			}
			_ = v_1.Args[1]
			v_1_0 := v_1.Args[0]
			if v_1_0.Op != OpAMD64MOVQconst {
				break
			}
			if v_1_0.AuxInt != 1 {
				break
			}
			x := v_1.Args[1]
			if !(!config.nacl) {
				break
			}
			b.Kind = BlockAMD64ULT
			v0 := b.NewValue0(v.Pos, OpAMD64BTQ, types.TypeFlags)
			v0.AddArg(x)
			v0.AddArg(y)
			b.SetControl(v0)
			return true
		}
		// match: (NE (TESTLconst [c] x))
		// cond: isPowerOfTwo(c) && log2(c) < 32 && !config.nacl
		// result: (ULT (BTLconst [log2(c)] x))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTLconst {
				break
			}
			c := v.AuxInt
			x := v.Args[0]
			if !(isPowerOfTwo(c) && log2(c) < 32 && !config.nacl) {
				break
			}
			b.Kind = BlockAMD64ULT
			v0 := b.NewValue0(v.Pos, OpAMD64BTLconst, types.TypeFlags)
			v0.AuxInt = log2(c)
			v0.AddArg(x)
			b.SetControl(v0)
			return true
		}
		// match: (NE (TESTQconst [c] x))
		// cond: isPowerOfTwo(c) && log2(c) < 64 && !config.nacl
		// result: (ULT (BTQconst [log2(c)] x))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTQconst {
				break
			}
			c := v.AuxInt
			x := v.Args[0]
			if !(isPowerOfTwo(c) && log2(c) < 64 && !config.nacl) {
				break
			}
			b.Kind = BlockAMD64ULT
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = log2(c)
			v0.AddArg(x)
			b.SetControl(v0)
			return true
		}
		// match: (NE (TESTQ (MOVQconst [c]) x))
		// cond: isPowerOfTwo(c) && log2(c) < 64 && !config.nacl
		// result: (ULT (BTQconst [log2(c)] x))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTQ {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64MOVQconst {
				break
			}
			c := v_0.AuxInt
			x := v.Args[1]
			if !(isPowerOfTwo(c) && log2(c) < 64 && !config.nacl) {
				break
			}
			b.Kind = BlockAMD64ULT
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = log2(c)
			v0.AddArg(x)
			b.SetControl(v0)
			return true
		}
		// match: (NE (TESTQ x (MOVQconst [c])))
		// cond: isPowerOfTwo(c) && log2(c) < 64 && !config.nacl
		// result: (ULT (BTQconst [log2(c)] x))
		for {
			v := b.Control
			if v.Op != OpAMD64TESTQ {
				break
			}
			_ = v.Args[1]
			x := v.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64MOVQconst {
				break
			}
			c := v_1.AuxInt
			if !(isPowerOfTwo(c) && log2(c) < 64 && !config.nacl) {
				break
			}
			b.Kind = BlockAMD64ULT
			v0 := b.NewValue0(v.Pos, OpAMD64BTQconst, types.TypeFlags)
			v0.AuxInt = log2(c)
			v0.AddArg(x)
			b.SetControl(v0)
			return true
		}
		// match: (NE (TESTB (SETGF cmp) (SETGF cmp)) yes no)
		// cond:
		// result: (UGT  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETGF {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETGF {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64UGT
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETGF cmp) (SETGF cmp)) yes no)
		// cond:
		// result: (UGT  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETGF {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETGF {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64UGT
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETGEF cmp) (SETGEF cmp)) yes no)
		// cond:
		// result: (UGE  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETGEF {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETGEF {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64UGE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETGEF cmp) (SETGEF cmp)) yes no)
		// cond:
		// result: (UGE  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETGEF {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETGEF {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64UGE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETEQF cmp) (SETEQF cmp)) yes no)
		// cond:
		// result: (EQF  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETEQF {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETEQF {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64EQF
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETEQF cmp) (SETEQF cmp)) yes no)
		// cond:
		// result: (EQF  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETEQF {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETEQF {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64EQF
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETNEF cmp) (SETNEF cmp)) yes no)
		// cond:
		// result: (NEF  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETNEF {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETNEF {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64NEF
			b.SetControl(cmp)
			return true
		}
		// match: (NE (TESTB (SETNEF cmp) (SETNEF cmp)) yes no)
		// cond:
		// result: (NEF  cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64TESTB {
				break
			}
			_ = v.Args[1]
			v_0 := v.Args[0]
			if v_0.Op != OpAMD64SETNEF {
				break
			}
			cmp := v_0.Args[0]
			v_1 := v.Args[1]
			if v_1.Op != OpAMD64SETNEF {
				break
			}
			if cmp != v_1.Args[0] {
				break
			}
			b.Kind = BlockAMD64NEF
			b.SetControl(cmp)
			return true
		}
		// match: (NE (InvertFlags cmp) yes no)
		// cond:
		// result: (NE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64NE
			b.SetControl(cmp)
			return true
		}
		// match: (NE (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagEQ {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (NE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (NE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (NE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (NE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
	case BlockAMD64UGE:
		// match: (UGE (InvertFlags cmp) yes no)
		// cond:
		// result: (ULE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64ULE
			b.SetControl(cmp)
			return true
		}
		// match: (UGE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagEQ {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (UGE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (UGE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (UGE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (UGE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
	case BlockAMD64UGT:
		// match: (UGT (InvertFlags cmp) yes no)
		// cond:
		// result: (ULT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64ULT
			b.SetControl(cmp)
			return true
		}
		// match: (UGT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagEQ {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (UGT (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (UGT (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
	case BlockAMD64ULE:
		// match: (ULE (InvertFlags cmp) yes no)
		// cond:
		// result: (UGE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64UGE
			b.SetControl(cmp)
			return true
		}
		// match: (ULE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagEQ {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (ULE (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (ULE (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (ULE (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (ULE (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
	case BlockAMD64ULT:
		// match: (ULT (InvertFlags cmp) yes no)
		// cond:
		// result: (UGT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockAMD64UGT
			b.SetControl(cmp)
			return true
		}
		// match: (ULT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagEQ {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (ULT (FlagLT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (ULT (FlagLT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagLT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (ULT (FlagGT_ULT) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_ULT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (ULT (FlagGT_UGT) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpAMD64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
	}
	return false
}
