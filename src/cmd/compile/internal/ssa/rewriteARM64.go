// Code generated from gen/ARM64.rules; DO NOT EDIT.
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

func rewriteValueARM64(v *Value) bool {
	switch v.Op {
	case OpARM64ADD:
		return rewriteValueARM64_OpARM64ADD_0(v)
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
	case OpARM64BICconst:
		return rewriteValueARM64_OpARM64BICconst_0(v)
	case OpARM64BICshiftLL:
		return rewriteValueARM64_OpARM64BICshiftLL_0(v)
	case OpARM64BICshiftRA:
		return rewriteValueARM64_OpARM64BICshiftRA_0(v)
	case OpARM64BICshiftRL:
		return rewriteValueARM64_OpARM64BICshiftRL_0(v)
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
	case OpARM64CSELULT:
		return rewriteValueARM64_OpARM64CSELULT_0(v)
	case OpARM64CSELULT0:
		return rewriteValueARM64_OpARM64CSELULT0_0(v)
	case OpARM64DIV:
		return rewriteValueARM64_OpARM64DIV_0(v)
	case OpARM64DIVW:
		return rewriteValueARM64_OpARM64DIVW_0(v)
	case OpARM64Equal:
		return rewriteValueARM64_OpARM64Equal_0(v)
	case OpARM64FMOVDload:
		return rewriteValueARM64_OpARM64FMOVDload_0(v)
	case OpARM64FMOVDstore:
		return rewriteValueARM64_OpARM64FMOVDstore_0(v)
	case OpARM64FMOVSload:
		return rewriteValueARM64_OpARM64FMOVSload_0(v)
	case OpARM64FMOVSstore:
		return rewriteValueARM64_OpARM64FMOVSstore_0(v)
	case OpARM64GreaterEqual:
		return rewriteValueARM64_OpARM64GreaterEqual_0(v)
	case OpARM64GreaterEqualU:
		return rewriteValueARM64_OpARM64GreaterEqualU_0(v)
	case OpARM64GreaterThan:
		return rewriteValueARM64_OpARM64GreaterThan_0(v)
	case OpARM64GreaterThanU:
		return rewriteValueARM64_OpARM64GreaterThanU_0(v)
	case OpARM64LessEqual:
		return rewriteValueARM64_OpARM64LessEqual_0(v)
	case OpARM64LessEqualU:
		return rewriteValueARM64_OpARM64LessEqualU_0(v)
	case OpARM64LessThan:
		return rewriteValueARM64_OpARM64LessThan_0(v)
	case OpARM64LessThanU:
		return rewriteValueARM64_OpARM64LessThanU_0(v)
	case OpARM64MOD:
		return rewriteValueARM64_OpARM64MOD_0(v)
	case OpARM64MODW:
		return rewriteValueARM64_OpARM64MODW_0(v)
	case OpARM64MOVBUload:
		return rewriteValueARM64_OpARM64MOVBUload_0(v)
	case OpARM64MOVBUreg:
		return rewriteValueARM64_OpARM64MOVBUreg_0(v)
	case OpARM64MOVBload:
		return rewriteValueARM64_OpARM64MOVBload_0(v)
	case OpARM64MOVBreg:
		return rewriteValueARM64_OpARM64MOVBreg_0(v)
	case OpARM64MOVBstore:
		return rewriteValueARM64_OpARM64MOVBstore_0(v)
	case OpARM64MOVBstorezero:
		return rewriteValueARM64_OpARM64MOVBstorezero_0(v)
	case OpARM64MOVDload:
		return rewriteValueARM64_OpARM64MOVDload_0(v)
	case OpARM64MOVDreg:
		return rewriteValueARM64_OpARM64MOVDreg_0(v)
	case OpARM64MOVDstore:
		return rewriteValueARM64_OpARM64MOVDstore_0(v)
	case OpARM64MOVDstorezero:
		return rewriteValueARM64_OpARM64MOVDstorezero_0(v)
	case OpARM64MOVHUload:
		return rewriteValueARM64_OpARM64MOVHUload_0(v)
	case OpARM64MOVHUreg:
		return rewriteValueARM64_OpARM64MOVHUreg_0(v)
	case OpARM64MOVHload:
		return rewriteValueARM64_OpARM64MOVHload_0(v)
	case OpARM64MOVHreg:
		return rewriteValueARM64_OpARM64MOVHreg_0(v)
	case OpARM64MOVHstore:
		return rewriteValueARM64_OpARM64MOVHstore_0(v)
	case OpARM64MOVHstorezero:
		return rewriteValueARM64_OpARM64MOVHstorezero_0(v)
	case OpARM64MOVWUload:
		return rewriteValueARM64_OpARM64MOVWUload_0(v)
	case OpARM64MOVWUreg:
		return rewriteValueARM64_OpARM64MOVWUreg_0(v)
	case OpARM64MOVWload:
		return rewriteValueARM64_OpARM64MOVWload_0(v)
	case OpARM64MOVWreg:
		return rewriteValueARM64_OpARM64MOVWreg_0(v) || rewriteValueARM64_OpARM64MOVWreg_10(v)
	case OpARM64MOVWstore:
		return rewriteValueARM64_OpARM64MOVWstore_0(v)
	case OpARM64MOVWstorezero:
		return rewriteValueARM64_OpARM64MOVWstorezero_0(v)
	case OpARM64MUL:
		return rewriteValueARM64_OpARM64MUL_0(v) || rewriteValueARM64_OpARM64MUL_10(v) || rewriteValueARM64_OpARM64MUL_20(v)
	case OpARM64MULW:
		return rewriteValueARM64_OpARM64MULW_0(v) || rewriteValueARM64_OpARM64MULW_10(v) || rewriteValueARM64_OpARM64MULW_20(v)
	case OpARM64MVN:
		return rewriteValueARM64_OpARM64MVN_0(v)
	case OpARM64NEG:
		return rewriteValueARM64_OpARM64NEG_0(v)
	case OpARM64NotEqual:
		return rewriteValueARM64_OpARM64NotEqual_0(v)
	case OpARM64OR:
		return rewriteValueARM64_OpARM64OR_0(v) || rewriteValueARM64_OpARM64OR_10(v)
	case OpARM64ORconst:
		return rewriteValueARM64_OpARM64ORconst_0(v)
	case OpARM64ORshiftLL:
		return rewriteValueARM64_OpARM64ORshiftLL_0(v) || rewriteValueARM64_OpARM64ORshiftLL_10(v)
	case OpARM64ORshiftRA:
		return rewriteValueARM64_OpARM64ORshiftRA_0(v)
	case OpARM64ORshiftRL:
		return rewriteValueARM64_OpARM64ORshiftRL_0(v)
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
		return rewriteValueARM64_OpARM64SRLconst_0(v)
	case OpARM64SUB:
		return rewriteValueARM64_OpARM64SUB_0(v)
	case OpARM64SUBconst:
		return rewriteValueARM64_OpARM64SUBconst_0(v)
	case OpARM64SUBshiftLL:
		return rewriteValueARM64_OpARM64SUBshiftLL_0(v)
	case OpARM64SUBshiftRA:
		return rewriteValueARM64_OpARM64SUBshiftRA_0(v)
	case OpARM64SUBshiftRL:
		return rewriteValueARM64_OpARM64SUBshiftRL_0(v)
	case OpARM64UDIV:
		return rewriteValueARM64_OpARM64UDIV_0(v)
	case OpARM64UDIVW:
		return rewriteValueARM64_OpARM64UDIVW_0(v)
	case OpARM64UMOD:
		return rewriteValueARM64_OpARM64UMOD_0(v)
	case OpARM64UMODW:
		return rewriteValueARM64_OpARM64UMODW_0(v)
	case OpARM64XOR:
		return rewriteValueARM64_OpARM64XOR_0(v)
	case OpARM64XORconst:
		return rewriteValueARM64_OpARM64XORconst_0(v)
	case OpARM64XORshiftLL:
		return rewriteValueARM64_OpARM64XORshiftLL_0(v)
	case OpARM64XORshiftRA:
		return rewriteValueARM64_OpARM64XORshiftRA_0(v)
	case OpARM64XORshiftRL:
		return rewriteValueARM64_OpARM64XORshiftRL_0(v)
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
	case OpAtomicAdd64:
		return rewriteValueARM64_OpAtomicAdd64_0(v)
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
	case OpConvert:
		return rewriteValueARM64_OpConvert_0(v)
	case OpCtz32:
		return rewriteValueARM64_OpCtz32_0(v)
	case OpCtz64:
		return rewriteValueARM64_OpCtz64_0(v)
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
	case OpRound32F:
		return rewriteValueARM64_OpRound32F_0(v)
	case OpRound64F:
		return rewriteValueARM64_OpRound64F_0(v)
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
	case OpXor16:
		return rewriteValueARM64_OpXor16_0(v)
	case OpXor32:
		return rewriteValueARM64_OpXor32_0(v)
	case OpXor64:
		return rewriteValueARM64_OpXor64_0(v)
	case OpXor8:
		return rewriteValueARM64_OpXor8_0(v)
	case OpZero:
		return rewriteValueARM64_OpZero_0(v) || rewriteValueARM64_OpZero_10(v)
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARM64ADDconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64NEG {
			break
		}
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD x (SLLconst [c] y))
	// cond:
	// result: (ADDshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD (SLLconst [c] y) x)
	// cond:
	// result: (ADDshiftLL x y [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64ADDshiftLL)
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
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64ADDshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD (SRLconst [c] y) x)
	// cond:
	// result: (ADDshiftRL x y [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64ADDshiftRL)
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
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64ADDshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (ADD (SRAconst [c] y) x)
	// cond:
	// result: (ADDshiftRA x y [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64ADDshiftRA)
		v.AuxInt = c
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
	_ = b
	// match: (ADDshiftLL (MOVDconst [c]) x [d])
	// cond:
	// result: (ADDconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		x := v_0.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = 64 - c
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftLL <t> [c] (SRLconst (MOVWUreg x) [32-c]) x)
	// cond: c < 32 && t.Size() == 4
	// result: (RORWconst [32-c] x)
	for {
		t := v.Type
		c := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != 32-c {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0_0.Args[0]
		if x != v.Args[1] {
			break
		}
		if !(c < 32 && t.Size() == 4) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = 32 - c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADDshiftRA_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (ADDshiftRA (MOVDconst [c]) x [d])
	// cond:
	// result: (ADDconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	// result: (ADDconst x [int64(int64(c)>>uint64(d))])
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
		v.AuxInt = int64(int64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ADDshiftRL_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (ADDshiftRL (MOVDconst [c]) x [d])
	// cond:
	// result: (ADDconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	// result: (RORconst [   c] x)
	for {
		c := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		x := v_0.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ADDshiftRL <t> [c] (SLLconst x [32-c]) (MOVWUreg x))
	// cond: c < 32 && t.Size() == 4
	// result: (RORWconst [   c] x)
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARM64ANDconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (AND x x)
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MVN {
			break
		}
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64BIC)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND x (SLLconst [c] y))
	// cond:
	// result: (ANDshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64ANDshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND (SLLconst [c] y) x)
	// cond:
	// result: (ANDshiftLL x y [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64ANDshiftLL)
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
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64ANDshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (AND (SRLconst [c] y) x)
	// cond:
	// result: (ANDshiftRL x y [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64ANDshiftRL)
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
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64ANDshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64AND_10(v *Value) bool {
	// match: (AND (SRAconst [c] y) x)
	// cond:
	// result: (ANDshiftRA x y [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64ANDshiftRA)
		v.AuxInt = c
		v.AddArg(x)
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
	return false
}
func rewriteValueARM64_OpARM64ANDshiftLL_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (ANDshiftLL (MOVDconst [c]) x [d])
	// cond:
	// result: (ANDconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	_ = b
	// match: (ANDshiftRA (MOVDconst [c]) x [d])
	// cond:
	// result: (ANDconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	// result: (ANDconst x [int64(int64(c)>>uint64(d))])
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
		v.AuxInt = int64(int64(c) >> uint64(d))
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
	_ = b
	// match: (ANDshiftRL (MOVDconst [c]) x [d])
	// cond:
	// result: (ANDconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	// result: (BICconst [c] x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64BICconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (BIC x x)
	// cond:
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (BIC x (SLLconst [c] y))
	// cond:
	// result: (BICshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64BICshiftLL)
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
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64BICshiftRL)
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
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64BICshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64BICconst_0(v *Value) bool {
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
	// match: (BICconst [-1] _)
	// cond:
	// result: (MOVDconst [0])
	for {
		if v.AuxInt != -1 {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (BICconst [c] (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [d&^c])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = d &^ c
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64BICshiftLL_0(v *Value) bool {
	// match: (BICshiftLL x (MOVDconst [c]) [d])
	// cond:
	// result: (BICconst x [int64(uint64(c)<<uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64BICconst)
		v.AuxInt = int64(uint64(c) << uint64(d))
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
	// result: (BICconst x [int64(int64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64BICconst)
		v.AuxInt = int64(int64(c) >> uint64(d))
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
	// result: (BICconst x [int64(uint64(c)>>uint64(d))])
	for {
		d := v.AuxInt
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpARM64BICconst)
		v.AuxInt = int64(uint64(c) >> uint64(d))
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
func rewriteValueARM64_OpARM64CMP_0(v *Value) bool {
	b := v.Block
	_ = b
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPconst, types.TypeFlags)
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
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64CMPshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMP (SLLconst [c] y) x)
	// cond:
	// result: (InvertFlags (CMPshiftLL x y [c]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPshiftLL, types.TypeFlags)
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
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64CMPshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMP (SRLconst [c] y) x)
	// cond:
	// result: (InvertFlags (CMPshiftRL x y [c]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPshiftRL, types.TypeFlags)
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
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64CMPshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (CMP (SRAconst [c] y) x)
	// cond:
	// result: (InvertFlags (CMPshiftRA x y [c]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64InvertFlags)
		v0 := b.NewValue0(v.Pos, OpARM64CMPshiftRA, types.TypeFlags)
		v0.AuxInt = c
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPW_0(v *Value) bool {
	b := v.Block
	_ = b
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	// cond: int64(x)<int64(y) && uint64(x)<uint64(y)
	// result: (FlagLT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int64(x) < int64(y) && uint64(x) < uint64(y)) {
			break
		}
		v.reset(OpARM64FlagLT_ULT)
		return true
	}
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: int64(x)<int64(y) && uint64(x)>uint64(y)
	// result: (FlagLT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int64(x) < int64(y) && uint64(x) > uint64(y)) {
			break
		}
		v.reset(OpARM64FlagLT_UGT)
		return true
	}
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: int64(x)>int64(y) && uint64(x)<uint64(y)
	// result: (FlagGT_ULT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int64(x) > int64(y) && uint64(x) < uint64(y)) {
			break
		}
		v.reset(OpARM64FlagGT_ULT)
		return true
	}
	// match: (CMPconst (MOVDconst [x]) [y])
	// cond: int64(x)>int64(y) && uint64(x)>uint64(y)
	// result: (FlagGT_UGT)
	for {
		y := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		x := v_0.AuxInt
		if !(int64(x) > int64(y) && uint64(x) > uint64(y)) {
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
	_ = b
	// match: (CMPshiftLL (MOVDconst [c]) x [d])
	// cond:
	// result: (InvertFlags (CMPconst [c] (SLLconst <x.Type> x [d])))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	_ = b
	// match: (CMPshiftRA (MOVDconst [c]) x [d])
	// cond:
	// result: (InvertFlags (CMPconst [c] (SRAconst <x.Type> x [d])))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	// result: (CMPconst x [int64(int64(c)>>uint64(d))])
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
		v.AuxInt = int64(int64(c) >> uint64(d))
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CMPshiftRL_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (CMPshiftRL (MOVDconst [c]) x [d])
	// cond:
	// result: (InvertFlags (CMPconst [c] (SRLconst <x.Type> x [d])))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
func rewriteValueARM64_OpARM64CSELULT_0(v *Value) bool {
	// match: (CSELULT x (MOVDconst [0]) flag)
	// cond:
	// result: (CSELULT0 x flag)
	for {
		_ = v.Args[2]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		flag := v.Args[2]
		v.reset(OpARM64CSELULT0)
		v.AddArg(x)
		v.AddArg(flag)
		return true
	}
	// match: (CSELULT _ y (FlagEQ))
	// cond:
	// result: y
	for {
		_ = v.Args[2]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64FlagEQ {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (CSELULT x _ (FlagLT_ULT))
	// cond:
	// result: x
	for {
		_ = v.Args[2]
		x := v.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64FlagLT_ULT {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (CSELULT _ y (FlagLT_UGT))
	// cond:
	// result: y
	for {
		_ = v.Args[2]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64FlagLT_UGT {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (CSELULT x _ (FlagGT_ULT))
	// cond:
	// result: x
	for {
		_ = v.Args[2]
		x := v.Args[0]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64FlagGT_ULT {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (CSELULT _ y (FlagGT_UGT))
	// cond:
	// result: y
	for {
		_ = v.Args[2]
		y := v.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpARM64FlagGT_UGT {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64CSELULT0_0(v *Value) bool {
	// match: (CSELULT0 _ (FlagEQ))
	// cond:
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FlagEQ {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (CSELULT0 x (FlagLT_ULT))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FlagLT_ULT {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (CSELULT0 _ (FlagLT_UGT))
	// cond:
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FlagLT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (CSELULT0 x (FlagGT_ULT))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FlagGT_ULT {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (CSELULT0 _ (FlagGT_UGT))
	// cond:
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64FlagGT_UGT {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64DIV_0(v *Value) bool {
	// match: (DIV (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [int64(c)/int64(d)])
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
		v.AuxInt = int64(c) / int64(d)
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
func rewriteValueARM64_OpARM64FMOVDload_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (FMOVDload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVDload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// match: (FMOVDload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVDload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
func rewriteValueARM64_OpARM64FMOVDstore_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (FMOVDstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVDstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
	// match: (FMOVDstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVDstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
func rewriteValueARM64_OpARM64FMOVSload_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (FMOVSload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVSload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// match: (FMOVSload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVSload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
func rewriteValueARM64_OpARM64FMOVSstore_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (FMOVSstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVSstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
	// match: (FMOVSstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (FMOVSstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
func rewriteValueARM64_OpARM64MOD_0(v *Value) bool {
	// match: (MOD (MOVDconst [c]) (MOVDconst [d]))
	// cond:
	// result: (MOVDconst [int64(c)%int64(d)])
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
		v.AuxInt = int64(c) % int64(d)
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
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVBUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBUload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// match: (MOVBUload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBUload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpARM64MOVBload_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVBload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// match: (MOVBload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpARM64MOVBstore_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVBstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
	// match: (MOVBstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVBreg {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVBUreg {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
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
func rewriteValueARM64_OpARM64MOVBstorezero_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVBstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstorezero [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVBstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpARM64MOVDload_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVDload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// match: (MOVDload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVDstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
	// match: (MOVDstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		mem := v.Args[2]
		v.reset(OpARM64MOVDstorezero)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVDstorezero_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVDstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDstorezero [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVDstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpARM64MOVHUload_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVHUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHUload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// match: (MOVHUload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHUload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpARM64MOVHload_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVHload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// match: (MOVHload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpARM64MOVHstore_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVHstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
	// match: (MOVHstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVHreg {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVHUreg {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpARM64MOVHstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVHstorezero_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVHstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHstorezero [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVHstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpARM64MOVWUload_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVWUload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWUload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// match: (MOVWUload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWUload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpARM64MOVWload_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVWload [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWload [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// match: (MOVWload [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWload [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpARM64MOVWreg_10(v *Value) bool {
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
	return false
}
func rewriteValueARM64_OpARM64MOVWstore_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVWstore [off1] {sym} (ADDconst [off2] ptr) val mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstore [off1+off2] {sym} ptr val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
	// match: (MOVWstore [off1] {sym1} (MOVDaddr [off2] {sym2} ptr) val mem)
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstore [off1+off2] {mergeSym(sym1,sym2)} ptr val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWreg {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVWUreg {
			break
		}
		x := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpARM64MOVWstore)
		v.AuxInt = off
		v.Aux = sym
		v.AddArg(ptr)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MOVWstorezero_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (MOVWstorezero [off1] {sym} (ADDconst [off2] ptr) mem)
	// cond: is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstorezero [off1+off2] {sym} ptr mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64ADDconst {
			break
		}
		off2 := v_0.AuxInt
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	// cond: canMergeSym(sym1,sym2) && is32Bit(off1+off2) 	&& (ptr.Op != OpSB || !config.ctxt.Flag_shared)
	// result: (MOVWstorezero [off1+off2] {mergeSym(sym1,sym2)} ptr mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		ptr := v_0.Args[0]
		mem := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpARM64MUL_0(v *Value) bool {
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c)
		v.AddArg(x)
		return true
	}
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(isPowerOfTwo(c-1) && c >= 3) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c - 1)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MUL_10(v *Value) bool {
	b := v.Block
	_ = b
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpARM64MUL_20(v *Value) bool {
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = log2(c)
		v.AddArg(x)
		return true
	}
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		if !(isPowerOfTwo(c-1) && int32(c) >= 3) {
			break
		}
		v.reset(OpARM64ADDshiftLL)
		v.AuxInt = log2(c - 1)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64MULW_10(v *Value) bool {
	b := v.Block
	_ = b
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpARM64MULW_20(v *Value) bool {
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
	return false
}
func rewriteValueARM64_OpARM64NEG_0(v *Value) bool {
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
	b := v.Block
	_ = b
	// match: (OR x (MOVDconst [c]))
	// cond:
	// result: (ORconst  [c] x)
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
	// result: (ORconst  [c] x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARM64ORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (OR x x)
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
	// match: (OR x (SLLconst [c] y))
	// cond:
	// result: (ORshiftLL  x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64ORshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR (SLLconst [c] y) x)
	// cond:
	// result: (ORshiftLL  x y [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64ORshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR x (SRLconst [c] y))
	// cond:
	// result: (ORshiftRL  x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64ORshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR (SRLconst [c] y) x)
	// cond:
	// result: (ORshiftRL  x y [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64ORshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR x (SRAconst [c] y))
	// cond:
	// result: (ORshiftRA  x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64ORshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR (SRAconst [c] y) x)
	// cond:
	// result: (ORshiftRA  x y [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64ORshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [i3] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i2] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i1] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 	&& i2 == i0+2 	&& i3 == i0+3 	&& x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 	&& y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 	&& o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 	&& mergePoint(b,x0,x1,x2,x3) != nil 	&& clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) 	&& clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) 	&& clobber(o0) && clobber(o1) && clobber(s0)
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
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUload, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.Aux = s
		v1 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v1.AuxInt = i0
		v1.AddArg(p)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64OR_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (OR <t> y3:(MOVDnop x3:(MOVBUload [i0] {s} p mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [i3] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i2] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i1] {s} p mem))))
	// cond: i1 == i0+1 	&& i2 == i0+2 	&& i3 == i0+3 	&& x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 	&& y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 	&& o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 	&& mergePoint(b,x0,x1,x2,x3) != nil 	&& clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) 	&& clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) 	&& clobber(o0) && clobber(o1) && clobber(s0)
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
		_ = x3.Args[1]
		p := x3.Args[0]
		mem := x3.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUload, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.Aux = s
		v1 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v1.AuxInt = i0
		v1.AddArg(p)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [i7] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i6] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i4] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i3] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [i2] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [i1] {s} p mem))) y7:(MOVDnop x7:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 	&& i2 == i0+2 	&& i3 == i0+3 	&& i4 == i0+4 	&& i5 == i0+5 	&& i6 == i0+6 	&& i7 == i0+7 	&& x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 	&& x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 	&& y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 	&& y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 	&& o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 	&& o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 	&& mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil 	&& clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) 	&& clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) 	&& clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) 	&& clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) 	&& clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) 	&& clobber(o4) && clobber(o5) && clobber(s0)
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
		i7 := x0.AuxInt
		s := x0.Aux
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDload, t)
		v1.Aux = s
		v2 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> y7:(MOVDnop x7:(MOVBUload [i0] {s} p mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [i7] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i6] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i4] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i3] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [i2] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [i1] {s} p mem))))
	// cond: i1 == i0+1 	&& i2 == i0+2 	&& i3 == i0+3 	&& i4 == i0+4 	&& i5 == i0+5 	&& i6 == i0+6 	&& i7 == i0+7 	&& x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 	&& x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 	&& y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 	&& y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 	&& o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 	&& o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 	&& mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil 	&& clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) 	&& clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) 	&& clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) 	&& clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) 	&& clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) 	&& clobber(o4) && clobber(o5) && clobber(s0)
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
		i0 := x7.AuxInt
		s := x7.Aux
		_ = x7.Args[1]
		p := x7.Args[0]
		mem := x7.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDload, t)
		v1.Aux = s
		v2 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i3] {s} p mem)))
	// cond: i1 == i0+1 	&& i2 == i0+2 	&& i3 == i0+3 	&& x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 	&& y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 	&& o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 	&& mergePoint(b,x0,x1,x2,x3) != nil 	&& clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) 	&& clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) 	&& clobber(o0) && clobber(o1) && clobber(s0)
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
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64REVW, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWUload, t)
		v1.Aux = s
		v2 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> y3:(MOVDnop x3:(MOVBUload [i3] {s} p mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] s0:(SLLconst [24] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))))
	// cond: i1 == i0+1 	&& i2 == i0+2 	&& i3 == i0+3 	&& x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 	&& y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 	&& o0.Uses == 1 && o1.Uses == 1 && s0.Uses == 1 	&& mergePoint(b,x0,x1,x2,x3) != nil 	&& clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) 	&& clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) 	&& clobber(o0) && clobber(o1) && clobber(s0)
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
		_ = x3.Args[1]
		p := x3.Args[0]
		mem := x3.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64REVW, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWUload, t)
		v1.Aux = s
		v2 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i3] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i4] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [i5] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [i6] {s} p mem))) y7:(MOVDnop x7:(MOVBUload [i7] {s} p mem)))
	// cond: i1 == i0+1 	&& i2 == i0+2 	&& i3 == i0+3 	&& i4 == i0+4 	&& i5 == i0+5 	&& i6 == i0+6 	&& i7 == i0+7 	&& x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 	&& x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 	&& y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 	&& y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 	&& o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 	&& o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 	&& mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil 	&& clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) 	&& clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) 	&& clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) 	&& clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) 	&& clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) 	&& clobber(o4) && clobber(o5) && clobber(s0)
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
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDload, t)
		v1.Aux = s
		v2 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (OR <t> y7:(MOVDnop x7:(MOVBUload [i7] {s} p mem)) o0:(ORshiftLL [8] o1:(ORshiftLL [16] o2:(ORshiftLL [24] o3:(ORshiftLL [32] o4:(ORshiftLL [40] o5:(ORshiftLL [48] s0:(SLLconst [56] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem))) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i3] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i4] {s} p mem))) y5:(MOVDnop x5:(MOVBUload [i5] {s} p mem))) y6:(MOVDnop x6:(MOVBUload [i6] {s} p mem))))
	// cond: i1 == i0+1 	&& i2 == i0+2 	&& i3 == i0+3 	&& i4 == i0+4 	&& i5 == i0+5 	&& i6 == i0+6 	&& i7 == i0+7 	&& x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 	&& x4.Uses == 1 && x5.Uses == 1 && x6.Uses == 1 && x7.Uses == 1 	&& y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 	&& y4.Uses == 1 && y5.Uses == 1 && y6.Uses == 1 && y7.Uses == 1 	&& o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 && o3.Uses == 1 	&& o4.Uses == 1 && o5.Uses == 1 && s0.Uses == 1 	&& mergePoint(b,x0,x1,x2,x3,x4,x5,x6,x7) != nil 	&& clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) 	&& clobber(x4) && clobber(x5) && clobber(x6) && clobber(x7) 	&& clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) 	&& clobber(y4) && clobber(y5) && clobber(y6) && clobber(y7) 	&& clobber(o0) && clobber(o1) && clobber(o2) && clobber(o3) 	&& clobber(o4) && clobber(o5) && clobber(s0)
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
		_ = x7.Args[1]
		p := x7.Args[0]
		mem := x7.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDload, t)
		v1.Aux = s
		v2 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
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
	return false
}
func rewriteValueARM64_OpARM64ORshiftLL_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (ORshiftLL (MOVDconst [c]) x [d])
	// cond:
	// result: (ORconst  [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	// result: (ORconst  x [int64(uint64(c)<<uint64(d))])
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		x := v_0.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = 64 - c
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL <t> [c] (SRLconst (MOVWUreg x) [32-c]) x)
	// cond: c < 32 && t.Size() == 4
	// result: (RORWconst [32-c] x)
	for {
		t := v.Type
		c := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != 32-c {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0_0.Args[0]
		if x != v.Args[1] {
			break
		}
		if !(c < 32 && t.Size() == 4) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = 32 - c
		v.AddArg(x)
		return true
	}
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUload [i0] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem)))
	// cond: i1 == i0+1 	&& x0.Uses == 1 && x1.Uses == 1 	&& y0.Uses == 1 && y1.Uses == 1 	&& mergePoint(b,x0,x1) != nil 	&& clobber(x0) && clobber(x1) 	&& clobber(y0) && clobber(y1)
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
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64MOVHUload, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.Aux = s
		v1 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v1.AuxInt = i0
		v1.AddArg(p)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] x0:(MOVHUload [i0] {s} p mem) y1:(MOVDnop x1:(MOVBUload [i2] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i3] {s} p mem)))
	// cond: i2 == i0+2 	&& i3 == i0+3 	&& x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 	&& y1.Uses == 1 && y2.Uses == 1 	&& o0.Uses == 1 	&& mergePoint(b,x0,x1,x2) != nil 	&& clobber(x0) && clobber(x1) && clobber(x2) 	&& clobber(y1) && clobber(y2) 	&& clobber(o0)
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
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64MOVWUload, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.Aux = s
		v1 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v1.AuxInt = i0
		v1.AddArg(p)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] x0:(MOVWUload [i0] {s} p mem) y1:(MOVDnop x1:(MOVBUload [i4] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i5] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i6] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i7] {s} p mem)))
	// cond: i4 == i0+4 	&& i5 == i0+5 	&& i6 == i0+6 	&& i7 == i0+7 	&& x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 	&& y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 	&& o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 	&& mergePoint(b,x0,x1,x2,x3,x4) != nil 	&& clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) 	&& clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) 	&& clobber(o0) && clobber(o1) && clobber(o2)
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
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64MOVDload, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v0.Aux = s
		v1 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v1.AuxInt = i0
		v1.AddArg(p)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	// match: (ORshiftLL <t> [8] y0:(MOVDnop x0:(MOVBUload [i1] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 	&& x0.Uses == 1 && x1.Uses == 1 	&& y0.Uses == 1 && y1.Uses == 1 	&& mergePoint(b,x0,x1) != nil 	&& clobber(x0) && clobber(x1) 	&& clobber(y0) && clobber(y1)
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
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64REV16W, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVHUload, t)
		v1.AuxInt = i0
		v1.Aux = s
		v1.AddArg(p)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	// match: (ORshiftLL <t> [24] o0:(ORshiftLL [16] y0:(REV16W x0:(MOVHUload [i2] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [i1] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 	&& i2 == i0+2 	&& x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 	&& y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 	&& o0.Uses == 1 	&& mergePoint(b,x0,x1,x2) != nil 	&& clobber(x0) && clobber(x1) && clobber(x2) 	&& clobber(y0) && clobber(y1) && clobber(y2) 	&& clobber(o0)
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
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64REVW, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVWUload, t)
		v1.Aux = s
		v2 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORshiftLL_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (ORshiftLL <t> [56] o0:(ORshiftLL [48] o1:(ORshiftLL [40] o2:(ORshiftLL [32] y0:(REVW x0:(MOVWUload [i4] {s} p mem)) y1:(MOVDnop x1:(MOVBUload [i3] {s} p mem))) y2:(MOVDnop x2:(MOVBUload [i2] {s} p mem))) y3:(MOVDnop x3:(MOVBUload [i1] {s} p mem))) y4:(MOVDnop x4:(MOVBUload [i0] {s} p mem)))
	// cond: i1 == i0+1 	&& i2 == i0+2 	&& i3 == i0+3 	&& i4 == i0+4 	&& x0.Uses == 1 && x1.Uses == 1 && x2.Uses == 1 && x3.Uses == 1 && x4.Uses == 1 	&& y0.Uses == 1 && y1.Uses == 1 && y2.Uses == 1 && y3.Uses == 1 && y4.Uses == 1 	&& o0.Uses == 1 && o1.Uses == 1 && o2.Uses == 1 	&& mergePoint(b,x0,x1,x2,x3,x4) != nil 	&& clobber(x0) && clobber(x1) && clobber(x2) && clobber(x3) && clobber(x4) 	&& clobber(y0) && clobber(y1) && clobber(y2) && clobber(y3) && clobber(y4) 	&& clobber(o0) && clobber(o1) && clobber(o2)
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
		_ = x0.Args[1]
		p := x0.Args[0]
		mem := x0.Args[1]
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
		v0 := b.NewValue0(v.Pos, OpARM64REV, t)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDload, t)
		v1.Aux = s
		v2 := b.NewValue0(v.Pos, OpOffPtr, p.Type)
		v2.AuxInt = i0
		v2.AddArg(p)
		v1.AddArg(v2)
		v1.AddArg(mem)
		v0.AddArg(v1)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64ORshiftRA_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (ORshiftRA (MOVDconst [c]) x [d])
	// cond:
	// result: (ORconst  [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	// result: (ORconst  x [int64(int64(c)>>uint64(d))])
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
		v.AuxInt = int64(int64(c) >> uint64(d))
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
	_ = b
	// match: (ORshiftRL (MOVDconst [c]) x [d])
	// cond:
	// result: (ORconst  [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	// result: (ORconst  x [int64(uint64(c)>>uint64(d))])
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
	// result: (RORconst [   c] x)
	for {
		c := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		x := v_0.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (ORshiftRL <t> [c] (SLLconst x [32-c]) (MOVWUreg x))
	// cond: c < 32 && t.Size() == 4
	// result: (RORWconst [   c] x)
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
	// result: (MOVDconst [int64(d)<<uint64(c)])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(d) << uint64(c)
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
	// result: (MOVDconst [int64(d)>>uint64(c)])
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		d := v_0.AuxInt
		v.reset(OpARM64MOVDconst)
		v.AuxInt = int64(d) >> uint64(c)
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
	return false
}
func rewriteValueARM64_OpARM64SUB_0(v *Value) bool {
	b := v.Block
	_ = b
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
	// match: (SUB x x)
	// cond:
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
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
		_ = v_1.Args[1]
		y := v_1.Args[0]
		z := v_1.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SUB {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		z := v.Args[1]
		v.reset(OpARM64SUB)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64ADD, y.Type)
		v0.AddArg(y)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (SUB x (SLLconst [c] y))
	// cond:
	// result: (SUBshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64SUBshiftLL)
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
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64SUBshiftRL)
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
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64SUBshiftRA)
		v.AuxInt = c
		v.AddArg(x)
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
	// result: (SUBconst x [int64(int64(c)>>uint64(d))])
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
		v.AuxInt = int64(int64(c) >> uint64(d))
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
		v.reset(OpARM64XORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XOR x x)
	// cond:
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (XOR x (SLLconst [c] y))
	// cond:
	// result: (XORshiftLL x y [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64SLLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64XORshiftLL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR (SLLconst [c] y) x)
	// cond:
	// result: (XORshiftLL x y [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64XORshiftLL)
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
		if v_1.Op != OpARM64SRLconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64XORshiftRL)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR (SRLconst [c] y) x)
	// cond:
	// result: (XORshiftRL x y [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64XORshiftRL)
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
		if v_1.Op != OpARM64SRAconst {
			break
		}
		c := v_1.AuxInt
		y := v_1.Args[0]
		v.reset(OpARM64XORshiftRA)
		v.AuxInt = c
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (XOR (SRAconst [c] y) x)
	// cond:
	// result: (XORshiftRA x y [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRAconst {
			break
		}
		c := v_0.AuxInt
		y := v_0.Args[0]
		x := v.Args[1]
		v.reset(OpARM64XORshiftRA)
		v.AuxInt = c
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
	_ = b
	// match: (XORshiftLL (MOVDconst [c]) x [d])
	// cond:
	// result: (XORconst [c] (SLLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		x := v_0.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = 64 - c
		v.AddArg(x)
		return true
	}
	// match: (XORshiftLL <t> [c] (SRLconst (MOVWUreg x) [32-c]) x)
	// cond: c < 32 && t.Size() == 4
	// result: (RORWconst [32-c] x)
	for {
		t := v.Type
		c := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SRLconst {
			break
		}
		if v_0.AuxInt != 32-c {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpARM64MOVWUreg {
			break
		}
		x := v_0_0.Args[0]
		if x != v.Args[1] {
			break
		}
		if !(c < 32 && t.Size() == 4) {
			break
		}
		v.reset(OpARM64RORWconst)
		v.AuxInt = 32 - c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValueARM64_OpARM64XORshiftRA_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (XORshiftRA (MOVDconst [c]) x [d])
	// cond:
	// result: (XORconst [c] (SRAconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	// result: (XORconst x [int64(int64(c)>>uint64(d))])
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
		v.AuxInt = int64(int64(c) >> uint64(d))
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
	_ = b
	// match: (XORshiftRL (MOVDconst [c]) x [d])
	// cond:
	// result: (XORconst [c] (SRLconst <x.Type> x [d]))
	for {
		d := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64MOVDconst {
			break
		}
		c := v_0.AuxInt
		x := v.Args[1]
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
	// result: (RORconst [   c] x)
	for {
		c := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpARM64SLLconst {
			break
		}
		if v_0.AuxInt != 64-c {
			break
		}
		x := v_0.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpARM64RORconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (XORshiftRL <t> [c] (SLLconst x [32-c]) (MOVWUreg x))
	// cond: c < 32 && t.Size() == 4
	// result: (RORWconst [   c] x)
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
func rewriteValueARM64_OpAdd16_0(v *Value) bool {
	// match: (Add16 x y)
	// cond:
	// result: (ADD x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpARM64LoweredAtomicAdd32)
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
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpARM64LoweredAtomicAdd64)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicAnd8_0(v *Value) bool {
	// match: (AtomicAnd8 ptr val mem)
	// cond:
	// result: (LoweredAtomicAnd8 ptr val mem)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpARM64LoweredAtomicAnd8)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicCompareAndSwap32_0(v *Value) bool {
	// match: (AtomicCompareAndSwap32 ptr old new_ mem)
	// cond:
	// result: (LoweredAtomicCas32 ptr old new_ mem)
	for {
		_ = v.Args[3]
		ptr := v.Args[0]
		old := v.Args[1]
		new_ := v.Args[2]
		mem := v.Args[3]
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
		_ = v.Args[3]
		ptr := v.Args[0]
		old := v.Args[1]
		new_ := v.Args[2]
		mem := v.Args[3]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpARM64LDARW)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicLoad64_0(v *Value) bool {
	// match: (AtomicLoad64 ptr mem)
	// cond:
	// result: (LDAR  ptr mem)
	for {
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpARM64LDAR)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicLoadPtr_0(v *Value) bool {
	// match: (AtomicLoadPtr ptr mem)
	// cond:
	// result: (LDAR  ptr mem)
	for {
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpARM64LDAR)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicOr8_0(v *Value) bool {
	// match: (AtomicOr8 ptr val mem)
	// cond:
	// result: (LoweredAtomicOr8  ptr val mem)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpARM64LoweredAtomicOr8)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAtomicStore32_0(v *Value) bool {
	// match: (AtomicStore32 ptr val mem)
	// cond:
	// result: (STLRW ptr val mem)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
	// result: (STLR  ptr val mem)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
	// result: (STLR  ptr val mem)
	for {
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
		v.reset(OpARM64STLR)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpAvg64u_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Avg64u <t> x y)
	// cond:
	// result: (ADD (SRLconst <t> (SUB <t> x y) [1]) y)
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
func rewriteValueARM64_OpBitLen64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
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
func rewriteValueARM64_OpClosureCall_0(v *Value) bool {
	// match: (ClosureCall [argwid] entry closure mem)
	// cond:
	// result: (CALLclosure [argwid] entry closure mem)
	for {
		argwid := v.AuxInt
		_ = v.Args[2]
		entry := v.Args[0]
		closure := v.Args[1]
		mem := v.Args[2]
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
func rewriteValueARM64_OpConvert_0(v *Value) bool {
	// match: (Convert x mem)
	// cond:
	// result: (MOVDconvert x mem)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		mem := v.Args[1]
		v.reset(OpARM64MOVDconvert)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpCtz32_0(v *Value) bool {
	b := v.Block
	_ = b
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
func rewriteValueARM64_OpCtz64_0(v *Value) bool {
	b := v.Block
	_ = b
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div16 x y)
	// cond:
	// result: (DIVW (SignExt16to32 x) (SignExt16to32 y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div16u x y)
	// cond:
	// result: (UDIVW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64UDIV)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpDiv8_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div8 x y)
	// cond:
	// result: (DIVW (SignExt8to32 x) (SignExt8to32 y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div8u x y)
	// cond:
	// result: (UDIVW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Eq16 x y)
	// cond:
	// result: (Equal (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Eq32 x y)
	// cond:
	// result: (Equal (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Eq32F x y)
	// cond:
	// result: (Equal (FCMPS x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Eq64 x y)
	// cond:
	// result: (Equal (CMP x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Eq64F x y)
	// cond:
	// result: (Equal (FCMPD x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Eq8 x y)
	// cond:
	// result: (Equal (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (EqB x y)
	// cond:
	// result: (XOR (MOVDconst [1]) (XOR <typ.Bool> x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (EqPtr x y)
	// cond:
	// result: (Equal (CMP x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64Equal)
		v0 := b.NewValue0(v.Pos, OpARM64CMP, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGeq16_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Geq16 x y)
	// cond:
	// result: (GreaterEqual (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Geq16U x y)
	// cond:
	// result: (GreaterEqualU (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Geq32 x y)
	// cond:
	// result: (GreaterEqual (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Geq32F x y)
	// cond:
	// result: (GreaterEqual (FCMPS x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGeq32U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Geq32U x y)
	// cond:
	// result: (GreaterEqualU (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Geq64 x y)
	// cond:
	// result: (GreaterEqual (CMP x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Geq64F x y)
	// cond:
	// result: (GreaterEqual (FCMPD x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGeq64U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Geq64U x y)
	// cond:
	// result: (GreaterEqualU (CMP x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Geq8 x y)
	// cond:
	// result: (GreaterEqual (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Geq8U x y)
	// cond:
	// result: (GreaterEqualU (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Greater16 x y)
	// cond:
	// result: (GreaterThan (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Greater16U x y)
	// cond:
	// result: (GreaterThanU (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Greater32 x y)
	// cond:
	// result: (GreaterThan (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Greater32F x y)
	// cond:
	// result: (GreaterThan (FCMPS x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGreater32U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Greater32U x y)
	// cond:
	// result: (GreaterThanU (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Greater64 x y)
	// cond:
	// result: (GreaterThan (CMP x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Greater64F x y)
	// cond:
	// result: (GreaterThan (FCMPD x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpGreater64U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Greater64U x y)
	// cond:
	// result: (GreaterThanU (CMP x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Greater8 x y)
	// cond:
	// result: (GreaterThan (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Greater8U x y)
	// cond:
	// result: (GreaterThanU (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Hmul32 x y)
	// cond:
	// result: (SRAconst (MULL <typ.Int64> x y) [32])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Hmul32u x y)
	// cond:
	// result: (SRAconst (UMULL <typ.UInt64> x y) [32])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		entry := v.Args[0]
		mem := v.Args[1]
		v.reset(OpARM64CALLinter)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpIsInBounds_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (IsInBounds idx len)
	// cond:
	// result: (LessThanU (CMP idx len))
	for {
		_ = v.Args[1]
		idx := v.Args[0]
		len := v.Args[1]
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
	_ = b
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
	_ = b
	// match: (IsSliceInBounds idx len)
	// cond:
	// result: (LessEqualU (CMP idx len))
	for {
		_ = v.Args[1]
		idx := v.Args[0]
		len := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Leq16 x y)
	// cond:
	// result: (LessEqual (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Leq16U x y)
	// cond:
	// result: (LessEqualU (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Leq32 x y)
	// cond:
	// result: (LessEqual (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Leq32F x y)
	// cond:
	// result: (GreaterEqual (FCMPS y x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq32U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Leq32U x y)
	// cond:
	// result: (LessEqualU (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Leq64 x y)
	// cond:
	// result: (LessEqual (CMP x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Leq64F x y)
	// cond:
	// result: (GreaterEqual (FCMPD y x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64GreaterEqual)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLeq64U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Leq64U x y)
	// cond:
	// result: (LessEqualU (CMP x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Leq8 x y)
	// cond:
	// result: (LessEqual (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Leq8U x y)
	// cond:
	// result: (LessEqualU (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Less16 x y)
	// cond:
	// result: (LessThan (CMPW (SignExt16to32 x) (SignExt16to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Less16U x y)
	// cond:
	// result: (LessThanU (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Less32 x y)
	// cond:
	// result: (LessThan (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Less32F x y)
	// cond:
	// result: (GreaterThan (FCMPS y x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPS, types.TypeFlags)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess32U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Less32U x y)
	// cond:
	// result: (LessThanU (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Less64 x y)
	// cond:
	// result: (LessThan (CMP x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Less64F x y)
	// cond:
	// result: (GreaterThan (FCMPD y x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64GreaterThan)
		v0 := b.NewValue0(v.Pos, OpARM64FCMPD, types.TypeFlags)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueARM64_OpLess64U_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Less64U x y)
	// cond:
	// result: (LessThanU (CMP x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Less8 x y)
	// cond:
	// result: (LessThan (CMPW (SignExt8to32 x) (SignExt8to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Less8U x y)
	// cond:
	// result: (LessThanU (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
func rewriteValueARM64_OpLsh16x16_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh16x16 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh16x32 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	// match: (Lsh16x64 x (MOVDconst [c]))
	// cond: uint64(c) < 16
	// result: (SLLconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 16) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (Lsh16x64 _ (MOVDconst [c]))
	// cond: uint64(c) >= 16
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 16) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (Lsh16x64 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh16x8 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x (ZeroExt8to64  y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64  y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh32x16 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh32x32 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	// match: (Lsh32x64 x (MOVDconst [c]))
	// cond: uint64(c) < 32
	// result: (SLLconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 32) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (Lsh32x64 _ (MOVDconst [c]))
	// cond: uint64(c) >= 32
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 32) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (Lsh32x64 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh32x8 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x (ZeroExt8to64  y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64  y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh64x16 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh64x32 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	// match: (Lsh64x64 x (MOVDconst [c]))
	// cond: uint64(c) < 64
	// result: (SLLconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 64) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (Lsh64x64 _ (MOVDconst [c]))
	// cond: uint64(c) >= 64
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (Lsh64x64 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh64x8 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x (ZeroExt8to64  y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64  y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh8x16 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh8x32 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	// match: (Lsh8x64 x (MOVDconst [c]))
	// cond: uint64(c) < 8
	// result: (SLLconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 8) {
			break
		}
		v.reset(OpARM64SLLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (Lsh8x64 _ (MOVDconst [c]))
	// cond: uint64(c) >= 8
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 8) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (Lsh8x64 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh8x8 <t> x y)
	// cond:
	// result: (CSELULT (SLL <t> x (ZeroExt8to64  y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64  y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mod16 x y)
	// cond:
	// result: (MODW (SignExt16to32 x) (SignExt16to32 y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mod16u x y)
	// cond:
	// result: (UMODW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64UMOD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpMod8_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mod8 x y)
	// cond:
	// result: (MODW (SignExt8to32 x) (SignExt8to32 y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mod8u x y)
	// cond:
	// result: (UMODW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	// result: (MOVBstore dst (MOVBUload src mem) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
	// result: (MOVBstore [2] dst (MOVBUload [2] src mem) 		(MOVHstore dst (MOVHUload src mem) mem))
	for {
		if v.AuxInt != 3 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
	// result: (MOVBstore [4] dst (MOVBUload [4] src mem) 		(MOVWstore dst (MOVWUload src mem) mem))
	for {
		if v.AuxInt != 5 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
	// result: (MOVHstore [4] dst (MOVHUload [4] src mem) 		(MOVWstore dst (MOVWUload src mem) mem))
	for {
		if v.AuxInt != 6 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
	// result: (MOVBstore [6] dst (MOVBUload [6] src mem) 		(MOVHstore [4] dst (MOVHUload [4] src mem) 			(MOVWstore dst (MOVWUload src mem) mem)))
	for {
		if v.AuxInt != 7 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
	// result: (MOVWstore [8] dst (MOVWUload [8] src mem) 		(MOVDstore dst (MOVDload src mem) mem))
	for {
		if v.AuxInt != 12 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Move [16] dst src mem)
	// cond:
	// result: (MOVDstore [8] dst (MOVDload [8] src mem) 		(MOVDstore dst (MOVDload src mem) mem))
	for {
		if v.AuxInt != 16 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
	// result: (MOVDstore [16] dst (MOVDload [16] src mem) 		(MOVDstore [8] dst (MOVDload [8] src mem) 			(MOVDstore dst (MOVDload src mem) mem)))
	for {
		if v.AuxInt != 24 {
			break
		}
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
	// result: (Move [s%8] 		(OffPtr <dst.Type> dst [s-s%8]) 		(OffPtr <src.Type> src [s-s%8]) 		(Move [s-s%8] dst src mem))
	for {
		s := v.AuxInt
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
	// cond: s%8 == 0 && s > 24 && s <= 8*128 	&& !config.noDuffDevice
	// result: (DUFFCOPY [8 * (128 - int64(s/8))] dst src mem)
	for {
		s := v.AuxInt
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
		if !(s%8 == 0 && s > 24 && s <= 8*128 && !config.noDuffDevice) {
			break
		}
		v.reset(OpARM64DUFFCOPY)
		v.AuxInt = 8 * (128 - int64(s/8))
		v.AddArg(dst)
		v.AddArg(src)
		v.AddArg(mem)
		return true
	}
	// match: (Move [s] dst src mem)
	// cond: s > 24 && s%8 == 0
	// result: (LoweredMove 		dst 		src 		(ADDconst <src.Type> src [s-8]) 		mem)
	for {
		s := v.AuxInt
		_ = v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64FMULD)
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Neq16 x y)
	// cond:
	// result: (NotEqual (CMPW (ZeroExt16to32 x) (ZeroExt16to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Neq32 x y)
	// cond:
	// result: (NotEqual (CMPW x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Neq32F x y)
	// cond:
	// result: (NotEqual (FCMPS x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Neq64 x y)
	// cond:
	// result: (NotEqual (CMP x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	// match: (Neq64F x y)
	// cond:
	// result: (NotEqual (FCMPD x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Neq8 x y)
	// cond:
	// result: (NotEqual (CMPW (ZeroExt8to32 x) (ZeroExt8to32 y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpNeqPtr_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (NeqPtr x y)
	// cond:
	// result: (NotEqual (CMP x y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpARM64LoweredNilCheck)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueARM64_OpNot_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpRound32F_0(v *Value) bool {
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
func rewriteValueARM64_OpRound64F_0(v *Value) bool {
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
func rewriteValueARM64_OpRsh16Ux16_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh16Ux16 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> (ZeroExt16to64 x) (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh16Ux32 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> (ZeroExt16to64 x) (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh16Ux64 x (MOVDconst [c]))
	// cond: uint64(c) < 16
	// result: (SRLconst (ZeroExt16to64 x) [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 16) {
			break
		}
		v.reset(OpARM64SRLconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16Ux64 _ (MOVDconst [c]))
	// cond: uint64(c) >= 16
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 16) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (Rsh16Ux64 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> (ZeroExt16to64 x) y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh16Ux8 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> (ZeroExt16to64 x) (ZeroExt8to64  y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64  y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh16x16 x y)
	// cond:
	// result: (SRA (SignExt16to64 x) (CSELULT <y.Type> (ZeroExt16to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh16x32 x y)
	// cond:
	// result: (SRA (SignExt16to64 x) (CSELULT <y.Type> (ZeroExt32to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh16x64 x (MOVDconst [c]))
	// cond: uint64(c) < 16
	// result: (SRAconst (SignExt16to64 x) [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 16) {
			break
		}
		v.reset(OpARM64SRAconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x64 x (MOVDconst [c]))
	// cond: uint64(c) >= 16
	// result: (SRAconst (SignExt16to64 x) [63])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 16) {
			break
		}
		v.reset(OpARM64SRAconst)
		v.AuxInt = 63
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x64 x y)
	// cond:
	// result: (SRA (SignExt16to64 x) (CSELULT <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh16x8 x y)
	// cond:
	// result: (SRA (SignExt16to64 x) (CSELULT <y.Type> (ZeroExt8to64  y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64  y))))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh32Ux16 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> (ZeroExt32to64 x) (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh32Ux32 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> (ZeroExt32to64 x) (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh32Ux64 x (MOVDconst [c]))
	// cond: uint64(c) < 32
	// result: (SRLconst (ZeroExt32to64 x) [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 32) {
			break
		}
		v.reset(OpARM64SRLconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32Ux64 _ (MOVDconst [c]))
	// cond: uint64(c) >= 32
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 32) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (Rsh32Ux64 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> (ZeroExt32to64 x) y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh32Ux8 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> (ZeroExt32to64 x) (ZeroExt8to64  y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64  y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh32x16 x y)
	// cond:
	// result: (SRA (SignExt32to64 x) (CSELULT <y.Type> (ZeroExt16to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh32x32 x y)
	// cond:
	// result: (SRA (SignExt32to64 x) (CSELULT <y.Type> (ZeroExt32to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh32x64 x (MOVDconst [c]))
	// cond: uint64(c) < 32
	// result: (SRAconst (SignExt32to64 x) [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 32) {
			break
		}
		v.reset(OpARM64SRAconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32x64 x (MOVDconst [c]))
	// cond: uint64(c) >= 32
	// result: (SRAconst (SignExt32to64 x) [63])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 32) {
			break
		}
		v.reset(OpARM64SRAconst)
		v.AuxInt = 63
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32x64 x y)
	// cond:
	// result: (SRA (SignExt32to64 x) (CSELULT <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh32x8 x y)
	// cond:
	// result: (SRA (SignExt32to64 x) (CSELULT <y.Type> (ZeroExt8to64  y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64  y))))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh64Ux16 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> x (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh64Ux32 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> x (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	// match: (Rsh64Ux64 x (MOVDconst [c]))
	// cond: uint64(c) < 64
	// result: (SRLconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 64) {
			break
		}
		v.reset(OpARM64SRLconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (Rsh64Ux64 _ (MOVDconst [c]))
	// cond: uint64(c) >= 64
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (Rsh64Ux64 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> x y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh64Ux8 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> x (ZeroExt8to64  y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64  y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh64x16 x y)
	// cond:
	// result: (SRA x (CSELULT <y.Type> (ZeroExt16to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh64x32 x y)
	// cond:
	// result: (SRA x (CSELULT <y.Type> (ZeroExt32to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	// match: (Rsh64x64 x (MOVDconst [c]))
	// cond: uint64(c) < 64
	// result: (SRAconst x [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 64) {
			break
		}
		v.reset(OpARM64SRAconst)
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (Rsh64x64 x (MOVDconst [c]))
	// cond: uint64(c) >= 64
	// result: (SRAconst x [63])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpARM64SRAconst)
		v.AuxInt = 63
		v.AddArg(x)
		return true
	}
	// match: (Rsh64x64 x y)
	// cond:
	// result: (SRA x (CSELULT <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh64x8 x y)
	// cond:
	// result: (SRA x (CSELULT <y.Type> (ZeroExt8to64  y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64  y))))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh8Ux16 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> (ZeroExt8to64 x) (ZeroExt16to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt16to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh8Ux32 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> (ZeroExt8to64 x) (ZeroExt32to64 y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt32to64 y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh8Ux64 x (MOVDconst [c]))
	// cond: uint64(c) < 8
	// result: (SRLconst (ZeroExt8to64  x) [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 8) {
			break
		}
		v.reset(OpARM64SRLconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8Ux64 _ (MOVDconst [c]))
	// cond: uint64(c) >= 8
	// result: (MOVDconst [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 8) {
			break
		}
		v.reset(OpARM64MOVDconst)
		v.AuxInt = 0
		return true
	}
	// match: (Rsh8Ux64 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> (ZeroExt8to64 x) y) (Const64 <t> [0]) (CMPconst [64] y))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh8Ux8 <t> x y)
	// cond:
	// result: (CSELULT (SRL <t> (ZeroExt8to64 x) (ZeroExt8to64  y)) (Const64 <t> [0]) (CMPconst [64] (ZeroExt8to64  y)))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64CSELULT)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh8x16 x y)
	// cond:
	// result: (SRA (SignExt8to64 x) (CSELULT <y.Type> (ZeroExt16to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt16to64 y))))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh8x32 x y)
	// cond:
	// result: (SRA (SignExt8to64 x) (CSELULT <y.Type> (ZeroExt32to64 y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt32to64 y))))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh8x64 x (MOVDconst [c]))
	// cond: uint64(c) < 8
	// result: (SRAconst (SignExt8to64  x) [c])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) < 8) {
			break
		}
		v.reset(OpARM64SRAconst)
		v.AuxInt = c
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8x64 x (MOVDconst [c]))
	// cond: uint64(c) >= 8
	// result: (SRAconst (SignExt8to64  x) [63])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpARM64MOVDconst {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 8) {
			break
		}
		v.reset(OpARM64SRAconst)
		v.AuxInt = 63
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8x64 x y)
	// cond:
	// result: (SRA (SignExt8to64 x) (CSELULT <y.Type> y (Const64 <y.Type> [63]) (CMPconst [64] y)))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh8x8 x y)
	// cond:
	// result: (SRA (SignExt8to64 x) (CSELULT <y.Type> (ZeroExt8to64  y) (Const64 <y.Type> [63]) (CMPconst [64] (ZeroExt8to64  y))))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SRA)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64CSELULT, y.Type)
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
	_ = b
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
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		mem := v.Args[2]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64SUB)
		v.AddArg(x)
		v.AddArg(y)
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
func rewriteValueARM64_OpXor16_0(v *Value) bool {
	// match: (Xor16 x y)
	// cond:
	// result: (XOR x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
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
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpARM64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueARM64_OpZero_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
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
	// match: (Zero [1] ptr mem)
	// cond:
	// result: (MOVBstore ptr (MOVDconst [0]) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
	// result: (MOVBstore [2] ptr (MOVDconst [0]) 		(MOVHstore ptr (MOVDconst [0]) mem))
	for {
		if v.AuxInt != 3 {
			break
		}
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
	// result: (MOVBstore [4] ptr (MOVDconst [0]) 		(MOVWstore ptr (MOVDconst [0]) mem))
	for {
		if v.AuxInt != 5 {
			break
		}
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
	// result: (MOVHstore [4] ptr (MOVDconst [0]) 		(MOVWstore ptr (MOVDconst [0]) mem))
	for {
		if v.AuxInt != 6 {
			break
		}
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
	// result: (MOVBstore [6] ptr (MOVDconst [0]) 		(MOVHstore [4] ptr (MOVDconst [0]) 			(MOVWstore ptr (MOVDconst [0]) mem)))
	for {
		if v.AuxInt != 7 {
			break
		}
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
	// match: (Zero [12] ptr mem)
	// cond:
	// result: (MOVWstore [8] ptr (MOVDconst [0]) 		(MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if v.AuxInt != 12 {
			break
		}
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
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
	return false
}
func rewriteValueARM64_OpZero_10(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Zero [16] ptr mem)
	// cond:
	// result: (MOVDstore [8] ptr (MOVDconst [0]) 		(MOVDstore ptr (MOVDconst [0]) mem))
	for {
		if v.AuxInt != 16 {
			break
		}
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpARM64MOVDstore)
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
	// match: (Zero [24] ptr mem)
	// cond:
	// result: (MOVDstore [16] ptr (MOVDconst [0]) 		(MOVDstore [8] ptr (MOVDconst [0]) 			(MOVDstore ptr (MOVDconst [0]) mem)))
	for {
		if v.AuxInt != 24 {
			break
		}
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		v.reset(OpARM64MOVDstore)
		v.AuxInt = 16
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64MOVDconst, typ.UInt64)
		v0.AuxInt = 0
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpARM64MOVDstore, types.TypeMem)
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
	// match: (Zero [s] ptr mem)
	// cond: s%8 != 0 && s > 8
	// result: (Zero [s%8] 		(OffPtr <ptr.Type> ptr [s-s%8]) 		(Zero [s-s%8] ptr mem))
	for {
		s := v.AuxInt
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(s%8 != 0 && s > 8) {
			break
		}
		v.reset(OpZero)
		v.AuxInt = s % 8
		v0 := b.NewValue0(v.Pos, OpOffPtr, ptr.Type)
		v0.AuxInt = s - s%8
		v0.AddArg(ptr)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZero, types.TypeMem)
		v1.AuxInt = s - s%8
		v1.AddArg(ptr)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s%8 == 0 && s > 24 && s <= 8*128 	&& !config.noDuffDevice
	// result: (DUFFZERO [4 * (128 - int64(s/8))] ptr mem)
	for {
		s := v.AuxInt
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(s%8 == 0 && s > 24 && s <= 8*128 && !config.noDuffDevice) {
			break
		}
		v.reset(OpARM64DUFFZERO)
		v.AuxInt = 4 * (128 - int64(s/8))
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [s] ptr mem)
	// cond: s%8 == 0 && (s > 8*128 || config.noDuffDevice)
	// result: (LoweredZero 		ptr 		(ADDconst <ptr.Type> [s-8] ptr) 		mem)
	for {
		s := v.AuxInt
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(s%8 == 0 && (s > 8*128 || config.noDuffDevice)) {
			break
		}
		v.reset(OpARM64LoweredZero)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpARM64ADDconst, ptr.Type)
		v0.AuxInt = s - 8
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
	_ = config
	fe := b.Func.fe
	_ = fe
	typ := &config.Types
	_ = typ
	switch b.Kind {
	case BlockARM64EQ:
		// match: (EQ (CMPconst [0] x) yes no)
		// cond:
		// result: (Z x yes no)
		for {
			v := b.Control
			if v.Op != OpARM64CMPconst {
				break
			}
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			b.Kind = BlockARM64Z
			b.SetControl(x)
			return true
		}
		// match: (EQ (CMPWconst [0] x) yes no)
		// cond:
		// result: (ZW x yes no)
		for {
			v := b.Control
			if v.Op != OpARM64CMPWconst {
				break
			}
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			b.Kind = BlockARM64ZW
			b.SetControl(x)
			return true
		}
		// match: (EQ (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpARM64FlagEQ {
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
			if v.Op != OpARM64FlagLT_ULT {
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
			if v.Op != OpARM64FlagLT_UGT {
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
			if v.Op != OpARM64FlagGT_ULT {
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
			if v.Op != OpARM64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (EQ (InvertFlags cmp) yes no)
		// cond:
		// result: (EQ cmp yes no)
		for {
			v := b.Control
			if v.Op != OpARM64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockARM64EQ
			b.SetControl(cmp)
			return true
		}
	case BlockARM64GE:
		// match: (GE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpARM64FlagEQ {
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
			if v.Op != OpARM64FlagLT_ULT {
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
			if v.Op != OpARM64FlagLT_UGT {
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
			if v.Op != OpARM64FlagGT_ULT {
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
			if v.Op != OpARM64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (GE (InvertFlags cmp) yes no)
		// cond:
		// result: (LE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpARM64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockARM64LE
			b.SetControl(cmp)
			return true
		}
	case BlockARM64GT:
		// match: (GT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpARM64FlagEQ {
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
			if v.Op != OpARM64FlagLT_ULT {
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
			if v.Op != OpARM64FlagLT_UGT {
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
			if v.Op != OpARM64FlagGT_ULT {
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
			if v.Op != OpARM64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (GT (InvertFlags cmp) yes no)
		// cond:
		// result: (LT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpARM64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockARM64LT
			b.SetControl(cmp)
			return true
		}
	case BlockIf:
		// match: (If (Equal cc) yes no)
		// cond:
		// result: (EQ cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64Equal {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64EQ
			b.SetControl(cc)
			return true
		}
		// match: (If (NotEqual cc) yes no)
		// cond:
		// result: (NE cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64NotEqual {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64NE
			b.SetControl(cc)
			return true
		}
		// match: (If (LessThan cc) yes no)
		// cond:
		// result: (LT cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64LessThan {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64LT
			b.SetControl(cc)
			return true
		}
		// match: (If (LessThanU cc) yes no)
		// cond:
		// result: (ULT cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64LessThanU {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64ULT
			b.SetControl(cc)
			return true
		}
		// match: (If (LessEqual cc) yes no)
		// cond:
		// result: (LE cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64LessEqual {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64LE
			b.SetControl(cc)
			return true
		}
		// match: (If (LessEqualU cc) yes no)
		// cond:
		// result: (ULE cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64LessEqualU {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64ULE
			b.SetControl(cc)
			return true
		}
		// match: (If (GreaterThan cc) yes no)
		// cond:
		// result: (GT cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64GreaterThan {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64GT
			b.SetControl(cc)
			return true
		}
		// match: (If (GreaterThanU cc) yes no)
		// cond:
		// result: (UGT cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64GreaterThanU {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64UGT
			b.SetControl(cc)
			return true
		}
		// match: (If (GreaterEqual cc) yes no)
		// cond:
		// result: (GE cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64GreaterEqual {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64GE
			b.SetControl(cc)
			return true
		}
		// match: (If (GreaterEqualU cc) yes no)
		// cond:
		// result: (UGE cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64GreaterEqualU {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64UGE
			b.SetControl(cc)
			return true
		}
		// match: (If cond yes no)
		// cond:
		// result: (NZ cond yes no)
		for {
			v := b.Control
			_ = v
			cond := b.Control
			b.Kind = BlockARM64NZ
			b.SetControl(cond)
			return true
		}
	case BlockARM64LE:
		// match: (LE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpARM64FlagEQ {
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
			if v.Op != OpARM64FlagLT_ULT {
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
			if v.Op != OpARM64FlagLT_UGT {
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
			if v.Op != OpARM64FlagGT_ULT {
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
			if v.Op != OpARM64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (LE (InvertFlags cmp) yes no)
		// cond:
		// result: (GE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpARM64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockARM64GE
			b.SetControl(cmp)
			return true
		}
	case BlockARM64LT:
		// match: (LT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpARM64FlagEQ {
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
			if v.Op != OpARM64FlagLT_ULT {
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
			if v.Op != OpARM64FlagLT_UGT {
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
			if v.Op != OpARM64FlagGT_ULT {
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
			if v.Op != OpARM64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (LT (InvertFlags cmp) yes no)
		// cond:
		// result: (GT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpARM64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockARM64GT
			b.SetControl(cmp)
			return true
		}
	case BlockARM64NE:
		// match: (NE (CMPconst [0] x) yes no)
		// cond:
		// result: (NZ x yes no)
		for {
			v := b.Control
			if v.Op != OpARM64CMPconst {
				break
			}
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			b.Kind = BlockARM64NZ
			b.SetControl(x)
			return true
		}
		// match: (NE (CMPWconst [0] x) yes no)
		// cond:
		// result: (NZW x yes no)
		for {
			v := b.Control
			if v.Op != OpARM64CMPWconst {
				break
			}
			if v.AuxInt != 0 {
				break
			}
			x := v.Args[0]
			b.Kind = BlockARM64NZW
			b.SetControl(x)
			return true
		}
		// match: (NE (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpARM64FlagEQ {
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
			if v.Op != OpARM64FlagLT_ULT {
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
			if v.Op != OpARM64FlagLT_UGT {
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
			if v.Op != OpARM64FlagGT_ULT {
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
			if v.Op != OpARM64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (NE (InvertFlags cmp) yes no)
		// cond:
		// result: (NE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpARM64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockARM64NE
			b.SetControl(cmp)
			return true
		}
	case BlockARM64NZ:
		// match: (NZ (Equal cc) yes no)
		// cond:
		// result: (EQ cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64Equal {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64EQ
			b.SetControl(cc)
			return true
		}
		// match: (NZ (NotEqual cc) yes no)
		// cond:
		// result: (NE cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64NotEqual {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64NE
			b.SetControl(cc)
			return true
		}
		// match: (NZ (LessThan cc) yes no)
		// cond:
		// result: (LT cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64LessThan {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64LT
			b.SetControl(cc)
			return true
		}
		// match: (NZ (LessThanU cc) yes no)
		// cond:
		// result: (ULT cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64LessThanU {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64ULT
			b.SetControl(cc)
			return true
		}
		// match: (NZ (LessEqual cc) yes no)
		// cond:
		// result: (LE cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64LessEqual {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64LE
			b.SetControl(cc)
			return true
		}
		// match: (NZ (LessEqualU cc) yes no)
		// cond:
		// result: (ULE cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64LessEqualU {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64ULE
			b.SetControl(cc)
			return true
		}
		// match: (NZ (GreaterThan cc) yes no)
		// cond:
		// result: (GT cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64GreaterThan {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64GT
			b.SetControl(cc)
			return true
		}
		// match: (NZ (GreaterThanU cc) yes no)
		// cond:
		// result: (UGT cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64GreaterThanU {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64UGT
			b.SetControl(cc)
			return true
		}
		// match: (NZ (GreaterEqual cc) yes no)
		// cond:
		// result: (GE cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64GreaterEqual {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64GE
			b.SetControl(cc)
			return true
		}
		// match: (NZ (GreaterEqualU cc) yes no)
		// cond:
		// result: (UGE cc yes no)
		for {
			v := b.Control
			if v.Op != OpARM64GreaterEqualU {
				break
			}
			cc := v.Args[0]
			b.Kind = BlockARM64UGE
			b.SetControl(cc)
			return true
		}
		// match: (NZ (MOVDconst [0]) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpARM64MOVDconst {
				break
			}
			if v.AuxInt != 0 {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (NZ (MOVDconst [c]) yes no)
		// cond: c != 0
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpARM64MOVDconst {
				break
			}
			c := v.AuxInt
			if !(c != 0) {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
	case BlockARM64NZW:
		// match: (NZW (MOVDconst [c]) yes no)
		// cond: int32(c) == 0
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpARM64MOVDconst {
				break
			}
			c := v.AuxInt
			if !(int32(c) == 0) {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (NZW (MOVDconst [c]) yes no)
		// cond: int32(c) != 0
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpARM64MOVDconst {
				break
			}
			c := v.AuxInt
			if !(int32(c) != 0) {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
	case BlockARM64UGE:
		// match: (UGE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpARM64FlagEQ {
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
			if v.Op != OpARM64FlagLT_ULT {
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
			if v.Op != OpARM64FlagLT_UGT {
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
			if v.Op != OpARM64FlagGT_ULT {
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
			if v.Op != OpARM64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (UGE (InvertFlags cmp) yes no)
		// cond:
		// result: (ULE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpARM64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockARM64ULE
			b.SetControl(cmp)
			return true
		}
	case BlockARM64UGT:
		// match: (UGT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpARM64FlagEQ {
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
			if v.Op != OpARM64FlagLT_ULT {
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
			if v.Op != OpARM64FlagLT_UGT {
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
			if v.Op != OpARM64FlagGT_ULT {
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
			if v.Op != OpARM64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (UGT (InvertFlags cmp) yes no)
		// cond:
		// result: (ULT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpARM64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockARM64ULT
			b.SetControl(cmp)
			return true
		}
	case BlockARM64ULE:
		// match: (ULE (FlagEQ) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpARM64FlagEQ {
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
			if v.Op != OpARM64FlagLT_ULT {
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
			if v.Op != OpARM64FlagLT_UGT {
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
			if v.Op != OpARM64FlagGT_ULT {
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
			if v.Op != OpARM64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (ULE (InvertFlags cmp) yes no)
		// cond:
		// result: (UGE cmp yes no)
		for {
			v := b.Control
			if v.Op != OpARM64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockARM64UGE
			b.SetControl(cmp)
			return true
		}
	case BlockARM64ULT:
		// match: (ULT (FlagEQ) yes no)
		// cond:
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpARM64FlagEQ {
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
			if v.Op != OpARM64FlagLT_ULT {
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
			if v.Op != OpARM64FlagLT_UGT {
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
			if v.Op != OpARM64FlagGT_ULT {
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
			if v.Op != OpARM64FlagGT_UGT {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
		// match: (ULT (InvertFlags cmp) yes no)
		// cond:
		// result: (UGT cmp yes no)
		for {
			v := b.Control
			if v.Op != OpARM64InvertFlags {
				break
			}
			cmp := v.Args[0]
			b.Kind = BlockARM64UGT
			b.SetControl(cmp)
			return true
		}
	case BlockARM64Z:
		// match: (Z (MOVDconst [0]) yes no)
		// cond:
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpARM64MOVDconst {
				break
			}
			if v.AuxInt != 0 {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (Z (MOVDconst [c]) yes no)
		// cond: c != 0
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpARM64MOVDconst {
				break
			}
			c := v.AuxInt
			if !(c != 0) {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			b.swapSuccessors()
			return true
		}
	case BlockARM64ZW:
		// match: (ZW (MOVDconst [c]) yes no)
		// cond: int32(c) == 0
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpARM64MOVDconst {
				break
			}
			c := v.AuxInt
			if !(int32(c) == 0) {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (ZW (MOVDconst [c]) yes no)
		// cond: int32(c) != 0
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpARM64MOVDconst {
				break
			}
			c := v.AuxInt
			if !(int32(c) != 0) {
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
