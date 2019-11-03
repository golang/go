// Code generated from gen/RISCV64.rules; DO NOT EDIT.
// generated with: cd gen; go run *.go

package ssa

import "math"
import "cmd/compile/internal/types"

func rewriteValueRISCV64(v *Value) bool {
	switch v.Op {
	case OpAdd16:
		return rewriteValueRISCV64_OpAdd16_0(v)
	case OpAdd32:
		return rewriteValueRISCV64_OpAdd32_0(v)
	case OpAdd32F:
		return rewriteValueRISCV64_OpAdd32F_0(v)
	case OpAdd64:
		return rewriteValueRISCV64_OpAdd64_0(v)
	case OpAdd64F:
		return rewriteValueRISCV64_OpAdd64F_0(v)
	case OpAdd8:
		return rewriteValueRISCV64_OpAdd8_0(v)
	case OpAddPtr:
		return rewriteValueRISCV64_OpAddPtr_0(v)
	case OpAddr:
		return rewriteValueRISCV64_OpAddr_0(v)
	case OpAnd16:
		return rewriteValueRISCV64_OpAnd16_0(v)
	case OpAnd32:
		return rewriteValueRISCV64_OpAnd32_0(v)
	case OpAnd64:
		return rewriteValueRISCV64_OpAnd64_0(v)
	case OpAnd8:
		return rewriteValueRISCV64_OpAnd8_0(v)
	case OpAndB:
		return rewriteValueRISCV64_OpAndB_0(v)
	case OpAvg64u:
		return rewriteValueRISCV64_OpAvg64u_0(v)
	case OpClosureCall:
		return rewriteValueRISCV64_OpClosureCall_0(v)
	case OpCom16:
		return rewriteValueRISCV64_OpCom16_0(v)
	case OpCom32:
		return rewriteValueRISCV64_OpCom32_0(v)
	case OpCom64:
		return rewriteValueRISCV64_OpCom64_0(v)
	case OpCom8:
		return rewriteValueRISCV64_OpCom8_0(v)
	case OpConst16:
		return rewriteValueRISCV64_OpConst16_0(v)
	case OpConst32:
		return rewriteValueRISCV64_OpConst32_0(v)
	case OpConst32F:
		return rewriteValueRISCV64_OpConst32F_0(v)
	case OpConst64:
		return rewriteValueRISCV64_OpConst64_0(v)
	case OpConst64F:
		return rewriteValueRISCV64_OpConst64F_0(v)
	case OpConst8:
		return rewriteValueRISCV64_OpConst8_0(v)
	case OpConstBool:
		return rewriteValueRISCV64_OpConstBool_0(v)
	case OpConstNil:
		return rewriteValueRISCV64_OpConstNil_0(v)
	case OpConvert:
		return rewriteValueRISCV64_OpConvert_0(v)
	case OpCvt32Fto32:
		return rewriteValueRISCV64_OpCvt32Fto32_0(v)
	case OpCvt32Fto64:
		return rewriteValueRISCV64_OpCvt32Fto64_0(v)
	case OpCvt32Fto64F:
		return rewriteValueRISCV64_OpCvt32Fto64F_0(v)
	case OpCvt32to32F:
		return rewriteValueRISCV64_OpCvt32to32F_0(v)
	case OpCvt32to64F:
		return rewriteValueRISCV64_OpCvt32to64F_0(v)
	case OpCvt64Fto32:
		return rewriteValueRISCV64_OpCvt64Fto32_0(v)
	case OpCvt64Fto32F:
		return rewriteValueRISCV64_OpCvt64Fto32F_0(v)
	case OpCvt64Fto64:
		return rewriteValueRISCV64_OpCvt64Fto64_0(v)
	case OpCvt64to32F:
		return rewriteValueRISCV64_OpCvt64to32F_0(v)
	case OpCvt64to64F:
		return rewriteValueRISCV64_OpCvt64to64F_0(v)
	case OpDiv16:
		return rewriteValueRISCV64_OpDiv16_0(v)
	case OpDiv16u:
		return rewriteValueRISCV64_OpDiv16u_0(v)
	case OpDiv32:
		return rewriteValueRISCV64_OpDiv32_0(v)
	case OpDiv32F:
		return rewriteValueRISCV64_OpDiv32F_0(v)
	case OpDiv32u:
		return rewriteValueRISCV64_OpDiv32u_0(v)
	case OpDiv64:
		return rewriteValueRISCV64_OpDiv64_0(v)
	case OpDiv64F:
		return rewriteValueRISCV64_OpDiv64F_0(v)
	case OpDiv64u:
		return rewriteValueRISCV64_OpDiv64u_0(v)
	case OpDiv8:
		return rewriteValueRISCV64_OpDiv8_0(v)
	case OpDiv8u:
		return rewriteValueRISCV64_OpDiv8u_0(v)
	case OpEq16:
		return rewriteValueRISCV64_OpEq16_0(v)
	case OpEq32:
		return rewriteValueRISCV64_OpEq32_0(v)
	case OpEq32F:
		return rewriteValueRISCV64_OpEq32F_0(v)
	case OpEq64:
		return rewriteValueRISCV64_OpEq64_0(v)
	case OpEq64F:
		return rewriteValueRISCV64_OpEq64F_0(v)
	case OpEq8:
		return rewriteValueRISCV64_OpEq8_0(v)
	case OpEqB:
		return rewriteValueRISCV64_OpEqB_0(v)
	case OpEqPtr:
		return rewriteValueRISCV64_OpEqPtr_0(v)
	case OpGeq16:
		return rewriteValueRISCV64_OpGeq16_0(v)
	case OpGeq16U:
		return rewriteValueRISCV64_OpGeq16U_0(v)
	case OpGeq32:
		return rewriteValueRISCV64_OpGeq32_0(v)
	case OpGeq32F:
		return rewriteValueRISCV64_OpGeq32F_0(v)
	case OpGeq32U:
		return rewriteValueRISCV64_OpGeq32U_0(v)
	case OpGeq64:
		return rewriteValueRISCV64_OpGeq64_0(v)
	case OpGeq64F:
		return rewriteValueRISCV64_OpGeq64F_0(v)
	case OpGeq64U:
		return rewriteValueRISCV64_OpGeq64U_0(v)
	case OpGeq8:
		return rewriteValueRISCV64_OpGeq8_0(v)
	case OpGeq8U:
		return rewriteValueRISCV64_OpGeq8U_0(v)
	case OpGetCallerPC:
		return rewriteValueRISCV64_OpGetCallerPC_0(v)
	case OpGetCallerSP:
		return rewriteValueRISCV64_OpGetCallerSP_0(v)
	case OpGetClosurePtr:
		return rewriteValueRISCV64_OpGetClosurePtr_0(v)
	case OpGreater16:
		return rewriteValueRISCV64_OpGreater16_0(v)
	case OpGreater16U:
		return rewriteValueRISCV64_OpGreater16U_0(v)
	case OpGreater32:
		return rewriteValueRISCV64_OpGreater32_0(v)
	case OpGreater32F:
		return rewriteValueRISCV64_OpGreater32F_0(v)
	case OpGreater32U:
		return rewriteValueRISCV64_OpGreater32U_0(v)
	case OpGreater64:
		return rewriteValueRISCV64_OpGreater64_0(v)
	case OpGreater64F:
		return rewriteValueRISCV64_OpGreater64F_0(v)
	case OpGreater64U:
		return rewriteValueRISCV64_OpGreater64U_0(v)
	case OpGreater8:
		return rewriteValueRISCV64_OpGreater8_0(v)
	case OpGreater8U:
		return rewriteValueRISCV64_OpGreater8U_0(v)
	case OpHmul32:
		return rewriteValueRISCV64_OpHmul32_0(v)
	case OpHmul32u:
		return rewriteValueRISCV64_OpHmul32u_0(v)
	case OpHmul64:
		return rewriteValueRISCV64_OpHmul64_0(v)
	case OpHmul64u:
		return rewriteValueRISCV64_OpHmul64u_0(v)
	case OpInterCall:
		return rewriteValueRISCV64_OpInterCall_0(v)
	case OpIsInBounds:
		return rewriteValueRISCV64_OpIsInBounds_0(v)
	case OpIsNonNil:
		return rewriteValueRISCV64_OpIsNonNil_0(v)
	case OpIsSliceInBounds:
		return rewriteValueRISCV64_OpIsSliceInBounds_0(v)
	case OpLeq16:
		return rewriteValueRISCV64_OpLeq16_0(v)
	case OpLeq16U:
		return rewriteValueRISCV64_OpLeq16U_0(v)
	case OpLeq32:
		return rewriteValueRISCV64_OpLeq32_0(v)
	case OpLeq32F:
		return rewriteValueRISCV64_OpLeq32F_0(v)
	case OpLeq32U:
		return rewriteValueRISCV64_OpLeq32U_0(v)
	case OpLeq64:
		return rewriteValueRISCV64_OpLeq64_0(v)
	case OpLeq64F:
		return rewriteValueRISCV64_OpLeq64F_0(v)
	case OpLeq64U:
		return rewriteValueRISCV64_OpLeq64U_0(v)
	case OpLeq8:
		return rewriteValueRISCV64_OpLeq8_0(v)
	case OpLeq8U:
		return rewriteValueRISCV64_OpLeq8U_0(v)
	case OpLess16:
		return rewriteValueRISCV64_OpLess16_0(v)
	case OpLess16U:
		return rewriteValueRISCV64_OpLess16U_0(v)
	case OpLess32:
		return rewriteValueRISCV64_OpLess32_0(v)
	case OpLess32F:
		return rewriteValueRISCV64_OpLess32F_0(v)
	case OpLess32U:
		return rewriteValueRISCV64_OpLess32U_0(v)
	case OpLess64:
		return rewriteValueRISCV64_OpLess64_0(v)
	case OpLess64F:
		return rewriteValueRISCV64_OpLess64F_0(v)
	case OpLess64U:
		return rewriteValueRISCV64_OpLess64U_0(v)
	case OpLess8:
		return rewriteValueRISCV64_OpLess8_0(v)
	case OpLess8U:
		return rewriteValueRISCV64_OpLess8U_0(v)
	case OpLoad:
		return rewriteValueRISCV64_OpLoad_0(v)
	case OpLocalAddr:
		return rewriteValueRISCV64_OpLocalAddr_0(v)
	case OpLsh16x16:
		return rewriteValueRISCV64_OpLsh16x16_0(v)
	case OpLsh16x32:
		return rewriteValueRISCV64_OpLsh16x32_0(v)
	case OpLsh16x64:
		return rewriteValueRISCV64_OpLsh16x64_0(v)
	case OpLsh16x8:
		return rewriteValueRISCV64_OpLsh16x8_0(v)
	case OpLsh32x16:
		return rewriteValueRISCV64_OpLsh32x16_0(v)
	case OpLsh32x32:
		return rewriteValueRISCV64_OpLsh32x32_0(v)
	case OpLsh32x64:
		return rewriteValueRISCV64_OpLsh32x64_0(v)
	case OpLsh32x8:
		return rewriteValueRISCV64_OpLsh32x8_0(v)
	case OpLsh64x16:
		return rewriteValueRISCV64_OpLsh64x16_0(v)
	case OpLsh64x32:
		return rewriteValueRISCV64_OpLsh64x32_0(v)
	case OpLsh64x64:
		return rewriteValueRISCV64_OpLsh64x64_0(v)
	case OpLsh64x8:
		return rewriteValueRISCV64_OpLsh64x8_0(v)
	case OpLsh8x16:
		return rewriteValueRISCV64_OpLsh8x16_0(v)
	case OpLsh8x32:
		return rewriteValueRISCV64_OpLsh8x32_0(v)
	case OpLsh8x64:
		return rewriteValueRISCV64_OpLsh8x64_0(v)
	case OpLsh8x8:
		return rewriteValueRISCV64_OpLsh8x8_0(v)
	case OpMod16:
		return rewriteValueRISCV64_OpMod16_0(v)
	case OpMod16u:
		return rewriteValueRISCV64_OpMod16u_0(v)
	case OpMod32:
		return rewriteValueRISCV64_OpMod32_0(v)
	case OpMod32u:
		return rewriteValueRISCV64_OpMod32u_0(v)
	case OpMod64:
		return rewriteValueRISCV64_OpMod64_0(v)
	case OpMod64u:
		return rewriteValueRISCV64_OpMod64u_0(v)
	case OpMod8:
		return rewriteValueRISCV64_OpMod8_0(v)
	case OpMod8u:
		return rewriteValueRISCV64_OpMod8u_0(v)
	case OpMove:
		return rewriteValueRISCV64_OpMove_0(v)
	case OpMul16:
		return rewriteValueRISCV64_OpMul16_0(v)
	case OpMul32:
		return rewriteValueRISCV64_OpMul32_0(v)
	case OpMul32F:
		return rewriteValueRISCV64_OpMul32F_0(v)
	case OpMul64:
		return rewriteValueRISCV64_OpMul64_0(v)
	case OpMul64F:
		return rewriteValueRISCV64_OpMul64F_0(v)
	case OpMul8:
		return rewriteValueRISCV64_OpMul8_0(v)
	case OpNeg16:
		return rewriteValueRISCV64_OpNeg16_0(v)
	case OpNeg32:
		return rewriteValueRISCV64_OpNeg32_0(v)
	case OpNeg32F:
		return rewriteValueRISCV64_OpNeg32F_0(v)
	case OpNeg64:
		return rewriteValueRISCV64_OpNeg64_0(v)
	case OpNeg64F:
		return rewriteValueRISCV64_OpNeg64F_0(v)
	case OpNeg8:
		return rewriteValueRISCV64_OpNeg8_0(v)
	case OpNeq16:
		return rewriteValueRISCV64_OpNeq16_0(v)
	case OpNeq32:
		return rewriteValueRISCV64_OpNeq32_0(v)
	case OpNeq32F:
		return rewriteValueRISCV64_OpNeq32F_0(v)
	case OpNeq64:
		return rewriteValueRISCV64_OpNeq64_0(v)
	case OpNeq64F:
		return rewriteValueRISCV64_OpNeq64F_0(v)
	case OpNeq8:
		return rewriteValueRISCV64_OpNeq8_0(v)
	case OpNeqB:
		return rewriteValueRISCV64_OpNeqB_0(v)
	case OpNeqPtr:
		return rewriteValueRISCV64_OpNeqPtr_0(v)
	case OpNilCheck:
		return rewriteValueRISCV64_OpNilCheck_0(v)
	case OpNot:
		return rewriteValueRISCV64_OpNot_0(v)
	case OpOffPtr:
		return rewriteValueRISCV64_OpOffPtr_0(v)
	case OpOr16:
		return rewriteValueRISCV64_OpOr16_0(v)
	case OpOr32:
		return rewriteValueRISCV64_OpOr32_0(v)
	case OpOr64:
		return rewriteValueRISCV64_OpOr64_0(v)
	case OpOr8:
		return rewriteValueRISCV64_OpOr8_0(v)
	case OpOrB:
		return rewriteValueRISCV64_OpOrB_0(v)
	case OpPanicBounds:
		return rewriteValueRISCV64_OpPanicBounds_0(v)
	case OpRISCV64ADD:
		return rewriteValueRISCV64_OpRISCV64ADD_0(v)
	case OpRISCV64ADDI:
		return rewriteValueRISCV64_OpRISCV64ADDI_0(v)
	case OpRISCV64MOVBUload:
		return rewriteValueRISCV64_OpRISCV64MOVBUload_0(v)
	case OpRISCV64MOVBload:
		return rewriteValueRISCV64_OpRISCV64MOVBload_0(v)
	case OpRISCV64MOVBstore:
		return rewriteValueRISCV64_OpRISCV64MOVBstore_0(v)
	case OpRISCV64MOVDconst:
		return rewriteValueRISCV64_OpRISCV64MOVDconst_0(v)
	case OpRISCV64MOVDload:
		return rewriteValueRISCV64_OpRISCV64MOVDload_0(v)
	case OpRISCV64MOVDstore:
		return rewriteValueRISCV64_OpRISCV64MOVDstore_0(v)
	case OpRISCV64MOVHUload:
		return rewriteValueRISCV64_OpRISCV64MOVHUload_0(v)
	case OpRISCV64MOVHload:
		return rewriteValueRISCV64_OpRISCV64MOVHload_0(v)
	case OpRISCV64MOVHstore:
		return rewriteValueRISCV64_OpRISCV64MOVHstore_0(v)
	case OpRISCV64MOVWUload:
		return rewriteValueRISCV64_OpRISCV64MOVWUload_0(v)
	case OpRISCV64MOVWload:
		return rewriteValueRISCV64_OpRISCV64MOVWload_0(v)
	case OpRISCV64MOVWstore:
		return rewriteValueRISCV64_OpRISCV64MOVWstore_0(v)
	case OpRotateLeft16:
		return rewriteValueRISCV64_OpRotateLeft16_0(v)
	case OpRotateLeft32:
		return rewriteValueRISCV64_OpRotateLeft32_0(v)
	case OpRotateLeft64:
		return rewriteValueRISCV64_OpRotateLeft64_0(v)
	case OpRotateLeft8:
		return rewriteValueRISCV64_OpRotateLeft8_0(v)
	case OpRound32F:
		return rewriteValueRISCV64_OpRound32F_0(v)
	case OpRound64F:
		return rewriteValueRISCV64_OpRound64F_0(v)
	case OpRsh16Ux16:
		return rewriteValueRISCV64_OpRsh16Ux16_0(v)
	case OpRsh16Ux32:
		return rewriteValueRISCV64_OpRsh16Ux32_0(v)
	case OpRsh16Ux64:
		return rewriteValueRISCV64_OpRsh16Ux64_0(v)
	case OpRsh16Ux8:
		return rewriteValueRISCV64_OpRsh16Ux8_0(v)
	case OpRsh16x16:
		return rewriteValueRISCV64_OpRsh16x16_0(v)
	case OpRsh16x32:
		return rewriteValueRISCV64_OpRsh16x32_0(v)
	case OpRsh16x64:
		return rewriteValueRISCV64_OpRsh16x64_0(v)
	case OpRsh16x8:
		return rewriteValueRISCV64_OpRsh16x8_0(v)
	case OpRsh32Ux16:
		return rewriteValueRISCV64_OpRsh32Ux16_0(v)
	case OpRsh32Ux32:
		return rewriteValueRISCV64_OpRsh32Ux32_0(v)
	case OpRsh32Ux64:
		return rewriteValueRISCV64_OpRsh32Ux64_0(v)
	case OpRsh32Ux8:
		return rewriteValueRISCV64_OpRsh32Ux8_0(v)
	case OpRsh32x16:
		return rewriteValueRISCV64_OpRsh32x16_0(v)
	case OpRsh32x32:
		return rewriteValueRISCV64_OpRsh32x32_0(v)
	case OpRsh32x64:
		return rewriteValueRISCV64_OpRsh32x64_0(v)
	case OpRsh32x8:
		return rewriteValueRISCV64_OpRsh32x8_0(v)
	case OpRsh64Ux16:
		return rewriteValueRISCV64_OpRsh64Ux16_0(v)
	case OpRsh64Ux32:
		return rewriteValueRISCV64_OpRsh64Ux32_0(v)
	case OpRsh64Ux64:
		return rewriteValueRISCV64_OpRsh64Ux64_0(v)
	case OpRsh64Ux8:
		return rewriteValueRISCV64_OpRsh64Ux8_0(v)
	case OpRsh64x16:
		return rewriteValueRISCV64_OpRsh64x16_0(v)
	case OpRsh64x32:
		return rewriteValueRISCV64_OpRsh64x32_0(v)
	case OpRsh64x64:
		return rewriteValueRISCV64_OpRsh64x64_0(v)
	case OpRsh64x8:
		return rewriteValueRISCV64_OpRsh64x8_0(v)
	case OpRsh8Ux16:
		return rewriteValueRISCV64_OpRsh8Ux16_0(v)
	case OpRsh8Ux32:
		return rewriteValueRISCV64_OpRsh8Ux32_0(v)
	case OpRsh8Ux64:
		return rewriteValueRISCV64_OpRsh8Ux64_0(v)
	case OpRsh8Ux8:
		return rewriteValueRISCV64_OpRsh8Ux8_0(v)
	case OpRsh8x16:
		return rewriteValueRISCV64_OpRsh8x16_0(v)
	case OpRsh8x32:
		return rewriteValueRISCV64_OpRsh8x32_0(v)
	case OpRsh8x64:
		return rewriteValueRISCV64_OpRsh8x64_0(v)
	case OpRsh8x8:
		return rewriteValueRISCV64_OpRsh8x8_0(v)
	case OpSignExt16to32:
		return rewriteValueRISCV64_OpSignExt16to32_0(v)
	case OpSignExt16to64:
		return rewriteValueRISCV64_OpSignExt16to64_0(v)
	case OpSignExt32to64:
		return rewriteValueRISCV64_OpSignExt32to64_0(v)
	case OpSignExt8to16:
		return rewriteValueRISCV64_OpSignExt8to16_0(v)
	case OpSignExt8to32:
		return rewriteValueRISCV64_OpSignExt8to32_0(v)
	case OpSignExt8to64:
		return rewriteValueRISCV64_OpSignExt8to64_0(v)
	case OpSlicemask:
		return rewriteValueRISCV64_OpSlicemask_0(v)
	case OpSqrt:
		return rewriteValueRISCV64_OpSqrt_0(v)
	case OpStaticCall:
		return rewriteValueRISCV64_OpStaticCall_0(v)
	case OpStore:
		return rewriteValueRISCV64_OpStore_0(v)
	case OpSub16:
		return rewriteValueRISCV64_OpSub16_0(v)
	case OpSub32:
		return rewriteValueRISCV64_OpSub32_0(v)
	case OpSub32F:
		return rewriteValueRISCV64_OpSub32F_0(v)
	case OpSub64:
		return rewriteValueRISCV64_OpSub64_0(v)
	case OpSub64F:
		return rewriteValueRISCV64_OpSub64F_0(v)
	case OpSub8:
		return rewriteValueRISCV64_OpSub8_0(v)
	case OpSubPtr:
		return rewriteValueRISCV64_OpSubPtr_0(v)
	case OpTrunc16to8:
		return rewriteValueRISCV64_OpTrunc16to8_0(v)
	case OpTrunc32to16:
		return rewriteValueRISCV64_OpTrunc32to16_0(v)
	case OpTrunc32to8:
		return rewriteValueRISCV64_OpTrunc32to8_0(v)
	case OpTrunc64to16:
		return rewriteValueRISCV64_OpTrunc64to16_0(v)
	case OpTrunc64to32:
		return rewriteValueRISCV64_OpTrunc64to32_0(v)
	case OpTrunc64to8:
		return rewriteValueRISCV64_OpTrunc64to8_0(v)
	case OpWB:
		return rewriteValueRISCV64_OpWB_0(v)
	case OpXor16:
		return rewriteValueRISCV64_OpXor16_0(v)
	case OpXor32:
		return rewriteValueRISCV64_OpXor32_0(v)
	case OpXor64:
		return rewriteValueRISCV64_OpXor64_0(v)
	case OpXor8:
		return rewriteValueRISCV64_OpXor8_0(v)
	case OpZero:
		return rewriteValueRISCV64_OpZero_0(v)
	case OpZeroExt16to32:
		return rewriteValueRISCV64_OpZeroExt16to32_0(v)
	case OpZeroExt16to64:
		return rewriteValueRISCV64_OpZeroExt16to64_0(v)
	case OpZeroExt32to64:
		return rewriteValueRISCV64_OpZeroExt32to64_0(v)
	case OpZeroExt8to16:
		return rewriteValueRISCV64_OpZeroExt8to16_0(v)
	case OpZeroExt8to32:
		return rewriteValueRISCV64_OpZeroExt8to32_0(v)
	case OpZeroExt8to64:
		return rewriteValueRISCV64_OpZeroExt8to64_0(v)
	}
	return false
}
func rewriteValueRISCV64_OpAdd16_0(v *Value) bool {
	// match: (Add16 x y)
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAdd32_0(v *Value) bool {
	// match: (Add32 x y)
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAdd32F_0(v *Value) bool {
	// match: (Add32F x y)
	// result: (FADDS x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FADDS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAdd64_0(v *Value) bool {
	// match: (Add64 x y)
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAdd64F_0(v *Value) bool {
	// match: (Add64F x y)
	// result: (FADDD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FADDD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAdd8_0(v *Value) bool {
	// match: (Add8 x y)
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAddPtr_0(v *Value) bool {
	// match: (AddPtr x y)
	// result: (ADD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64ADD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAddr_0(v *Value) bool {
	// match: (Addr {sym} base)
	// result: (MOVaddr {sym} base)
	for {
		sym := v.Aux
		base := v.Args[0]
		v.reset(OpRISCV64MOVaddr)
		v.Aux = sym
		v.AddArg(base)
		return true
	}
}
func rewriteValueRISCV64_OpAnd16_0(v *Value) bool {
	// match: (And16 x y)
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAnd32_0(v *Value) bool {
	// match: (And32 x y)
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAnd64_0(v *Value) bool {
	// match: (And64 x y)
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAnd8_0(v *Value) bool {
	// match: (And8 x y)
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAndB_0(v *Value) bool {
	// match: (AndB x y)
	// result: (AND x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpAvg64u_0(v *Value) bool {
	b := v.Block
	// match: (Avg64u <t> x y)
	// result: (ADD (ADD <t> (SRLI <t> [1] x) (SRLI <t> [1] y)) (ANDI <t> [1] (AND <t> x y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64ADD)
		v0 := b.NewValue0(v.Pos, OpRISCV64ADD, t)
		v1 := b.NewValue0(v.Pos, OpRISCV64SRLI, t)
		v1.AuxInt = 1
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpRISCV64SRLI, t)
		v2.AuxInt = 1
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpRISCV64ANDI, t)
		v3.AuxInt = 1
		v4 := b.NewValue0(v.Pos, OpRISCV64AND, t)
		v4.AddArg(x)
		v4.AddArg(y)
		v3.AddArg(v4)
		v.AddArg(v3)
		return true
	}
}
func rewriteValueRISCV64_OpClosureCall_0(v *Value) bool {
	// match: (ClosureCall [argwid] entry closure mem)
	// result: (CALLclosure [argwid] entry closure mem)
	for {
		argwid := v.AuxInt
		mem := v.Args[2]
		entry := v.Args[0]
		closure := v.Args[1]
		v.reset(OpRISCV64CALLclosure)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(closure)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpCom16_0(v *Value) bool {
	// match: (Com16 x)
	// result: (XORI [int64(-1)] x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64XORI)
		v.AuxInt = int64(-1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCom32_0(v *Value) bool {
	// match: (Com32 x)
	// result: (XORI [int64(-1)] x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64XORI)
		v.AuxInt = int64(-1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCom64_0(v *Value) bool {
	// match: (Com64 x)
	// result: (XORI [int64(-1)] x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64XORI)
		v.AuxInt = int64(-1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCom8_0(v *Value) bool {
	// match: (Com8 x)
	// result: (XORI [int64(-1)] x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64XORI)
		v.AuxInt = int64(-1)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpConst16_0(v *Value) bool {
	// match: (Const16 [val])
	// result: (MOVHconst [val])
	for {
		val := v.AuxInt
		v.reset(OpRISCV64MOVHconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueRISCV64_OpConst32_0(v *Value) bool {
	// match: (Const32 [val])
	// result: (MOVWconst [val])
	for {
		val := v.AuxInt
		v.reset(OpRISCV64MOVWconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueRISCV64_OpConst32F_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Const32F [val])
	// result: (FMVSX (MOVWconst [int64(int32(math.Float32bits(float32(math.Float64frombits(uint64(val))))))]))
	for {
		val := v.AuxInt
		v.reset(OpRISCV64FMVSX)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVWconst, typ.UInt32)
		v0.AuxInt = int64(int32(math.Float32bits(float32(math.Float64frombits(uint64(val))))))
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpConst64_0(v *Value) bool {
	// match: (Const64 [val])
	// result: (MOVDconst [val])
	for {
		val := v.AuxInt
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueRISCV64_OpConst64F_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Const64F [val])
	// result: (FMVDX (MOVDconst [val]))
	for {
		val := v.AuxInt
		v.reset(OpRISCV64FMVDX)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = val
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpConst8_0(v *Value) bool {
	// match: (Const8 [val])
	// result: (MOVBconst [val])
	for {
		val := v.AuxInt
		v.reset(OpRISCV64MOVBconst)
		v.AuxInt = val
		return true
	}
}
func rewriteValueRISCV64_OpConstBool_0(v *Value) bool {
	// match: (ConstBool [b])
	// result: (MOVBconst [b])
	for {
		b := v.AuxInt
		v.reset(OpRISCV64MOVBconst)
		v.AuxInt = b
		return true
	}
}
func rewriteValueRISCV64_OpConstNil_0(v *Value) bool {
	// match: (ConstNil)
	// result: (MOVDconst [0])
	for {
		v.reset(OpRISCV64MOVDconst)
		v.AuxInt = 0
		return true
	}
}
func rewriteValueRISCV64_OpConvert_0(v *Value) bool {
	// match: (Convert x mem)
	// result: (MOVconvert x mem)
	for {
		mem := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64MOVconvert)
		v.AddArg(x)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpCvt32Fto32_0(v *Value) bool {
	// match: (Cvt32Fto32 x)
	// result: (FCVTWS x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FCVTWS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt32Fto64_0(v *Value) bool {
	// match: (Cvt32Fto64 x)
	// result: (FCVTLS x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FCVTLS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt32Fto64F_0(v *Value) bool {
	// match: (Cvt32Fto64F x)
	// result: (FCVTDS x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FCVTDS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt32to32F_0(v *Value) bool {
	// match: (Cvt32to32F x)
	// result: (FCVTSW x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FCVTSW)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt32to64F_0(v *Value) bool {
	// match: (Cvt32to64F x)
	// result: (FCVTDW x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FCVTDW)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt64Fto32_0(v *Value) bool {
	// match: (Cvt64Fto32 x)
	// result: (FCVTWD x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FCVTWD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt64Fto32F_0(v *Value) bool {
	// match: (Cvt64Fto32F x)
	// result: (FCVTSD x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FCVTSD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt64Fto64_0(v *Value) bool {
	// match: (Cvt64Fto64 x)
	// result: (FCVTLD x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FCVTLD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt64to32F_0(v *Value) bool {
	// match: (Cvt64to32F x)
	// result: (FCVTSL x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FCVTSL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpCvt64to64F_0(v *Value) bool {
	// match: (Cvt64to64F x)
	// result: (FCVTDL x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FCVTDL)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpDiv16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16 x y)
	// result: (DIVW (SignExt16to32 x) (SignExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64DIVW)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpDiv16u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div16u x y)
	// result: (DIVUW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64DIVUW)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpDiv32_0(v *Value) bool {
	// match: (Div32 x y)
	// result: (DIVW x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64DIVW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpDiv32F_0(v *Value) bool {
	// match: (Div32F x y)
	// result: (FDIVS x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FDIVS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpDiv32u_0(v *Value) bool {
	// match: (Div32u x y)
	// result: (DIVUW x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64DIVUW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpDiv64_0(v *Value) bool {
	// match: (Div64 x y)
	// result: (DIV x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64DIV)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpDiv64F_0(v *Value) bool {
	// match: (Div64F x y)
	// result: (FDIVD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FDIVD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpDiv64u_0(v *Value) bool {
	// match: (Div64u x y)
	// result: (DIVU x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64DIVU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpDiv8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8 x y)
	// result: (DIVW (SignExt8to32 x) (SignExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64DIVW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpDiv8u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Div8u x y)
	// result: (DIVUW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64DIVUW)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpEq16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq16 x y)
	// result: (SEQZ (ZeroExt16to64 (SUB <x.Type> x y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v1.AddArg(x)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpEq32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq32 x y)
	// result: (SEQZ (ZeroExt32to64 (SUB <x.Type> x y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v1.AddArg(x)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpEq32F_0(v *Value) bool {
	// match: (Eq32F x y)
	// result: (FEQS x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FEQS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpEq64_0(v *Value) bool {
	b := v.Block
	// match: (Eq64 x y)
	// result: (SEQZ (SUB <x.Type> x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpEq64F_0(v *Value) bool {
	// match: (Eq64F x y)
	// result: (FEQD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FEQD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpEq8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Eq8 x y)
	// result: (SEQZ (ZeroExt8to64 (SUB <x.Type> x y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v1.AddArg(x)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpEqB_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (EqB x y)
	// result: (XORI [1] (XOR <typ.Bool> x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64XORI)
		v.AuxInt = 1
		v0 := b.NewValue0(v.Pos, OpRISCV64XOR, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpEqPtr_0(v *Value) bool {
	b := v.Block
	// match: (EqPtr x y)
	// result: (SEQZ (SUB <x.Type> x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SEQZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq16 x y)
	// result: (Not (Less16 x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess16, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq16U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq16U x y)
	// result: (Not (Less16U x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess16U, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq32 x y)
	// result: (Not (Less32 x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess32, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq32F_0(v *Value) bool {
	// match: (Geq32F x y)
	// result: (FLES y x)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FLES)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGeq32U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq32U x y)
	// result: (Not (Less32U x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess32U, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq64 x y)
	// result: (Not (Less64 x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess64, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq64F_0(v *Value) bool {
	// match: (Geq64F x y)
	// result: (FLED y x)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FLED)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGeq64U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq64U x y)
	// result: (Not (Less64U x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess64U, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq8 x y)
	// result: (Not (Less8 x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess8, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGeq8U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Geq8U x y)
	// result: (Not (Less8U x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess8U, typ.Bool)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpGetCallerPC_0(v *Value) bool {
	// match: (GetCallerPC)
	// result: (LoweredGetCallerPC)
	for {
		v.reset(OpRISCV64LoweredGetCallerPC)
		return true
	}
}
func rewriteValueRISCV64_OpGetCallerSP_0(v *Value) bool {
	// match: (GetCallerSP)
	// result: (LoweredGetCallerSP)
	for {
		v.reset(OpRISCV64LoweredGetCallerSP)
		return true
	}
}
func rewriteValueRISCV64_OpGetClosurePtr_0(v *Value) bool {
	// match: (GetClosurePtr)
	// result: (LoweredGetClosurePtr)
	for {
		v.reset(OpRISCV64LoweredGetClosurePtr)
		return true
	}
}
func rewriteValueRISCV64_OpGreater16_0(v *Value) bool {
	// match: (Greater16 x y)
	// result: (Less16 y x)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpLess16)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater16U_0(v *Value) bool {
	// match: (Greater16U x y)
	// result: (Less16U y x)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpLess16U)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater32_0(v *Value) bool {
	// match: (Greater32 x y)
	// result: (Less32 y x)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpLess32)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater32F_0(v *Value) bool {
	// match: (Greater32F x y)
	// result: (FLTS y x)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FLTS)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater32U_0(v *Value) bool {
	// match: (Greater32U x y)
	// result: (Less32U y x)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpLess32U)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater64_0(v *Value) bool {
	// match: (Greater64 x y)
	// result: (Less64 y x)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpLess64)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater64F_0(v *Value) bool {
	// match: (Greater64F x y)
	// result: (FLTD y x)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FLTD)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater64U_0(v *Value) bool {
	// match: (Greater64U x y)
	// result: (Less64U y x)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpLess64U)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater8_0(v *Value) bool {
	// match: (Greater8 x y)
	// result: (Less8 y x)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpLess8)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpGreater8U_0(v *Value) bool {
	// match: (Greater8U x y)
	// result: (Less8U y x)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpLess8U)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpHmul32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32 x y)
	// result: (SRAI [32] (MUL (SignExt32to64 x) (SignExt32to64 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 32
		v0 := b.NewValue0(v.Pos, OpRISCV64MUL, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpHmul32u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Hmul32u x y)
	// result: (SRLI [32] (MUL (ZeroExt32to64 x) (ZeroExt32to64 y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRLI)
		v.AuxInt = 32
		v0 := b.NewValue0(v.Pos, OpRISCV64MUL, typ.Int64)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v2.AddArg(y)
		v0.AddArg(v2)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpHmul64_0(v *Value) bool {
	// match: (Hmul64 x y)
	// result: (MULH x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64MULH)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpHmul64u_0(v *Value) bool {
	// match: (Hmul64u x y)
	// result: (MULHU x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64MULHU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpInterCall_0(v *Value) bool {
	// match: (InterCall [argwid] entry mem)
	// result: (CALLinter [argwid] entry mem)
	for {
		argwid := v.AuxInt
		mem := v.Args[1]
		entry := v.Args[0]
		v.reset(OpRISCV64CALLinter)
		v.AuxInt = argwid
		v.AddArg(entry)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpIsInBounds_0(v *Value) bool {
	// match: (IsInBounds idx len)
	// result: (Less64U idx len)
	for {
		len := v.Args[1]
		idx := v.Args[0]
		v.reset(OpLess64U)
		v.AddArg(idx)
		v.AddArg(len)
		return true
	}
}
func rewriteValueRISCV64_OpIsNonNil_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (IsNonNil p)
	// result: (NeqPtr (MOVDconst) p)
	for {
		p := v.Args[0]
		v.reset(OpNeqPtr)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v.AddArg(v0)
		v.AddArg(p)
		return true
	}
}
func rewriteValueRISCV64_OpIsSliceInBounds_0(v *Value) bool {
	// match: (IsSliceInBounds idx len)
	// result: (Leq64U idx len)
	for {
		len := v.Args[1]
		idx := v.Args[0]
		v.reset(OpLeq64U)
		v.AddArg(idx)
		v.AddArg(len)
		return true
	}
}
func rewriteValueRISCV64_OpLeq16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16 x y)
	// result: (Not (Less16 y x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess16, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq16U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq16U x y)
	// result: (Not (Less16U y x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess16U, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32 x y)
	// result: (Not (Less32 y x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess32, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq32F_0(v *Value) bool {
	// match: (Leq32F x y)
	// result: (FLES x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FLES)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpLeq32U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq32U x y)
	// result: (Not (Less32U y x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess32U, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64 x y)
	// result: (Not (Less64 y x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess64, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq64F_0(v *Value) bool {
	// match: (Leq64F x y)
	// result: (FLED x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FLED)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpLeq64U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq64U x y)
	// result: (Not (Less64U y x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess64U, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8 x y)
	// result: (Not (Less8 y x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess8, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLeq8U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Leq8U x y)
	// result: (Not (Less8U y x))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpLess8U, typ.Bool)
		v0.AddArg(y)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpLess16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16 x y)
	// result: (SLT (SignExt16to64 x) (SignExt16to64 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SLT)
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLess16U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less16U x y)
	// result: (SLTU (ZeroExt16to64 x) (ZeroExt16to64 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SLTU)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLess32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32 x y)
	// result: (SLT (SignExt32to64 x) (SignExt32to64 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SLT)
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLess32F_0(v *Value) bool {
	// match: (Less32F x y)
	// result: (FLTS x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FLTS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpLess32U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less32U x y)
	// result: (SLTU (ZeroExt32to64 x) (ZeroExt32to64 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SLTU)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLess64_0(v *Value) bool {
	// match: (Less64 x y)
	// result: (SLT x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SLT)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpLess64F_0(v *Value) bool {
	// match: (Less64F x y)
	// result: (FLTD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FLTD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpLess64U_0(v *Value) bool {
	// match: (Less64U x y)
	// result: (SLTU x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SLTU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpLess8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8 x y)
	// result: (SLT (SignExt8to64 x) (SignExt8to64 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SLT)
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLess8U_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Less8U x y)
	// result: (SLTU (ZeroExt8to64 x) (ZeroExt8to64 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SLTU)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLoad_0(v *Value) bool {
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
		v.reset(OpRISCV64MOVBUload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ( is8BitInt(t) && isSigned(t))
	// result: (MOVBload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is8BitInt(t) && isSigned(t)) {
			break
		}
		v.reset(OpRISCV64MOVBload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: ( is8BitInt(t) && !isSigned(t))
	// result: (MOVBUload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is8BitInt(t) && !isSigned(t)) {
			break
		}
		v.reset(OpRISCV64MOVBUload)
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
		v.reset(OpRISCV64MOVHload)
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
		v.reset(OpRISCV64MOVHUload)
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
		v.reset(OpRISCV64MOVWload)
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
		v.reset(OpRISCV64MOVWUload)
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
		v.reset(OpRISCV64MOVDload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: is32BitFloat(t)
	// result: (FMOVWload ptr mem)
	for {
		t := v.Type
		mem := v.Args[1]
		ptr := v.Args[0]
		if !(is32BitFloat(t)) {
			break
		}
		v.reset(OpRISCV64FMOVWload)
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
		v.reset(OpRISCV64FMOVDload)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpLocalAddr_0(v *Value) bool {
	// match: (LocalAddr {sym} base _)
	// result: (MOVaddr {sym} base)
	for {
		sym := v.Aux
		_ = v.Args[1]
		base := v.Args[0]
		v.reset(OpRISCV64MOVaddr)
		v.Aux = sym
		v.AddArg(base)
		return true
	}
}
func rewriteValueRISCV64_OpLsh16x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x16 <t> x y)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg16, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh16x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x32 <t> x y)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg16, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh16x64_0(v *Value) bool {
	b := v.Block
	// match: (Lsh16x64 <t> x y)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg16, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh16x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh16x8 <t> x y)
	// result: (AND (SLL <t> x y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg16, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh32x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x16 <t> x y)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg32, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh32x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x32 <t> x y)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg32, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh32x64_0(v *Value) bool {
	b := v.Block
	// match: (Lsh32x64 <t> x y)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg32, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh32x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh32x8 <t> x y)
	// result: (AND (SLL <t> x y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg32, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh64x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x16 <t> x y)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh64x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x32 <t> x y)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh64x64_0(v *Value) bool {
	b := v.Block
	// match: (Lsh64x64 <t> x y)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh64x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh64x8 <t> x y)
	// result: (AND (SLL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh8x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x16 <t> x y)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg8, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh8x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x32 <t> x y)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg8, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh8x64_0(v *Value) bool {
	b := v.Block
	// match: (Lsh8x64 <t> x y)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg8, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpLsh8x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Lsh8x8 <t> x y)
	// result: (AND (SLL <t> x y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg8, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpMod16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16 x y)
	// result: (REMW (SignExt16to32 x) (SignExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64REMW)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpMod16u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod16u x y)
	// result: (REMUW (ZeroExt16to32 x) (ZeroExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64REMUW)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpMod32_0(v *Value) bool {
	// match: (Mod32 x y)
	// result: (REMW x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64REMW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMod32u_0(v *Value) bool {
	// match: (Mod32u x y)
	// result: (REMUW x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64REMUW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMod64_0(v *Value) bool {
	// match: (Mod64 x y)
	// result: (REM x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64REM)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMod64u_0(v *Value) bool {
	// match: (Mod64u x y)
	// result: (REMU x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64REMU)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMod8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8 x y)
	// result: (REMW (SignExt8to32 x) (SignExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64REMW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpMod8u_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mod8u x y)
	// result: (REMUW (ZeroExt8to32 x) (ZeroExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64REMUW)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpMove_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Move [0] _ _ mem)
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
	// result: (MOVBstore dst (MOVBload src mem) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpRISCV64MOVBstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVBload, typ.Int8)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [2] dst src mem)
	// result: (MOVHstore dst (MOVHload src mem) mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpRISCV64MOVHstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVHload, typ.Int16)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [4] dst src mem)
	// result: (MOVWstore dst (MOVWload src mem) mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpRISCV64MOVWstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVWload, typ.Int32)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [8] dst src mem)
	// result: (MOVDstore dst (MOVDload src mem) mem)
	for {
		if v.AuxInt != 8 {
			break
		}
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpRISCV64MOVDstore)
		v.AddArg(dst)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDload, typ.Int64)
		v0.AddArg(src)
		v0.AddArg(mem)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Move [s] {t} dst src mem)
	// result: (LoweredMove [t.(*types.Type).Alignment()] dst src (ADDI <src.Type> [s-moveSize(t.(*types.Type).Alignment(), config)] src) mem)
	for {
		s := v.AuxInt
		t := v.Aux
		mem := v.Args[2]
		dst := v.Args[0]
		src := v.Args[1]
		v.reset(OpRISCV64LoweredMove)
		v.AuxInt = t.(*types.Type).Alignment()
		v.AddArg(dst)
		v.AddArg(src)
		v0 := b.NewValue0(v.Pos, OpRISCV64ADDI, src.Type)
		v0.AuxInt = s - moveSize(t.(*types.Type).Alignment(), config)
		v0.AddArg(src)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpMul16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul16 x y)
	// result: (MULW (SignExt16to32 x) (SignExt16to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64MULW)
		v0 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpMul32_0(v *Value) bool {
	// match: (Mul32 x y)
	// result: (MULW x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64MULW)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMul32F_0(v *Value) bool {
	// match: (Mul32F x y)
	// result: (FMULS x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FMULS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMul64_0(v *Value) bool {
	// match: (Mul64 x y)
	// result: (MUL x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64MUL)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMul64F_0(v *Value) bool {
	// match: (Mul64F x y)
	// result: (FMULD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FMULD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpMul8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Mul8 x y)
	// result: (MULW (SignExt8to32 x) (SignExt8to32 y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64MULW)
		v0 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpNeg16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg16 x)
	// result: (SUB (MOVHconst) x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVHconst, typ.UInt16)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpNeg32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg32 x)
	// result: (SUB (MOVWconst) x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVWconst, typ.UInt32)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpNeg32F_0(v *Value) bool {
	// match: (Neg32F x)
	// result: (FNEGS x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FNEGS)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpNeg64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg64 x)
	// result: (SUB (MOVDconst) x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpNeg64F_0(v *Value) bool {
	// match: (Neg64F x)
	// result: (FNEGD x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FNEGD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpNeg8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neg8 x)
	// result: (SUB (MOVBconst) x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64SUB)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVBconst, typ.UInt8)
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpNeq16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq16 x y)
	// result: (SNEZ (ZeroExt16to64 (SUB <x.Type> x y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v1.AddArg(x)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpNeq32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq32 x y)
	// result: (SNEZ (ZeroExt32to64 (SUB <x.Type> x y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v1.AddArg(x)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpNeq32F_0(v *Value) bool {
	// match: (Neq32F x y)
	// result: (FNES x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FNES)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpNeq64_0(v *Value) bool {
	b := v.Block
	// match: (Neq64 x y)
	// result: (SNEZ (SUB <x.Type> x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpNeq64F_0(v *Value) bool {
	// match: (Neq64F x y)
	// result: (FNED x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FNED)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpNeq8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Neq8 x y)
	// result: (SNEZ (ZeroExt8to64 (SUB <x.Type> x y)))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v1.AddArg(x)
		v1.AddArg(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpNeqB_0(v *Value) bool {
	// match: (NeqB x y)
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpNeqPtr_0(v *Value) bool {
	b := v.Block
	// match: (NeqPtr x y)
	// result: (SNEZ (SUB <x.Type> x y))
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SNEZ)
		v0 := b.NewValue0(v.Pos, OpRISCV64SUB, x.Type)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpNilCheck_0(v *Value) bool {
	// match: (NilCheck ptr mem)
	// result: (LoweredNilCheck ptr mem)
	for {
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpRISCV64LoweredNilCheck)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpNot_0(v *Value) bool {
	// match: (Not x)
	// result: (XORI [1] x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64XORI)
		v.AuxInt = 1
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpOffPtr_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (OffPtr [off] ptr:(SP))
	// result: (MOVaddr [off] ptr)
	for {
		off := v.AuxInt
		ptr := v.Args[0]
		if ptr.Op != OpSP {
			break
		}
		v.reset(OpRISCV64MOVaddr)
		v.AuxInt = off
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// cond: is32Bit(off)
	// result: (ADDI [off] ptr)
	for {
		off := v.AuxInt
		ptr := v.Args[0]
		if !(is32Bit(off)) {
			break
		}
		v.reset(OpRISCV64ADDI)
		v.AuxInt = off
		v.AddArg(ptr)
		return true
	}
	// match: (OffPtr [off] ptr)
	// result: (ADD (MOVDconst [off]) ptr)
	for {
		off := v.AuxInt
		ptr := v.Args[0]
		v.reset(OpRISCV64ADD)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = off
		v.AddArg(v0)
		v.AddArg(ptr)
		return true
	}
}
func rewriteValueRISCV64_OpOr16_0(v *Value) bool {
	// match: (Or16 x y)
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpOr32_0(v *Value) bool {
	// match: (Or32 x y)
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpOr64_0(v *Value) bool {
	// match: (Or64 x y)
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpOr8_0(v *Value) bool {
	// match: (Or8 x y)
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpOrB_0(v *Value) bool {
	// match: (OrB x y)
	// result: (OR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64OR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpPanicBounds_0(v *Value) bool {
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
		v.reset(OpRISCV64LoweredPanicBoundsA)
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
		v.reset(OpRISCV64LoweredPanicBoundsB)
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
		v.reset(OpRISCV64LoweredPanicBoundsC)
		v.AuxInt = kind
		v.AddArg(x)
		v.AddArg(y)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64ADD_0(v *Value) bool {
	// match: (ADD (MOVDconst [off]) ptr)
	// cond: is32Bit(off)
	// result: (ADDI [off] ptr)
	for {
		ptr := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVDconst {
			break
		}
		off := v_0.AuxInt
		if !(is32Bit(off)) {
			break
		}
		v.reset(OpRISCV64ADDI)
		v.AuxInt = off
		v.AddArg(ptr)
		return true
	}
	// match: (ADD ptr (MOVDconst [off]))
	// cond: is32Bit(off)
	// result: (ADDI [off] ptr)
	for {
		_ = v.Args[1]
		ptr := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpRISCV64MOVDconst {
			break
		}
		off := v_1.AuxInt
		if !(is32Bit(off)) {
			break
		}
		v.reset(OpRISCV64ADDI)
		v.AuxInt = off
		v.AddArg(ptr)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64ADDI_0(v *Value) bool {
	// match: (ADDI [c] (MOVaddr [d] {s} x))
	// cond: is32Bit(c+d)
	// result: (MOVaddr [c+d] {s} x)
	for {
		c := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		d := v_0.AuxInt
		s := v_0.Aux
		x := v_0.Args[0]
		if !(is32Bit(c + d)) {
			break
		}
		v.reset(OpRISCV64MOVaddr)
		v.AuxInt = c + d
		v.Aux = s
		v.AddArg(x)
		return true
	}
	// match: (ADDI [0] x)
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
func rewriteValueRISCV64_OpRISCV64MOVBUload_0(v *Value) bool {
	// match: (MOVBUload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVBUload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVBUload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBUload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVBUload [off1+off2] {sym} base mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpRISCV64MOVBUload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVBload_0(v *Value) bool {
	// match: (MOVBload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVBload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVBload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVBload [off1+off2] {sym} base mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpRISCV64MOVBload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVBstore_0(v *Value) bool {
	// match: (MOVBstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVBstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVBstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVBstore [off1+off2] {sym} base val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpRISCV64MOVBstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVDconst_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (MOVDconst <t> [c])
	// cond: !is32Bit(c) && int32(c) < 0
	// result: (ADD (SLLI <t> [32] (MOVDconst [c>>32+1])) (MOVDconst [int64(int32(c))]))
	for {
		t := v.Type
		c := v.AuxInt
		if !(!is32Bit(c) && int32(c) < 0) {
			break
		}
		v.reset(OpRISCV64ADD)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 32
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v1.AuxInt = c>>32 + 1
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v2.AuxInt = int64(int32(c))
		v.AddArg(v2)
		return true
	}
	// match: (MOVDconst <t> [c])
	// cond: !is32Bit(c) && int32(c) >= 0
	// result: (ADD (SLLI <t> [32] (MOVDconst [c>>32+0])) (MOVDconst [int64(int32(c))]))
	for {
		t := v.Type
		c := v.AuxInt
		if !(!is32Bit(c) && int32(c) >= 0) {
			break
		}
		v.reset(OpRISCV64ADD)
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 32
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v1.AuxInt = c>>32 + 0
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v2.AuxInt = int64(int32(c))
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVDload_0(v *Value) bool {
	// match: (MOVDload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVDload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVDload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVDload [off1+off2] {sym} base mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpRISCV64MOVDload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVDstore_0(v *Value) bool {
	// match: (MOVDstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVDstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVDstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVDstore [off1+off2] {sym} base val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpRISCV64MOVDstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVHUload_0(v *Value) bool {
	// match: (MOVHUload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVHUload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVHUload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHUload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVHUload [off1+off2] {sym} base mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpRISCV64MOVHUload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVHload_0(v *Value) bool {
	// match: (MOVHload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVHload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVHload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVHload [off1+off2] {sym} base mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpRISCV64MOVHload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVHstore_0(v *Value) bool {
	// match: (MOVHstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVHstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVHstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVHstore [off1+off2] {sym} base val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpRISCV64MOVHstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVWUload_0(v *Value) bool {
	// match: (MOVWUload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWUload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVWUload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWUload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVWUload [off1+off2] {sym} base mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpRISCV64MOVWUload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVWload_0(v *Value) bool {
	// match: (MOVWload [off1] {sym1} (MOVaddr [off2] {sym2} base) mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWload [off1+off2] {mergeSym(sym1,sym2)} base mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVWload)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWload [off1] {sym} (ADDI [off2] base) mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVWload [off1+off2] {sym} base mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpRISCV64MOVWload)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(base)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRISCV64MOVWstore_0(v *Value) bool {
	// match: (MOVWstore [off1] {sym1} (MOVaddr [off2] {sym2} base) val mem)
	// cond: is32Bit(off1+off2) && canMergeSym(sym1, sym2)
	// result: (MOVWstore [off1+off2] {mergeSym(sym1,sym2)} base val mem)
	for {
		off1 := v.AuxInt
		sym1 := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64MOVaddr {
			break
		}
		off2 := v_0.AuxInt
		sym2 := v_0.Aux
		base := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1+off2) && canMergeSym(sym1, sym2)) {
			break
		}
		v.reset(OpRISCV64MOVWstore)
		v.AuxInt = off1 + off2
		v.Aux = mergeSym(sym1, sym2)
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (MOVWstore [off1] {sym} (ADDI [off2] base) val mem)
	// cond: is32Bit(off1+off2)
	// result: (MOVWstore [off1+off2] {sym} base val mem)
	for {
		off1 := v.AuxInt
		sym := v.Aux
		mem := v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpRISCV64ADDI {
			break
		}
		off2 := v_0.AuxInt
		base := v_0.Args[0]
		val := v.Args[1]
		if !(is32Bit(off1 + off2)) {
			break
		}
		v.reset(OpRISCV64MOVWstore)
		v.AuxInt = off1 + off2
		v.Aux = sym
		v.AddArg(base)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRotateLeft16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft16 <t> x (MOVHconst [c]))
	// result: (Or16 (Lsh16x64 <t> x (MOVHconst [c&15])) (Rsh16Ux64 <t> x (MOVHconst [-c&15])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpRISCV64MOVHconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpLsh16x64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVHconst, typ.UInt16)
		v1.AuxInt = c & 15
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpRsh16Ux64, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVHconst, typ.UInt16)
		v3.AuxInt = -c & 15
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRotateLeft32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft32 <t> x (MOVWconst [c]))
	// result: (Or32 (Lsh32x64 <t> x (MOVWconst [c&31])) (Rsh32Ux64 <t> x (MOVWconst [-c&31])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpRISCV64MOVWconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr32)
		v0 := b.NewValue0(v.Pos, OpLsh32x64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVWconst, typ.UInt32)
		v1.AuxInt = c & 31
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpRsh32Ux64, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVWconst, typ.UInt32)
		v3.AuxInt = -c & 31
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRotateLeft64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft64 <t> x (MOVDconst [c]))
	// result: (Or64 (Lsh64x64 <t> x (MOVDconst [c&63])) (Rsh64Ux64 <t> x (MOVDconst [-c&63])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpRISCV64MOVDconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr64)
		v0 := b.NewValue0(v.Pos, OpLsh64x64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v1.AuxInt = c & 63
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpRsh64Ux64, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v3.AuxInt = -c & 63
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRotateLeft8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (RotateLeft8 <t> x (MOVBconst [c]))
	// result: (Or8 (Lsh8x64 <t> x (MOVBconst [c&7])) (Rsh8Ux64 <t> x (MOVBconst [-c&7])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpRISCV64MOVBconst {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpLsh8x64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVBconst, typ.UInt8)
		v1.AuxInt = c & 7
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpRsh8Ux64, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVBconst, typ.UInt8)
		v3.AuxInt = -c & 7
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpRound32F_0(v *Value) bool {
	// match: (Round32F x)
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpRound64F_0(v *Value) bool {
	// match: (Round64F x)
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpRsh16Ux16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux16 <t> x y)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpNeg16, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueRISCV64_OpRsh16Ux32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux32 <t> x y)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpNeg16, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueRISCV64_OpRsh16Ux64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux64 <t> x y)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpNeg16, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = 64
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueRISCV64_OpRsh16Ux8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16Ux8 <t> x y)
	// result: (AND (SRL <t> (ZeroExt16to64 x) y) (Neg16 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpNeg16, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueRISCV64_OpRsh16x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x16 <t> x y)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = -1
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh16x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x32 <t> x y)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = -1
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh16x64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x64 <t> x y)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = -1
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = 64
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh16x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh16x8 <t> x y)
	// result: (SRA <t> (SignExt16to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt16to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = -1
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh32Ux16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux16 <t> x y)
	// result: (AND (SRL <t> (ZeroExt32to64 x) y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpNeg32, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueRISCV64_OpRsh32Ux32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux32 <t> x y)
	// result: (AND (SRL <t> (ZeroExt32to64 x) y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpNeg32, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueRISCV64_OpRsh32Ux64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux64 <t> x y)
	// result: (AND (SRL <t> (ZeroExt32to64 x) y) (Neg32 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpNeg32, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = 64
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueRISCV64_OpRsh32Ux8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32Ux8 <t> x y)
	// result: (AND (SRL <t> (ZeroExt32to64 x) y) (Neg32 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpNeg32, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueRISCV64_OpRsh32x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x16 <t> x y)
	// result: (SRA <t> (SignExt32to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = -1
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh32x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x32 <t> x y)
	// result: (SRA <t> (SignExt32to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = -1
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh32x64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x64 <t> x y)
	// result: (SRA <t> (SignExt32to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = -1
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = 64
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh32x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh32x8 <t> x y)
	// result: (SRA <t> (SignExt32to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = -1
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh64Ux16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux16 <t> x y)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh64Ux32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux32 <t> x y)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh64Ux64_0(v *Value) bool {
	b := v.Block
	// match: (Rsh64Ux64 <t> x y)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh64Ux8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64Ux8 <t> x y)
	// result: (AND (SRL <t> x y) (Neg64 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v0.AddArg(x)
		v0.AddArg(y)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh64x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x16 <t> x y)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v1.AuxInt = -1
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpRsh64x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x32 <t> x y)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v1.AuxInt = -1
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpRsh64x64_0(v *Value) bool {
	b := v.Block
	// match: (Rsh64x64 <t> x y)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v1.AuxInt = -1
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v2.AuxInt = 64
		v2.AddArg(y)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpRsh64x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh64x8 <t> x y)
	// result: (SRA <t> x (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v0.AddArg(y)
		v1 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v1.AuxInt = -1
		v2 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v2.AuxInt = 64
		v3 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpRsh8Ux16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux16 <t> x y)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt16to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpNeg8, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueRISCV64_OpRsh8Ux32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux32 <t> x y)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt32to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpNeg8, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueRISCV64_OpRsh8Ux64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux64 <t> x y)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] y)))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpNeg8, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = 64
		v3.AddArg(y)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueRISCV64_OpRsh8Ux8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8Ux8 <t> x y)
	// result: (AND (SRL <t> (ZeroExt8to64 x) y) (Neg8 <t> (SLTIU <t> [64] (ZeroExt8to64 y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64AND)
		v0 := b.NewValue0(v.Pos, OpRISCV64SRL, t)
		v1 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v1.AddArg(x)
		v0.AddArg(v1)
		v0.AddArg(y)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpNeg8, t)
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, t)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v.AddArg(v2)
		return true
	}
}
func rewriteValueRISCV64_OpRsh8x16_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x16 <t> x y)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt16to64 y)))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = -1
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh8x32_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x32 <t> x y)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt32to64 y)))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = -1
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh8x64_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x64 <t> x y)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] y))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = -1
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = 64
		v3.AddArg(y)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpRsh8x8_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Rsh8x8 <t> x y)
	// result: (SRA <t> (SignExt8to64 x) (OR <y.Type> y (ADDI <y.Type> [-1] (SLTIU <y.Type> [64] (ZeroExt8to64 y)))))
	for {
		t := v.Type
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SRA)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpSignExt8to64, typ.Int64)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64OR, y.Type)
		v1.AddArg(y)
		v2 := b.NewValue0(v.Pos, OpRISCV64ADDI, y.Type)
		v2.AuxInt = -1
		v3 := b.NewValue0(v.Pos, OpRISCV64SLTIU, y.Type)
		v3.AuxInt = 64
		v4 := b.NewValue0(v.Pos, OpZeroExt8to64, typ.UInt64)
		v4.AddArg(y)
		v3.AddArg(v4)
		v2.AddArg(v3)
		v1.AddArg(v2)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpSignExt16to32_0(v *Value) bool {
	b := v.Block
	// match: (SignExt16to32 <t> x)
	// result: (SRAI [48] (SLLI <t> [48] x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 48
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 48
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpSignExt16to64_0(v *Value) bool {
	b := v.Block
	// match: (SignExt16to64 <t> x)
	// result: (SRAI [48] (SLLI <t> [48] x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 48
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 48
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpSignExt32to64_0(v *Value) bool {
	b := v.Block
	// match: (SignExt32to64 <t> x)
	// result: (SRAI [32] (SLLI <t> [32] x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 32
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 32
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpSignExt8to16_0(v *Value) bool {
	b := v.Block
	// match: (SignExt8to16 <t> x)
	// result: (SRAI [56] (SLLI <t> [56] x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 56
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 56
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpSignExt8to32_0(v *Value) bool {
	b := v.Block
	// match: (SignExt8to32 <t> x)
	// result: (SRAI [56] (SLLI <t> [56] x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 56
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 56
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpSignExt8to64_0(v *Value) bool {
	b := v.Block
	// match: (SignExt8to64 <t> x)
	// result: (SRAI [56] (SLLI <t> [56] x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64SRAI)
		v.AuxInt = 56
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 56
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpSlicemask_0(v *Value) bool {
	b := v.Block
	typ := &b.Func.Config.Types
	// match: (Slicemask <t> x)
	// result: (XOR (MOVDconst [-1]) (SRA <t> (SUB <t> x (MOVDconst [1])) (MOVDconst [63])))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64XOR)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v0.AuxInt = -1
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpRISCV64SRA, t)
		v2 := b.NewValue0(v.Pos, OpRISCV64SUB, t)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v3.AuxInt = 1
		v2.AddArg(v3)
		v1.AddArg(v2)
		v4 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v4.AuxInt = 63
		v1.AddArg(v4)
		v.AddArg(v1)
		return true
	}
}
func rewriteValueRISCV64_OpSqrt_0(v *Value) bool {
	// match: (Sqrt x)
	// result: (FSQRTD x)
	for {
		x := v.Args[0]
		v.reset(OpRISCV64FSQRTD)
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpStaticCall_0(v *Value) bool {
	// match: (StaticCall [argwid] {target} mem)
	// result: (CALLstatic [argwid] {target} mem)
	for {
		argwid := v.AuxInt
		target := v.Aux
		mem := v.Args[0]
		v.reset(OpRISCV64CALLstatic)
		v.AuxInt = argwid
		v.Aux = target
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpStore_0(v *Value) bool {
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
		v.reset(OpRISCV64MOVBstore)
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
		v.reset(OpRISCV64MOVHstore)
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
		v.reset(OpRISCV64MOVWstore)
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
		v.reset(OpRISCV64MOVDstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	// match: (Store {t} ptr val mem)
	// cond: t.(*types.Type).Size() == 4 && is32BitFloat(val.Type)
	// result: (FMOVWstore ptr val mem)
	for {
		t := v.Aux
		mem := v.Args[2]
		ptr := v.Args[0]
		val := v.Args[1]
		if !(t.(*types.Type).Size() == 4 && is32BitFloat(val.Type)) {
			break
		}
		v.reset(OpRISCV64FMOVWstore)
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
		v.reset(OpRISCV64FMOVDstore)
		v.AddArg(ptr)
		v.AddArg(val)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValueRISCV64_OpSub16_0(v *Value) bool {
	// match: (Sub16 x y)
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpSub32_0(v *Value) bool {
	// match: (Sub32 x y)
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpSub32F_0(v *Value) bool {
	// match: (Sub32F x y)
	// result: (FSUBS x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FSUBS)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpSub64_0(v *Value) bool {
	// match: (Sub64 x y)
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpSub64F_0(v *Value) bool {
	// match: (Sub64F x y)
	// result: (FSUBD x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64FSUBD)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpSub8_0(v *Value) bool {
	// match: (Sub8 x y)
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpSubPtr_0(v *Value) bool {
	// match: (SubPtr x y)
	// result: (SUB x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64SUB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpTrunc16to8_0(v *Value) bool {
	// match: (Trunc16to8 x)
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpTrunc32to16_0(v *Value) bool {
	// match: (Trunc32to16 x)
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpTrunc32to8_0(v *Value) bool {
	// match: (Trunc32to8 x)
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpTrunc64to16_0(v *Value) bool {
	// match: (Trunc64to16 x)
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpTrunc64to32_0(v *Value) bool {
	// match: (Trunc64to32 x)
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpTrunc64to8_0(v *Value) bool {
	// match: (Trunc64to8 x)
	// result: x
	for {
		x := v.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
}
func rewriteValueRISCV64_OpWB_0(v *Value) bool {
	// match: (WB {fn} destptr srcptr mem)
	// result: (LoweredWB {fn} destptr srcptr mem)
	for {
		fn := v.Aux
		mem := v.Args[2]
		destptr := v.Args[0]
		srcptr := v.Args[1]
		v.reset(OpRISCV64LoweredWB)
		v.Aux = fn
		v.AddArg(destptr)
		v.AddArg(srcptr)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpXor16_0(v *Value) bool {
	// match: (Xor16 x y)
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpXor32_0(v *Value) bool {
	// match: (Xor32 x y)
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpXor64_0(v *Value) bool {
	// match: (Xor64 x y)
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpXor8_0(v *Value) bool {
	// match: (Xor8 x y)
	// result: (XOR x y)
	for {
		y := v.Args[1]
		x := v.Args[0]
		v.reset(OpRISCV64XOR)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
}
func rewriteValueRISCV64_OpZero_0(v *Value) bool {
	b := v.Block
	config := b.Func.Config
	typ := &b.Func.Config.Types
	// match: (Zero [0] _ mem)
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
	// result: (MOVBstore ptr (MOVBconst) mem)
	for {
		if v.AuxInt != 1 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpRISCV64MOVBstore)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVBconst, typ.UInt8)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [2] ptr mem)
	// result: (MOVHstore ptr (MOVHconst) mem)
	for {
		if v.AuxInt != 2 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpRISCV64MOVHstore)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVHconst, typ.UInt16)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [4] ptr mem)
	// result: (MOVWstore ptr (MOVWconst) mem)
	for {
		if v.AuxInt != 4 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpRISCV64MOVWstore)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVWconst, typ.UInt32)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [8] ptr mem)
	// result: (MOVDstore ptr (MOVDconst) mem)
	for {
		if v.AuxInt != 8 {
			break
		}
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpRISCV64MOVDstore)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
	// match: (Zero [s] {t} ptr mem)
	// result: (LoweredZero [t.(*types.Type).Alignment()] ptr (ADD <ptr.Type> ptr (MOVDconst [s-moveSize(t.(*types.Type).Alignment(), config)])) mem)
	for {
		s := v.AuxInt
		t := v.Aux
		mem := v.Args[1]
		ptr := v.Args[0]
		v.reset(OpRISCV64LoweredZero)
		v.AuxInt = t.(*types.Type).Alignment()
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpRISCV64ADD, ptr.Type)
		v0.AddArg(ptr)
		v1 := b.NewValue0(v.Pos, OpRISCV64MOVDconst, typ.UInt64)
		v1.AuxInt = s - moveSize(t.(*types.Type).Alignment(), config)
		v0.AddArg(v1)
		v.AddArg(v0)
		v.AddArg(mem)
		return true
	}
}
func rewriteValueRISCV64_OpZeroExt16to32_0(v *Value) bool {
	b := v.Block
	// match: (ZeroExt16to32 <t> x)
	// result: (SRLI [48] (SLLI <t> [48] x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64SRLI)
		v.AuxInt = 48
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 48
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpZeroExt16to64_0(v *Value) bool {
	b := v.Block
	// match: (ZeroExt16to64 <t> x)
	// result: (SRLI [48] (SLLI <t> [48] x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64SRLI)
		v.AuxInt = 48
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 48
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpZeroExt32to64_0(v *Value) bool {
	b := v.Block
	// match: (ZeroExt32to64 <t> x)
	// result: (SRLI [32] (SLLI <t> [32] x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64SRLI)
		v.AuxInt = 32
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 32
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpZeroExt8to16_0(v *Value) bool {
	b := v.Block
	// match: (ZeroExt8to16 <t> x)
	// result: (SRLI [56] (SLLI <t> [56] x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64SRLI)
		v.AuxInt = 56
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 56
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpZeroExt8to32_0(v *Value) bool {
	b := v.Block
	// match: (ZeroExt8to32 <t> x)
	// result: (SRLI [56] (SLLI <t> [56] x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64SRLI)
		v.AuxInt = 56
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 56
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteValueRISCV64_OpZeroExt8to64_0(v *Value) bool {
	b := v.Block
	// match: (ZeroExt8to64 <t> x)
	// result: (SRLI [56] (SLLI <t> [56] x))
	for {
		t := v.Type
		x := v.Args[0]
		v.reset(OpRISCV64SRLI)
		v.AuxInt = 56
		v0 := b.NewValue0(v.Pos, OpRISCV64SLLI, t)
		v0.AuxInt = 56
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
}
func rewriteBlockRISCV64(b *Block) bool {
	switch b.Kind {
	case BlockIf:
		// match: (If cond yes no)
		// result: (BNE cond yes no)
		for {
			cond := b.Controls[0]
			b.Reset(BlockRISCV64BNE)
			b.AddControl(cond)
			return true
		}
	}
	return false
}
