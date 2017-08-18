// Code generated from gen/generic.rules; DO NOT EDIT.
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

func rewriteValuegeneric(v *Value) bool {
	switch v.Op {
	case OpAdd16:
		return rewriteValuegeneric_OpAdd16_0(v) || rewriteValuegeneric_OpAdd16_10(v) || rewriteValuegeneric_OpAdd16_20(v)
	case OpAdd32:
		return rewriteValuegeneric_OpAdd32_0(v) || rewriteValuegeneric_OpAdd32_10(v) || rewriteValuegeneric_OpAdd32_20(v)
	case OpAdd32F:
		return rewriteValuegeneric_OpAdd32F_0(v)
	case OpAdd64:
		return rewriteValuegeneric_OpAdd64_0(v) || rewriteValuegeneric_OpAdd64_10(v) || rewriteValuegeneric_OpAdd64_20(v)
	case OpAdd64F:
		return rewriteValuegeneric_OpAdd64F_0(v)
	case OpAdd8:
		return rewriteValuegeneric_OpAdd8_0(v) || rewriteValuegeneric_OpAdd8_10(v) || rewriteValuegeneric_OpAdd8_20(v)
	case OpAddPtr:
		return rewriteValuegeneric_OpAddPtr_0(v)
	case OpAnd16:
		return rewriteValuegeneric_OpAnd16_0(v) || rewriteValuegeneric_OpAnd16_10(v)
	case OpAnd32:
		return rewriteValuegeneric_OpAnd32_0(v) || rewriteValuegeneric_OpAnd32_10(v)
	case OpAnd64:
		return rewriteValuegeneric_OpAnd64_0(v) || rewriteValuegeneric_OpAnd64_10(v) || rewriteValuegeneric_OpAnd64_20(v)
	case OpAnd8:
		return rewriteValuegeneric_OpAnd8_0(v) || rewriteValuegeneric_OpAnd8_10(v)
	case OpArg:
		return rewriteValuegeneric_OpArg_0(v) || rewriteValuegeneric_OpArg_10(v)
	case OpArraySelect:
		return rewriteValuegeneric_OpArraySelect_0(v)
	case OpCom16:
		return rewriteValuegeneric_OpCom16_0(v)
	case OpCom32:
		return rewriteValuegeneric_OpCom32_0(v)
	case OpCom64:
		return rewriteValuegeneric_OpCom64_0(v)
	case OpCom8:
		return rewriteValuegeneric_OpCom8_0(v)
	case OpConstInterface:
		return rewriteValuegeneric_OpConstInterface_0(v)
	case OpConstSlice:
		return rewriteValuegeneric_OpConstSlice_0(v)
	case OpConstString:
		return rewriteValuegeneric_OpConstString_0(v)
	case OpConvert:
		return rewriteValuegeneric_OpConvert_0(v)
	case OpCvt32Fto64F:
		return rewriteValuegeneric_OpCvt32Fto64F_0(v)
	case OpCvt64Fto32F:
		return rewriteValuegeneric_OpCvt64Fto32F_0(v)
	case OpDiv16:
		return rewriteValuegeneric_OpDiv16_0(v)
	case OpDiv16u:
		return rewriteValuegeneric_OpDiv16u_0(v)
	case OpDiv32:
		return rewriteValuegeneric_OpDiv32_0(v)
	case OpDiv32F:
		return rewriteValuegeneric_OpDiv32F_0(v)
	case OpDiv32u:
		return rewriteValuegeneric_OpDiv32u_0(v)
	case OpDiv64:
		return rewriteValuegeneric_OpDiv64_0(v)
	case OpDiv64F:
		return rewriteValuegeneric_OpDiv64F_0(v)
	case OpDiv64u:
		return rewriteValuegeneric_OpDiv64u_0(v)
	case OpDiv8:
		return rewriteValuegeneric_OpDiv8_0(v)
	case OpDiv8u:
		return rewriteValuegeneric_OpDiv8u_0(v)
	case OpEq16:
		return rewriteValuegeneric_OpEq16_0(v)
	case OpEq32:
		return rewriteValuegeneric_OpEq32_0(v)
	case OpEq64:
		return rewriteValuegeneric_OpEq64_0(v)
	case OpEq8:
		return rewriteValuegeneric_OpEq8_0(v)
	case OpEqB:
		return rewriteValuegeneric_OpEqB_0(v)
	case OpEqInter:
		return rewriteValuegeneric_OpEqInter_0(v)
	case OpEqPtr:
		return rewriteValuegeneric_OpEqPtr_0(v)
	case OpEqSlice:
		return rewriteValuegeneric_OpEqSlice_0(v)
	case OpGeq16:
		return rewriteValuegeneric_OpGeq16_0(v)
	case OpGeq16U:
		return rewriteValuegeneric_OpGeq16U_0(v)
	case OpGeq32:
		return rewriteValuegeneric_OpGeq32_0(v)
	case OpGeq32U:
		return rewriteValuegeneric_OpGeq32U_0(v)
	case OpGeq64:
		return rewriteValuegeneric_OpGeq64_0(v)
	case OpGeq64U:
		return rewriteValuegeneric_OpGeq64U_0(v)
	case OpGeq8:
		return rewriteValuegeneric_OpGeq8_0(v)
	case OpGeq8U:
		return rewriteValuegeneric_OpGeq8U_0(v)
	case OpGreater16:
		return rewriteValuegeneric_OpGreater16_0(v)
	case OpGreater16U:
		return rewriteValuegeneric_OpGreater16U_0(v)
	case OpGreater32:
		return rewriteValuegeneric_OpGreater32_0(v)
	case OpGreater32U:
		return rewriteValuegeneric_OpGreater32U_0(v)
	case OpGreater64:
		return rewriteValuegeneric_OpGreater64_0(v)
	case OpGreater64U:
		return rewriteValuegeneric_OpGreater64U_0(v)
	case OpGreater8:
		return rewriteValuegeneric_OpGreater8_0(v)
	case OpGreater8U:
		return rewriteValuegeneric_OpGreater8U_0(v)
	case OpIMake:
		return rewriteValuegeneric_OpIMake_0(v)
	case OpInterCall:
		return rewriteValuegeneric_OpInterCall_0(v)
	case OpIsInBounds:
		return rewriteValuegeneric_OpIsInBounds_0(v) || rewriteValuegeneric_OpIsInBounds_10(v) || rewriteValuegeneric_OpIsInBounds_20(v)
	case OpIsNonNil:
		return rewriteValuegeneric_OpIsNonNil_0(v)
	case OpIsSliceInBounds:
		return rewriteValuegeneric_OpIsSliceInBounds_0(v)
	case OpLeq16:
		return rewriteValuegeneric_OpLeq16_0(v)
	case OpLeq16U:
		return rewriteValuegeneric_OpLeq16U_0(v)
	case OpLeq32:
		return rewriteValuegeneric_OpLeq32_0(v)
	case OpLeq32U:
		return rewriteValuegeneric_OpLeq32U_0(v)
	case OpLeq64:
		return rewriteValuegeneric_OpLeq64_0(v)
	case OpLeq64U:
		return rewriteValuegeneric_OpLeq64U_0(v)
	case OpLeq8:
		return rewriteValuegeneric_OpLeq8_0(v)
	case OpLeq8U:
		return rewriteValuegeneric_OpLeq8U_0(v)
	case OpLess16:
		return rewriteValuegeneric_OpLess16_0(v)
	case OpLess16U:
		return rewriteValuegeneric_OpLess16U_0(v)
	case OpLess32:
		return rewriteValuegeneric_OpLess32_0(v)
	case OpLess32U:
		return rewriteValuegeneric_OpLess32U_0(v)
	case OpLess64:
		return rewriteValuegeneric_OpLess64_0(v)
	case OpLess64U:
		return rewriteValuegeneric_OpLess64U_0(v)
	case OpLess8:
		return rewriteValuegeneric_OpLess8_0(v)
	case OpLess8U:
		return rewriteValuegeneric_OpLess8U_0(v)
	case OpLoad:
		return rewriteValuegeneric_OpLoad_0(v)
	case OpLsh16x16:
		return rewriteValuegeneric_OpLsh16x16_0(v)
	case OpLsh16x32:
		return rewriteValuegeneric_OpLsh16x32_0(v)
	case OpLsh16x64:
		return rewriteValuegeneric_OpLsh16x64_0(v)
	case OpLsh16x8:
		return rewriteValuegeneric_OpLsh16x8_0(v)
	case OpLsh32x16:
		return rewriteValuegeneric_OpLsh32x16_0(v)
	case OpLsh32x32:
		return rewriteValuegeneric_OpLsh32x32_0(v)
	case OpLsh32x64:
		return rewriteValuegeneric_OpLsh32x64_0(v)
	case OpLsh32x8:
		return rewriteValuegeneric_OpLsh32x8_0(v)
	case OpLsh64x16:
		return rewriteValuegeneric_OpLsh64x16_0(v)
	case OpLsh64x32:
		return rewriteValuegeneric_OpLsh64x32_0(v)
	case OpLsh64x64:
		return rewriteValuegeneric_OpLsh64x64_0(v)
	case OpLsh64x8:
		return rewriteValuegeneric_OpLsh64x8_0(v)
	case OpLsh8x16:
		return rewriteValuegeneric_OpLsh8x16_0(v)
	case OpLsh8x32:
		return rewriteValuegeneric_OpLsh8x32_0(v)
	case OpLsh8x64:
		return rewriteValuegeneric_OpLsh8x64_0(v)
	case OpLsh8x8:
		return rewriteValuegeneric_OpLsh8x8_0(v)
	case OpMod16:
		return rewriteValuegeneric_OpMod16_0(v)
	case OpMod16u:
		return rewriteValuegeneric_OpMod16u_0(v)
	case OpMod32:
		return rewriteValuegeneric_OpMod32_0(v)
	case OpMod32u:
		return rewriteValuegeneric_OpMod32u_0(v)
	case OpMod64:
		return rewriteValuegeneric_OpMod64_0(v)
	case OpMod64u:
		return rewriteValuegeneric_OpMod64u_0(v)
	case OpMod8:
		return rewriteValuegeneric_OpMod8_0(v)
	case OpMod8u:
		return rewriteValuegeneric_OpMod8u_0(v)
	case OpMul16:
		return rewriteValuegeneric_OpMul16_0(v) || rewriteValuegeneric_OpMul16_10(v)
	case OpMul32:
		return rewriteValuegeneric_OpMul32_0(v) || rewriteValuegeneric_OpMul32_10(v)
	case OpMul32F:
		return rewriteValuegeneric_OpMul32F_0(v)
	case OpMul64:
		return rewriteValuegeneric_OpMul64_0(v) || rewriteValuegeneric_OpMul64_10(v)
	case OpMul64F:
		return rewriteValuegeneric_OpMul64F_0(v)
	case OpMul8:
		return rewriteValuegeneric_OpMul8_0(v) || rewriteValuegeneric_OpMul8_10(v)
	case OpNeg16:
		return rewriteValuegeneric_OpNeg16_0(v)
	case OpNeg32:
		return rewriteValuegeneric_OpNeg32_0(v)
	case OpNeg32F:
		return rewriteValuegeneric_OpNeg32F_0(v)
	case OpNeg64:
		return rewriteValuegeneric_OpNeg64_0(v)
	case OpNeg64F:
		return rewriteValuegeneric_OpNeg64F_0(v)
	case OpNeg8:
		return rewriteValuegeneric_OpNeg8_0(v)
	case OpNeq16:
		return rewriteValuegeneric_OpNeq16_0(v)
	case OpNeq32:
		return rewriteValuegeneric_OpNeq32_0(v)
	case OpNeq64:
		return rewriteValuegeneric_OpNeq64_0(v)
	case OpNeq8:
		return rewriteValuegeneric_OpNeq8_0(v)
	case OpNeqB:
		return rewriteValuegeneric_OpNeqB_0(v)
	case OpNeqInter:
		return rewriteValuegeneric_OpNeqInter_0(v)
	case OpNeqPtr:
		return rewriteValuegeneric_OpNeqPtr_0(v)
	case OpNeqSlice:
		return rewriteValuegeneric_OpNeqSlice_0(v)
	case OpNilCheck:
		return rewriteValuegeneric_OpNilCheck_0(v)
	case OpNot:
		return rewriteValuegeneric_OpNot_0(v) || rewriteValuegeneric_OpNot_10(v) || rewriteValuegeneric_OpNot_20(v) || rewriteValuegeneric_OpNot_30(v) || rewriteValuegeneric_OpNot_40(v)
	case OpOffPtr:
		return rewriteValuegeneric_OpOffPtr_0(v)
	case OpOr16:
		return rewriteValuegeneric_OpOr16_0(v) || rewriteValuegeneric_OpOr16_10(v) || rewriteValuegeneric_OpOr16_20(v)
	case OpOr32:
		return rewriteValuegeneric_OpOr32_0(v) || rewriteValuegeneric_OpOr32_10(v) || rewriteValuegeneric_OpOr32_20(v)
	case OpOr64:
		return rewriteValuegeneric_OpOr64_0(v) || rewriteValuegeneric_OpOr64_10(v) || rewriteValuegeneric_OpOr64_20(v)
	case OpOr8:
		return rewriteValuegeneric_OpOr8_0(v) || rewriteValuegeneric_OpOr8_10(v) || rewriteValuegeneric_OpOr8_20(v)
	case OpPhi:
		return rewriteValuegeneric_OpPhi_0(v)
	case OpPtrIndex:
		return rewriteValuegeneric_OpPtrIndex_0(v)
	case OpRound32F:
		return rewriteValuegeneric_OpRound32F_0(v)
	case OpRound64F:
		return rewriteValuegeneric_OpRound64F_0(v)
	case OpRsh16Ux16:
		return rewriteValuegeneric_OpRsh16Ux16_0(v)
	case OpRsh16Ux32:
		return rewriteValuegeneric_OpRsh16Ux32_0(v)
	case OpRsh16Ux64:
		return rewriteValuegeneric_OpRsh16Ux64_0(v)
	case OpRsh16Ux8:
		return rewriteValuegeneric_OpRsh16Ux8_0(v)
	case OpRsh16x16:
		return rewriteValuegeneric_OpRsh16x16_0(v)
	case OpRsh16x32:
		return rewriteValuegeneric_OpRsh16x32_0(v)
	case OpRsh16x64:
		return rewriteValuegeneric_OpRsh16x64_0(v)
	case OpRsh16x8:
		return rewriteValuegeneric_OpRsh16x8_0(v)
	case OpRsh32Ux16:
		return rewriteValuegeneric_OpRsh32Ux16_0(v)
	case OpRsh32Ux32:
		return rewriteValuegeneric_OpRsh32Ux32_0(v)
	case OpRsh32Ux64:
		return rewriteValuegeneric_OpRsh32Ux64_0(v)
	case OpRsh32Ux8:
		return rewriteValuegeneric_OpRsh32Ux8_0(v)
	case OpRsh32x16:
		return rewriteValuegeneric_OpRsh32x16_0(v)
	case OpRsh32x32:
		return rewriteValuegeneric_OpRsh32x32_0(v)
	case OpRsh32x64:
		return rewriteValuegeneric_OpRsh32x64_0(v)
	case OpRsh32x8:
		return rewriteValuegeneric_OpRsh32x8_0(v)
	case OpRsh64Ux16:
		return rewriteValuegeneric_OpRsh64Ux16_0(v)
	case OpRsh64Ux32:
		return rewriteValuegeneric_OpRsh64Ux32_0(v)
	case OpRsh64Ux64:
		return rewriteValuegeneric_OpRsh64Ux64_0(v)
	case OpRsh64Ux8:
		return rewriteValuegeneric_OpRsh64Ux8_0(v)
	case OpRsh64x16:
		return rewriteValuegeneric_OpRsh64x16_0(v)
	case OpRsh64x32:
		return rewriteValuegeneric_OpRsh64x32_0(v)
	case OpRsh64x64:
		return rewriteValuegeneric_OpRsh64x64_0(v)
	case OpRsh64x8:
		return rewriteValuegeneric_OpRsh64x8_0(v)
	case OpRsh8Ux16:
		return rewriteValuegeneric_OpRsh8Ux16_0(v)
	case OpRsh8Ux32:
		return rewriteValuegeneric_OpRsh8Ux32_0(v)
	case OpRsh8Ux64:
		return rewriteValuegeneric_OpRsh8Ux64_0(v)
	case OpRsh8Ux8:
		return rewriteValuegeneric_OpRsh8Ux8_0(v)
	case OpRsh8x16:
		return rewriteValuegeneric_OpRsh8x16_0(v)
	case OpRsh8x32:
		return rewriteValuegeneric_OpRsh8x32_0(v)
	case OpRsh8x64:
		return rewriteValuegeneric_OpRsh8x64_0(v)
	case OpRsh8x8:
		return rewriteValuegeneric_OpRsh8x8_0(v)
	case OpSignExt16to32:
		return rewriteValuegeneric_OpSignExt16to32_0(v)
	case OpSignExt16to64:
		return rewriteValuegeneric_OpSignExt16to64_0(v)
	case OpSignExt32to64:
		return rewriteValuegeneric_OpSignExt32to64_0(v)
	case OpSignExt8to16:
		return rewriteValuegeneric_OpSignExt8to16_0(v)
	case OpSignExt8to32:
		return rewriteValuegeneric_OpSignExt8to32_0(v)
	case OpSignExt8to64:
		return rewriteValuegeneric_OpSignExt8to64_0(v)
	case OpSliceCap:
		return rewriteValuegeneric_OpSliceCap_0(v)
	case OpSliceLen:
		return rewriteValuegeneric_OpSliceLen_0(v)
	case OpSlicePtr:
		return rewriteValuegeneric_OpSlicePtr_0(v)
	case OpSlicemask:
		return rewriteValuegeneric_OpSlicemask_0(v)
	case OpSqrt:
		return rewriteValuegeneric_OpSqrt_0(v)
	case OpStore:
		return rewriteValuegeneric_OpStore_0(v) || rewriteValuegeneric_OpStore_10(v)
	case OpStringLen:
		return rewriteValuegeneric_OpStringLen_0(v)
	case OpStringPtr:
		return rewriteValuegeneric_OpStringPtr_0(v)
	case OpStructSelect:
		return rewriteValuegeneric_OpStructSelect_0(v) || rewriteValuegeneric_OpStructSelect_10(v)
	case OpSub16:
		return rewriteValuegeneric_OpSub16_0(v) || rewriteValuegeneric_OpSub16_10(v)
	case OpSub32:
		return rewriteValuegeneric_OpSub32_0(v) || rewriteValuegeneric_OpSub32_10(v)
	case OpSub32F:
		return rewriteValuegeneric_OpSub32F_0(v)
	case OpSub64:
		return rewriteValuegeneric_OpSub64_0(v) || rewriteValuegeneric_OpSub64_10(v)
	case OpSub64F:
		return rewriteValuegeneric_OpSub64F_0(v)
	case OpSub8:
		return rewriteValuegeneric_OpSub8_0(v) || rewriteValuegeneric_OpSub8_10(v)
	case OpTrunc16to8:
		return rewriteValuegeneric_OpTrunc16to8_0(v)
	case OpTrunc32to16:
		return rewriteValuegeneric_OpTrunc32to16_0(v)
	case OpTrunc32to8:
		return rewriteValuegeneric_OpTrunc32to8_0(v)
	case OpTrunc64to16:
		return rewriteValuegeneric_OpTrunc64to16_0(v)
	case OpTrunc64to32:
		return rewriteValuegeneric_OpTrunc64to32_0(v)
	case OpTrunc64to8:
		return rewriteValuegeneric_OpTrunc64to8_0(v)
	case OpXor16:
		return rewriteValuegeneric_OpXor16_0(v) || rewriteValuegeneric_OpXor16_10(v)
	case OpXor32:
		return rewriteValuegeneric_OpXor32_0(v) || rewriteValuegeneric_OpXor32_10(v)
	case OpXor64:
		return rewriteValuegeneric_OpXor64_0(v) || rewriteValuegeneric_OpXor64_10(v)
	case OpXor8:
		return rewriteValuegeneric_OpXor8_0(v) || rewriteValuegeneric_OpXor8_10(v)
	case OpZero:
		return rewriteValuegeneric_OpZero_0(v)
	case OpZeroExt16to32:
		return rewriteValuegeneric_OpZeroExt16to32_0(v)
	case OpZeroExt16to64:
		return rewriteValuegeneric_OpZeroExt16to64_0(v)
	case OpZeroExt32to64:
		return rewriteValuegeneric_OpZeroExt32to64_0(v)
	case OpZeroExt8to16:
		return rewriteValuegeneric_OpZeroExt8to16_0(v)
	case OpZeroExt8to32:
		return rewriteValuegeneric_OpZeroExt8to32_0(v)
	case OpZeroExt8to64:
		return rewriteValuegeneric_OpZeroExt8to64_0(v)
	}
	return false
}
func rewriteValuegeneric_OpAdd16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Add16 (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (Const16 [int64(int16(c+d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c + d))
		return true
	}
	// match: (Add16 (Const16 [d]) (Const16 [c]))
	// cond:
	// result: (Const16 [int64(int16(c+d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c + d))
		return true
	}
	// match: (Add16 (Const16 [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Add16 x (Const16 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Add16 (Const16 [1]) (Com16 x))
	// cond:
	// result: (Neg16 x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpCom16 {
			break
		}
		x := v_1.Args[0]
		v.reset(OpNeg16)
		v.AddArg(x)
		return true
	}
	// match: (Add16 (Com16 x) (Const16 [1]))
	// cond:
	// result: (Neg16 x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpCom16 {
			break
		}
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpNeg16)
		v.AddArg(x)
		return true
	}
	// match: (Add16 (Add16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Add16 i (Add16 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAdd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Add16 (Add16 z i:(Const16 <t>)) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Add16 i (Add16 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAdd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Add16 x (Add16 i:(Const16 <t>) z))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Add16 i (Add16 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAdd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Add16 x (Add16 z i:(Const16 <t>)))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Add16 i (Add16 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAdd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd16_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Add16 (Sub16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Add16 i (Sub16 <t> x z))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub16 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAdd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub16, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add16 x (Sub16 i:(Const16 <t>) z))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Add16 i (Sub16 <t> x z))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub16 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAdd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub16, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add16 x (Sub16 i:(Const16 <t>) z))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Add16 i (Sub16 <t> x z))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub16 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAdd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub16, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add16 (Sub16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Add16 i (Sub16 <t> x z))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub16 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAdd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub16, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add16 (Sub16 z i:(Const16 <t>)) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Sub16 (Add16 <t> x z) i)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub16 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpSub16)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add16 x (Sub16 z i:(Const16 <t>)))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Sub16 (Add16 <t> x z) i)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub16 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpSub16)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add16 x (Sub16 z i:(Const16 <t>)))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Sub16 (Add16 <t> x z) i)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub16 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpSub16)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add16 (Sub16 z i:(Const16 <t>)) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Sub16 (Add16 <t> x z) i)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub16 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpSub16)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add16 (Const16 <t> [c]) (Add16 (Const16 <t> [d]) x))
	// cond:
	// result: (Add16 (Const16 <t> [int64(int16(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAdd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add16 (Const16 <t> [c]) (Add16 x (Const16 <t> [d])))
	// cond:
	// result: (Add16 (Const16 <t> [int64(int16(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst16 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAdd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd16_20(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Add16 (Add16 (Const16 <t> [d]) x) (Const16 <t> [c]))
	// cond:
	// result: (Add16 (Const16 <t> [int64(int16(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst16 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add16 (Add16 x (Const16 <t> [d])) (Const16 <t> [c]))
	// cond:
	// result: (Add16 (Const16 <t> [int64(int16(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst16 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add16 (Const16 <t> [c]) (Sub16 (Const16 <t> [d]) x))
	// cond:
	// result: (Sub16 (Const16 <t> [int64(int16(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpSub16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add16 (Sub16 (Const16 <t> [d]) x) (Const16 <t> [c]))
	// cond:
	// result: (Sub16 (Const16 <t> [int64(int16(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst16 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpSub16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add16 (Const16 <t> [c]) (Sub16 x (Const16 <t> [d])))
	// cond:
	// result: (Add16 (Const16 <t> [int64(int16(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub16 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst16 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAdd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add16 (Sub16 x (Const16 <t> [d])) (Const16 <t> [c]))
	// cond:
	// result: (Add16 (Const16 <t> [int64(int16(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst16 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Add32 (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (Const32 [int64(int32(c+d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c + d))
		return true
	}
	// match: (Add32 (Const32 [d]) (Const32 [c]))
	// cond:
	// result: (Const32 [int64(int32(c+d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c + d))
		return true
	}
	// match: (Add32 (Const32 [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Add32 x (Const32 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Add32 (Const32 [1]) (Com32 x))
	// cond:
	// result: (Neg32 x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpCom32 {
			break
		}
		x := v_1.Args[0]
		v.reset(OpNeg32)
		v.AddArg(x)
		return true
	}
	// match: (Add32 (Com32 x) (Const32 [1]))
	// cond:
	// result: (Neg32 x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpCom32 {
			break
		}
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpNeg32)
		v.AddArg(x)
		return true
	}
	// match: (Add32 (Add32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Add32 i (Add32 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAdd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Add32 (Add32 z i:(Const32 <t>)) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Add32 i (Add32 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAdd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Add32 x (Add32 i:(Const32 <t>) z))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Add32 i (Add32 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAdd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Add32 x (Add32 z i:(Const32 <t>)))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Add32 i (Add32 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAdd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd32_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Add32 (Sub32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Add32 i (Sub32 <t> x z))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub32 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAdd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub32, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add32 x (Sub32 i:(Const32 <t>) z))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Add32 i (Sub32 <t> x z))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub32 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAdd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub32, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add32 x (Sub32 i:(Const32 <t>) z))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Add32 i (Sub32 <t> x z))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub32 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAdd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub32, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add32 (Sub32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Add32 i (Sub32 <t> x z))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub32 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAdd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub32, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add32 (Sub32 z i:(Const32 <t>)) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Sub32 (Add32 <t> x z) i)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub32 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpSub32)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add32 x (Sub32 z i:(Const32 <t>)))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Sub32 (Add32 <t> x z) i)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub32 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpSub32)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add32 x (Sub32 z i:(Const32 <t>)))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Sub32 (Add32 <t> x z) i)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub32 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpSub32)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add32 (Sub32 z i:(Const32 <t>)) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Sub32 (Add32 <t> x z) i)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub32 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpSub32)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add32 (Const32 <t> [c]) (Add32 (Const32 <t> [d]) x))
	// cond:
	// result: (Add32 (Const32 <t> [int64(int32(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add32 (Const32 <t> [c]) (Add32 x (Const32 <t> [d])))
	// cond:
	// result: (Add32 (Const32 <t> [int64(int32(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd32_20(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Add32 (Add32 (Const32 <t> [d]) x) (Const32 <t> [c]))
	// cond:
	// result: (Add32 (Const32 <t> [int64(int32(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add32 (Add32 x (Const32 <t> [d])) (Const32 <t> [c]))
	// cond:
	// result: (Add32 (Const32 <t> [int64(int32(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add32 (Const32 <t> [c]) (Sub32 (Const32 <t> [d]) x))
	// cond:
	// result: (Sub32 (Const32 <t> [int64(int32(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpSub32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add32 (Sub32 (Const32 <t> [d]) x) (Const32 <t> [c]))
	// cond:
	// result: (Sub32 (Const32 <t> [int64(int32(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpSub32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add32 (Const32 <t> [c]) (Sub32 x (Const32 <t> [d])))
	// cond:
	// result: (Add32 (Const32 <t> [int64(int32(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub32 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add32 (Sub32 x (Const32 <t> [d])) (Const32 <t> [c]))
	// cond:
	// result: (Add32 (Const32 <t> [int64(int32(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd32F_0(v *Value) bool {
	// match: (Add32F (Const32F [c]) (Const32F [d]))
	// cond:
	// result: (Const32F [f2i(float64(i2f32(c) + i2f32(d)))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32F {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32F {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32F)
		v.AuxInt = f2i(float64(i2f32(c) + i2f32(d)))
		return true
	}
	// match: (Add32F (Const32F [d]) (Const32F [c]))
	// cond:
	// result: (Const32F [f2i(float64(i2f32(c) + i2f32(d)))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32F {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32F {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst32F)
		v.AuxInt = f2i(float64(i2f32(c) + i2f32(d)))
		return true
	}
	// match: (Add32F x (Const32F [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32F {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Add32F (Const32F [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32F {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Add64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (Const64 [c+d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c + d
		return true
	}
	// match: (Add64 (Const64 [d]) (Const64 [c]))
	// cond:
	// result: (Const64 [c+d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c + d
		return true
	}
	// match: (Add64 (Const64 [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Add64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Add64 (Const64 [1]) (Com64 x))
	// cond:
	// result: (Neg64 x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpCom64 {
			break
		}
		x := v_1.Args[0]
		v.reset(OpNeg64)
		v.AddArg(x)
		return true
	}
	// match: (Add64 (Com64 x) (Const64 [1]))
	// cond:
	// result: (Neg64 x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpCom64 {
			break
		}
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpNeg64)
		v.AddArg(x)
		return true
	}
	// match: (Add64 (Add64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Add64 i (Add64 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAdd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Add64 (Add64 z i:(Const64 <t>)) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Add64 i (Add64 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAdd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Add64 x (Add64 i:(Const64 <t>) z))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Add64 i (Add64 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAdd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Add64 x (Add64 z i:(Const64 <t>)))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Add64 i (Add64 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAdd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd64_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Add64 (Sub64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Add64 i (Sub64 <t> x z))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub64 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAdd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub64, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add64 x (Sub64 i:(Const64 <t>) z))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Add64 i (Sub64 <t> x z))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub64 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAdd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub64, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add64 x (Sub64 i:(Const64 <t>) z))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Add64 i (Sub64 <t> x z))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub64 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAdd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub64, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add64 (Sub64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Add64 i (Sub64 <t> x z))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub64 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAdd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub64, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add64 (Sub64 z i:(Const64 <t>)) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Sub64 (Add64 <t> x z) i)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub64 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpSub64)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add64 x (Sub64 z i:(Const64 <t>)))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Sub64 (Add64 <t> x z) i)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub64 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpSub64)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add64 x (Sub64 z i:(Const64 <t>)))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Sub64 (Add64 <t> x z) i)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub64 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpSub64)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add64 (Sub64 z i:(Const64 <t>)) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Sub64 (Add64 <t> x z) i)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub64 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpSub64)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add64 (Const64 <t> [c]) (Add64 (Const64 <t> [d]) x))
	// cond:
	// result: (Add64 (Const64 <t> [c+d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add64 (Const64 <t> [c]) (Add64 x (Const64 <t> [d])))
	// cond:
	// result: (Add64 (Const64 <t> [c+d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd64_20(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Add64 (Add64 (Const64 <t> [d]) x) (Const64 <t> [c]))
	// cond:
	// result: (Add64 (Const64 <t> [c+d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add64 (Add64 x (Const64 <t> [d])) (Const64 <t> [c]))
	// cond:
	// result: (Add64 (Const64 <t> [c+d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add64 (Const64 <t> [c]) (Sub64 (Const64 <t> [d]) x))
	// cond:
	// result: (Sub64 (Const64 <t> [c+d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpSub64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add64 (Sub64 (Const64 <t> [d]) x) (Const64 <t> [c]))
	// cond:
	// result: (Sub64 (Const64 <t> [c+d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpSub64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add64 (Const64 <t> [c]) (Sub64 x (Const64 <t> [d])))
	// cond:
	// result: (Add64 (Const64 <t> [c-d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub64 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c - d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add64 (Sub64 x (Const64 <t> [d])) (Const64 <t> [c]))
	// cond:
	// result: (Add64 (Const64 <t> [c-d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c - d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd64F_0(v *Value) bool {
	// match: (Add64F (Const64F [c]) (Const64F [d]))
	// cond:
	// result: (Const64F [f2i(i2f(c) + i2f(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64F {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64F)
		v.AuxInt = f2i(i2f(c) + i2f(d))
		return true
	}
	// match: (Add64F (Const64F [d]) (Const64F [c]))
	// cond:
	// result: (Const64F [f2i(i2f(c) + i2f(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64F {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst64F)
		v.AuxInt = f2i(i2f(c) + i2f(d))
		return true
	}
	// match: (Add64F x (Const64F [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64F {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Add64F (Const64F [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Add8 (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (Const8  [int64(int8(c+d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c + d))
		return true
	}
	// match: (Add8 (Const8 [d]) (Const8 [c]))
	// cond:
	// result: (Const8  [int64(int8(c+d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c + d))
		return true
	}
	// match: (Add8 (Const8 [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Add8 x (Const8 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Add8 (Const8 [1]) (Com8 x))
	// cond:
	// result: (Neg8  x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpCom8 {
			break
		}
		x := v_1.Args[0]
		v.reset(OpNeg8)
		v.AddArg(x)
		return true
	}
	// match: (Add8 (Com8 x) (Const8 [1]))
	// cond:
	// result: (Neg8  x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpCom8 {
			break
		}
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpNeg8)
		v.AddArg(x)
		return true
	}
	// match: (Add8 (Add8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Add8  i (Add8  <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAdd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Add8 (Add8 z i:(Const8 <t>)) x)
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Add8  i (Add8  <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAdd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Add8 x (Add8 i:(Const8 <t>) z))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Add8  i (Add8  <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAdd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Add8 x (Add8 z i:(Const8 <t>)))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Add8  i (Add8  <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAdd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd8_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Add8 (Sub8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Add8  i (Sub8  <t> x z))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub8 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAdd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub8, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add8 x (Sub8 i:(Const8 <t>) z))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Add8  i (Sub8  <t> x z))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub8 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAdd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub8, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add8 x (Sub8 i:(Const8 <t>) z))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Add8  i (Sub8  <t> x z))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub8 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAdd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub8, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add8 (Sub8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Add8  i (Sub8  <t> x z))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub8 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAdd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub8, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Add8 (Sub8 z i:(Const8 <t>)) x)
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Sub8  (Add8  <t> x z) i)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub8 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpSub8)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add8 x (Sub8 z i:(Const8 <t>)))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Sub8  (Add8  <t> x z) i)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub8 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpSub8)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add8 x (Sub8 z i:(Const8 <t>)))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Sub8  (Add8  <t> x z) i)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub8 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpSub8)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add8 (Sub8 z i:(Const8 <t>)) x)
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Sub8  (Add8  <t> x z) i)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub8 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpSub8)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Add8 (Const8 <t> [c]) (Add8 (Const8 <t> [d]) x))
	// cond:
	// result: (Add8  (Const8  <t> [int64(int8(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAdd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add8 (Const8 <t> [c]) (Add8 x (Const8 <t> [d])))
	// cond:
	// result: (Add8  (Const8  <t> [int64(int8(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst8 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAdd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAdd8_20(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Add8 (Add8 (Const8 <t> [d]) x) (Const8 <t> [c]))
	// cond:
	// result: (Add8  (Const8  <t> [int64(int8(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst8 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add8 (Add8 x (Const8 <t> [d])) (Const8 <t> [c]))
	// cond:
	// result: (Add8  (Const8  <t> [int64(int8(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst8 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add8 (Const8 <t> [c]) (Sub8 (Const8 <t> [d]) x))
	// cond:
	// result: (Sub8  (Const8  <t> [int64(int8(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpSub8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add8 (Sub8 (Const8 <t> [d]) x) (Const8 <t> [c]))
	// cond:
	// result: (Sub8  (Const8  <t> [int64(int8(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst8 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpSub8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add8 (Const8 <t> [c]) (Sub8 x (Const8 <t> [d])))
	// cond:
	// result: (Add8  (Const8  <t> [int64(int8(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub8 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst8 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAdd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Add8 (Sub8 x (Const8 <t> [d])) (Const8 <t> [c]))
	// cond:
	// result: (Add8  (Const8  <t> [int64(int8(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSub8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst8 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAddPtr_0(v *Value) bool {
	// match: (AddPtr <t> x (Const64 [c]))
	// cond:
	// result: (OffPtr <t> x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOffPtr)
		v.Type = t
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	// match: (AddPtr <t> x (Const32 [c]))
	// cond:
	// result: (OffPtr <t> x [c])
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOffPtr)
		v.Type = t
		v.AuxInt = c
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAnd16_0(v *Value) bool {
	// match: (And16 (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (Const16 [int64(int16(c&d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c & d))
		return true
	}
	// match: (And16 (Const16 [d]) (Const16 [c]))
	// cond:
	// result: (Const16 [int64(int16(c&d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c & d))
		return true
	}
	// match: (And16 x x)
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
	// match: (And16 (Const16 [-1]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (And16 x (Const16 [-1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
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
	// match: (And16 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	// match: (And16 _ (Const16 [0]))
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	// match: (And16 x (And16 x y))
	// cond:
	// result: (And16 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpAnd16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (And16 x (And16 y x))
	// cond:
	// result: (And16 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpAnd16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (And16 (And16 x y) x)
	// cond:
	// result: (And16 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAnd16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAnd16_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (And16 (And16 y x) x)
	// cond:
	// result: (And16 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAnd16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (And16 (And16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (And16 i (And16 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAnd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And16 (And16 z i:(Const16 <t>)) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (And16 i (And16 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAnd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And16 x (And16 i:(Const16 <t>) z))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (And16 i (And16 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAnd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And16 x (And16 z i:(Const16 <t>)))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (And16 i (And16 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAnd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And16 (Const16 <t> [c]) (And16 (Const16 <t> [d]) x))
	// cond:
	// result: (And16 (Const16 <t> [int64(int16(c&d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAnd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c & d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (And16 (Const16 <t> [c]) (And16 x (Const16 <t> [d])))
	// cond:
	// result: (And16 (Const16 <t> [int64(int16(c&d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst16 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAnd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c & d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (And16 (And16 (Const16 <t> [d]) x) (Const16 <t> [c]))
	// cond:
	// result: (And16 (Const16 <t> [int64(int16(c&d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst16 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAnd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c & d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (And16 (And16 x (Const16 <t> [d])) (Const16 <t> [c]))
	// cond:
	// result: (And16 (Const16 <t> [int64(int16(c&d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst16 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAnd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c & d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAnd32_0(v *Value) bool {
	// match: (And32 (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (Const32 [int64(int32(c&d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c & d))
		return true
	}
	// match: (And32 (Const32 [d]) (Const32 [c]))
	// cond:
	// result: (Const32 [int64(int32(c&d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c & d))
		return true
	}
	// match: (And32 x x)
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
	// match: (And32 (Const32 [-1]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (And32 x (Const32 [-1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
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
	// match: (And32 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	// match: (And32 _ (Const32 [0]))
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	// match: (And32 x (And32 x y))
	// cond:
	// result: (And32 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpAnd32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (And32 x (And32 y x))
	// cond:
	// result: (And32 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpAnd32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (And32 (And32 x y) x)
	// cond:
	// result: (And32 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAnd32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAnd32_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (And32 (And32 y x) x)
	// cond:
	// result: (And32 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAnd32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (And32 (And32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (And32 i (And32 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAnd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And32 (And32 z i:(Const32 <t>)) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (And32 i (And32 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAnd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And32 x (And32 i:(Const32 <t>) z))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (And32 i (And32 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAnd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And32 x (And32 z i:(Const32 <t>)))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (And32 i (And32 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAnd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And32 (Const32 <t> [c]) (And32 (Const32 <t> [d]) x))
	// cond:
	// result: (And32 (Const32 <t> [int64(int32(c&d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAnd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c & d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (And32 (Const32 <t> [c]) (And32 x (Const32 <t> [d])))
	// cond:
	// result: (And32 (Const32 <t> [int64(int32(c&d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAnd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c & d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (And32 (And32 (Const32 <t> [d]) x) (Const32 <t> [c]))
	// cond:
	// result: (And32 (Const32 <t> [int64(int32(c&d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAnd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c & d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (And32 (And32 x (Const32 <t> [d])) (Const32 <t> [c]))
	// cond:
	// result: (And32 (Const32 <t> [int64(int32(c&d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAnd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c & d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAnd64_0(v *Value) bool {
	// match: (And64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (Const64 [c&d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c & d
		return true
	}
	// match: (And64 (Const64 [d]) (Const64 [c]))
	// cond:
	// result: (Const64 [c&d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c & d
		return true
	}
	// match: (And64 x x)
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
	// match: (And64 (Const64 [-1]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (And64 x (Const64 [-1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
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
	// match: (And64 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	// match: (And64 _ (Const64 [0]))
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	// match: (And64 x (And64 x y))
	// cond:
	// result: (And64 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpAnd64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (And64 x (And64 y x))
	// cond:
	// result: (And64 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpAnd64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (And64 (And64 x y) x)
	// cond:
	// result: (And64 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAnd64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAnd64_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (And64 (And64 y x) x)
	// cond:
	// result: (And64 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAnd64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (And64 <t> (Const64 [y]) x)
	// cond: nlz(y) + nto(y) == 64 && nto(y) >= 32
	// result: (Rsh64Ux64 (Lsh64x64 <t> x (Const64 <t> [nlz(y)])) (Const64 <t> [nlz(y)]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		y := v_0.AuxInt
		x := v.Args[1]
		if !(nlz(y)+nto(y) == 64 && nto(y) >= 32) {
			break
		}
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpLsh64x64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = nlz(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = nlz(y)
		v.AddArg(v2)
		return true
	}
	// match: (And64 <t> x (Const64 [y]))
	// cond: nlz(y) + nto(y) == 64 && nto(y) >= 32
	// result: (Rsh64Ux64 (Lsh64x64 <t> x (Const64 <t> [nlz(y)])) (Const64 <t> [nlz(y)]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		y := v_1.AuxInt
		if !(nlz(y)+nto(y) == 64 && nto(y) >= 32) {
			break
		}
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpLsh64x64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = nlz(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = nlz(y)
		v.AddArg(v2)
		return true
	}
	// match: (And64 <t> (Const64 [y]) x)
	// cond: nlo(y) + ntz(y) == 64 && ntz(y) >= 32
	// result: (Lsh64x64 (Rsh64Ux64 <t> x (Const64 <t> [ntz(y)])) (Const64 <t> [ntz(y)]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		y := v_0.AuxInt
		x := v.Args[1]
		if !(nlo(y)+ntz(y) == 64 && ntz(y) >= 32) {
			break
		}
		v.reset(OpLsh64x64)
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = ntz(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = ntz(y)
		v.AddArg(v2)
		return true
	}
	// match: (And64 <t> x (Const64 [y]))
	// cond: nlo(y) + ntz(y) == 64 && ntz(y) >= 32
	// result: (Lsh64x64 (Rsh64Ux64 <t> x (Const64 <t> [ntz(y)])) (Const64 <t> [ntz(y)]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		y := v_1.AuxInt
		if !(nlo(y)+ntz(y) == 64 && ntz(y) >= 32) {
			break
		}
		v.reset(OpLsh64x64)
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = ntz(y)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = ntz(y)
		v.AddArg(v2)
		return true
	}
	// match: (And64 (And64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (And64 i (And64 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAnd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And64 (And64 z i:(Const64 <t>)) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (And64 i (And64 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAnd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And64 x (And64 i:(Const64 <t>) z))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (And64 i (And64 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAnd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And64 x (And64 z i:(Const64 <t>)))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (And64 i (And64 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAnd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And64 (Const64 <t> [c]) (And64 (Const64 <t> [d]) x))
	// cond:
	// result: (And64 (Const64 <t> [c&d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAnd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c & d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAnd64_20(v *Value) bool {
	b := v.Block
	_ = b
	// match: (And64 (Const64 <t> [c]) (And64 x (Const64 <t> [d])))
	// cond:
	// result: (And64 (Const64 <t> [c&d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAnd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c & d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (And64 (And64 (Const64 <t> [d]) x) (Const64 <t> [c]))
	// cond:
	// result: (And64 (Const64 <t> [c&d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAnd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c & d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (And64 (And64 x (Const64 <t> [d])) (Const64 <t> [c]))
	// cond:
	// result: (And64 (Const64 <t> [c&d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAnd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c & d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAnd8_0(v *Value) bool {
	// match: (And8 (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (Const8  [int64(int8(c&d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c & d))
		return true
	}
	// match: (And8 (Const8 [d]) (Const8 [c]))
	// cond:
	// result: (Const8  [int64(int8(c&d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c & d))
		return true
	}
	// match: (And8 x x)
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
	// match: (And8 (Const8 [-1]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (And8 x (Const8 [-1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
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
	// match: (And8 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	// match: (And8 _ (Const8 [0]))
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	// match: (And8 x (And8 x y))
	// cond:
	// result: (And8  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpAnd8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (And8 x (And8 y x))
	// cond:
	// result: (And8  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpAnd8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (And8 (And8 x y) x)
	// cond:
	// result: (And8  x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAnd8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpAnd8_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (And8 (And8 y x) x)
	// cond:
	// result: (And8  x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpAnd8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (And8 (And8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (And8  i (And8  <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAnd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And8 (And8 z i:(Const8 <t>)) x)
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (And8  i (And8  <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAnd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And8 x (And8 i:(Const8 <t>) z))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (And8  i (And8  <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAnd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And8 x (And8 z i:(Const8 <t>)))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (And8  i (And8  <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAnd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpAnd8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (And8 (Const8 <t> [c]) (And8 (Const8 <t> [d]) x))
	// cond:
	// result: (And8  (Const8  <t> [int64(int8(c&d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAnd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c & d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (And8 (Const8 <t> [c]) (And8 x (Const8 <t> [d])))
	// cond:
	// result: (And8  (Const8  <t> [int64(int8(c&d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst8 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAnd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c & d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (And8 (And8 (Const8 <t> [d]) x) (Const8 <t> [c]))
	// cond:
	// result: (And8  (Const8  <t> [int64(int8(c&d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst8 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAnd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c & d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (And8 (And8 x (Const8 <t> [d])) (Const8 <t> [c]))
	// cond:
	// result: (And8  (Const8  <t> [int64(int8(c&d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst8 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAnd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c & d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpArg_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	fe := b.Func.fe
	_ = fe
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Arg {n} [off])
	// cond: v.Type.IsString()
	// result: (StringMake     (Arg <typ.BytePtr> {n} [off])     (Arg <typ.Int> {n} [off+config.PtrSize]))
	for {
		off := v.AuxInt
		n := v.Aux
		if !(v.Type.IsString()) {
			break
		}
		v.reset(OpStringMake)
		v0 := b.NewValue0(v.Pos, OpArg, typ.BytePtr)
		v0.AuxInt = off
		v0.Aux = n
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpArg, typ.Int)
		v1.AuxInt = off + config.PtrSize
		v1.Aux = n
		v.AddArg(v1)
		return true
	}
	// match: (Arg {n} [off])
	// cond: v.Type.IsSlice()
	// result: (SliceMake     (Arg <v.Type.ElemType().PtrTo()> {n} [off])     (Arg <typ.Int> {n} [off+config.PtrSize])     (Arg <typ.Int> {n} [off+2*config.PtrSize]))
	for {
		off := v.AuxInt
		n := v.Aux
		if !(v.Type.IsSlice()) {
			break
		}
		v.reset(OpSliceMake)
		v0 := b.NewValue0(v.Pos, OpArg, v.Type.ElemType().PtrTo())
		v0.AuxInt = off
		v0.Aux = n
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpArg, typ.Int)
		v1.AuxInt = off + config.PtrSize
		v1.Aux = n
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpArg, typ.Int)
		v2.AuxInt = off + 2*config.PtrSize
		v2.Aux = n
		v.AddArg(v2)
		return true
	}
	// match: (Arg {n} [off])
	// cond: v.Type.IsInterface()
	// result: (IMake     (Arg <typ.BytePtr> {n} [off])     (Arg <typ.BytePtr> {n} [off+config.PtrSize]))
	for {
		off := v.AuxInt
		n := v.Aux
		if !(v.Type.IsInterface()) {
			break
		}
		v.reset(OpIMake)
		v0 := b.NewValue0(v.Pos, OpArg, typ.BytePtr)
		v0.AuxInt = off
		v0.Aux = n
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpArg, typ.BytePtr)
		v1.AuxInt = off + config.PtrSize
		v1.Aux = n
		v.AddArg(v1)
		return true
	}
	// match: (Arg {n} [off])
	// cond: v.Type.IsComplex() && v.Type.Size() == 16
	// result: (ComplexMake     (Arg <typ.Float64> {n} [off])     (Arg <typ.Float64> {n} [off+8]))
	for {
		off := v.AuxInt
		n := v.Aux
		if !(v.Type.IsComplex() && v.Type.Size() == 16) {
			break
		}
		v.reset(OpComplexMake)
		v0 := b.NewValue0(v.Pos, OpArg, typ.Float64)
		v0.AuxInt = off
		v0.Aux = n
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpArg, typ.Float64)
		v1.AuxInt = off + 8
		v1.Aux = n
		v.AddArg(v1)
		return true
	}
	// match: (Arg {n} [off])
	// cond: v.Type.IsComplex() && v.Type.Size() == 8
	// result: (ComplexMake     (Arg <typ.Float32> {n} [off])     (Arg <typ.Float32> {n} [off+4]))
	for {
		off := v.AuxInt
		n := v.Aux
		if !(v.Type.IsComplex() && v.Type.Size() == 8) {
			break
		}
		v.reset(OpComplexMake)
		v0 := b.NewValue0(v.Pos, OpArg, typ.Float32)
		v0.AuxInt = off
		v0.Aux = n
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpArg, typ.Float32)
		v1.AuxInt = off + 4
		v1.Aux = n
		v.AddArg(v1)
		return true
	}
	// match: (Arg <t>)
	// cond: t.IsStruct() && t.NumFields() == 0 && fe.CanSSA(t)
	// result: (StructMake0)
	for {
		t := v.Type
		if !(t.IsStruct() && t.NumFields() == 0 && fe.CanSSA(t)) {
			break
		}
		v.reset(OpStructMake0)
		return true
	}
	// match: (Arg <t> {n} [off])
	// cond: t.IsStruct() && t.NumFields() == 1 && fe.CanSSA(t)
	// result: (StructMake1     (Arg <t.FieldType(0)> {n} [off+t.FieldOff(0)]))
	for {
		t := v.Type
		off := v.AuxInt
		n := v.Aux
		if !(t.IsStruct() && t.NumFields() == 1 && fe.CanSSA(t)) {
			break
		}
		v.reset(OpStructMake1)
		v0 := b.NewValue0(v.Pos, OpArg, t.FieldType(0))
		v0.AuxInt = off + t.FieldOff(0)
		v0.Aux = n
		v.AddArg(v0)
		return true
	}
	// match: (Arg <t> {n} [off])
	// cond: t.IsStruct() && t.NumFields() == 2 && fe.CanSSA(t)
	// result: (StructMake2     (Arg <t.FieldType(0)> {n} [off+t.FieldOff(0)])     (Arg <t.FieldType(1)> {n} [off+t.FieldOff(1)]))
	for {
		t := v.Type
		off := v.AuxInt
		n := v.Aux
		if !(t.IsStruct() && t.NumFields() == 2 && fe.CanSSA(t)) {
			break
		}
		v.reset(OpStructMake2)
		v0 := b.NewValue0(v.Pos, OpArg, t.FieldType(0))
		v0.AuxInt = off + t.FieldOff(0)
		v0.Aux = n
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpArg, t.FieldType(1))
		v1.AuxInt = off + t.FieldOff(1)
		v1.Aux = n
		v.AddArg(v1)
		return true
	}
	// match: (Arg <t> {n} [off])
	// cond: t.IsStruct() && t.NumFields() == 3 && fe.CanSSA(t)
	// result: (StructMake3     (Arg <t.FieldType(0)> {n} [off+t.FieldOff(0)])     (Arg <t.FieldType(1)> {n} [off+t.FieldOff(1)])     (Arg <t.FieldType(2)> {n} [off+t.FieldOff(2)]))
	for {
		t := v.Type
		off := v.AuxInt
		n := v.Aux
		if !(t.IsStruct() && t.NumFields() == 3 && fe.CanSSA(t)) {
			break
		}
		v.reset(OpStructMake3)
		v0 := b.NewValue0(v.Pos, OpArg, t.FieldType(0))
		v0.AuxInt = off + t.FieldOff(0)
		v0.Aux = n
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpArg, t.FieldType(1))
		v1.AuxInt = off + t.FieldOff(1)
		v1.Aux = n
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpArg, t.FieldType(2))
		v2.AuxInt = off + t.FieldOff(2)
		v2.Aux = n
		v.AddArg(v2)
		return true
	}
	// match: (Arg <t> {n} [off])
	// cond: t.IsStruct() && t.NumFields() == 4 && fe.CanSSA(t)
	// result: (StructMake4     (Arg <t.FieldType(0)> {n} [off+t.FieldOff(0)])     (Arg <t.FieldType(1)> {n} [off+t.FieldOff(1)])     (Arg <t.FieldType(2)> {n} [off+t.FieldOff(2)])     (Arg <t.FieldType(3)> {n} [off+t.FieldOff(3)]))
	for {
		t := v.Type
		off := v.AuxInt
		n := v.Aux
		if !(t.IsStruct() && t.NumFields() == 4 && fe.CanSSA(t)) {
			break
		}
		v.reset(OpStructMake4)
		v0 := b.NewValue0(v.Pos, OpArg, t.FieldType(0))
		v0.AuxInt = off + t.FieldOff(0)
		v0.Aux = n
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpArg, t.FieldType(1))
		v1.AuxInt = off + t.FieldOff(1)
		v1.Aux = n
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpArg, t.FieldType(2))
		v2.AuxInt = off + t.FieldOff(2)
		v2.Aux = n
		v.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpArg, t.FieldType(3))
		v3.AuxInt = off + t.FieldOff(3)
		v3.Aux = n
		v.AddArg(v3)
		return true
	}
	return false
}
func rewriteValuegeneric_OpArg_10(v *Value) bool {
	b := v.Block
	_ = b
	fe := b.Func.fe
	_ = fe
	// match: (Arg <t>)
	// cond: t.IsArray() && t.NumElem() == 0
	// result: (ArrayMake0)
	for {
		t := v.Type
		if !(t.IsArray() && t.NumElem() == 0) {
			break
		}
		v.reset(OpArrayMake0)
		return true
	}
	// match: (Arg <t> {n} [off])
	// cond: t.IsArray() && t.NumElem() == 1 && fe.CanSSA(t)
	// result: (ArrayMake1 (Arg <t.ElemType()> {n} [off]))
	for {
		t := v.Type
		off := v.AuxInt
		n := v.Aux
		if !(t.IsArray() && t.NumElem() == 1 && fe.CanSSA(t)) {
			break
		}
		v.reset(OpArrayMake1)
		v0 := b.NewValue0(v.Pos, OpArg, t.ElemType())
		v0.AuxInt = off
		v0.Aux = n
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpArraySelect_0(v *Value) bool {
	// match: (ArraySelect (ArrayMake1 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpArrayMake1 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (ArraySelect [0] (Load ptr mem))
	// cond:
	// result: (Load ptr mem)
	for {
		if v.AuxInt != 0 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpLoad {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_0.Args[1]
		v.reset(OpLoad)
		v.AddArg(ptr)
		v.AddArg(mem)
		return true
	}
	// match: (ArraySelect [0] x:(IData _))
	// cond:
	// result: x
	for {
		if v.AuxInt != 0 {
			break
		}
		x := v.Args[0]
		if x.Op != OpIData {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpCom16_0(v *Value) bool {
	// match: (Com16 (Com16 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpCom16 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Com16 (Const16 [c]))
	// cond:
	// result: (Const16 [^c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst16)
		v.AuxInt = ^c
		return true
	}
	return false
}
func rewriteValuegeneric_OpCom32_0(v *Value) bool {
	// match: (Com32 (Com32 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpCom32 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Com32 (Const32 [c]))
	// cond:
	// result: (Const32 [^c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst32)
		v.AuxInt = ^c
		return true
	}
	return false
}
func rewriteValuegeneric_OpCom64_0(v *Value) bool {
	// match: (Com64 (Com64 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpCom64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Com64 (Const64 [c]))
	// cond:
	// result: (Const64 [^c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst64)
		v.AuxInt = ^c
		return true
	}
	return false
}
func rewriteValuegeneric_OpCom8_0(v *Value) bool {
	// match: (Com8 (Com8 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpCom8 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Com8 (Const8 [c]))
	// cond:
	// result: (Const8  [^c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst8)
		v.AuxInt = ^c
		return true
	}
	return false
}
func rewriteValuegeneric_OpConstInterface_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ConstInterface)
	// cond:
	// result: (IMake     (ConstNil <typ.BytePtr>)     (ConstNil <typ.BytePtr>))
	for {
		v.reset(OpIMake)
		v0 := b.NewValue0(v.Pos, OpConstNil, typ.BytePtr)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpConstNil, typ.BytePtr)
		v.AddArg(v1)
		return true
	}
}
func rewriteValuegeneric_OpConstSlice_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ConstSlice)
	// cond: config.PtrSize == 4
	// result: (SliceMake     (ConstNil <v.Type.ElemType().PtrTo()>)     (Const32 <typ.Int> [0])     (Const32 <typ.Int> [0]))
	for {
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpSliceMake)
		v0 := b.NewValue0(v.Pos, OpConstNil, v.Type.ElemType().PtrTo())
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpConst32, typ.Int)
		v1.AuxInt = 0
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.Int)
		v2.AuxInt = 0
		v.AddArg(v2)
		return true
	}
	// match: (ConstSlice)
	// cond: config.PtrSize == 8
	// result: (SliceMake     (ConstNil <v.Type.ElemType().PtrTo()>)     (Const64 <typ.Int> [0])     (Const64 <typ.Int> [0]))
	for {
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpSliceMake)
		v0 := b.NewValue0(v.Pos, OpConstNil, v.Type.ElemType().PtrTo())
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.Int)
		v1.AuxInt = 0
		v.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.Int)
		v2.AuxInt = 0
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValuegeneric_OpConstString_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	fe := b.Func.fe
	_ = fe
	typ := &b.Func.Config.Types
	_ = typ
	// match: (ConstString {s})
	// cond: config.PtrSize == 4 && s.(string) == ""
	// result: (StringMake (ConstNil) (Const32 <typ.Int> [0]))
	for {
		s := v.Aux
		if !(config.PtrSize == 4 && s.(string) == "") {
			break
		}
		v.reset(OpStringMake)
		v0 := b.NewValue0(v.Pos, OpConstNil, typ.BytePtr)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpConst32, typ.Int)
		v1.AuxInt = 0
		v.AddArg(v1)
		return true
	}
	// match: (ConstString {s})
	// cond: config.PtrSize == 8 && s.(string) == ""
	// result: (StringMake (ConstNil) (Const64 <typ.Int> [0]))
	for {
		s := v.Aux
		if !(config.PtrSize == 8 && s.(string) == "") {
			break
		}
		v.reset(OpStringMake)
		v0 := b.NewValue0(v.Pos, OpConstNil, typ.BytePtr)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.Int)
		v1.AuxInt = 0
		v.AddArg(v1)
		return true
	}
	// match: (ConstString {s})
	// cond: config.PtrSize == 4 && s.(string) != ""
	// result: (StringMake     (Addr <typ.BytePtr> {fe.StringData(s.(string))}       (SB))     (Const32 <typ.Int> [int64(len(s.(string)))]))
	for {
		s := v.Aux
		if !(config.PtrSize == 4 && s.(string) != "") {
			break
		}
		v.reset(OpStringMake)
		v0 := b.NewValue0(v.Pos, OpAddr, typ.BytePtr)
		v0.Aux = fe.StringData(s.(string))
		v1 := b.NewValue0(v.Pos, OpSB, typ.Uintptr)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.Int)
		v2.AuxInt = int64(len(s.(string)))
		v.AddArg(v2)
		return true
	}
	// match: (ConstString {s})
	// cond: config.PtrSize == 8 && s.(string) != ""
	// result: (StringMake     (Addr <typ.BytePtr> {fe.StringData(s.(string))}       (SB))     (Const64 <typ.Int> [int64(len(s.(string)))]))
	for {
		s := v.Aux
		if !(config.PtrSize == 8 && s.(string) != "") {
			break
		}
		v.reset(OpStringMake)
		v0 := b.NewValue0(v.Pos, OpAddr, typ.BytePtr)
		v0.Aux = fe.StringData(s.(string))
		v1 := b.NewValue0(v.Pos, OpSB, typ.Uintptr)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.Int)
		v2.AuxInt = int64(len(s.(string)))
		v.AddArg(v2)
		return true
	}
	return false
}
func rewriteValuegeneric_OpConvert_0(v *Value) bool {
	// match: (Convert (Add64 (Convert ptr mem) off) mem)
	// cond:
	// result: (Add64 ptr off)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConvert {
			break
		}
		_ = v_0_0.Args[1]
		ptr := v_0_0.Args[0]
		mem := v_0_0.Args[1]
		off := v_0.Args[1]
		if mem != v.Args[1] {
			break
		}
		v.reset(OpAdd64)
		v.AddArg(ptr)
		v.AddArg(off)
		return true
	}
	// match: (Convert (Add64 off (Convert ptr mem)) mem)
	// cond:
	// result: (Add64 ptr off)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		off := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConvert {
			break
		}
		_ = v_0_1.Args[1]
		ptr := v_0_1.Args[0]
		mem := v_0_1.Args[1]
		if mem != v.Args[1] {
			break
		}
		v.reset(OpAdd64)
		v.AddArg(ptr)
		v.AddArg(off)
		return true
	}
	// match: (Convert (Convert ptr mem) mem)
	// cond:
	// result: ptr
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConvert {
			break
		}
		_ = v_0.Args[1]
		ptr := v_0.Args[0]
		mem := v_0.Args[1]
		if mem != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = ptr.Type
		v.AddArg(ptr)
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvt32Fto64F_0(v *Value) bool {
	// match: (Cvt32Fto64F (Const32F [c]))
	// cond:
	// result: (Const64F [c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst32F {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst64F)
		v.AuxInt = c
		return true
	}
	return false
}
func rewriteValuegeneric_OpCvt64Fto32F_0(v *Value) bool {
	// match: (Cvt64Fto32F (Const64F [c]))
	// cond:
	// result: (Const32F [f2i(float64(i2f32(c)))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst32F)
		v.AuxInt = f2i(float64(i2f32(c)))
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv16_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div16 (Const16 [c]) (Const16 [d]))
	// cond: d != 0
	// result: (Const16 [int64(int16(c)/int16(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c) / int16(d))
		return true
	}
	// match: (Div16 <t> n (Const16 [c]))
	// cond: c < 0 && c != -1<<15
	// result: (Neg16 (Div16 <t> n (Const16 <t> [-c])))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(c < 0 && c != -1<<15) {
			break
		}
		v.reset(OpNeg16)
		v0 := b.NewValue0(v.Pos, OpDiv16, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpConst16, t)
		v1.AuxInt = -c
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Div16 <t> x (Const16 [-1<<15]))
	// cond:
	// result: (Rsh16Ux64 (And16 <t> x (Neg16 <t> x)) (Const64 <typ.UInt64> [15]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.AuxInt != -1<<15 {
			break
		}
		v.reset(OpRsh16Ux64)
		v0 := b.NewValue0(v.Pos, OpAnd16, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpNeg16, t)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = 15
		v.AddArg(v2)
		return true
	}
	// match: (Div16 <t> n (Const16 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh16x64     (Add16 <t> n (Rsh16Ux64 <t> (Rsh16x64 <t> n (Const64 <typ.UInt64> [15])) (Const64 <typ.UInt64> [16-log2(c)])))     (Const64 <typ.UInt64> [log2(c)]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh16x64)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpRsh16Ux64, t)
		v2 := b.NewValue0(v.Pos, OpRsh16x64, t)
		v2.AddArg(n)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = 15
		v2.AddArg(v3)
		v1.AddArg(v2)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 16 - log2(c)
		v1.AddArg(v4)
		v0.AddArg(v1)
		v.AddArg(v0)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = log2(c)
		v.AddArg(v5)
		return true
	}
	// match: (Div16 <t> x (Const16 [c]))
	// cond: smagicOK(16,c)
	// result: (Sub16 <t>     (Rsh32x64 <t>       (Mul32 <typ.UInt32>         (Const32 <typ.UInt32> [int64(smagic(16,c).m)])         (SignExt16to32 x))       (Const64 <typ.UInt64> [16+smagic(16,c).s]))     (Rsh32x64 <t>       (SignExt16to32 x)       (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(smagicOK(16, c)) {
			break
		}
		v.reset(OpSub16)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int64(smagic(16, c).m)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v3.AddArg(x)
		v1.AddArg(v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 16 + smagic(16, c).s
		v0.AddArg(v4)
		v.AddArg(v0)
		v5 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v6 := b.NewValue0(v.Pos, OpSignExt16to32, typ.Int32)
		v6.AddArg(x)
		v5.AddArg(v6)
		v7 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v7.AuxInt = 31
		v5.AddArg(v7)
		v.AddArg(v5)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv16u_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div16u (Const16 [c]) (Const16 [d]))
	// cond: d != 0
	// result: (Const16 [int64(int16(uint16(c)/uint16(d)))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int64(int16(uint16(c) / uint16(d)))
		return true
	}
	// match: (Div16u n (Const16 [c]))
	// cond: isPowerOfTwo(c&0xffff)
	// result: (Rsh16Ux64 n (Const64 <typ.UInt64> [log2(c&0xffff)]))
	for {
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c & 0xffff)) {
			break
		}
		v.reset(OpRsh16Ux64)
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = log2(c & 0xffff)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u x (Const16 [c]))
	// cond: umagicOK(16, c) && config.RegSize == 8
	// result: (Trunc64to16     (Rsh64Ux64 <typ.UInt64>       (Mul64 <typ.UInt64>         (Const64 <typ.UInt64> [int64(1<<16+umagic(16,c).m)])         (ZeroExt16to64 x))       (Const64 <typ.UInt64> [16+umagic(16,c).s])))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(16, c) && config.RegSize == 8) {
			break
		}
		v.reset(OpTrunc64to16)
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64(1<<16 + umagic(16, c).m)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to64, typ.UInt64)
		v3.AddArg(x)
		v1.AddArg(v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 16 + umagic(16, c).s
		v0.AddArg(v4)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u x (Const16 [c]))
	// cond: umagicOK(16, c) && config.RegSize == 4 && umagic(16,c).m&1 == 0
	// result: (Trunc32to16     (Rsh32Ux64 <typ.UInt32>       (Mul32 <typ.UInt32>         (Const32 <typ.UInt32> [int64(1<<15+umagic(16,c).m/2)])         (ZeroExt16to32 x))       (Const64 <typ.UInt64> [16+umagic(16,c).s-1])))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(16, c) && config.RegSize == 4 && umagic(16, c).m&1 == 0) {
			break
		}
		v.reset(OpTrunc32to16)
		v0 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int64(1<<15 + umagic(16, c).m/2)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v3.AddArg(x)
		v1.AddArg(v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 16 + umagic(16, c).s - 1
		v0.AddArg(v4)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u x (Const16 [c]))
	// cond: umagicOK(16, c) && config.RegSize == 4 && c&1 == 0
	// result: (Trunc32to16     (Rsh32Ux64 <typ.UInt32>       (Mul32 <typ.UInt32>         (Const32 <typ.UInt32> [int64(1<<15+(umagic(16,c).m+1)/2)])         (Rsh32Ux64 <typ.UInt32> (ZeroExt16to32 x) (Const64 <typ.UInt64> [1])))       (Const64 <typ.UInt64> [16+umagic(16,c).s-2])))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(16, c) && config.RegSize == 4 && c&1 == 0) {
			break
		}
		v.reset(OpTrunc32to16)
		v0 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int64(1<<15 + (umagic(16, c).m+1)/2)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v4 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v4.AddArg(x)
		v3.AddArg(v4)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = 1
		v3.AddArg(v5)
		v1.AddArg(v3)
		v0.AddArg(v1)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = 16 + umagic(16, c).s - 2
		v0.AddArg(v6)
		v.AddArg(v0)
		return true
	}
	// match: (Div16u x (Const16 [c]))
	// cond: umagicOK(16, c) && config.RegSize == 4
	// result: (Trunc32to16     (Rsh32Ux64 <typ.UInt32>       (Avg32u         (Lsh32x64 <typ.UInt32> (ZeroExt16to32 x) (Const64 <typ.UInt64> [16]))         (Mul32 <typ.UInt32>           (Const32 <typ.UInt32> [int64(umagic(16,c).m)])           (ZeroExt16to32 x)))       (Const64 <typ.UInt64> [16+umagic(16,c).s-1])))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(16, c) && config.RegSize == 4) {
			break
		}
		v.reset(OpTrunc32to16)
		v0 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpAvg32u, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpLsh32x64, typ.UInt32)
		v3 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v3.AddArg(x)
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 16
		v2.AddArg(v4)
		v1.AddArg(v2)
		v5 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v6 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v6.AuxInt = int64(umagic(16, c).m)
		v5.AddArg(v6)
		v7 := b.NewValue0(v.Pos, OpZeroExt16to32, typ.UInt32)
		v7.AddArg(x)
		v5.AddArg(v7)
		v1.AddArg(v5)
		v0.AddArg(v1)
		v8 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v8.AuxInt = 16 + umagic(16, c).s - 1
		v0.AddArg(v8)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv32_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div32 (Const32 [c]) (Const32 [d]))
	// cond: d != 0
	// result: (Const32 [int64(int32(c)/int32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c) / int32(d))
		return true
	}
	// match: (Div32 <t> n (Const32 [c]))
	// cond: c < 0 && c != -1<<31
	// result: (Neg32 (Div32 <t> n (Const32 <t> [-c])))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(c < 0 && c != -1<<31) {
			break
		}
		v.reset(OpNeg32)
		v0 := b.NewValue0(v.Pos, OpDiv32, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpConst32, t)
		v1.AuxInt = -c
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Div32 <t> x (Const32 [-1<<31]))
	// cond:
	// result: (Rsh32Ux64 (And32 <t> x (Neg32 <t> x)) (Const64 <typ.UInt64> [31]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.AuxInt != -1<<31 {
			break
		}
		v.reset(OpRsh32Ux64)
		v0 := b.NewValue0(v.Pos, OpAnd32, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpNeg32, t)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = 31
		v.AddArg(v2)
		return true
	}
	// match: (Div32 <t> n (Const32 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh32x64     (Add32 <t> n (Rsh32Ux64 <t> (Rsh32x64 <t> n (Const64 <typ.UInt64> [31])) (Const64 <typ.UInt64> [32-log2(c)])))     (Const64 <typ.UInt64> [log2(c)]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh32x64)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpRsh32Ux64, t)
		v2 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v2.AddArg(n)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = 31
		v2.AddArg(v3)
		v1.AddArg(v2)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 32 - log2(c)
		v1.AddArg(v4)
		v0.AddArg(v1)
		v.AddArg(v0)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = log2(c)
		v.AddArg(v5)
		return true
	}
	// match: (Div32 <t> x (Const32 [c]))
	// cond: smagicOK(32,c) && config.RegSize == 8
	// result: (Sub32 <t>     (Rsh64x64 <t>       (Mul64 <typ.UInt64>         (Const64 <typ.UInt64> [int64(smagic(32,c).m)])         (SignExt32to64 x))       (Const64 <typ.UInt64> [32+smagic(32,c).s]))     (Rsh64x64 <t>       (SignExt32to64 x)       (Const64 <typ.UInt64> [63])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(smagicOK(32, c) && config.RegSize == 8) {
			break
		}
		v.reset(OpSub32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64(smagic(32, c).m)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v3.AddArg(x)
		v1.AddArg(v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 32 + smagic(32, c).s
		v0.AddArg(v4)
		v.AddArg(v0)
		v5 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v6 := b.NewValue0(v.Pos, OpSignExt32to64, typ.Int64)
		v6.AddArg(x)
		v5.AddArg(v6)
		v7 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v7.AuxInt = 63
		v5.AddArg(v7)
		v.AddArg(v5)
		return true
	}
	// match: (Div32 <t> x (Const32 [c]))
	// cond: smagicOK(32,c) && config.RegSize == 4 && smagic(32,c).m&1 == 0
	// result: (Sub32 <t>     (Rsh32x64 <t>       (Hmul32 <t>         (Const32 <typ.UInt32> [int64(int32(smagic(32,c).m/2))])         x)       (Const64 <typ.UInt64> [smagic(32,c).s-1]))     (Rsh32x64 <t>       x       (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(smagicOK(32, c) && config.RegSize == 4 && smagic(32, c).m&1 == 0) {
			break
		}
		v.reset(OpSub32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpHmul32, t)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int64(int32(smagic(32, c).m / 2))
		v1.AddArg(v2)
		v1.AddArg(x)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = smagic(32, c).s - 1
		v0.AddArg(v3)
		v.AddArg(v0)
		v4 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v4.AddArg(x)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = 31
		v4.AddArg(v5)
		v.AddArg(v4)
		return true
	}
	// match: (Div32 <t> x (Const32 [c]))
	// cond: smagicOK(32,c) && config.RegSize == 4 && smagic(32,c).m&1 != 0
	// result: (Sub32 <t>     (Rsh32x64 <t>       (Add32 <t>         (Hmul32 <t>           (Const32 <typ.UInt32> [int64(int32(smagic(32,c).m))])           x)         x)       (Const64 <typ.UInt64> [smagic(32,c).s]))     (Rsh32x64 <t>       x       (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(smagicOK(32, c) && config.RegSize == 4 && smagic(32, c).m&1 != 0) {
			break
		}
		v.reset(OpSub32)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpAdd32, t)
		v2 := b.NewValue0(v.Pos, OpHmul32, t)
		v3 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v3.AuxInt = int64(int32(smagic(32, c).m))
		v2.AddArg(v3)
		v2.AddArg(x)
		v1.AddArg(v2)
		v1.AddArg(x)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = smagic(32, c).s
		v0.AddArg(v4)
		v.AddArg(v0)
		v5 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v5.AddArg(x)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = 31
		v5.AddArg(v6)
		v.AddArg(v5)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv32F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Div32F (Const32F [c]) (Const32F [d]))
	// cond:
	// result: (Const32F [f2i(float64(i2f32(c) / i2f32(d)))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32F {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32F {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32F)
		v.AuxInt = f2i(float64(i2f32(c) / i2f32(d)))
		return true
	}
	// match: (Div32F x (Const32F <t> [c]))
	// cond: reciprocalExact32(float32(i2f(c)))
	// result: (Mul32F x (Const32F <t> [f2i(1/i2f(c))]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32F {
			break
		}
		t := v_1.Type
		c := v_1.AuxInt
		if !(reciprocalExact32(float32(i2f(c)))) {
			break
		}
		v.reset(OpMul32F)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst32F, t)
		v0.AuxInt = f2i(1 / i2f(c))
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv32u_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div32u (Const32 [c]) (Const32 [d]))
	// cond: d != 0
	// result: (Const32 [int64(int32(uint32(c)/uint32(d)))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int64(int32(uint32(c) / uint32(d)))
		return true
	}
	// match: (Div32u n (Const32 [c]))
	// cond: isPowerOfTwo(c&0xffffffff)
	// result: (Rsh32Ux64 n (Const64 <typ.UInt64> [log2(c&0xffffffff)]))
	for {
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c & 0xffffffff)) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = log2(c & 0xffffffff)
		v.AddArg(v0)
		return true
	}
	// match: (Div32u x (Const32 [c]))
	// cond: umagicOK(32, c) && config.RegSize == 4 && umagic(32,c).m&1 == 0
	// result: (Rsh32Ux64 <typ.UInt32>     (Hmul32u <typ.UInt32>       (Const32 <typ.UInt32> [int64(int32(1<<31+umagic(32,c).m/2))])       x)     (Const64 <typ.UInt64> [umagic(32,c).s-1]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(32, c) && config.RegSize == 4 && umagic(32, c).m&1 == 0) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, OpHmul32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v1.AuxInt = int64(int32(1<<31 + umagic(32, c).m/2))
		v0.AddArg(v1)
		v0.AddArg(x)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = umagic(32, c).s - 1
		v.AddArg(v2)
		return true
	}
	// match: (Div32u x (Const32 [c]))
	// cond: umagicOK(32, c) && config.RegSize == 4 && c&1 == 0
	// result: (Rsh32Ux64 <typ.UInt32>     (Hmul32u <typ.UInt32>       (Const32 <typ.UInt32> [int64(int32(1<<31+(umagic(32,c).m+1)/2))])       (Rsh32Ux64 <typ.UInt32> x (Const64 <typ.UInt64> [1])))     (Const64 <typ.UInt64> [umagic(32,c).s-2]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(32, c) && config.RegSize == 4 && c&1 == 0) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, OpHmul32u, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v1.AuxInt = int64(int32(1<<31 + (umagic(32, c).m+1)/2))
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = 1
		v2.AddArg(v3)
		v0.AddArg(v2)
		v.AddArg(v0)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = umagic(32, c).s - 2
		v.AddArg(v4)
		return true
	}
	// match: (Div32u x (Const32 [c]))
	// cond: umagicOK(32, c) && config.RegSize == 4
	// result: (Rsh32Ux64 <typ.UInt32>     (Avg32u       x       (Hmul32u <typ.UInt32>         (Const32 <typ.UInt32> [int64(int32(umagic(32,c).m))])         x))     (Const64 <typ.UInt64> [umagic(32,c).s-1]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(32, c) && config.RegSize == 4) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.Type = typ.UInt32
		v0 := b.NewValue0(v.Pos, OpAvg32u, typ.UInt32)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpHmul32u, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int64(int32(umagic(32, c).m))
		v1.AddArg(v2)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = umagic(32, c).s - 1
		v.AddArg(v3)
		return true
	}
	// match: (Div32u x (Const32 [c]))
	// cond: umagicOK(32, c) && config.RegSize == 8 && umagic(32,c).m&1 == 0
	// result: (Trunc64to32     (Rsh64Ux64 <typ.UInt64>       (Mul64 <typ.UInt64>         (Const64 <typ.UInt64> [int64(1<<31+umagic(32,c).m/2)])         (ZeroExt32to64 x))       (Const64 <typ.UInt64> [32+umagic(32,c).s-1])))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(32, c) && config.RegSize == 8 && umagic(32, c).m&1 == 0) {
			break
		}
		v.reset(OpTrunc64to32)
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64(1<<31 + umagic(32, c).m/2)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(x)
		v1.AddArg(v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 32 + umagic(32, c).s - 1
		v0.AddArg(v4)
		v.AddArg(v0)
		return true
	}
	// match: (Div32u x (Const32 [c]))
	// cond: umagicOK(32, c) && config.RegSize == 8 && c&1 == 0
	// result: (Trunc64to32     (Rsh64Ux64 <typ.UInt64>       (Mul64 <typ.UInt64>         (Const64 <typ.UInt64> [int64(1<<31+(umagic(32,c).m+1)/2)])         (Rsh64Ux64 <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt64> [1])))       (Const64 <typ.UInt64> [32+umagic(32,c).s-2])))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(32, c) && config.RegSize == 8 && c&1 == 0) {
			break
		}
		v.reset(OpTrunc64to32)
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64(1<<31 + (umagic(32, c).m+1)/2)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v4 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v4.AddArg(x)
		v3.AddArg(v4)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = 1
		v3.AddArg(v5)
		v1.AddArg(v3)
		v0.AddArg(v1)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = 32 + umagic(32, c).s - 2
		v0.AddArg(v6)
		v.AddArg(v0)
		return true
	}
	// match: (Div32u x (Const32 [c]))
	// cond: umagicOK(32, c) && config.RegSize == 8
	// result: (Trunc64to32     (Rsh64Ux64 <typ.UInt64>       (Avg64u         (Lsh64x64 <typ.UInt64> (ZeroExt32to64 x) (Const64 <typ.UInt64> [32]))         (Mul64 <typ.UInt64>           (Const64 <typ.UInt32> [int64(umagic(32,c).m)])           (ZeroExt32to64 x)))       (Const64 <typ.UInt64> [32+umagic(32,c).s-1])))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(32, c) && config.RegSize == 8) {
			break
		}
		v.reset(OpTrunc64to32)
		v0 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpAvg64u, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpLsh64x64, typ.UInt64)
		v3 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v3.AddArg(x)
		v2.AddArg(v3)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 32
		v2.AddArg(v4)
		v1.AddArg(v2)
		v5 := b.NewValue0(v.Pos, OpMul64, typ.UInt64)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt32)
		v6.AuxInt = int64(umagic(32, c).m)
		v5.AddArg(v6)
		v7 := b.NewValue0(v.Pos, OpZeroExt32to64, typ.UInt64)
		v7.AddArg(x)
		v5.AddArg(v7)
		v1.AddArg(v5)
		v0.AddArg(v1)
		v8 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v8.AuxInt = 32 + umagic(32, c).s - 1
		v0.AddArg(v8)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div64 (Const64 [c]) (Const64 [d]))
	// cond: d != 0
	// result: (Const64 [c/d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = c / d
		return true
	}
	// match: (Div64 <t> n (Const64 [c]))
	// cond: c < 0 && c != -1<<63
	// result: (Neg64 (Div64 <t> n (Const64 <t> [-c])))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(c < 0 && c != -1<<63) {
			break
		}
		v.reset(OpNeg64)
		v0 := b.NewValue0(v.Pos, OpDiv64, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpConst64, t)
		v1.AuxInt = -c
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Div64 <t> x (Const64 [-1<<63]))
	// cond:
	// result: (Rsh64Ux64 (And64 <t> x (Neg64 <t> x)) (Const64 <typ.UInt64> [63]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != -1<<63 {
			break
		}
		v.reset(OpRsh64Ux64)
		v0 := b.NewValue0(v.Pos, OpAnd64, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpNeg64, t)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = 63
		v.AddArg(v2)
		return true
	}
	// match: (Div64 <t> n (Const64 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh64x64     (Add64 <t> n (Rsh64Ux64 <t> (Rsh64x64 <t> n (Const64 <typ.UInt64> [63])) (Const64 <typ.UInt64> [64-log2(c)])))     (Const64 <typ.UInt64> [log2(c)]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh64x64)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpRsh64Ux64, t)
		v2 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v2.AddArg(n)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = 63
		v2.AddArg(v3)
		v1.AddArg(v2)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 64 - log2(c)
		v1.AddArg(v4)
		v0.AddArg(v1)
		v.AddArg(v0)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = log2(c)
		v.AddArg(v5)
		return true
	}
	// match: (Div64 <t> x (Const64 [c]))
	// cond: smagicOK(64,c) && smagic(64,c).m&1 == 0
	// result: (Sub64 <t>     (Rsh64x64 <t>       (Hmul64 <t>         (Const64 <typ.UInt64> [int64(smagic(64,c).m/2)])         x)       (Const64 <typ.UInt64> [smagic(64,c).s-1]))     (Rsh64x64 <t>       x       (Const64 <typ.UInt64> [63])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(smagicOK(64, c) && smagic(64, c).m&1 == 0) {
			break
		}
		v.reset(OpSub64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v1 := b.NewValue0(v.Pos, OpHmul64, t)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64(smagic(64, c).m / 2)
		v1.AddArg(v2)
		v1.AddArg(x)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = smagic(64, c).s - 1
		v0.AddArg(v3)
		v.AddArg(v0)
		v4 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v4.AddArg(x)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = 63
		v4.AddArg(v5)
		v.AddArg(v4)
		return true
	}
	// match: (Div64 <t> x (Const64 [c]))
	// cond: smagicOK(64,c) && smagic(64,c).m&1 != 0
	// result: (Sub64 <t>     (Rsh64x64 <t>       (Add64 <t>         (Hmul64 <t>           (Const64 <typ.UInt64> [int64(smagic(64,c).m)])           x)         x)       (Const64 <typ.UInt64> [smagic(64,c).s]))     (Rsh64x64 <t>       x       (Const64 <typ.UInt64> [63])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(smagicOK(64, c) && smagic(64, c).m&1 != 0) {
			break
		}
		v.reset(OpSub64)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v1 := b.NewValue0(v.Pos, OpAdd64, t)
		v2 := b.NewValue0(v.Pos, OpHmul64, t)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = int64(smagic(64, c).m)
		v2.AddArg(v3)
		v2.AddArg(x)
		v1.AddArg(v2)
		v1.AddArg(x)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = smagic(64, c).s
		v0.AddArg(v4)
		v.AddArg(v0)
		v5 := b.NewValue0(v.Pos, OpRsh64x64, t)
		v5.AddArg(x)
		v6 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v6.AuxInt = 63
		v5.AddArg(v6)
		v.AddArg(v5)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv64F_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Div64F (Const64F [c]) (Const64F [d]))
	// cond:
	// result: (Const64F [f2i(i2f(c) / i2f(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64F {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64F)
		v.AuxInt = f2i(i2f(c) / i2f(d))
		return true
	}
	// match: (Div64F x (Const64F <t> [c]))
	// cond: reciprocalExact64(i2f(c))
	// result: (Mul64F x (Const64F <t> [f2i(1/i2f(c))]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64F {
			break
		}
		t := v_1.Type
		c := v_1.AuxInt
		if !(reciprocalExact64(i2f(c))) {
			break
		}
		v.reset(OpMul64F)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64F, t)
		v0.AuxInt = f2i(1 / i2f(c))
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv64u_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div64u (Const64 [c]) (Const64 [d]))
	// cond: d != 0
	// result: (Const64 [int64(uint64(c)/uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64(uint64(c) / uint64(d))
		return true
	}
	// match: (Div64u n (Const64 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh64Ux64 n (Const64 <typ.UInt64> [log2(c)]))
	for {
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = log2(c)
		v.AddArg(v0)
		return true
	}
	// match: (Div64u n (Const64 [-1<<63]))
	// cond:
	// result: (Rsh64Ux64 n (Const64 <typ.UInt64> [63]))
	for {
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != -1<<63 {
			break
		}
		v.reset(OpRsh64Ux64)
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = 63
		v.AddArg(v0)
		return true
	}
	// match: (Div64u x (Const64 [c]))
	// cond: umagicOK(64, c) && config.RegSize == 8 && umagic(64,c).m&1 == 0
	// result: (Rsh64Ux64 <typ.UInt64>     (Hmul64u <typ.UInt64>       (Const64 <typ.UInt64> [int64(1<<63+umagic(64,c).m/2)])       x)     (Const64 <typ.UInt64> [umagic(64,c).s-1]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(64, c) && config.RegSize == 8 && umagic(64, c).m&1 == 0) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpHmul64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = int64(1<<63 + umagic(64, c).m/2)
		v0.AddArg(v1)
		v0.AddArg(x)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = umagic(64, c).s - 1
		v.AddArg(v2)
		return true
	}
	// match: (Div64u x (Const64 [c]))
	// cond: umagicOK(64, c) && config.RegSize == 8 && c&1 == 0
	// result: (Rsh64Ux64 <typ.UInt64>     (Hmul64u <typ.UInt64>       (Const64 <typ.UInt64> [int64(1<<63+(umagic(64,c).m+1)/2)])       (Rsh64Ux64 <typ.UInt64> x (Const64 <typ.UInt64> [1])))     (Const64 <typ.UInt64> [umagic(64,c).s-2]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(64, c) && config.RegSize == 8 && c&1 == 0) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpHmul64u, typ.UInt64)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = int64(1<<63 + (umagic(64, c).m+1)/2)
		v0.AddArg(v1)
		v2 := b.NewValue0(v.Pos, OpRsh64Ux64, typ.UInt64)
		v2.AddArg(x)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = 1
		v2.AddArg(v3)
		v0.AddArg(v2)
		v.AddArg(v0)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = umagic(64, c).s - 2
		v.AddArg(v4)
		return true
	}
	// match: (Div64u x (Const64 [c]))
	// cond: umagicOK(64, c) && config.RegSize == 8
	// result: (Rsh64Ux64 <typ.UInt64>     (Avg64u       x       (Hmul64u <typ.UInt64>         (Const64 <typ.UInt64> [int64(umagic(64,c).m)])         x))     (Const64 <typ.UInt64> [umagic(64,c).s-1]))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(64, c) && config.RegSize == 8) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.Type = typ.UInt64
		v0 := b.NewValue0(v.Pos, OpAvg64u, typ.UInt64)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpHmul64u, typ.UInt64)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = int64(umagic(64, c).m)
		v1.AddArg(v2)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = umagic(64, c).s - 1
		v.AddArg(v3)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv8_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div8 (Const8 [c]) (Const8 [d]))
	// cond: d != 0
	// result: (Const8  [int64(int8(c)/int8(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c) / int8(d))
		return true
	}
	// match: (Div8 <t> n (Const8 [c]))
	// cond: c < 0 && c != -1<<7
	// result: (Neg8  (Div8  <t> n (Const8  <t> [-c])))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		if !(c < 0 && c != -1<<7) {
			break
		}
		v.reset(OpNeg8)
		v0 := b.NewValue0(v.Pos, OpDiv8, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpConst8, t)
		v1.AuxInt = -c
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Div8 <t> x (Const8 [-1<<7 ]))
	// cond:
	// result: (Rsh8Ux64  (And8  <t> x (Neg8  <t> x)) (Const64 <typ.UInt64> [7 ]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.AuxInt != -1<<7 {
			break
		}
		v.reset(OpRsh8Ux64)
		v0 := b.NewValue0(v.Pos, OpAnd8, t)
		v0.AddArg(x)
		v1 := b.NewValue0(v.Pos, OpNeg8, t)
		v1.AddArg(x)
		v0.AddArg(v1)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v2.AuxInt = 7
		v.AddArg(v2)
		return true
	}
	// match: (Div8 <t> n (Const8 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Rsh8x64     (Add8  <t> n (Rsh8Ux64  <t> (Rsh8x64  <t> n (Const64 <typ.UInt64> [ 7])) (Const64 <typ.UInt64> [ 8-log2(c)])))     (Const64 <typ.UInt64> [log2(c)]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpRsh8x64)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpRsh8Ux64, t)
		v2 := b.NewValue0(v.Pos, OpRsh8x64, t)
		v2.AddArg(n)
		v3 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v3.AuxInt = 7
		v2.AddArg(v3)
		v1.AddArg(v2)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 8 - log2(c)
		v1.AddArg(v4)
		v0.AddArg(v1)
		v.AddArg(v0)
		v5 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v5.AuxInt = log2(c)
		v.AddArg(v5)
		return true
	}
	// match: (Div8 <t> x (Const8 [c]))
	// cond: smagicOK(8,c)
	// result: (Sub8 <t>     (Rsh32x64 <t>       (Mul32 <typ.UInt32>         (Const32 <typ.UInt32> [int64(smagic(8,c).m)])         (SignExt8to32 x))       (Const64 <typ.UInt64> [8+smagic(8,c).s]))     (Rsh32x64 <t>       (SignExt8to32 x)       (Const64 <typ.UInt64> [31])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		if !(smagicOK(8, c)) {
			break
		}
		v.reset(OpSub8)
		v.Type = t
		v0 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int64(smagic(8, c).m)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v3.AddArg(x)
		v1.AddArg(v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 8 + smagic(8, c).s
		v0.AddArg(v4)
		v.AddArg(v0)
		v5 := b.NewValue0(v.Pos, OpRsh32x64, t)
		v6 := b.NewValue0(v.Pos, OpSignExt8to32, typ.Int32)
		v6.AddArg(x)
		v5.AddArg(v6)
		v7 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v7.AuxInt = 31
		v5.AddArg(v7)
		v.AddArg(v5)
		return true
	}
	return false
}
func rewriteValuegeneric_OpDiv8u_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Div8u (Const8 [c]) (Const8 [d]))
	// cond: d != 0
	// result: (Const8  [int64(int8(uint8(c)/uint8(d)))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int64(int8(uint8(c) / uint8(d)))
		return true
	}
	// match: (Div8u n (Const8 [c]))
	// cond: isPowerOfTwo(c&0xff)
	// result: (Rsh8Ux64 n  (Const64 <typ.UInt64> [log2(c&0xff)]))
	for {
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c & 0xff)) {
			break
		}
		v.reset(OpRsh8Ux64)
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = log2(c & 0xff)
		v.AddArg(v0)
		return true
	}
	// match: (Div8u x (Const8 [c]))
	// cond: umagicOK(8, c)
	// result: (Trunc32to8     (Rsh32Ux64 <typ.UInt32>       (Mul32 <typ.UInt32>         (Const32 <typ.UInt32> [int64(1<<8+umagic(8,c).m)])         (ZeroExt8to32 x))       (Const64 <typ.UInt64> [8+umagic(8,c).s])))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		if !(umagicOK(8, c)) {
			break
		}
		v.reset(OpTrunc32to8)
		v0 := b.NewValue0(v.Pos, OpRsh32Ux64, typ.UInt32)
		v1 := b.NewValue0(v.Pos, OpMul32, typ.UInt32)
		v2 := b.NewValue0(v.Pos, OpConst32, typ.UInt32)
		v2.AuxInt = int64(1<<8 + umagic(8, c).m)
		v1.AddArg(v2)
		v3 := b.NewValue0(v.Pos, OpZeroExt8to32, typ.UInt32)
		v3.AddArg(x)
		v1.AddArg(v3)
		v0.AddArg(v1)
		v4 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v4.AuxInt = 8 + umagic(8, c).s
		v0.AddArg(v4)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpEq16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Eq16 x x)
	// cond:
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (Eq16 (Const16 <t> [c]) (Add16 (Const16 <t> [d]) x))
	// cond:
	// result: (Eq16 (Const16 <t> [int64(int16(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpEq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq16 (Const16 <t> [c]) (Add16 x (Const16 <t> [d])))
	// cond:
	// result: (Eq16 (Const16 <t> [int64(int16(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst16 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpEq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq16 (Add16 (Const16 <t> [d]) x) (Const16 <t> [c]))
	// cond:
	// result: (Eq16 (Const16 <t> [int64(int16(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst16 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpEq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq16 (Add16 x (Const16 <t> [d])) (Const16 <t> [c]))
	// cond:
	// result: (Eq16 (Const16 <t> [int64(int16(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst16 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpEq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq16 (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (ConstBool [b2i(c == d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c == d)
		return true
	}
	// match: (Eq16 (Const16 [d]) (Const16 [c]))
	// cond:
	// result: (ConstBool [b2i(c == d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c == d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpEq32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Eq32 x x)
	// cond:
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (Eq32 (Const32 <t> [c]) (Add32 (Const32 <t> [d]) x))
	// cond:
	// result: (Eq32 (Const32 <t> [int64(int32(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpEq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq32 (Const32 <t> [c]) (Add32 x (Const32 <t> [d])))
	// cond:
	// result: (Eq32 (Const32 <t> [int64(int32(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpEq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq32 (Add32 (Const32 <t> [d]) x) (Const32 <t> [c]))
	// cond:
	// result: (Eq32 (Const32 <t> [int64(int32(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpEq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq32 (Add32 x (Const32 <t> [d])) (Const32 <t> [c]))
	// cond:
	// result: (Eq32 (Const32 <t> [int64(int32(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpEq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq32 (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (ConstBool [b2i(c == d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c == d)
		return true
	}
	// match: (Eq32 (Const32 [d]) (Const32 [c]))
	// cond:
	// result: (ConstBool [b2i(c == d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c == d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpEq64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Eq64 x x)
	// cond:
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (Eq64 (Const64 <t> [c]) (Add64 (Const64 <t> [d]) x))
	// cond:
	// result: (Eq64 (Const64 <t> [c-d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpEq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c - d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq64 (Const64 <t> [c]) (Add64 x (Const64 <t> [d])))
	// cond:
	// result: (Eq64 (Const64 <t> [c-d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpEq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c - d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq64 (Add64 (Const64 <t> [d]) x) (Const64 <t> [c]))
	// cond:
	// result: (Eq64 (Const64 <t> [c-d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpEq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c - d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq64 (Add64 x (Const64 <t> [d])) (Const64 <t> [c]))
	// cond:
	// result: (Eq64 (Const64 <t> [c-d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpEq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c - d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (ConstBool [b2i(c == d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c == d)
		return true
	}
	// match: (Eq64 (Const64 [d]) (Const64 [c]))
	// cond:
	// result: (ConstBool [b2i(c == d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c == d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpEq8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Eq8 x x)
	// cond:
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (Eq8 (Const8 <t> [c]) (Add8 (Const8 <t> [d]) x))
	// cond:
	// result: (Eq8  (Const8 <t> [int64(int8(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpEq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq8 (Const8 <t> [c]) (Add8 x (Const8 <t> [d])))
	// cond:
	// result: (Eq8  (Const8 <t> [int64(int8(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst8 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpEq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq8 (Add8 (Const8 <t> [d]) x) (Const8 <t> [c]))
	// cond:
	// result: (Eq8  (Const8 <t> [int64(int8(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst8 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpEq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq8 (Add8 x (Const8 <t> [d])) (Const8 <t> [c]))
	// cond:
	// result: (Eq8  (Const8 <t> [int64(int8(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst8 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpEq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Eq8 (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (ConstBool [b2i(c == d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c == d)
		return true
	}
	// match: (Eq8 (Const8 [d]) (Const8 [c]))
	// cond:
	// result: (ConstBool [b2i(c == d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c == d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpEqB_0(v *Value) bool {
	// match: (EqB (ConstBool [c]) (ConstBool [d]))
	// cond:
	// result: (ConstBool [b2i(c == d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConstBool {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConstBool {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c == d)
		return true
	}
	// match: (EqB (ConstBool [d]) (ConstBool [c]))
	// cond:
	// result: (ConstBool [b2i(c == d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConstBool {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConstBool {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c == d)
		return true
	}
	// match: (EqB (ConstBool [0]) x)
	// cond:
	// result: (Not x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConstBool {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpNot)
		v.AddArg(x)
		return true
	}
	// match: (EqB x (ConstBool [0]))
	// cond:
	// result: (Not x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConstBool {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpNot)
		v.AddArg(x)
		return true
	}
	// match: (EqB (ConstBool [1]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConstBool {
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
	// match: (EqB x (ConstBool [1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConstBool {
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
	return false
}
func rewriteValuegeneric_OpEqInter_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (EqInter x y)
	// cond:
	// result: (EqPtr  (ITab x) (ITab y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpEqPtr)
		v0 := b.NewValue0(v.Pos, OpITab, typ.BytePtr)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpITab, typ.BytePtr)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValuegeneric_OpEqPtr_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (EqPtr p (ConstNil))
	// cond:
	// result: (Not (IsNonNil p))
	for {
		_ = v.Args[1]
		p := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConstNil {
			break
		}
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpIsNonNil, typ.Bool)
		v0.AddArg(p)
		v.AddArg(v0)
		return true
	}
	// match: (EqPtr (ConstNil) p)
	// cond:
	// result: (Not (IsNonNil p))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConstNil {
			break
		}
		p := v.Args[1]
		v.reset(OpNot)
		v0 := b.NewValue0(v.Pos, OpIsNonNil, typ.Bool)
		v0.AddArg(p)
		v.AddArg(v0)
		return true
	}
	// match: (EqPtr x x)
	// cond:
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (EqPtr (Addr {a} x) (Addr {b} x))
	// cond:
	// result: (ConstBool [b2i(a == b)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAddr {
			break
		}
		a := v_0.Aux
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAddr {
			break
		}
		b := v_1.Aux
		if x != v_1.Args[0] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = b2i(a == b)
		return true
	}
	// match: (EqPtr (Addr {b} x) (Addr {a} x))
	// cond:
	// result: (ConstBool [b2i(a == b)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAddr {
			break
		}
		b := v_0.Aux
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpAddr {
			break
		}
		a := v_1.Aux
		if x != v_1.Args[0] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = b2i(a == b)
		return true
	}
	return false
}
func rewriteValuegeneric_OpEqSlice_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (EqSlice x y)
	// cond:
	// result: (EqPtr  (SlicePtr x) (SlicePtr y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpEqPtr)
		v0 := b.NewValue0(v.Pos, OpSlicePtr, typ.BytePtr)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSlicePtr, typ.BytePtr)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValuegeneric_OpGeq16_0(v *Value) bool {
	// match: (Geq16 (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (ConstBool [b2i(c >= d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c >= d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpGeq16U_0(v *Value) bool {
	// match: (Geq16U (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (ConstBool [b2i(uint16(c) >= uint16(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint16(c) >= uint16(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpGeq32_0(v *Value) bool {
	// match: (Geq32 (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (ConstBool [b2i(c >= d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c >= d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpGeq32U_0(v *Value) bool {
	// match: (Geq32U (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (ConstBool [b2i(uint32(c) >= uint32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint32(c) >= uint32(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpGeq64_0(v *Value) bool {
	// match: (Geq64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (ConstBool [b2i(c >= d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c >= d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpGeq64U_0(v *Value) bool {
	// match: (Geq64U (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (ConstBool [b2i(uint64(c) >= uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint64(c) >= uint64(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpGeq8_0(v *Value) bool {
	// match: (Geq8 (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (ConstBool [b2i(c >= d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c >= d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpGeq8U_0(v *Value) bool {
	// match: (Geq8U (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (ConstBool [b2i(uint8(c)  >= uint8(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint8(c) >= uint8(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpGreater16_0(v *Value) bool {
	// match: (Greater16 (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (ConstBool [b2i(c > d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c > d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpGreater16U_0(v *Value) bool {
	// match: (Greater16U (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (ConstBool [b2i(uint16(c) > uint16(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint16(c) > uint16(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpGreater32_0(v *Value) bool {
	// match: (Greater32 (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (ConstBool [b2i(c > d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c > d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpGreater32U_0(v *Value) bool {
	// match: (Greater32U (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (ConstBool [b2i(uint32(c) > uint32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint32(c) > uint32(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpGreater64_0(v *Value) bool {
	// match: (Greater64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (ConstBool [b2i(c > d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c > d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpGreater64U_0(v *Value) bool {
	// match: (Greater64U (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (ConstBool [b2i(uint64(c) > uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint64(c) > uint64(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpGreater8_0(v *Value) bool {
	// match: (Greater8 (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (ConstBool [b2i(c > d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c > d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpGreater8U_0(v *Value) bool {
	// match: (Greater8U (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (ConstBool [b2i(uint8(c)  > uint8(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint8(c) > uint8(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpIMake_0(v *Value) bool {
	// match: (IMake typ (StructMake1 val))
	// cond:
	// result: (IMake typ val)
	for {
		_ = v.Args[1]
		typ := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpStructMake1 {
			break
		}
		val := v_1.Args[0]
		v.reset(OpIMake)
		v.AddArg(typ)
		v.AddArg(val)
		return true
	}
	// match: (IMake typ (ArrayMake1 val))
	// cond:
	// result: (IMake typ val)
	for {
		_ = v.Args[1]
		typ := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpArrayMake1 {
			break
		}
		val := v_1.Args[0]
		v.reset(OpIMake)
		v.AddArg(typ)
		v.AddArg(val)
		return true
	}
	return false
}
func rewriteValuegeneric_OpInterCall_0(v *Value) bool {
	// match: (InterCall [argsize] (Load (OffPtr [off] (ITab (IMake (Addr {itab} (SB)) _))) _) mem)
	// cond: devirt(v, itab, off) != nil
	// result: (StaticCall [argsize] {devirt(v, itab, off)} mem)
	for {
		argsize := v.AuxInt
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLoad {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpOffPtr {
			break
		}
		off := v_0_0.AuxInt
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpITab {
			break
		}
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpIMake {
			break
		}
		_ = v_0_0_0_0.Args[1]
		v_0_0_0_0_0 := v_0_0_0_0.Args[0]
		if v_0_0_0_0_0.Op != OpAddr {
			break
		}
		itab := v_0_0_0_0_0.Aux
		v_0_0_0_0_0_0 := v_0_0_0_0_0.Args[0]
		if v_0_0_0_0_0_0.Op != OpSB {
			break
		}
		mem := v.Args[1]
		if !(devirt(v, itab, off) != nil) {
			break
		}
		v.reset(OpStaticCall)
		v.AuxInt = argsize
		v.Aux = devirt(v, itab, off)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValuegeneric_OpIsInBounds_0(v *Value) bool {
	// match: (IsInBounds (ZeroExt8to32 _) (Const32 [c]))
	// cond: (1 << 8)  <= c
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to32 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !((1 << 8) <= c) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt8to64 _) (Const64 [c]))
	// cond: (1 << 8)  <= c
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to64 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !((1 << 8) <= c) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt16to32 _) (Const32 [c]))
	// cond: (1 << 16) <= c
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt16to32 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !((1 << 16) <= c) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt16to64 _) (Const64 [c]))
	// cond: (1 << 16) <= c
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt16to64 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !((1 << 16) <= c) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds x x)
	// cond:
	// result: (ConstBool [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 0
		return true
	}
	// match: (IsInBounds (And8 (Const8 [c]) _) (Const8 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst8 {
			break
		}
		c := v_0_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (And8 _ (Const8 [c])) (Const8 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst8 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt8to16 (And8 (Const8 [c]) _)) (Const16 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to16 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd8 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpConst8 {
			break
		}
		c := v_0_0_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt8to16 (And8 _ (Const8 [c]))) (Const16 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to16 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd8 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst8 {
			break
		}
		c := v_0_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt8to32 (And8 (Const8 [c]) _)) (Const32 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to32 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd8 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpConst8 {
			break
		}
		c := v_0_0_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	return false
}
func rewriteValuegeneric_OpIsInBounds_10(v *Value) bool {
	// match: (IsInBounds (ZeroExt8to32 (And8 _ (Const8 [c]))) (Const32 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to32 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd8 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst8 {
			break
		}
		c := v_0_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt8to64 (And8 (Const8 [c]) _)) (Const64 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to64 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd8 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpConst8 {
			break
		}
		c := v_0_0_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt8to64 (And8 _ (Const8 [c]))) (Const64 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to64 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd8 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst8 {
			break
		}
		c := v_0_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (And16 (Const16 [c]) _) (Const16 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst16 {
			break
		}
		c := v_0_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (And16 _ (Const16 [c])) (Const16 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst16 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt16to32 (And16 (Const16 [c]) _)) (Const32 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt16to32 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd16 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpConst16 {
			break
		}
		c := v_0_0_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt16to32 (And16 _ (Const16 [c]))) (Const32 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt16to32 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd16 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst16 {
			break
		}
		c := v_0_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt16to64 (And16 (Const16 [c]) _)) (Const64 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt16to64 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd16 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpConst16 {
			break
		}
		c := v_0_0_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt16to64 (And16 _ (Const16 [c]))) (Const64 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt16to64 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd16 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst16 {
			break
		}
		c := v_0_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (And32 (Const32 [c]) _) (Const32 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		c := v_0_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	return false
}
func rewriteValuegeneric_OpIsInBounds_20(v *Value) bool {
	// match: (IsInBounds (And32 _ (Const32 [c])) (Const32 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt32to64 (And32 (Const32 [c]) _)) (Const64 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt32to64 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd32 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpConst32 {
			break
		}
		c := v_0_0_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (ZeroExt32to64 (And32 _ (Const32 [c]))) (Const64 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt32to64 {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpAnd32 {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst32 {
			break
		}
		c := v_0_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (And64 (Const64 [c]) _) (Const64 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		c := v_0_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (And64 _ (Const64 [c])) (Const64 [d]))
	// cond: 0 <= c && c < d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c < d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (ConstBool [b2i(0 <= c && c < d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(0 <= c && c < d)
		return true
	}
	// match: (IsInBounds (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (ConstBool [b2i(0 <= c && c < d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(0 <= c && c < d)
		return true
	}
	// match: (IsInBounds (Mod32u _ y) y)
	// cond:
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpMod32u {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[1]
		if y != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsInBounds (Mod64u _ y) y)
	// cond:
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpMod64u {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[1]
		if y != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	return false
}
func rewriteValuegeneric_OpIsNonNil_0(v *Value) bool {
	// match: (IsNonNil (ConstNil))
	// cond:
	// result: (ConstBool [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConstNil {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpIsSliceInBounds_0(v *Value) bool {
	// match: (IsSliceInBounds x x)
	// cond:
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsSliceInBounds (And32 (Const32 [c]) _) (Const32 [d]))
	// cond: 0 <= c && c <= d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		c := v_0_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c <= d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsSliceInBounds (And32 _ (Const32 [c])) (Const32 [d]))
	// cond: 0 <= c && c <= d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c <= d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsSliceInBounds (And64 (Const64 [c]) _) (Const64 [d]))
	// cond: 0 <= c && c <= d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		c := v_0_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c <= d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsSliceInBounds (And64 _ (Const64 [c])) (Const64 [d]))
	// cond: 0 <= c && c <= d
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(0 <= c && c <= d) {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsSliceInBounds (Const32 [0]) _)
	// cond:
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsSliceInBounds (Const64 [0]) _)
	// cond:
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	// match: (IsSliceInBounds (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (ConstBool [b2i(0 <= c && c <= d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(0 <= c && c <= d)
		return true
	}
	// match: (IsSliceInBounds (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (ConstBool [b2i(0 <= c && c <= d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(0 <= c && c <= d)
		return true
	}
	// match: (IsSliceInBounds (SliceLen x) (SliceCap x))
	// cond:
	// result: (ConstBool [1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpSliceLen {
			break
		}
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSliceCap {
			break
		}
		if x != v_1.Args[0] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 1
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq16_0(v *Value) bool {
	// match: (Leq16 (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (ConstBool [b2i(c <= d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c <= d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq16U_0(v *Value) bool {
	// match: (Leq16U (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (ConstBool [b2i(uint16(c) <= uint16(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint16(c) <= uint16(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq32_0(v *Value) bool {
	// match: (Leq32 (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (ConstBool [b2i(c <= d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c <= d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq32U_0(v *Value) bool {
	// match: (Leq32U (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (ConstBool [b2i(uint32(c) <= uint32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint32(c) <= uint32(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq64_0(v *Value) bool {
	// match: (Leq64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (ConstBool [b2i(c <= d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c <= d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq64U_0(v *Value) bool {
	// match: (Leq64U (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (ConstBool [b2i(uint64(c) <= uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint64(c) <= uint64(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq8_0(v *Value) bool {
	// match: (Leq8 (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (ConstBool [b2i(c <= d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c <= d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLeq8U_0(v *Value) bool {
	// match: (Leq8U (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (ConstBool [b2i(uint8(c)  <= uint8(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint8(c) <= uint8(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess16_0(v *Value) bool {
	// match: (Less16 (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (ConstBool [b2i(c < d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c < d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess16U_0(v *Value) bool {
	// match: (Less16U (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (ConstBool [b2i(uint16(c) < uint16(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint16(c) < uint16(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess32_0(v *Value) bool {
	// match: (Less32 (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (ConstBool [b2i(c < d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c < d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess32U_0(v *Value) bool {
	// match: (Less32U (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (ConstBool [b2i(uint32(c) < uint32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint32(c) < uint32(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess64_0(v *Value) bool {
	// match: (Less64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (ConstBool [b2i(c < d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c < d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess64U_0(v *Value) bool {
	// match: (Less64U (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (ConstBool [b2i(uint64(c) < uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint64(c) < uint64(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess8_0(v *Value) bool {
	// match: (Less8 (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (ConstBool [b2i(c < d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c < d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLess8U_0(v *Value) bool {
	// match: (Less8U (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (ConstBool [b2i(uint8(c)  < uint8(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(uint8(c) < uint8(d))
		return true
	}
	return false
}
func rewriteValuegeneric_OpLoad_0(v *Value) bool {
	b := v.Block
	_ = b
	fe := b.Func.fe
	_ = fe
	// match: (Load <t1> p1 (Store {t2} p2 x _))
	// cond: isSamePtr(p1,p2) && t1.Compare(x.Type) == types.CMPeq && t1.Size() == t2.(*types.Type).Size()
	// result: x
	for {
		t1 := v.Type
		_ = v.Args[1]
		p1 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpStore {
			break
		}
		t2 := v_1.Aux
		_ = v_1.Args[2]
		p2 := v_1.Args[0]
		x := v_1.Args[1]
		if !(isSamePtr(p1, p2) && t1.Compare(x.Type) == types.CMPeq && t1.Size() == t2.(*types.Type).Size()) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Load <t> _ _)
	// cond: t.IsStruct() && t.NumFields() == 0 && fe.CanSSA(t)
	// result: (StructMake0)
	for {
		t := v.Type
		_ = v.Args[1]
		if !(t.IsStruct() && t.NumFields() == 0 && fe.CanSSA(t)) {
			break
		}
		v.reset(OpStructMake0)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsStruct() && t.NumFields() == 1 && fe.CanSSA(t)
	// result: (StructMake1     (Load <t.FieldType(0)> (OffPtr <t.FieldType(0).PtrTo()> [0] ptr) mem))
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(t.IsStruct() && t.NumFields() == 1 && fe.CanSSA(t)) {
			break
		}
		v.reset(OpStructMake1)
		v0 := b.NewValue0(v.Pos, OpLoad, t.FieldType(0))
		v1 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(0).PtrTo())
		v1.AuxInt = 0
		v1.AddArg(ptr)
		v0.AddArg(v1)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsStruct() && t.NumFields() == 2 && fe.CanSSA(t)
	// result: (StructMake2     (Load <t.FieldType(0)> (OffPtr <t.FieldType(0).PtrTo()> [0]             ptr) mem)     (Load <t.FieldType(1)> (OffPtr <t.FieldType(1).PtrTo()> [t.FieldOff(1)] ptr) mem))
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(t.IsStruct() && t.NumFields() == 2 && fe.CanSSA(t)) {
			break
		}
		v.reset(OpStructMake2)
		v0 := b.NewValue0(v.Pos, OpLoad, t.FieldType(0))
		v1 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(0).PtrTo())
		v1.AuxInt = 0
		v1.AddArg(ptr)
		v0.AddArg(v1)
		v0.AddArg(mem)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpLoad, t.FieldType(1))
		v3 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(1).PtrTo())
		v3.AuxInt = t.FieldOff(1)
		v3.AddArg(ptr)
		v2.AddArg(v3)
		v2.AddArg(mem)
		v.AddArg(v2)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsStruct() && t.NumFields() == 3 && fe.CanSSA(t)
	// result: (StructMake3     (Load <t.FieldType(0)> (OffPtr <t.FieldType(0).PtrTo()> [0]             ptr) mem)     (Load <t.FieldType(1)> (OffPtr <t.FieldType(1).PtrTo()> [t.FieldOff(1)] ptr) mem)     (Load <t.FieldType(2)> (OffPtr <t.FieldType(2).PtrTo()> [t.FieldOff(2)] ptr) mem))
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(t.IsStruct() && t.NumFields() == 3 && fe.CanSSA(t)) {
			break
		}
		v.reset(OpStructMake3)
		v0 := b.NewValue0(v.Pos, OpLoad, t.FieldType(0))
		v1 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(0).PtrTo())
		v1.AuxInt = 0
		v1.AddArg(ptr)
		v0.AddArg(v1)
		v0.AddArg(mem)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpLoad, t.FieldType(1))
		v3 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(1).PtrTo())
		v3.AuxInt = t.FieldOff(1)
		v3.AddArg(ptr)
		v2.AddArg(v3)
		v2.AddArg(mem)
		v.AddArg(v2)
		v4 := b.NewValue0(v.Pos, OpLoad, t.FieldType(2))
		v5 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(2).PtrTo())
		v5.AuxInt = t.FieldOff(2)
		v5.AddArg(ptr)
		v4.AddArg(v5)
		v4.AddArg(mem)
		v.AddArg(v4)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsStruct() && t.NumFields() == 4 && fe.CanSSA(t)
	// result: (StructMake4     (Load <t.FieldType(0)> (OffPtr <t.FieldType(0).PtrTo()> [0]             ptr) mem)     (Load <t.FieldType(1)> (OffPtr <t.FieldType(1).PtrTo()> [t.FieldOff(1)] ptr) mem)     (Load <t.FieldType(2)> (OffPtr <t.FieldType(2).PtrTo()> [t.FieldOff(2)] ptr) mem)     (Load <t.FieldType(3)> (OffPtr <t.FieldType(3).PtrTo()> [t.FieldOff(3)] ptr) mem))
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(t.IsStruct() && t.NumFields() == 4 && fe.CanSSA(t)) {
			break
		}
		v.reset(OpStructMake4)
		v0 := b.NewValue0(v.Pos, OpLoad, t.FieldType(0))
		v1 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(0).PtrTo())
		v1.AuxInt = 0
		v1.AddArg(ptr)
		v0.AddArg(v1)
		v0.AddArg(mem)
		v.AddArg(v0)
		v2 := b.NewValue0(v.Pos, OpLoad, t.FieldType(1))
		v3 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(1).PtrTo())
		v3.AuxInt = t.FieldOff(1)
		v3.AddArg(ptr)
		v2.AddArg(v3)
		v2.AddArg(mem)
		v.AddArg(v2)
		v4 := b.NewValue0(v.Pos, OpLoad, t.FieldType(2))
		v5 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(2).PtrTo())
		v5.AuxInt = t.FieldOff(2)
		v5.AddArg(ptr)
		v4.AddArg(v5)
		v4.AddArg(mem)
		v.AddArg(v4)
		v6 := b.NewValue0(v.Pos, OpLoad, t.FieldType(3))
		v7 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(3).PtrTo())
		v7.AuxInt = t.FieldOff(3)
		v7.AddArg(ptr)
		v6.AddArg(v7)
		v6.AddArg(mem)
		v.AddArg(v6)
		return true
	}
	// match: (Load <t> _ _)
	// cond: t.IsArray() && t.NumElem() == 0
	// result: (ArrayMake0)
	for {
		t := v.Type
		_ = v.Args[1]
		if !(t.IsArray() && t.NumElem() == 0) {
			break
		}
		v.reset(OpArrayMake0)
		return true
	}
	// match: (Load <t> ptr mem)
	// cond: t.IsArray() && t.NumElem() == 1 && fe.CanSSA(t)
	// result: (ArrayMake1 (Load <t.ElemType()> ptr mem))
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		mem := v.Args[1]
		if !(t.IsArray() && t.NumElem() == 1 && fe.CanSSA(t)) {
			break
		}
		v.reset(OpArrayMake1)
		v0 := b.NewValue0(v.Pos, OpLoad, t.ElemType())
		v0.AddArg(ptr)
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh16x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh16x16 <t> x (Const16 [c]))
	// cond:
	// result: (Lsh16x64  x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpLsh16x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint16(c))
		v.AddArg(v0)
		return true
	}
	// match: (Lsh16x16 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh16x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh16x32 <t> x (Const32 [c]))
	// cond:
	// result: (Lsh16x64  x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpLsh16x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint32(c))
		v.AddArg(v0)
		return true
	}
	// match: (Lsh16x32 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh16x64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh16x64 (Const16 [c]) (Const64 [d]))
	// cond:
	// result: (Const16 [int64(int16(c) << uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c) << uint64(d))
		return true
	}
	// match: (Lsh16x64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Lsh16x64 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
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
	// match: (Lsh16x64 <t> (Lsh16x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Lsh16x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh16x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpLsh16x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		return true
	}
	// match: (Lsh16x64 (Rsh16Ux64 (Lsh16x64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Lsh16x64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRsh16Ux64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpLsh16x64 {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c1 := v_0_0_1.AuxInt
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c3 := v_1.AuxInt
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpLsh16x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = c1 - c2 + c3
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh16x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh16x8 <t> x (Const8 [c]))
	// cond:
	// result: (Lsh16x64  x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpLsh16x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint8(c))
		v.AddArg(v0)
		return true
	}
	// match: (Lsh16x8 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh32x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh32x16 <t> x (Const16 [c]))
	// cond:
	// result: (Lsh32x64  x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpLsh32x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint16(c))
		v.AddArg(v0)
		return true
	}
	// match: (Lsh32x16 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh32x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh32x32 <t> x (Const32 [c]))
	// cond:
	// result: (Lsh32x64  x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpLsh32x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint32(c))
		v.AddArg(v0)
		return true
	}
	// match: (Lsh32x32 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh32x64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh32x64 (Const32 [c]) (Const64 [d]))
	// cond:
	// result: (Const32 [int64(int32(c) << uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c) << uint64(d))
		return true
	}
	// match: (Lsh32x64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Lsh32x64 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
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
	// match: (Lsh32x64 <t> (Lsh32x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Lsh32x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh32x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpLsh32x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		return true
	}
	// match: (Lsh32x64 (Rsh32Ux64 (Lsh32x64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Lsh32x64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRsh32Ux64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpLsh32x64 {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c1 := v_0_0_1.AuxInt
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c3 := v_1.AuxInt
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpLsh32x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = c1 - c2 + c3
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh32x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh32x8 <t> x (Const8 [c]))
	// cond:
	// result: (Lsh32x64  x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpLsh32x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint8(c))
		v.AddArg(v0)
		return true
	}
	// match: (Lsh32x8 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh64x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh64x16 <t> x (Const16 [c]))
	// cond:
	// result: (Lsh64x64  x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpLsh64x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint16(c))
		v.AddArg(v0)
		return true
	}
	// match: (Lsh64x16 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh64x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh64x32 <t> x (Const32 [c]))
	// cond:
	// result: (Lsh64x64  x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpLsh64x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint32(c))
		v.AddArg(v0)
		return true
	}
	// match: (Lsh64x32 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh64x64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh64x64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (Const64 [c << uint64(d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c << uint64(d)
		return true
	}
	// match: (Lsh64x64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Lsh64x64 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	// match: (Lsh64x64 _ (Const64 [c]))
	// cond: uint64(c) >= 64
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	// match: (Lsh64x64 <t> (Lsh64x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Lsh64x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpLsh64x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		return true
	}
	// match: (Lsh64x64 (Rsh64Ux64 (Lsh64x64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Lsh64x64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRsh64Ux64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c1 := v_0_0_1.AuxInt
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c3 := v_1.AuxInt
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpLsh64x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = c1 - c2 + c3
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh64x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh64x8 <t> x (Const8 [c]))
	// cond:
	// result: (Lsh64x64  x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpLsh64x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint8(c))
		v.AddArg(v0)
		return true
	}
	// match: (Lsh64x8 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh8x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh8x16 <t> x (Const16 [c]))
	// cond:
	// result: (Lsh8x64  x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpLsh8x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint16(c))
		v.AddArg(v0)
		return true
	}
	// match: (Lsh8x16 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh8x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh8x32 <t> x (Const32 [c]))
	// cond:
	// result: (Lsh8x64  x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpLsh8x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint32(c))
		v.AddArg(v0)
		return true
	}
	// match: (Lsh8x32 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh8x64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Lsh8x64 (Const8 [c]) (Const64 [d]))
	// cond:
	// result: (Const8  [int64(int8(c) << uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c) << uint64(d))
		return true
	}
	// match: (Lsh8x64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Lsh8x64 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	// match: (Lsh8x64 _ (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (Const8  [0])
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
	// match: (Lsh8x64 <t> (Lsh8x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Lsh8x64  x (Const64 <t> [c+d]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh8x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpLsh8x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		return true
	}
	// match: (Lsh8x64 (Rsh8Ux64 (Lsh8x64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Lsh8x64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRsh8Ux64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpLsh8x64 {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c1 := v_0_0_1.AuxInt
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c3 := v_1.AuxInt
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpLsh8x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = c1 - c2 + c3
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpLsh8x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Lsh8x8 <t> x (Const8 [c]))
	// cond:
	// result: (Lsh8x64  x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpLsh8x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint8(c))
		v.AddArg(v0)
		return true
	}
	// match: (Lsh8x8 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Mod16 (Const16 [c]) (Const16 [d]))
	// cond: d != 0
	// result: (Const16 [int64(int16(c % d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c % d))
		return true
	}
	// match: (Mod16 <t> n (Const16 [c]))
	// cond: c < 0 && c != -1<<15
	// result: (Mod16 <t> n (Const16 <t> [-c]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(c < 0 && c != -1<<15) {
			break
		}
		v.reset(OpMod16)
		v.Type = t
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = -c
		v.AddArg(v0)
		return true
	}
	// match: (Mod16 <t> x (Const16 [c]))
	// cond: x.Op != OpConst16 && (c > 0 || c == -1<<15)
	// result: (Sub16 x (Mul16 <t> (Div16  <t> x (Const16 <t> [c])) (Const16 <t> [c])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(x.Op != OpConst16 && (c > 0 || c == -1<<15)) {
			break
		}
		v.reset(OpSub16)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpMul16, t)
		v1 := b.NewValue0(v.Pos, OpDiv16, t)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpConst16, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpConst16, t)
		v3.AuxInt = c
		v0.AddArg(v3)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod16u_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Mod16u (Const16 [c]) (Const16 [d]))
	// cond: d != 0
	// result: (Const16 [int64(uint16(c) % uint16(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = int64(uint16(c) % uint16(d))
		return true
	}
	// match: (Mod16u <t> n (Const16 [c]))
	// cond: isPowerOfTwo(c&0xffff)
	// result: (And16 n (Const16 <t> [(c&0xffff)-1]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c & 0xffff)) {
			break
		}
		v.reset(OpAnd16)
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = (c & 0xffff) - 1
		v.AddArg(v0)
		return true
	}
	// match: (Mod16u <t> x (Const16 [c]))
	// cond: x.Op != OpConst16 && c > 0 && umagicOK(16,c)
	// result: (Sub16 x (Mul16 <t> (Div16u <t> x (Const16 <t> [c])) (Const16 <t> [c])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(x.Op != OpConst16 && c > 0 && umagicOK(16, c)) {
			break
		}
		v.reset(OpSub16)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpMul16, t)
		v1 := b.NewValue0(v.Pos, OpDiv16u, t)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpConst16, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpConst16, t)
		v3.AuxInt = c
		v0.AddArg(v3)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Mod32 (Const32 [c]) (Const32 [d]))
	// cond: d != 0
	// result: (Const32 [int64(int32(c % d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c % d))
		return true
	}
	// match: (Mod32 <t> n (Const32 [c]))
	// cond: c < 0 && c != -1<<31
	// result: (Mod32 <t> n (Const32 <t> [-c]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(c < 0 && c != -1<<31) {
			break
		}
		v.reset(OpMod32)
		v.Type = t
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = -c
		v.AddArg(v0)
		return true
	}
	// match: (Mod32 <t> x (Const32 [c]))
	// cond: x.Op != OpConst32 && (c > 0 || c == -1<<31)
	// result: (Sub32 x (Mul32 <t> (Div32  <t> x (Const32 <t> [c])) (Const32 <t> [c])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(x.Op != OpConst32 && (c > 0 || c == -1<<31)) {
			break
		}
		v.reset(OpSub32)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpMul32, t)
		v1 := b.NewValue0(v.Pos, OpDiv32, t)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpConst32, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpConst32, t)
		v3.AuxInt = c
		v0.AddArg(v3)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod32u_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Mod32u (Const32 [c]) (Const32 [d]))
	// cond: d != 0
	// result: (Const32 [int64(uint32(c) % uint32(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = int64(uint32(c) % uint32(d))
		return true
	}
	// match: (Mod32u <t> n (Const32 [c]))
	// cond: isPowerOfTwo(c&0xffffffff)
	// result: (And32 n (Const32 <t> [(c&0xffffffff)-1]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c & 0xffffffff)) {
			break
		}
		v.reset(OpAnd32)
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = (c & 0xffffffff) - 1
		v.AddArg(v0)
		return true
	}
	// match: (Mod32u <t> x (Const32 [c]))
	// cond: x.Op != OpConst32 && c > 0 && umagicOK(32,c)
	// result: (Sub32 x (Mul32 <t> (Div32u <t> x (Const32 <t> [c])) (Const32 <t> [c])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(x.Op != OpConst32 && c > 0 && umagicOK(32, c)) {
			break
		}
		v.reset(OpSub32)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpMul32, t)
		v1 := b.NewValue0(v.Pos, OpDiv32u, t)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpConst32, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpConst32, t)
		v3.AuxInt = c
		v0.AddArg(v3)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Mod64 (Const64 [c]) (Const64 [d]))
	// cond: d != 0
	// result: (Const64 [c % d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = c % d
		return true
	}
	// match: (Mod64 <t> n (Const64 [c]))
	// cond: c < 0 && c != -1<<63
	// result: (Mod64 <t> n (Const64 <t> [-c]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(c < 0 && c != -1<<63) {
			break
		}
		v.reset(OpMod64)
		v.Type = t
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = -c
		v.AddArg(v0)
		return true
	}
	// match: (Mod64 <t> x (Const64 [c]))
	// cond: x.Op != OpConst64 && (c > 0 || c == -1<<63)
	// result: (Sub64 x (Mul64 <t> (Div64  <t> x (Const64 <t> [c])) (Const64 <t> [c])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(x.Op != OpConst64 && (c > 0 || c == -1<<63)) {
			break
		}
		v.reset(OpSub64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpMul64, t)
		v1 := b.NewValue0(v.Pos, OpDiv64, t)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = c
		v0.AddArg(v3)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod64u_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Mod64u (Const64 [c]) (Const64 [d]))
	// cond: d != 0
	// result: (Const64 [int64(uint64(c) % uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = int64(uint64(c) % uint64(d))
		return true
	}
	// match: (Mod64u <t> n (Const64 [c]))
	// cond: isPowerOfTwo(c)
	// result: (And64 n (Const64 <t> [c-1]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpAnd64)
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c - 1
		v.AddArg(v0)
		return true
	}
	// match: (Mod64u <t> n (Const64 [-1<<63]))
	// cond:
	// result: (And64 n (Const64 <t> [1<<63-1]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != -1<<63 {
			break
		}
		v.reset(OpAnd64)
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = 1<<63 - 1
		v.AddArg(v0)
		return true
	}
	// match: (Mod64u <t> x (Const64 [c]))
	// cond: x.Op != OpConst64 && c > 0 && umagicOK(64,c)
	// result: (Sub64 x (Mul64 <t> (Div64u <t> x (Const64 <t> [c])) (Const64 <t> [c])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(x.Op != OpConst64 && c > 0 && umagicOK(64, c)) {
			break
		}
		v.reset(OpSub64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpMul64, t)
		v1 := b.NewValue0(v.Pos, OpDiv64u, t)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpConst64, t)
		v3.AuxInt = c
		v0.AddArg(v3)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Mod8 (Const8 [c]) (Const8 [d]))
	// cond: d != 0
	// result: (Const8  [int64(int8(c % d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c % d))
		return true
	}
	// match: (Mod8 <t> n (Const8 [c]))
	// cond: c < 0 && c != -1<<7
	// result: (Mod8  <t> n (Const8  <t> [-c]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		if !(c < 0 && c != -1<<7) {
			break
		}
		v.reset(OpMod8)
		v.Type = t
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = -c
		v.AddArg(v0)
		return true
	}
	// match: (Mod8 <t> x (Const8 [c]))
	// cond: x.Op != OpConst8  && (c > 0 || c == -1<<7)
	// result: (Sub8  x (Mul8  <t> (Div8   <t> x (Const8  <t> [c])) (Const8  <t> [c])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		if !(x.Op != OpConst8 && (c > 0 || c == -1<<7)) {
			break
		}
		v.reset(OpSub8)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpMul8, t)
		v1 := b.NewValue0(v.Pos, OpDiv8, t)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpConst8, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpConst8, t)
		v3.AuxInt = c
		v0.AddArg(v3)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMod8u_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Mod8u (Const8 [c]) (Const8 [d]))
	// cond: d != 0
	// result: (Const8  [int64(uint8(c) % uint8(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		if !(d != 0) {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = int64(uint8(c) % uint8(d))
		return true
	}
	// match: (Mod8u <t> n (Const8 [c]))
	// cond: isPowerOfTwo(c&0xff)
	// result: (And8 n (Const8 <t> [(c&0xff)-1]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c & 0xff)) {
			break
		}
		v.reset(OpAnd8)
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = (c & 0xff) - 1
		v.AddArg(v0)
		return true
	}
	// match: (Mod8u <t> x (Const8 [c]))
	// cond: x.Op != OpConst8  && c > 0 && umagicOK(8 ,c)
	// result: (Sub8  x (Mul8  <t> (Div8u  <t> x (Const8  <t> [c])) (Const8  <t> [c])))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		if !(x.Op != OpConst8 && c > 0 && umagicOK(8, c)) {
			break
		}
		v.reset(OpSub8)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpMul8, t)
		v1 := b.NewValue0(v.Pos, OpDiv8u, t)
		v1.AddArg(x)
		v2 := b.NewValue0(v.Pos, OpConst8, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v0.AddArg(v1)
		v3 := b.NewValue0(v.Pos, OpConst8, t)
		v3.AuxInt = c
		v0.AddArg(v3)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMul16_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mul16 (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (Const16 [int64(int16(c*d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c * d))
		return true
	}
	// match: (Mul16 (Const16 [d]) (Const16 [c]))
	// cond:
	// result: (Const16 [int64(int16(c*d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c * d))
		return true
	}
	// match: (Mul16 (Const16 [1]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
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
	// match: (Mul16 x (Const16 [1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
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
	// match: (Mul16 (Const16 [-1]) x)
	// cond:
	// result: (Neg16 x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		x := v.Args[1]
		v.reset(OpNeg16)
		v.AddArg(x)
		return true
	}
	// match: (Mul16 x (Const16 [-1]))
	// cond:
	// result: (Neg16 x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.AuxInt != -1 {
			break
		}
		v.reset(OpNeg16)
		v.AddArg(x)
		return true
	}
	// match: (Mul16 <t> n (Const16 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Lsh16x64 <t> n (Const64 <typ.UInt64> [log2(c)]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpLsh16x64)
		v.Type = t
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = log2(c)
		v.AddArg(v0)
		return true
	}
	// match: (Mul16 <t> (Const16 [c]) n)
	// cond: isPowerOfTwo(c)
	// result: (Lsh16x64 <t> n (Const64 <typ.UInt64> [log2(c)]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		n := v.Args[1]
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpLsh16x64)
		v.Type = t
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = log2(c)
		v.AddArg(v0)
		return true
	}
	// match: (Mul16 <t> n (Const16 [c]))
	// cond: t.IsSigned() && isPowerOfTwo(-c)
	// result: (Neg16 (Lsh16x64 <t> n (Const64 <typ.UInt64> [log2(-c)])))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		if !(t.IsSigned() && isPowerOfTwo(-c)) {
			break
		}
		v.reset(OpNeg16)
		v0 := b.NewValue0(v.Pos, OpLsh16x64, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = log2(-c)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Mul16 <t> (Const16 [c]) n)
	// cond: t.IsSigned() && isPowerOfTwo(-c)
	// result: (Neg16 (Lsh16x64 <t> n (Const64 <typ.UInt64> [log2(-c)])))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		n := v.Args[1]
		if !(t.IsSigned() && isPowerOfTwo(-c)) {
			break
		}
		v.reset(OpNeg16)
		v0 := b.NewValue0(v.Pos, OpLsh16x64, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = log2(-c)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMul16_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Mul16 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	// match: (Mul16 _ (Const16 [0]))
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	// match: (Mul16 (Const16 <t> [c]) (Mul16 (Const16 <t> [d]) x))
	// cond:
	// result: (Mul16 (Const16 <t> [int64(int16(c*d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpMul16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpMul16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c * d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Mul16 (Const16 <t> [c]) (Mul16 x (Const16 <t> [d])))
	// cond:
	// result: (Mul16 (Const16 <t> [int64(int16(c*d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpMul16 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst16 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpMul16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c * d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Mul16 (Mul16 (Const16 <t> [d]) x) (Const16 <t> [c]))
	// cond:
	// result: (Mul16 (Const16 <t> [int64(int16(c*d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpMul16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst16 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpMul16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c * d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Mul16 (Mul16 x (Const16 <t> [d])) (Const16 <t> [c]))
	// cond:
	// result: (Mul16 (Const16 <t> [int64(int16(c*d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpMul16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst16 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpMul16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c * d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMul32_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mul32 (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (Const32 [int64(int32(c*d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c * d))
		return true
	}
	// match: (Mul32 (Const32 [d]) (Const32 [c]))
	// cond:
	// result: (Const32 [int64(int32(c*d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c * d))
		return true
	}
	// match: (Mul32 (Const32 [1]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
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
	// match: (Mul32 x (Const32 [1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
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
	// match: (Mul32 (Const32 [-1]) x)
	// cond:
	// result: (Neg32 x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		x := v.Args[1]
		v.reset(OpNeg32)
		v.AddArg(x)
		return true
	}
	// match: (Mul32 x (Const32 [-1]))
	// cond:
	// result: (Neg32 x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.AuxInt != -1 {
			break
		}
		v.reset(OpNeg32)
		v.AddArg(x)
		return true
	}
	// match: (Mul32 <t> n (Const32 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Lsh32x64 <t> n (Const64 <typ.UInt64> [log2(c)]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpLsh32x64)
		v.Type = t
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = log2(c)
		v.AddArg(v0)
		return true
	}
	// match: (Mul32 <t> (Const32 [c]) n)
	// cond: isPowerOfTwo(c)
	// result: (Lsh32x64 <t> n (Const64 <typ.UInt64> [log2(c)]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		n := v.Args[1]
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpLsh32x64)
		v.Type = t
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = log2(c)
		v.AddArg(v0)
		return true
	}
	// match: (Mul32 <t> n (Const32 [c]))
	// cond: t.IsSigned() && isPowerOfTwo(-c)
	// result: (Neg32 (Lsh32x64 <t> n (Const64 <typ.UInt64> [log2(-c)])))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		if !(t.IsSigned() && isPowerOfTwo(-c)) {
			break
		}
		v.reset(OpNeg32)
		v0 := b.NewValue0(v.Pos, OpLsh32x64, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = log2(-c)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Mul32 <t> (Const32 [c]) n)
	// cond: t.IsSigned() && isPowerOfTwo(-c)
	// result: (Neg32 (Lsh32x64 <t> n (Const64 <typ.UInt64> [log2(-c)])))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		n := v.Args[1]
		if !(t.IsSigned() && isPowerOfTwo(-c)) {
			break
		}
		v.reset(OpNeg32)
		v0 := b.NewValue0(v.Pos, OpLsh32x64, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = log2(-c)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMul32_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Mul32 (Const32 <t> [c]) (Add32 <t> (Const32 <t> [d]) x))
	// cond:
	// result: (Add32 (Const32 <t> [int64(int32(c*d))]) (Mul32 <t> (Const32 <t> [c]) x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd32 {
			break
		}
		if v_1.Type != t {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c * d))
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpMul32, t)
		v2 := b.NewValue0(v.Pos, OpConst32, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v1.AddArg(x)
		v.AddArg(v1)
		return true
	}
	// match: (Mul32 (Const32 <t> [c]) (Add32 <t> x (Const32 <t> [d])))
	// cond:
	// result: (Add32 (Const32 <t> [int64(int32(c*d))]) (Mul32 <t> (Const32 <t> [c]) x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd32 {
			break
		}
		if v_1.Type != t {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c * d))
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpMul32, t)
		v2 := b.NewValue0(v.Pos, OpConst32, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v1.AddArg(x)
		v.AddArg(v1)
		return true
	}
	// match: (Mul32 (Add32 <t> (Const32 <t> [d]) x) (Const32 <t> [c]))
	// cond:
	// result: (Add32 (Const32 <t> [int64(int32(c*d))]) (Mul32 <t> (Const32 <t> [c]) x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		if v_0_0.Type != t {
			break
		}
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c * d))
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpMul32, t)
		v2 := b.NewValue0(v.Pos, OpConst32, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v1.AddArg(x)
		v.AddArg(v1)
		return true
	}
	// match: (Mul32 (Add32 <t> x (Const32 <t> [d])) (Const32 <t> [c]))
	// cond:
	// result: (Add32 (Const32 <t> [int64(int32(c*d))]) (Mul32 <t> (Const32 <t> [c]) x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		if v_0_1.Type != t {
			break
		}
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c * d))
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpMul32, t)
		v2 := b.NewValue0(v.Pos, OpConst32, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v1.AddArg(x)
		v.AddArg(v1)
		return true
	}
	// match: (Mul32 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	// match: (Mul32 _ (Const32 [0]))
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	// match: (Mul32 (Const32 <t> [c]) (Mul32 (Const32 <t> [d]) x))
	// cond:
	// result: (Mul32 (Const32 <t> [int64(int32(c*d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpMul32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpMul32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c * d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Mul32 (Const32 <t> [c]) (Mul32 x (Const32 <t> [d])))
	// cond:
	// result: (Mul32 (Const32 <t> [int64(int32(c*d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpMul32 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpMul32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c * d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Mul32 (Mul32 (Const32 <t> [d]) x) (Const32 <t> [c]))
	// cond:
	// result: (Mul32 (Const32 <t> [int64(int32(c*d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpMul32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpMul32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c * d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Mul32 (Mul32 x (Const32 <t> [d])) (Const32 <t> [c]))
	// cond:
	// result: (Mul32 (Const32 <t> [int64(int32(c*d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpMul32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpMul32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c * d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMul32F_0(v *Value) bool {
	// match: (Mul32F (Const32F [c]) (Const32F [d]))
	// cond:
	// result: (Const32F [f2i(float64(i2f32(c) * i2f32(d)))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32F {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32F {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32F)
		v.AuxInt = f2i(float64(i2f32(c) * i2f32(d)))
		return true
	}
	// match: (Mul32F (Const32F [d]) (Const32F [c]))
	// cond:
	// result: (Const32F [f2i(float64(i2f32(c) * i2f32(d)))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32F {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32F {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst32F)
		v.AuxInt = f2i(float64(i2f32(c) * i2f32(d)))
		return true
	}
	// match: (Mul32F x (Const32F [f2i(1)]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32F {
			break
		}
		if v_1.AuxInt != f2i(1) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Mul32F (Const32F [f2i(1)]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32F {
			break
		}
		if v_0.AuxInt != f2i(1) {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Mul32F x (Const32F [f2i(-1)]))
	// cond:
	// result: (Neg32F x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32F {
			break
		}
		if v_1.AuxInt != f2i(-1) {
			break
		}
		v.reset(OpNeg32F)
		v.AddArg(x)
		return true
	}
	// match: (Mul32F (Const32F [f2i(-1)]) x)
	// cond:
	// result: (Neg32F x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32F {
			break
		}
		if v_0.AuxInt != f2i(-1) {
			break
		}
		x := v.Args[1]
		v.reset(OpNeg32F)
		v.AddArg(x)
		return true
	}
	// match: (Mul32F x (Const32F [f2i(2)]))
	// cond:
	// result: (Add32F x x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32F {
			break
		}
		if v_1.AuxInt != f2i(2) {
			break
		}
		v.reset(OpAdd32F)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	// match: (Mul32F (Const32F [f2i(2)]) x)
	// cond:
	// result: (Add32F x x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32F {
			break
		}
		if v_0.AuxInt != f2i(2) {
			break
		}
		x := v.Args[1]
		v.reset(OpAdd32F)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMul64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mul64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (Const64 [c*d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c * d
		return true
	}
	// match: (Mul64 (Const64 [d]) (Const64 [c]))
	// cond:
	// result: (Const64 [c*d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c * d
		return true
	}
	// match: (Mul64 (Const64 [1]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
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
	// match: (Mul64 x (Const64 [1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
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
	// match: (Mul64 (Const64 [-1]) x)
	// cond:
	// result: (Neg64 x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		x := v.Args[1]
		v.reset(OpNeg64)
		v.AddArg(x)
		return true
	}
	// match: (Mul64 x (Const64 [-1]))
	// cond:
	// result: (Neg64 x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != -1 {
			break
		}
		v.reset(OpNeg64)
		v.AddArg(x)
		return true
	}
	// match: (Mul64 <t> n (Const64 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Lsh64x64 <t> n (Const64 <typ.UInt64> [log2(c)]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpLsh64x64)
		v.Type = t
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = log2(c)
		v.AddArg(v0)
		return true
	}
	// match: (Mul64 <t> (Const64 [c]) n)
	// cond: isPowerOfTwo(c)
	// result: (Lsh64x64 <t> n (Const64 <typ.UInt64> [log2(c)]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		n := v.Args[1]
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpLsh64x64)
		v.Type = t
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = log2(c)
		v.AddArg(v0)
		return true
	}
	// match: (Mul64 <t> n (Const64 [c]))
	// cond: t.IsSigned() && isPowerOfTwo(-c)
	// result: (Neg64 (Lsh64x64 <t> n (Const64 <typ.UInt64> [log2(-c)])))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(t.IsSigned() && isPowerOfTwo(-c)) {
			break
		}
		v.reset(OpNeg64)
		v0 := b.NewValue0(v.Pos, OpLsh64x64, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = log2(-c)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Mul64 <t> (Const64 [c]) n)
	// cond: t.IsSigned() && isPowerOfTwo(-c)
	// result: (Neg64 (Lsh64x64 <t> n (Const64 <typ.UInt64> [log2(-c)])))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		n := v.Args[1]
		if !(t.IsSigned() && isPowerOfTwo(-c)) {
			break
		}
		v.reset(OpNeg64)
		v0 := b.NewValue0(v.Pos, OpLsh64x64, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = log2(-c)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMul64_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Mul64 (Const64 <t> [c]) (Add64 <t> (Const64 <t> [d]) x))
	// cond:
	// result: (Add64 (Const64 <t> [c*d]) (Mul64 <t> (Const64 <t> [c]) x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd64 {
			break
		}
		if v_1.Type != t {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c * d
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpMul64, t)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v1.AddArg(x)
		v.AddArg(v1)
		return true
	}
	// match: (Mul64 (Const64 <t> [c]) (Add64 <t> x (Const64 <t> [d])))
	// cond:
	// result: (Add64 (Const64 <t> [c*d]) (Mul64 <t> (Const64 <t> [c]) x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd64 {
			break
		}
		if v_1.Type != t {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c * d
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpMul64, t)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v1.AddArg(x)
		v.AddArg(v1)
		return true
	}
	// match: (Mul64 (Add64 <t> (Const64 <t> [d]) x) (Const64 <t> [c]))
	// cond:
	// result: (Add64 (Const64 <t> [c*d]) (Mul64 <t> (Const64 <t> [c]) x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		if v_0_0.Type != t {
			break
		}
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c * d
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpMul64, t)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v1.AddArg(x)
		v.AddArg(v1)
		return true
	}
	// match: (Mul64 (Add64 <t> x (Const64 <t> [d])) (Const64 <t> [c]))
	// cond:
	// result: (Add64 (Const64 <t> [c*d]) (Mul64 <t> (Const64 <t> [c]) x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		t := v_0.Type
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.Type != t {
			break
		}
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c * d
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpMul64, t)
		v2 := b.NewValue0(v.Pos, OpConst64, t)
		v2.AuxInt = c
		v1.AddArg(v2)
		v1.AddArg(x)
		v.AddArg(v1)
		return true
	}
	// match: (Mul64 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	// match: (Mul64 _ (Const64 [0]))
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	// match: (Mul64 (Const64 <t> [c]) (Mul64 (Const64 <t> [d]) x))
	// cond:
	// result: (Mul64 (Const64 <t> [c*d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpMul64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpMul64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c * d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Mul64 (Const64 <t> [c]) (Mul64 x (Const64 <t> [d])))
	// cond:
	// result: (Mul64 (Const64 <t> [c*d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpMul64 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpMul64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c * d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Mul64 (Mul64 (Const64 <t> [d]) x) (Const64 <t> [c]))
	// cond:
	// result: (Mul64 (Const64 <t> [c*d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpMul64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpMul64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c * d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Mul64 (Mul64 x (Const64 <t> [d])) (Const64 <t> [c]))
	// cond:
	// result: (Mul64 (Const64 <t> [c*d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpMul64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpMul64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c * d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMul64F_0(v *Value) bool {
	// match: (Mul64F (Const64F [c]) (Const64F [d]))
	// cond:
	// result: (Const64F [f2i(i2f(c) * i2f(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64F {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64F)
		v.AuxInt = f2i(i2f(c) * i2f(d))
		return true
	}
	// match: (Mul64F (Const64F [d]) (Const64F [c]))
	// cond:
	// result: (Const64F [f2i(i2f(c) * i2f(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64F {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst64F)
		v.AuxInt = f2i(i2f(c) * i2f(d))
		return true
	}
	// match: (Mul64F x (Const64F [f2i(1)]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64F {
			break
		}
		if v_1.AuxInt != f2i(1) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Mul64F (Const64F [f2i(1)]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		if v_0.AuxInt != f2i(1) {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Mul64F x (Const64F [f2i(-1)]))
	// cond:
	// result: (Neg64F x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64F {
			break
		}
		if v_1.AuxInt != f2i(-1) {
			break
		}
		v.reset(OpNeg64F)
		v.AddArg(x)
		return true
	}
	// match: (Mul64F (Const64F [f2i(-1)]) x)
	// cond:
	// result: (Neg64F x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		if v_0.AuxInt != f2i(-1) {
			break
		}
		x := v.Args[1]
		v.reset(OpNeg64F)
		v.AddArg(x)
		return true
	}
	// match: (Mul64F x (Const64F [f2i(2)]))
	// cond:
	// result: (Add64F x x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64F {
			break
		}
		if v_1.AuxInt != f2i(2) {
			break
		}
		v.reset(OpAdd64F)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	// match: (Mul64F (Const64F [f2i(2)]) x)
	// cond:
	// result: (Add64F x x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		if v_0.AuxInt != f2i(2) {
			break
		}
		x := v.Args[1]
		v.reset(OpAdd64F)
		v.AddArg(x)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMul8_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Mul8 (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (Const8  [int64(int8(c*d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c * d))
		return true
	}
	// match: (Mul8 (Const8 [d]) (Const8 [c]))
	// cond:
	// result: (Const8  [int64(int8(c*d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c * d))
		return true
	}
	// match: (Mul8 (Const8 [1]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
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
	// match: (Mul8 x (Const8 [1]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
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
	// match: (Mul8 (Const8 [-1]) x)
	// cond:
	// result: (Neg8  x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		x := v.Args[1]
		v.reset(OpNeg8)
		v.AddArg(x)
		return true
	}
	// match: (Mul8 x (Const8 [-1]))
	// cond:
	// result: (Neg8  x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.AuxInt != -1 {
			break
		}
		v.reset(OpNeg8)
		v.AddArg(x)
		return true
	}
	// match: (Mul8 <t> n (Const8 [c]))
	// cond: isPowerOfTwo(c)
	// result: (Lsh8x64  <t> n (Const64 <typ.UInt64> [log2(c)]))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpLsh8x64)
		v.Type = t
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = log2(c)
		v.AddArg(v0)
		return true
	}
	// match: (Mul8 <t> (Const8 [c]) n)
	// cond: isPowerOfTwo(c)
	// result: (Lsh8x64  <t> n (Const64 <typ.UInt64> [log2(c)]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		n := v.Args[1]
		if !(isPowerOfTwo(c)) {
			break
		}
		v.reset(OpLsh8x64)
		v.Type = t
		v.AddArg(n)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = log2(c)
		v.AddArg(v0)
		return true
	}
	// match: (Mul8 <t> n (Const8 [c]))
	// cond: t.IsSigned() && isPowerOfTwo(-c)
	// result: (Neg8  (Lsh8x64  <t> n (Const64 <typ.UInt64> [log2(-c)])))
	for {
		t := v.Type
		_ = v.Args[1]
		n := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		if !(t.IsSigned() && isPowerOfTwo(-c)) {
			break
		}
		v.reset(OpNeg8)
		v0 := b.NewValue0(v.Pos, OpLsh8x64, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = log2(-c)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (Mul8 <t> (Const8 [c]) n)
	// cond: t.IsSigned() && isPowerOfTwo(-c)
	// result: (Neg8  (Lsh8x64  <t> n (Const64 <typ.UInt64> [log2(-c)])))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		n := v.Args[1]
		if !(t.IsSigned() && isPowerOfTwo(-c)) {
			break
		}
		v.reset(OpNeg8)
		v0 := b.NewValue0(v.Pos, OpLsh8x64, t)
		v0.AddArg(n)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v1.AuxInt = log2(-c)
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpMul8_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Mul8 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	// match: (Mul8 _ (Const8 [0]))
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	// match: (Mul8 (Const8 <t> [c]) (Mul8 (Const8 <t> [d]) x))
	// cond:
	// result: (Mul8  (Const8  <t> [int64(int8(c*d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpMul8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpMul8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c * d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Mul8 (Const8 <t> [c]) (Mul8 x (Const8 <t> [d])))
	// cond:
	// result: (Mul8  (Const8  <t> [int64(int8(c*d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpMul8 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst8 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpMul8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c * d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Mul8 (Mul8 (Const8 <t> [d]) x) (Const8 <t> [c]))
	// cond:
	// result: (Mul8  (Const8  <t> [int64(int8(c*d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpMul8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst8 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpMul8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c * d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Mul8 (Mul8 x (Const8 <t> [d])) (Const8 <t> [c]))
	// cond:
	// result: (Mul8  (Const8  <t> [int64(int8(c*d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpMul8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst8 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpMul8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c * d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeg16_0(v *Value) bool {
	// match: (Neg16 (Const16 [c]))
	// cond:
	// result: (Const16  [int64(-int16(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(-int16(c))
		return true
	}
	// match: (Neg16 (Sub16 x y))
	// cond:
	// result: (Sub16 y x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSub16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpSub16)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeg32_0(v *Value) bool {
	// match: (Neg32 (Const32 [c]))
	// cond:
	// result: (Const32  [int64(-int32(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(-int32(c))
		return true
	}
	// match: (Neg32 (Sub32 x y))
	// cond:
	// result: (Sub32 y x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSub32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpSub32)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeg32F_0(v *Value) bool {
	// match: (Neg32F (Const32F [c]))
	// cond: i2f(c) != 0
	// result: (Const32F [f2i(-i2f(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst32F {
			break
		}
		c := v_0.AuxInt
		if !(i2f(c) != 0) {
			break
		}
		v.reset(OpConst32F)
		v.AuxInt = f2i(-i2f(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeg64_0(v *Value) bool {
	// match: (Neg64 (Const64 [c]))
	// cond:
	// result: (Const64  [-c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst64)
		v.AuxInt = -c
		return true
	}
	// match: (Neg64 (Sub64 x y))
	// cond:
	// result: (Sub64 y x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSub64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpSub64)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeg64F_0(v *Value) bool {
	// match: (Neg64F (Const64F [c]))
	// cond: i2f(c) != 0
	// result: (Const64F [f2i(-i2f(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		c := v_0.AuxInt
		if !(i2f(c) != 0) {
			break
		}
		v.reset(OpConst64F)
		v.AuxInt = f2i(-i2f(c))
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeg8_0(v *Value) bool {
	// match: (Neg8 (Const8 [c]))
	// cond:
	// result: (Const8   [int64( -int8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(-int8(c))
		return true
	}
	// match: (Neg8 (Sub8 x y))
	// cond:
	// result: (Sub8  y x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSub8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpSub8)
		v.AddArg(y)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeq16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Neq16 x x)
	// cond:
	// result: (ConstBool [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 0
		return true
	}
	// match: (Neq16 (Const16 <t> [c]) (Add16 (Const16 <t> [d]) x))
	// cond:
	// result: (Neq16 (Const16 <t> [int64(int16(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpNeq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq16 (Const16 <t> [c]) (Add16 x (Const16 <t> [d])))
	// cond:
	// result: (Neq16 (Const16 <t> [int64(int16(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd16 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst16 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpNeq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq16 (Add16 (Const16 <t> [d]) x) (Const16 <t> [c]))
	// cond:
	// result: (Neq16 (Const16 <t> [int64(int16(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst16 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpNeq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq16 (Add16 x (Const16 <t> [d])) (Const16 <t> [c]))
	// cond:
	// result: (Neq16 (Const16 <t> [int64(int16(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst16 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpNeq16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq16 (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (ConstBool [b2i(c != d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c != d)
		return true
	}
	// match: (Neq16 (Const16 [d]) (Const16 [c]))
	// cond:
	// result: (ConstBool [b2i(c != d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c != d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeq32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Neq32 x x)
	// cond:
	// result: (ConstBool [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 0
		return true
	}
	// match: (Neq32 (Const32 <t> [c]) (Add32 (Const32 <t> [d]) x))
	// cond:
	// result: (Neq32 (Const32 <t> [int64(int32(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpNeq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq32 (Const32 <t> [c]) (Add32 x (Const32 <t> [d])))
	// cond:
	// result: (Neq32 (Const32 <t> [int64(int32(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd32 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpNeq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq32 (Add32 (Const32 <t> [d]) x) (Const32 <t> [c]))
	// cond:
	// result: (Neq32 (Const32 <t> [int64(int32(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpNeq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq32 (Add32 x (Const32 <t> [d])) (Const32 <t> [c]))
	// cond:
	// result: (Neq32 (Const32 <t> [int64(int32(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpNeq32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq32 (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (ConstBool [b2i(c != d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c != d)
		return true
	}
	// match: (Neq32 (Const32 [d]) (Const32 [c]))
	// cond:
	// result: (ConstBool [b2i(c != d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c != d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeq64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Neq64 x x)
	// cond:
	// result: (ConstBool [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 0
		return true
	}
	// match: (Neq64 (Const64 <t> [c]) (Add64 (Const64 <t> [d]) x))
	// cond:
	// result: (Neq64 (Const64 <t> [c-d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpNeq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c - d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq64 (Const64 <t> [c]) (Add64 x (Const64 <t> [d])))
	// cond:
	// result: (Neq64 (Const64 <t> [c-d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd64 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpNeq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c - d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq64 (Add64 (Const64 <t> [d]) x) (Const64 <t> [c]))
	// cond:
	// result: (Neq64 (Const64 <t> [c-d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpNeq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c - d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq64 (Add64 x (Const64 <t> [d])) (Const64 <t> [c]))
	// cond:
	// result: (Neq64 (Const64 <t> [c-d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpNeq64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c - d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (ConstBool [b2i(c != d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c != d)
		return true
	}
	// match: (Neq64 (Const64 [d]) (Const64 [c]))
	// cond:
	// result: (ConstBool [b2i(c != d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c != d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeq8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Neq8 x x)
	// cond:
	// result: (ConstBool [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConstBool)
		v.AuxInt = 0
		return true
	}
	// match: (Neq8 (Const8 <t> [c]) (Add8 (Const8 <t> [d]) x))
	// cond:
	// result: (Neq8 (Const8 <t> [int64(int8(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpNeq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq8 (Const8 <t> [c]) (Add8 x (Const8 <t> [d])))
	// cond:
	// result: (Neq8 (Const8 <t> [int64(int8(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAdd8 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst8 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpNeq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq8 (Add8 (Const8 <t> [d]) x) (Const8 <t> [c]))
	// cond:
	// result: (Neq8 (Const8 <t> [int64(int8(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst8 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpNeq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq8 (Add8 x (Const8 <t> [d])) (Const8 <t> [c]))
	// cond:
	// result: (Neq8 (Const8 <t> [int64(int8(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst8 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpNeq8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Neq8 (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (ConstBool [b2i(c != d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c != d)
		return true
	}
	// match: (Neq8 (Const8 [d]) (Const8 [c]))
	// cond:
	// result: (ConstBool [b2i(c != d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c != d)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeqB_0(v *Value) bool {
	// match: (NeqB (ConstBool [c]) (ConstBool [d]))
	// cond:
	// result: (ConstBool [b2i(c != d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConstBool {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConstBool {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c != d)
		return true
	}
	// match: (NeqB (ConstBool [d]) (ConstBool [c]))
	// cond:
	// result: (ConstBool [b2i(c != d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConstBool {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConstBool {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConstBool)
		v.AuxInt = b2i(c != d)
		return true
	}
	// match: (NeqB (ConstBool [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConstBool {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (NeqB x (ConstBool [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConstBool {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (NeqB (ConstBool [1]) x)
	// cond:
	// result: (Not x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConstBool {
			break
		}
		if v_0.AuxInt != 1 {
			break
		}
		x := v.Args[1]
		v.reset(OpNot)
		v.AddArg(x)
		return true
	}
	// match: (NeqB x (ConstBool [1]))
	// cond:
	// result: (Not x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConstBool {
			break
		}
		if v_1.AuxInt != 1 {
			break
		}
		v.reset(OpNot)
		v.AddArg(x)
		return true
	}
	// match: (NeqB (Not x) (Not y))
	// cond:
	// result: (NeqB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpNot {
			break
		}
		x := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpNot {
			break
		}
		y := v_1.Args[0]
		v.reset(OpNeqB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (NeqB (Not y) (Not x))
	// cond:
	// result: (NeqB x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpNot {
			break
		}
		y := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpNot {
			break
		}
		x := v_1.Args[0]
		v.reset(OpNeqB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeqInter_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (NeqInter x y)
	// cond:
	// result: (NeqPtr (ITab x) (ITab y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpNeqPtr)
		v0 := b.NewValue0(v.Pos, OpITab, typ.BytePtr)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpITab, typ.BytePtr)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValuegeneric_OpNeqPtr_0(v *Value) bool {
	// match: (NeqPtr p (ConstNil))
	// cond:
	// result: (IsNonNil p)
	for {
		_ = v.Args[1]
		p := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConstNil {
			break
		}
		v.reset(OpIsNonNil)
		v.AddArg(p)
		return true
	}
	// match: (NeqPtr (ConstNil) p)
	// cond:
	// result: (IsNonNil p)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConstNil {
			break
		}
		p := v.Args[1]
		v.reset(OpIsNonNil)
		v.AddArg(p)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNeqSlice_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (NeqSlice x y)
	// cond:
	// result: (NeqPtr (SlicePtr x) (SlicePtr y))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		y := v.Args[1]
		v.reset(OpNeqPtr)
		v0 := b.NewValue0(v.Pos, OpSlicePtr, typ.BytePtr)
		v0.AddArg(x)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpSlicePtr, typ.BytePtr)
		v1.AddArg(y)
		v.AddArg(v1)
		return true
	}
}
func rewriteValuegeneric_OpNilCheck_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	fe := b.Func.fe
	_ = fe
	// match: (NilCheck (GetG mem) mem)
	// cond:
	// result: mem
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpGetG {
			break
		}
		mem := v_0.Args[0]
		if mem != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	// match: (NilCheck (Load (OffPtr [c] (SP)) mem) mem)
	// cond: mem.Op == OpStaticCall 	&& isSameSym(mem.Aux, "runtime.newobject") 	&& c == config.ctxt.FixedFrameSize() + config.RegSize 	&& warnRule(fe.Debug_checknil() && v.Pos.Line() > 1, v, "removed nil check")
	// result: (Invalid)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLoad {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpOffPtr {
			break
		}
		c := v_0_0.AuxInt
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpSP {
			break
		}
		mem := v_0.Args[1]
		if mem != v.Args[1] {
			break
		}
		if !(mem.Op == OpStaticCall && isSameSym(mem.Aux, "runtime.newobject") && c == config.ctxt.FixedFrameSize()+config.RegSize && warnRule(fe.Debug_checknil() && v.Pos.Line() > 1, v, "removed nil check")) {
			break
		}
		v.reset(OpInvalid)
		return true
	}
	// match: (NilCheck (OffPtr (Load (OffPtr [c] (SP)) mem)) mem)
	// cond: mem.Op == OpStaticCall 	&& isSameSym(mem.Aux, "runtime.newobject") 	&& c == config.ctxt.FixedFrameSize() + config.RegSize 	&& warnRule(fe.Debug_checknil() && v.Pos.Line() > 1, v, "removed nil check")
	// result: (Invalid)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOffPtr {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpLoad {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpOffPtr {
			break
		}
		c := v_0_0_0.AuxInt
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpSP {
			break
		}
		mem := v_0_0.Args[1]
		if mem != v.Args[1] {
			break
		}
		if !(mem.Op == OpStaticCall && isSameSym(mem.Aux, "runtime.newobject") && c == config.ctxt.FixedFrameSize()+config.RegSize && warnRule(fe.Debug_checknil() && v.Pos.Line() > 1, v, "removed nil check")) {
			break
		}
		v.reset(OpInvalid)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNot_0(v *Value) bool {
	// match: (Not (Eq64 x y))
	// cond:
	// result: (Neq64 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpEq64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpNeq64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Eq32 x y))
	// cond:
	// result: (Neq32 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpEq32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpNeq32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Eq16 x y))
	// cond:
	// result: (Neq16 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpEq16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpNeq16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Eq8 x y))
	// cond:
	// result: (Neq8  x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpEq8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpNeq8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (EqB x y))
	// cond:
	// result: (NeqB  x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpEqB {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpNeqB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Neq64 x y))
	// cond:
	// result: (Eq64 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpNeq64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpEq64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Neq32 x y))
	// cond:
	// result: (Eq32 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpNeq32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpEq32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Neq16 x y))
	// cond:
	// result: (Eq16 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpNeq16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpEq16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Neq8 x y))
	// cond:
	// result: (Eq8  x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpNeq8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpEq8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (NeqB x y))
	// cond:
	// result: (EqB  x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpNeqB {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpEqB)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNot_10(v *Value) bool {
	// match: (Not (Greater64 x y))
	// cond:
	// result: (Leq64 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGreater64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLeq64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Greater32 x y))
	// cond:
	// result: (Leq32 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGreater32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLeq32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Greater16 x y))
	// cond:
	// result: (Leq16 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGreater16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLeq16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Greater8 x y))
	// cond:
	// result: (Leq8  x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGreater8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLeq8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Greater64U x y))
	// cond:
	// result: (Leq64U x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGreater64U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLeq64U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Greater32U x y))
	// cond:
	// result: (Leq32U x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGreater32U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLeq32U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Greater16U x y))
	// cond:
	// result: (Leq16U x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGreater16U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLeq16U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Greater8U x y))
	// cond:
	// result: (Leq8U  x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGreater8U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLeq8U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Geq64 x y))
	// cond:
	// result: (Less64 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGeq64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLess64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Geq32 x y))
	// cond:
	// result: (Less32 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGeq32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLess32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNot_20(v *Value) bool {
	// match: (Not (Geq16 x y))
	// cond:
	// result: (Less16 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGeq16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLess16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Geq8 x y))
	// cond:
	// result: (Less8  x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGeq8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLess8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Geq64U x y))
	// cond:
	// result: (Less64U x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGeq64U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLess64U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Geq32U x y))
	// cond:
	// result: (Less32U x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGeq32U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLess32U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Geq16U x y))
	// cond:
	// result: (Less16U x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGeq16U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLess16U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Geq8U x y))
	// cond:
	// result: (Less8U  x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpGeq8U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpLess8U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Less64 x y))
	// cond:
	// result: (Geq64 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLess64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGeq64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Less32 x y))
	// cond:
	// result: (Geq32 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLess32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGeq32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Less16 x y))
	// cond:
	// result: (Geq16 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLess16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGeq16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Less8 x y))
	// cond:
	// result: (Geq8  x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLess8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGeq8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNot_30(v *Value) bool {
	// match: (Not (Less64U x y))
	// cond:
	// result: (Geq64U x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLess64U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGeq64U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Less32U x y))
	// cond:
	// result: (Geq32U x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLess32U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGeq32U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Less16U x y))
	// cond:
	// result: (Geq16U x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLess16U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGeq16U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Less8U x y))
	// cond:
	// result: (Geq8U  x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLess8U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGeq8U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Leq64 x y))
	// cond:
	// result: (Greater64 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLeq64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGreater64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Leq32 x y))
	// cond:
	// result: (Greater32 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLeq32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGreater32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Leq16 x y))
	// cond:
	// result: (Greater16 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLeq16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGreater16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Leq8 x y))
	// cond:
	// result: (Greater8 x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLeq8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGreater8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Leq64U x y))
	// cond:
	// result: (Greater64U x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLeq64U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGreater64U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Leq32U x y))
	// cond:
	// result: (Greater32U x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLeq32U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGreater32U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpNot_40(v *Value) bool {
	// match: (Not (Leq16U x y))
	// cond:
	// result: (Greater16U x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLeq16U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGreater16U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Not (Leq8U x y))
	// cond:
	// result: (Greater8U  x y)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpLeq8U {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		v.reset(OpGreater8U)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOffPtr_0(v *Value) bool {
	// match: (OffPtr (OffPtr p [b]) [a])
	// cond:
	// result: (OffPtr p [a+b])
	for {
		a := v.AuxInt
		v_0 := v.Args[0]
		if v_0.Op != OpOffPtr {
			break
		}
		b := v_0.AuxInt
		p := v_0.Args[0]
		v.reset(OpOffPtr)
		v.AuxInt = a + b
		v.AddArg(p)
		return true
	}
	// match: (OffPtr p [0])
	// cond: v.Type.Compare(p.Type) == types.CMPeq
	// result: p
	for {
		if v.AuxInt != 0 {
			break
		}
		p := v.Args[0]
		if !(v.Type.Compare(p.Type) == types.CMPeq) {
			break
		}
		v.reset(OpCopy)
		v.Type = p.Type
		v.AddArg(p)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr16_0(v *Value) bool {
	// match: (Or16 (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (Const16 [int64(int16(c|d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c | d))
		return true
	}
	// match: (Or16 (Const16 [d]) (Const16 [c]))
	// cond:
	// result: (Const16 [int64(int16(c|d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c | d))
		return true
	}
	// match: (Or16 x x)
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
	// match: (Or16 (Const16 [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Or16 x (Const16 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Or16 (Const16 [-1]) _)
	// cond:
	// result: (Const16 [-1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = -1
		return true
	}
	// match: (Or16 _ (Const16 [-1]))
	// cond:
	// result: (Const16 [-1])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.AuxInt != -1 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = -1
		return true
	}
	// match: (Or16 x (Or16 x y))
	// cond:
	// result: (Or16 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr16 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpOr16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Or16 x (Or16 y x))
	// cond:
	// result: (Or16 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr16 {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpOr16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Or16 (Or16 x y) x)
	// cond:
	// result: (Or16 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpOr16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr16_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Or16 (Or16 y x) x)
	// cond:
	// result: (Or16 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr16 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpOr16)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Or16 (And16 x (Const16 [c2])) (Const16 <t> [c1]))
	// cond: ^(c1 | c2) == 0
	// result: (Or16 (Const16 <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst16 {
			break
		}
		c2 := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		t := v_1.Type
		c1 := v_1.AuxInt
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or16 (And16 (Const16 [c2]) x) (Const16 <t> [c1]))
	// cond: ^(c1 | c2) == 0
	// result: (Or16 (Const16 <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst16 {
			break
		}
		c2 := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		t := v_1.Type
		c1 := v_1.AuxInt
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or16 (Const16 <t> [c1]) (And16 x (Const16 [c2])))
	// cond: ^(c1 | c2) == 0
	// result: (Or16 (Const16 <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c1 := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst16 {
			break
		}
		c2 := v_1_1.AuxInt
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or16 (Const16 <t> [c1]) (And16 (Const16 [c2]) x))
	// cond: ^(c1 | c2) == 0
	// result: (Or16 (Const16 <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c1 := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		c2 := v_1_0.AuxInt
		x := v_1.Args[1]
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or16 (Or16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Or16 i (Or16 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr16 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpOr16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or16 (Or16 z i:(Const16 <t>)) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Or16 i (Or16 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr16 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpOr16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or16 x (Or16 i:(Const16 <t>) z))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Or16 i (Or16 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr16 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpOr16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or16 x (Or16 z i:(Const16 <t>)))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Or16 i (Or16 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr16 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpOr16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or16 (Const16 <t> [c]) (Or16 (Const16 <t> [d]) x))
	// cond:
	// result: (Or16 (Const16 <t> [int64(int16(c|d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpOr16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c | d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr16_20(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Or16 (Const16 <t> [c]) (Or16 x (Const16 <t> [d])))
	// cond:
	// result: (Or16 (Const16 <t> [int64(int16(c|d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpOr16 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst16 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c | d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or16 (Or16 (Const16 <t> [d]) x) (Const16 <t> [c]))
	// cond:
	// result: (Or16 (Const16 <t> [int64(int16(c|d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst16 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c | d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or16 (Or16 x (Const16 <t> [d])) (Const16 <t> [c]))
	// cond:
	// result: (Or16 (Const16 <t> [int64(int16(c|d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst16 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c | d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr32_0(v *Value) bool {
	// match: (Or32 (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (Const32 [int64(int32(c|d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c | d))
		return true
	}
	// match: (Or32 (Const32 [d]) (Const32 [c]))
	// cond:
	// result: (Const32 [int64(int32(c|d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c | d))
		return true
	}
	// match: (Or32 x x)
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
	// match: (Or32 (Const32 [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Or32 x (Const32 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Or32 (Const32 [-1]) _)
	// cond:
	// result: (Const32 [-1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = -1
		return true
	}
	// match: (Or32 _ (Const32 [-1]))
	// cond:
	// result: (Const32 [-1])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.AuxInt != -1 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = -1
		return true
	}
	// match: (Or32 x (Or32 x y))
	// cond:
	// result: (Or32 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr32 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpOr32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Or32 x (Or32 y x))
	// cond:
	// result: (Or32 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr32 {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpOr32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Or32 (Or32 x y) x)
	// cond:
	// result: (Or32 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpOr32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr32_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Or32 (Or32 y x) x)
	// cond:
	// result: (Or32 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr32 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpOr32)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Or32 (And32 x (Const32 [c2])) (Const32 <t> [c1]))
	// cond: ^(c1 | c2) == 0
	// result: (Or32 (Const32 <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		c2 := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		t := v_1.Type
		c1 := v_1.AuxInt
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or32 (And32 (Const32 [c2]) x) (Const32 <t> [c1]))
	// cond: ^(c1 | c2) == 0
	// result: (Or32 (Const32 <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		c2 := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		t := v_1.Type
		c1 := v_1.AuxInt
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or32 (Const32 <t> [c1]) (And32 x (Const32 [c2])))
	// cond: ^(c1 | c2) == 0
	// result: (Or32 (Const32 <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c1 := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32 {
			break
		}
		c2 := v_1_1.AuxInt
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or32 (Const32 <t> [c1]) (And32 (Const32 [c2]) x))
	// cond: ^(c1 | c2) == 0
	// result: (Or32 (Const32 <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c1 := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		c2 := v_1_0.AuxInt
		x := v_1.Args[1]
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or32 (Or32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Or32 i (Or32 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr32 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpOr32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or32 (Or32 z i:(Const32 <t>)) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Or32 i (Or32 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr32 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpOr32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or32 x (Or32 i:(Const32 <t>) z))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Or32 i (Or32 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr32 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpOr32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or32 x (Or32 z i:(Const32 <t>)))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Or32 i (Or32 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr32 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpOr32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or32 (Const32 <t> [c]) (Or32 (Const32 <t> [d]) x))
	// cond:
	// result: (Or32 (Const32 <t> [int64(int32(c|d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpOr32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpOr32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c | d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr32_20(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Or32 (Const32 <t> [c]) (Or32 x (Const32 <t> [d])))
	// cond:
	// result: (Or32 (Const32 <t> [int64(int32(c|d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpOr32 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpOr32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c | d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or32 (Or32 (Const32 <t> [d]) x) (Const32 <t> [c]))
	// cond:
	// result: (Or32 (Const32 <t> [int64(int32(c|d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c | d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or32 (Or32 x (Const32 <t> [d])) (Const32 <t> [c]))
	// cond:
	// result: (Or32 (Const32 <t> [int64(int32(c|d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c | d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr64_0(v *Value) bool {
	// match: (Or64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (Const64 [c|d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c | d
		return true
	}
	// match: (Or64 (Const64 [d]) (Const64 [c]))
	// cond:
	// result: (Const64 [c|d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c | d
		return true
	}
	// match: (Or64 x x)
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
	// match: (Or64 (Const64 [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Or64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Or64 (Const64 [-1]) _)
	// cond:
	// result: (Const64 [-1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = -1
		return true
	}
	// match: (Or64 _ (Const64 [-1]))
	// cond:
	// result: (Const64 [-1])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != -1 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = -1
		return true
	}
	// match: (Or64 x (Or64 x y))
	// cond:
	// result: (Or64 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr64 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpOr64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Or64 x (Or64 y x))
	// cond:
	// result: (Or64 x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr64 {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpOr64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Or64 (Or64 x y) x)
	// cond:
	// result: (Or64 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpOr64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr64_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Or64 (Or64 y x) x)
	// cond:
	// result: (Or64 x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr64 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpOr64)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Or64 (And64 x (Const64 [c2])) (Const64 <t> [c1]))
	// cond: ^(c1 | c2) == 0
	// result: (Or64 (Const64 <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		c1 := v_1.AuxInt
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or64 (And64 (Const64 [c2]) x) (Const64 <t> [c1]))
	// cond: ^(c1 | c2) == 0
	// result: (Or64 (Const64 <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		c2 := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		c1 := v_1.AuxInt
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or64 (Const64 <t> [c1]) (And64 x (Const64 [c2])))
	// cond: ^(c1 | c2) == 0
	// result: (Or64 (Const64 <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c1 := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		c2 := v_1_1.AuxInt
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or64 (Const64 <t> [c1]) (And64 (Const64 [c2]) x))
	// cond: ^(c1 | c2) == 0
	// result: (Or64 (Const64 <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c1 := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		c2 := v_1_0.AuxInt
		x := v_1.Args[1]
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or64 (Or64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Or64 i (Or64 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr64 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpOr64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or64 (Or64 z i:(Const64 <t>)) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Or64 i (Or64 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr64 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpOr64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or64 x (Or64 i:(Const64 <t>) z))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Or64 i (Or64 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr64 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpOr64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or64 x (Or64 z i:(Const64 <t>)))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Or64 i (Or64 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr64 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpOr64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or64 (Const64 <t> [c]) (Or64 (Const64 <t> [d]) x))
	// cond:
	// result: (Or64 (Const64 <t> [c|d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpOr64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpOr64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c | d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr64_20(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Or64 (Const64 <t> [c]) (Or64 x (Const64 <t> [d])))
	// cond:
	// result: (Or64 (Const64 <t> [c|d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpOr64 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpOr64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c | d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or64 (Or64 (Const64 <t> [d]) x) (Const64 <t> [c]))
	// cond:
	// result: (Or64 (Const64 <t> [c|d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c | d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or64 (Or64 x (Const64 <t> [d])) (Const64 <t> [c]))
	// cond:
	// result: (Or64 (Const64 <t> [c|d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c | d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr8_0(v *Value) bool {
	// match: (Or8 (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (Const8  [int64(int8(c|d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c | d))
		return true
	}
	// match: (Or8 (Const8 [d]) (Const8 [c]))
	// cond:
	// result: (Const8  [int64(int8(c|d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c | d))
		return true
	}
	// match: (Or8 x x)
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
	// match: (Or8 (Const8 [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Or8 x (Const8 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Or8 (Const8 [-1]) _)
	// cond:
	// result: (Const8  [-1])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != -1 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = -1
		return true
	}
	// match: (Or8 _ (Const8 [-1]))
	// cond:
	// result: (Const8  [-1])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.AuxInt != -1 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = -1
		return true
	}
	// match: (Or8 x (Or8 x y))
	// cond:
	// result: (Or8  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr8 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpOr8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Or8 x (Or8 y x))
	// cond:
	// result: (Or8  x y)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr8 {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpOr8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Or8 (Or8 x y) x)
	// cond:
	// result: (Or8  x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpOr8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr8_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Or8 (Or8 y x) x)
	// cond:
	// result: (Or8  x y)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr8 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpOr8)
		v.AddArg(x)
		v.AddArg(y)
		return true
	}
	// match: (Or8 (And8 x (Const8 [c2])) (Const8 <t> [c1]))
	// cond: ^(c1 | c2) == 0
	// result: (Or8  (Const8  <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst8 {
			break
		}
		c2 := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		t := v_1.Type
		c1 := v_1.AuxInt
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or8 (And8 (Const8 [c2]) x) (Const8 <t> [c1]))
	// cond: ^(c1 | c2) == 0
	// result: (Or8  (Const8  <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAnd8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst8 {
			break
		}
		c2 := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		t := v_1.Type
		c1 := v_1.AuxInt
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or8 (Const8 <t> [c1]) (And8 x (Const8 [c2])))
	// cond: ^(c1 | c2) == 0
	// result: (Or8  (Const8  <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c1 := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst8 {
			break
		}
		c2 := v_1_1.AuxInt
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or8 (Const8 <t> [c1]) (And8 (Const8 [c2]) x))
	// cond: ^(c1 | c2) == 0
	// result: (Or8  (Const8  <t> [c1]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c1 := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpAnd8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		c2 := v_1_0.AuxInt
		x := v_1.Args[1]
		if !(^(c1 | c2) == 0) {
			break
		}
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = c1
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or8 (Or8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Or8  i (Or8  <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr8 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpOr8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or8 (Or8 z i:(Const8 <t>)) x)
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Or8  i (Or8  <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr8 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpOr8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or8 x (Or8 i:(Const8 <t>) z))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Or8  i (Or8  <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr8 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpOr8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or8 x (Or8 z i:(Const8 <t>)))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Or8  i (Or8  <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpOr8 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpOr8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpOr8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Or8 (Const8 <t> [c]) (Or8 (Const8 <t> [d]) x))
	// cond:
	// result: (Or8  (Const8  <t> [int64(int8(c|d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpOr8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c | d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpOr8_20(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Or8 (Const8 <t> [c]) (Or8 x (Const8 <t> [d])))
	// cond:
	// result: (Or8  (Const8  <t> [int64(int8(c|d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpOr8 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst8 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c | d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or8 (Or8 (Const8 <t> [d]) x) (Const8 <t> [c]))
	// cond:
	// result: (Or8  (Const8  <t> [int64(int8(c|d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst8 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c | d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Or8 (Or8 x (Const8 <t> [d])) (Const8 <t> [c]))
	// cond:
	// result: (Or8  (Const8  <t> [int64(int8(c|d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpOr8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst8 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpOr8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c | d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpPhi_0(v *Value) bool {
	// match: (Phi (Const8 [c]) (Const8 [c]))
	// cond:
	// result: (Const8  [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.AuxInt != c {
			break
		}
		if len(v.Args) != 2 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = c
		return true
	}
	// match: (Phi (Const16 [c]) (Const16 [c]))
	// cond:
	// result: (Const16 [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.AuxInt != c {
			break
		}
		if len(v.Args) != 2 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = c
		return true
	}
	// match: (Phi (Const32 [c]) (Const32 [c]))
	// cond:
	// result: (Const32 [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.AuxInt != c {
			break
		}
		if len(v.Args) != 2 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = c
		return true
	}
	// match: (Phi (Const64 [c]) (Const64 [c]))
	// cond:
	// result: (Const64 [c])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != c {
			break
		}
		if len(v.Args) != 2 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = c
		return true
	}
	return false
}
func rewriteValuegeneric_OpPtrIndex_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	typ := &b.Func.Config.Types
	_ = typ
	// match: (PtrIndex <t> ptr idx)
	// cond: config.PtrSize == 4
	// result: (AddPtr ptr (Mul32 <typ.Int> idx (Const32 <typ.Int> [t.ElemType().Size()])))
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		idx := v.Args[1]
		if !(config.PtrSize == 4) {
			break
		}
		v.reset(OpAddPtr)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpMul32, typ.Int)
		v0.AddArg(idx)
		v1 := b.NewValue0(v.Pos, OpConst32, typ.Int)
		v1.AuxInt = t.ElemType().Size()
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	// match: (PtrIndex <t> ptr idx)
	// cond: config.PtrSize == 8
	// result: (AddPtr ptr (Mul64 <typ.Int> idx (Const64 <typ.Int> [t.ElemType().Size()])))
	for {
		t := v.Type
		_ = v.Args[1]
		ptr := v.Args[0]
		idx := v.Args[1]
		if !(config.PtrSize == 8) {
			break
		}
		v.reset(OpAddPtr)
		v.AddArg(ptr)
		v0 := b.NewValue0(v.Pos, OpMul64, typ.Int)
		v0.AddArg(idx)
		v1 := b.NewValue0(v.Pos, OpConst64, typ.Int)
		v1.AuxInt = t.ElemType().Size()
		v0.AddArg(v1)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRound32F_0(v *Value) bool {
	// match: (Round32F x:(Const32F))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpConst32F {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRound64F_0(v *Value) bool {
	// match: (Round64F x:(Const64F))
	// cond:
	// result: x
	for {
		x := v.Args[0]
		if x.Op != OpConst64F {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16Ux16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16Ux16 <t> x (Const16 [c]))
	// cond:
	// result: (Rsh16Ux64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh16Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint16(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16Ux16 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16Ux32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16Ux32 <t> x (Const32 [c]))
	// cond:
	// result: (Rsh16Ux64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh16Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint32(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16Ux32 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16Ux64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh16Ux64 (Const16 [c]) (Const64 [d]))
	// cond:
	// result: (Const16 [int64(int16(uint16(c) >> uint64(d)))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(uint16(c) >> uint64(d)))
		return true
	}
	// match: (Rsh16Ux64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Rsh16Ux64 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
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
	// match: (Rsh16Ux64 <t> (Rsh16Ux64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh16Ux64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRsh16Ux64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh16Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16Ux64 (Lsh16x64 (Rsh16Ux64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Rsh16Ux64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh16x64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRsh16Ux64 {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c1 := v_0_0_1.AuxInt
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c3 := v_1.AuxInt
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpRsh16Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = c1 - c2 + c3
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16Ux64 (Lsh16x64 x (Const64 [8])) (Const64 [8]))
	// cond:
	// result: (ZeroExt8to16  (Trunc16to8  <typ.UInt8>  x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh16x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.AuxInt != 8 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 8 {
			break
		}
		v.reset(OpZeroExt8to16)
		v0 := b.NewValue0(v.Pos, OpTrunc16to8, typ.UInt8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16Ux8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16Ux8 <t> x (Const8 [c]))
	// cond:
	// result: (Rsh16Ux64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh16Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint8(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16Ux8 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16x16 <t> x (Const16 [c]))
	// cond:
	// result: (Rsh16x64  x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh16x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint16(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x16 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16x32 <t> x (Const32 [c]))
	// cond:
	// result: (Rsh16x64  x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh16x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint32(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x32 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16x64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh16x64 (Const16 [c]) (Const64 [d]))
	// cond:
	// result: (Const16 [int64(int16(c) >> uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c) >> uint64(d))
		return true
	}
	// match: (Rsh16x64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Rsh16x64 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	// match: (Rsh16x64 <t> (Rsh16x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh16x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRsh16x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh16x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x64 (Lsh16x64 x (Const64 [8])) (Const64 [8]))
	// cond:
	// result: (SignExt8to16  (Trunc16to8  <typ.Int8>  x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh16x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.AuxInt != 8 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 8 {
			break
		}
		v.reset(OpSignExt8to16)
		v0 := b.NewValue0(v.Pos, OpTrunc16to8, typ.Int8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh16x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh16x8 <t> x (Const8 [c]))
	// cond:
	// result: (Rsh16x64  x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh16x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint8(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh16x8 (Const16 [0]) _)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32Ux16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32Ux16 <t> x (Const16 [c]))
	// cond:
	// result: (Rsh32Ux64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh32Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint16(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32Ux16 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32Ux32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32Ux32 <t> x (Const32 [c]))
	// cond:
	// result: (Rsh32Ux64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh32Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint32(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32Ux32 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32Ux64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh32Ux64 (Const32 [c]) (Const64 [d]))
	// cond:
	// result: (Const32 [int64(int32(uint32(c) >> uint64(d)))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(uint32(c) >> uint64(d)))
		return true
	}
	// match: (Rsh32Ux64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Rsh32Ux64 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
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
	// match: (Rsh32Ux64 <t> (Rsh32Ux64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh32Ux64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRsh32Ux64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32Ux64 (Lsh32x64 (Rsh32Ux64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Rsh32Ux64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh32x64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRsh32Ux64 {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c1 := v_0_0_1.AuxInt
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c3 := v_1.AuxInt
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpRsh32Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = c1 - c2 + c3
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32Ux64 (Lsh32x64 x (Const64 [24])) (Const64 [24]))
	// cond:
	// result: (ZeroExt8to32  (Trunc32to8  <typ.UInt8>  x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh32x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.AuxInt != 24 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 24 {
			break
		}
		v.reset(OpZeroExt8to32)
		v0 := b.NewValue0(v.Pos, OpTrunc32to8, typ.UInt8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32Ux64 (Lsh32x64 x (Const64 [16])) (Const64 [16]))
	// cond:
	// result: (ZeroExt16to32 (Trunc32to16 <typ.UInt16> x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh32x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.AuxInt != 16 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 16 {
			break
		}
		v.reset(OpZeroExt16to32)
		v0 := b.NewValue0(v.Pos, OpTrunc32to16, typ.UInt16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32Ux8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32Ux8 <t> x (Const8 [c]))
	// cond:
	// result: (Rsh32Ux64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh32Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint8(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32Ux8 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32x16 <t> x (Const16 [c]))
	// cond:
	// result: (Rsh32x64  x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh32x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint16(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32x16 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32x32 <t> x (Const32 [c]))
	// cond:
	// result: (Rsh32x64  x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh32x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint32(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32x32 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32x64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh32x64 (Const32 [c]) (Const64 [d]))
	// cond:
	// result: (Const32 [int64(int32(c) >> uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c) >> uint64(d))
		return true
	}
	// match: (Rsh32x64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Rsh32x64 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	// match: (Rsh32x64 <t> (Rsh32x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh32x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRsh32x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh32x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32x64 (Lsh32x64 x (Const64 [24])) (Const64 [24]))
	// cond:
	// result: (SignExt8to32  (Trunc32to8  <typ.Int8>  x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh32x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.AuxInt != 24 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 24 {
			break
		}
		v.reset(OpSignExt8to32)
		v0 := b.NewValue0(v.Pos, OpTrunc32to8, typ.Int8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32x64 (Lsh32x64 x (Const64 [16])) (Const64 [16]))
	// cond:
	// result: (SignExt16to32 (Trunc32to16 <typ.Int16> x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh32x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.AuxInt != 16 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 16 {
			break
		}
		v.reset(OpSignExt16to32)
		v0 := b.NewValue0(v.Pos, OpTrunc32to16, typ.Int16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh32x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh32x8 <t> x (Const8 [c]))
	// cond:
	// result: (Rsh32x64  x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh32x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint8(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh32x8 (Const32 [0]) _)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64Ux16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64Ux16 <t> x (Const16 [c]))
	// cond:
	// result: (Rsh64Ux64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh64Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint16(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64Ux16 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64Ux32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64Ux32 <t> x (Const32 [c]))
	// cond:
	// result: (Rsh64Ux64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh64Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint32(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64Ux32 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64Ux64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh64Ux64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (Const64 [int64(uint64(c) >> uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = int64(uint64(c) >> uint64(d))
		return true
	}
	// match: (Rsh64Ux64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Rsh64Ux64 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	// match: (Rsh64Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 64
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		if !(uint64(c) >= 64) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	// match: (Rsh64Ux64 <t> (Rsh64Ux64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh64Ux64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRsh64Ux64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64Ux64 (Lsh64x64 (Rsh64Ux64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Rsh64Ux64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRsh64Ux64 {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c1 := v_0_0_1.AuxInt
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c3 := v_1.AuxInt
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpRsh64Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = c1 - c2 + c3
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64Ux64 (Lsh64x64 x (Const64 [56])) (Const64 [56]))
	// cond:
	// result: (ZeroExt8to64  (Trunc64to8  <typ.UInt8>  x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.AuxInt != 56 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 56 {
			break
		}
		v.reset(OpZeroExt8to64)
		v0 := b.NewValue0(v.Pos, OpTrunc64to8, typ.UInt8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64Ux64 (Lsh64x64 x (Const64 [48])) (Const64 [48]))
	// cond:
	// result: (ZeroExt16to64 (Trunc64to16 <typ.UInt16> x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.AuxInt != 48 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 48 {
			break
		}
		v.reset(OpZeroExt16to64)
		v0 := b.NewValue0(v.Pos, OpTrunc64to16, typ.UInt16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64Ux64 (Lsh64x64 x (Const64 [32])) (Const64 [32]))
	// cond:
	// result: (ZeroExt32to64 (Trunc64to32 <typ.UInt32> x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.AuxInt != 32 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 32 {
			break
		}
		v.reset(OpZeroExt32to64)
		v0 := b.NewValue0(v.Pos, OpTrunc64to32, typ.UInt32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64Ux8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64Ux8 <t> x (Const8 [c]))
	// cond:
	// result: (Rsh64Ux64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh64Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint8(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64Ux8 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64x16 <t> x (Const16 [c]))
	// cond:
	// result: (Rsh64x64  x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh64x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint16(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64x16 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64x32 <t> x (Const32 [c]))
	// cond:
	// result: (Rsh64x64  x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh64x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint32(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64x32 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64x64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh64x64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (Const64 [c >> uint64(d)])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c >> uint64(d)
		return true
	}
	// match: (Rsh64x64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Rsh64x64 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	// match: (Rsh64x64 <t> (Rsh64x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh64x64 x (Const64 <t> [c+d]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh64x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64x64 (Lsh64x64 x (Const64 [56])) (Const64 [56]))
	// cond:
	// result: (SignExt8to64  (Trunc64to8  <typ.Int8>  x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.AuxInt != 56 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 56 {
			break
		}
		v.reset(OpSignExt8to64)
		v0 := b.NewValue0(v.Pos, OpTrunc64to8, typ.Int8)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64x64 (Lsh64x64 x (Const64 [48])) (Const64 [48]))
	// cond:
	// result: (SignExt16to64 (Trunc64to16 <typ.Int16> x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.AuxInt != 48 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 48 {
			break
		}
		v.reset(OpSignExt16to64)
		v0 := b.NewValue0(v.Pos, OpTrunc64to16, typ.Int16)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64x64 (Lsh64x64 x (Const64 [32])) (Const64 [32]))
	// cond:
	// result: (SignExt32to64 (Trunc64to32 <typ.Int32> x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh64x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		if v_0_1.AuxInt != 32 {
			break
		}
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 32 {
			break
		}
		v.reset(OpSignExt32to64)
		v0 := b.NewValue0(v.Pos, OpTrunc64to32, typ.Int32)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh64x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh64x8 <t> x (Const8 [c]))
	// cond:
	// result: (Rsh64x64  x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh64x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint8(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh64x8 (Const64 [0]) _)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8Ux16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8Ux16 <t> x (Const16 [c]))
	// cond:
	// result: (Rsh8Ux64 x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh8Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint16(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8Ux16 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8Ux32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8Ux32 <t> x (Const32 [c]))
	// cond:
	// result: (Rsh8Ux64 x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh8Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint32(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8Ux32 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8Ux64_0(v *Value) bool {
	b := v.Block
	_ = b
	typ := &b.Func.Config.Types
	_ = typ
	// match: (Rsh8Ux64 (Const8 [c]) (Const64 [d]))
	// cond:
	// result: (Const8  [int64(int8(uint8(c) >> uint64(d)))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(uint8(c) >> uint64(d)))
		return true
	}
	// match: (Rsh8Ux64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Rsh8Ux64 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	// match: (Rsh8Ux64 _ (Const64 [c]))
	// cond: uint64(c) >= 8
	// result: (Const8  [0])
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
	// match: (Rsh8Ux64 <t> (Rsh8Ux64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh8Ux64  x (Const64 <t> [c+d]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRsh8Ux64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh8Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8Ux64 (Lsh8x64 (Rsh8Ux64 x (Const64 [c1])) (Const64 [c2])) (Const64 [c3]))
	// cond: uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)
	// result: (Rsh8Ux64 x (Const64 <typ.UInt64> [c1-c2+c3]))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLsh8x64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpRsh8Ux64 {
			break
		}
		_ = v_0_0.Args[1]
		x := v_0_0.Args[0]
		v_0_0_1 := v_0_0.Args[1]
		if v_0_0_1.Op != OpConst64 {
			break
		}
		c1 := v_0_0_1.AuxInt
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c2 := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c3 := v_1.AuxInt
		if !(uint64(c1) >= uint64(c2) && uint64(c3) >= uint64(c2) && !uaddOvf(c1-c2, c3)) {
			break
		}
		v.reset(OpRsh8Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, typ.UInt64)
		v0.AuxInt = c1 - c2 + c3
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8Ux8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8Ux8 <t> x (Const8 [c]))
	// cond:
	// result: (Rsh8Ux64 x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh8Ux64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint8(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8Ux8 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8x16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8x16 <t> x (Const16 [c]))
	// cond:
	// result: (Rsh8x64  x (Const64 <t> [int64(uint16(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh8x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint16(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8x16 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8x32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8x32 <t> x (Const32 [c]))
	// cond:
	// result: (Rsh8x64  x (Const64 <t> [int64(uint32(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh8x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint32(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8x32 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8x64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8x64 (Const8 [c]) (Const64 [d]))
	// cond:
	// result: (Const8  [int64(int8(c) >> uint64(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c) >> uint64(d))
		return true
	}
	// match: (Rsh8x64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Rsh8x64 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	// match: (Rsh8x64 <t> (Rsh8x64 x (Const64 [c])) (Const64 [d]))
	// cond: !uaddOvf(c,d)
	// result: (Rsh8x64  x (Const64 <t> [c+d]))
	for {
		t := v.Type
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpRsh8x64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		c := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		if !(!uaddOvf(c, d)) {
			break
		}
		v.reset(OpRsh8x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpRsh8x8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Rsh8x8 <t> x (Const8 [c]))
	// cond:
	// result: (Rsh8x64  x (Const64 <t> [int64(uint8(c))]))
	for {
		t := v.Type
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpRsh8x64)
		v.AddArg(x)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = int64(uint8(c))
		v.AddArg(v0)
		return true
	}
	// match: (Rsh8x8 (Const8 [0]) _)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpSignExt16to32_0(v *Value) bool {
	// match: (SignExt16to32 (Const16 [c]))
	// cond:
	// result: (Const32 [int64( int16(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int16(c))
		return true
	}
	// match: (SignExt16to32 (Trunc32to16 x:(Rsh32x64 _ (Const64 [s]))))
	// cond: s >= 16
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpTrunc32to16 {
			break
		}
		x := v_0.Args[0]
		if x.Op != OpRsh32x64 {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != OpConst64 {
			break
		}
		s := x_1.AuxInt
		if !(s >= 16) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSignExt16to64_0(v *Value) bool {
	// match: (SignExt16to64 (Const16 [c]))
	// cond:
	// result: (Const64 [int64( int16(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst64)
		v.AuxInt = int64(int16(c))
		return true
	}
	// match: (SignExt16to64 (Trunc64to16 x:(Rsh64x64 _ (Const64 [s]))))
	// cond: s >= 48
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpTrunc64to16 {
			break
		}
		x := v_0.Args[0]
		if x.Op != OpRsh64x64 {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != OpConst64 {
			break
		}
		s := x_1.AuxInt
		if !(s >= 48) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSignExt32to64_0(v *Value) bool {
	// match: (SignExt32to64 (Const32 [c]))
	// cond:
	// result: (Const64 [int64( int32(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst64)
		v.AuxInt = int64(int32(c))
		return true
	}
	// match: (SignExt32to64 (Trunc64to32 x:(Rsh64x64 _ (Const64 [s]))))
	// cond: s >= 32
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpTrunc64to32 {
			break
		}
		x := v_0.Args[0]
		if x.Op != OpRsh64x64 {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != OpConst64 {
			break
		}
		s := x_1.AuxInt
		if !(s >= 32) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSignExt8to16_0(v *Value) bool {
	// match: (SignExt8to16 (Const8 [c]))
	// cond:
	// result: (Const16 [int64(  int8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int8(c))
		return true
	}
	// match: (SignExt8to16 (Trunc16to8 x:(Rsh16x64 _ (Const64 [s]))))
	// cond: s >= 8
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpTrunc16to8 {
			break
		}
		x := v_0.Args[0]
		if x.Op != OpRsh16x64 {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != OpConst64 {
			break
		}
		s := x_1.AuxInt
		if !(s >= 8) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSignExt8to32_0(v *Value) bool {
	// match: (SignExt8to32 (Const8 [c]))
	// cond:
	// result: (Const32 [int64(  int8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int8(c))
		return true
	}
	// match: (SignExt8to32 (Trunc32to8 x:(Rsh32x64 _ (Const64 [s]))))
	// cond: s >= 24
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpTrunc32to8 {
			break
		}
		x := v_0.Args[0]
		if x.Op != OpRsh32x64 {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != OpConst64 {
			break
		}
		s := x_1.AuxInt
		if !(s >= 24) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSignExt8to64_0(v *Value) bool {
	// match: (SignExt8to64 (Const8 [c]))
	// cond:
	// result: (Const64 [int64(  int8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst64)
		v.AuxInt = int64(int8(c))
		return true
	}
	// match: (SignExt8to64 (Trunc64to8 x:(Rsh64x64 _ (Const64 [s]))))
	// cond: s >= 56
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpTrunc64to8 {
			break
		}
		x := v_0.Args[0]
		if x.Op != OpRsh64x64 {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != OpConst64 {
			break
		}
		s := x_1.AuxInt
		if !(s >= 56) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSliceCap_0(v *Value) bool {
	// match: (SliceCap (SliceMake _ _ (Const64 <t> [c])))
	// cond:
	// result: (Const64 <t> [c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		v_0_2 := v_0.Args[2]
		if v_0_2.Op != OpConst64 {
			break
		}
		t := v_0_2.Type
		c := v_0_2.AuxInt
		v.reset(OpConst64)
		v.Type = t
		v.AuxInt = c
		return true
	}
	// match: (SliceCap (SliceMake _ _ (Const32 <t> [c])))
	// cond:
	// result: (Const32 <t> [c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		v_0_2 := v_0.Args[2]
		if v_0_2.Op != OpConst32 {
			break
		}
		t := v_0_2.Type
		c := v_0_2.AuxInt
		v.reset(OpConst32)
		v.Type = t
		v.AuxInt = c
		return true
	}
	// match: (SliceCap (SliceMake _ _ (SliceCap x)))
	// cond:
	// result: (SliceCap x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		v_0_2 := v_0.Args[2]
		if v_0_2.Op != OpSliceCap {
			break
		}
		x := v_0_2.Args[0]
		v.reset(OpSliceCap)
		v.AddArg(x)
		return true
	}
	// match: (SliceCap (SliceMake _ _ (SliceLen x)))
	// cond:
	// result: (SliceLen x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		v_0_2 := v_0.Args[2]
		if v_0_2.Op != OpSliceLen {
			break
		}
		x := v_0_2.Args[0]
		v.reset(OpSliceLen)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSliceLen_0(v *Value) bool {
	// match: (SliceLen (SliceMake _ (Const64 <t> [c]) _))
	// cond:
	// result: (Const64 <t> [c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		t := v_0_1.Type
		c := v_0_1.AuxInt
		v.reset(OpConst64)
		v.Type = t
		v.AuxInt = c
		return true
	}
	// match: (SliceLen (SliceMake _ (Const32 <t> [c]) _))
	// cond:
	// result: (Const32 <t> [c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		t := v_0_1.Type
		c := v_0_1.AuxInt
		v.reset(OpConst32)
		v.Type = t
		v.AuxInt = c
		return true
	}
	// match: (SliceLen (SliceMake _ (SliceLen x) _))
	// cond:
	// result: (SliceLen x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpSliceLen {
			break
		}
		x := v_0_1.Args[0]
		v.reset(OpSliceLen)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSlicePtr_0(v *Value) bool {
	// match: (SlicePtr (SliceMake (SlicePtr x) _ _))
	// cond:
	// result: (SlicePtr x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSliceMake {
			break
		}
		_ = v_0.Args[2]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpSlicePtr {
			break
		}
		x := v_0_0.Args[0]
		v.reset(OpSlicePtr)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSlicemask_0(v *Value) bool {
	// match: (Slicemask (Const32 [x]))
	// cond: x > 0
	// result: (Const32 [-1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		x := v_0.AuxInt
		if !(x > 0) {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = -1
		return true
	}
	// match: (Slicemask (Const32 [0]))
	// cond:
	// result: (Const32 [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	// match: (Slicemask (Const64 [x]))
	// cond: x > 0
	// result: (Const64 [-1])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		x := v_0.AuxInt
		if !(x > 0) {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = -1
		return true
	}
	// match: (Slicemask (Const64 [0]))
	// cond:
	// result: (Const64 [0])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	return false
}
func rewriteValuegeneric_OpSqrt_0(v *Value) bool {
	// match: (Sqrt (Const64F [c]))
	// cond:
	// result: (Const64F [f2i(math.Sqrt(i2f(c)))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst64F)
		v.AuxInt = f2i(math.Sqrt(i2f(c)))
		return true
	}
	return false
}
func rewriteValuegeneric_OpStore_0(v *Value) bool {
	b := v.Block
	_ = b
	fe := b.Func.fe
	_ = fe
	// match: (Store {t1} p1 (Load <t2> p2 mem) mem)
	// cond: isSamePtr(p1, p2) && 	t2.Size() == t1.(*types.Type).Size()
	// result: mem
	for {
		t1 := v.Aux
		_ = v.Args[2]
		p1 := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpLoad {
			break
		}
		t2 := v_1.Type
		_ = v_1.Args[1]
		p2 := v_1.Args[0]
		mem := v_1.Args[1]
		if mem != v.Args[2] {
			break
		}
		if !(isSamePtr(p1, p2) && t2.Size() == t1.(*types.Type).Size()) {
			break
		}
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	// match: (Store {t1} (OffPtr [o1] p1) (Load <t2> (OffPtr [o1] p2) oldmem) mem:(Store {t3} (OffPtr [o3] p3) _ oldmem))
	// cond: isSamePtr(p1, p2) && 	isSamePtr(p1, p3) && 	t2.Size() == t1.(*types.Type).Size() && 	!overlap(o1, t2.Size(), o3, t3.(*types.Type).Size())
	// result: mem
	for {
		t1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpOffPtr {
			break
		}
		o1 := v_0.AuxInt
		p1 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpLoad {
			break
		}
		t2 := v_1.Type
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpOffPtr {
			break
		}
		if v_1_0.AuxInt != o1 {
			break
		}
		p2 := v_1_0.Args[0]
		oldmem := v_1.Args[1]
		mem := v.Args[2]
		if mem.Op != OpStore {
			break
		}
		t3 := mem.Aux
		_ = mem.Args[2]
		mem_0 := mem.Args[0]
		if mem_0.Op != OpOffPtr {
			break
		}
		o3 := mem_0.AuxInt
		p3 := mem_0.Args[0]
		if oldmem != mem.Args[2] {
			break
		}
		if !(isSamePtr(p1, p2) && isSamePtr(p1, p3) && t2.Size() == t1.(*types.Type).Size() && !overlap(o1, t2.Size(), o3, t3.(*types.Type).Size())) {
			break
		}
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	// match: (Store {t1} (OffPtr [o1] p1) (Load <t2> (OffPtr [o1] p2) oldmem) mem:(Store {t3} (OffPtr [o3] p3) _ (Store {t4} (OffPtr [o4] p4) _ oldmem)))
	// cond: isSamePtr(p1, p2) && 	isSamePtr(p1, p3) && 	isSamePtr(p1, p4) && 	t2.Size() == t1.(*types.Type).Size() && 	!overlap(o1, t2.Size(), o3, t3.(*types.Type).Size()) && 	!overlap(o1, t2.Size(), o4, t4.(*types.Type).Size())
	// result: mem
	for {
		t1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpOffPtr {
			break
		}
		o1 := v_0.AuxInt
		p1 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpLoad {
			break
		}
		t2 := v_1.Type
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpOffPtr {
			break
		}
		if v_1_0.AuxInt != o1 {
			break
		}
		p2 := v_1_0.Args[0]
		oldmem := v_1.Args[1]
		mem := v.Args[2]
		if mem.Op != OpStore {
			break
		}
		t3 := mem.Aux
		_ = mem.Args[2]
		mem_0 := mem.Args[0]
		if mem_0.Op != OpOffPtr {
			break
		}
		o3 := mem_0.AuxInt
		p3 := mem_0.Args[0]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpStore {
			break
		}
		t4 := mem_2.Aux
		_ = mem_2.Args[2]
		mem_2_0 := mem_2.Args[0]
		if mem_2_0.Op != OpOffPtr {
			break
		}
		o4 := mem_2_0.AuxInt
		p4 := mem_2_0.Args[0]
		if oldmem != mem_2.Args[2] {
			break
		}
		if !(isSamePtr(p1, p2) && isSamePtr(p1, p3) && isSamePtr(p1, p4) && t2.Size() == t1.(*types.Type).Size() && !overlap(o1, t2.Size(), o3, t3.(*types.Type).Size()) && !overlap(o1, t2.Size(), o4, t4.(*types.Type).Size())) {
			break
		}
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	// match: (Store {t1} (OffPtr [o1] p1) (Load <t2> (OffPtr [o1] p2) oldmem) mem:(Store {t3} (OffPtr [o3] p3) _ (Store {t4} (OffPtr [o4] p4) _ (Store {t5} (OffPtr [o5] p5) _ oldmem))))
	// cond: isSamePtr(p1, p2) && 	isSamePtr(p1, p3) && 	isSamePtr(p1, p4) && 	isSamePtr(p1, p5) && 	t2.Size() == t1.(*types.Type).Size() && 	!overlap(o1, t2.Size(), o3, t3.(*types.Type).Size()) && 	!overlap(o1, t2.Size(), o4, t4.(*types.Type).Size()) && 	!overlap(o1, t2.Size(), o5, t5.(*types.Type).Size())
	// result: mem
	for {
		t1 := v.Aux
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpOffPtr {
			break
		}
		o1 := v_0.AuxInt
		p1 := v_0.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpLoad {
			break
		}
		t2 := v_1.Type
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpOffPtr {
			break
		}
		if v_1_0.AuxInt != o1 {
			break
		}
		p2 := v_1_0.Args[0]
		oldmem := v_1.Args[1]
		mem := v.Args[2]
		if mem.Op != OpStore {
			break
		}
		t3 := mem.Aux
		_ = mem.Args[2]
		mem_0 := mem.Args[0]
		if mem_0.Op != OpOffPtr {
			break
		}
		o3 := mem_0.AuxInt
		p3 := mem_0.Args[0]
		mem_2 := mem.Args[2]
		if mem_2.Op != OpStore {
			break
		}
		t4 := mem_2.Aux
		_ = mem_2.Args[2]
		mem_2_0 := mem_2.Args[0]
		if mem_2_0.Op != OpOffPtr {
			break
		}
		o4 := mem_2_0.AuxInt
		p4 := mem_2_0.Args[0]
		mem_2_2 := mem_2.Args[2]
		if mem_2_2.Op != OpStore {
			break
		}
		t5 := mem_2_2.Aux
		_ = mem_2_2.Args[2]
		mem_2_2_0 := mem_2_2.Args[0]
		if mem_2_2_0.Op != OpOffPtr {
			break
		}
		o5 := mem_2_2_0.AuxInt
		p5 := mem_2_2_0.Args[0]
		if oldmem != mem_2_2.Args[2] {
			break
		}
		if !(isSamePtr(p1, p2) && isSamePtr(p1, p3) && isSamePtr(p1, p4) && isSamePtr(p1, p5) && t2.Size() == t1.(*types.Type).Size() && !overlap(o1, t2.Size(), o3, t3.(*types.Type).Size()) && !overlap(o1, t2.Size(), o4, t4.(*types.Type).Size()) && !overlap(o1, t2.Size(), o5, t5.(*types.Type).Size())) {
			break
		}
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	// match: (Store _ (StructMake0) mem)
	// cond:
	// result: mem
	for {
		_ = v.Args[2]
		v_1 := v.Args[1]
		if v_1.Op != OpStructMake0 {
			break
		}
		mem := v.Args[2]
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	// match: (Store dst (StructMake1 <t> f0) mem)
	// cond:
	// result: (Store {t.FieldType(0)} (OffPtr <t.FieldType(0).PtrTo()> [0] dst) f0 mem)
	for {
		_ = v.Args[2]
		dst := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpStructMake1 {
			break
		}
		t := v_1.Type
		f0 := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpStore)
		v.Aux = t.FieldType(0)
		v0 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(0).PtrTo())
		v0.AuxInt = 0
		v0.AddArg(dst)
		v.AddArg(v0)
		v.AddArg(f0)
		v.AddArg(mem)
		return true
	}
	// match: (Store dst (StructMake2 <t> f0 f1) mem)
	// cond:
	// result: (Store {t.FieldType(1)}     (OffPtr <t.FieldType(1).PtrTo()> [t.FieldOff(1)] dst)     f1     (Store {t.FieldType(0)}       (OffPtr <t.FieldType(0).PtrTo()> [0] dst)         f0 mem))
	for {
		_ = v.Args[2]
		dst := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpStructMake2 {
			break
		}
		t := v_1.Type
		_ = v_1.Args[1]
		f0 := v_1.Args[0]
		f1 := v_1.Args[1]
		mem := v.Args[2]
		v.reset(OpStore)
		v.Aux = t.FieldType(1)
		v0 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(1).PtrTo())
		v0.AuxInt = t.FieldOff(1)
		v0.AddArg(dst)
		v.AddArg(v0)
		v.AddArg(f1)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = t.FieldType(0)
		v2 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(0).PtrTo())
		v2.AuxInt = 0
		v2.AddArg(dst)
		v1.AddArg(v2)
		v1.AddArg(f0)
		v1.AddArg(mem)
		v.AddArg(v1)
		return true
	}
	// match: (Store dst (StructMake3 <t> f0 f1 f2) mem)
	// cond:
	// result: (Store {t.FieldType(2)}     (OffPtr <t.FieldType(2).PtrTo()> [t.FieldOff(2)] dst)     f2     (Store {t.FieldType(1)}       (OffPtr <t.FieldType(1).PtrTo()> [t.FieldOff(1)] dst)       f1       (Store {t.FieldType(0)}         (OffPtr <t.FieldType(0).PtrTo()> [0] dst)           f0 mem)))
	for {
		_ = v.Args[2]
		dst := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpStructMake3 {
			break
		}
		t := v_1.Type
		_ = v_1.Args[2]
		f0 := v_1.Args[0]
		f1 := v_1.Args[1]
		f2 := v_1.Args[2]
		mem := v.Args[2]
		v.reset(OpStore)
		v.Aux = t.FieldType(2)
		v0 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(2).PtrTo())
		v0.AuxInt = t.FieldOff(2)
		v0.AddArg(dst)
		v.AddArg(v0)
		v.AddArg(f2)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = t.FieldType(1)
		v2 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(1).PtrTo())
		v2.AuxInt = t.FieldOff(1)
		v2.AddArg(dst)
		v1.AddArg(v2)
		v1.AddArg(f1)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = t.FieldType(0)
		v4 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(0).PtrTo())
		v4.AuxInt = 0
		v4.AddArg(dst)
		v3.AddArg(v4)
		v3.AddArg(f0)
		v3.AddArg(mem)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Store dst (StructMake4 <t> f0 f1 f2 f3) mem)
	// cond:
	// result: (Store {t.FieldType(3)}     (OffPtr <t.FieldType(3).PtrTo()> [t.FieldOff(3)] dst)     f3     (Store {t.FieldType(2)}       (OffPtr <t.FieldType(2).PtrTo()> [t.FieldOff(2)] dst)       f2       (Store {t.FieldType(1)}         (OffPtr <t.FieldType(1).PtrTo()> [t.FieldOff(1)] dst)         f1         (Store {t.FieldType(0)}           (OffPtr <t.FieldType(0).PtrTo()> [0] dst)             f0 mem))))
	for {
		_ = v.Args[2]
		dst := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpStructMake4 {
			break
		}
		t := v_1.Type
		_ = v_1.Args[3]
		f0 := v_1.Args[0]
		f1 := v_1.Args[1]
		f2 := v_1.Args[2]
		f3 := v_1.Args[3]
		mem := v.Args[2]
		v.reset(OpStore)
		v.Aux = t.FieldType(3)
		v0 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(3).PtrTo())
		v0.AuxInt = t.FieldOff(3)
		v0.AddArg(dst)
		v.AddArg(v0)
		v.AddArg(f3)
		v1 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v1.Aux = t.FieldType(2)
		v2 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(2).PtrTo())
		v2.AuxInt = t.FieldOff(2)
		v2.AddArg(dst)
		v1.AddArg(v2)
		v1.AddArg(f2)
		v3 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v3.Aux = t.FieldType(1)
		v4 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(1).PtrTo())
		v4.AuxInt = t.FieldOff(1)
		v4.AddArg(dst)
		v3.AddArg(v4)
		v3.AddArg(f1)
		v5 := b.NewValue0(v.Pos, OpStore, types.TypeMem)
		v5.Aux = t.FieldType(0)
		v6 := b.NewValue0(v.Pos, OpOffPtr, t.FieldType(0).PtrTo())
		v6.AuxInt = 0
		v6.AddArg(dst)
		v5.AddArg(v6)
		v5.AddArg(f0)
		v5.AddArg(mem)
		v3.AddArg(v5)
		v1.AddArg(v3)
		v.AddArg(v1)
		return true
	}
	// match: (Store {t} dst (Load src mem) mem)
	// cond: !fe.CanSSA(t.(*types.Type))
	// result: (Move {t} [t.(*types.Type).Size()] dst src mem)
	for {
		t := v.Aux
		_ = v.Args[2]
		dst := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpLoad {
			break
		}
		_ = v_1.Args[1]
		src := v_1.Args[0]
		mem := v_1.Args[1]
		if mem != v.Args[2] {
			break
		}
		if !(!fe.CanSSA(t.(*types.Type))) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = t.(*types.Type).Size()
		v.Aux = t
		v.AddArg(dst)
		v.AddArg(src)
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValuegeneric_OpStore_10(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	fe := b.Func.fe
	_ = fe
	// match: (Store {t} dst (Load src mem) (VarDef {x} mem))
	// cond: !fe.CanSSA(t.(*types.Type))
	// result: (Move {t} [t.(*types.Type).Size()] dst src (VarDef {x} mem))
	for {
		t := v.Aux
		_ = v.Args[2]
		dst := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpLoad {
			break
		}
		_ = v_1.Args[1]
		src := v_1.Args[0]
		mem := v_1.Args[1]
		v_2 := v.Args[2]
		if v_2.Op != OpVarDef {
			break
		}
		x := v_2.Aux
		if mem != v_2.Args[0] {
			break
		}
		if !(!fe.CanSSA(t.(*types.Type))) {
			break
		}
		v.reset(OpMove)
		v.AuxInt = t.(*types.Type).Size()
		v.Aux = t
		v.AddArg(dst)
		v.AddArg(src)
		v0 := b.NewValue0(v.Pos, OpVarDef, types.TypeMem)
		v0.Aux = x
		v0.AddArg(mem)
		v.AddArg(v0)
		return true
	}
	// match: (Store _ (ArrayMake0) mem)
	// cond:
	// result: mem
	for {
		_ = v.Args[2]
		v_1 := v.Args[1]
		if v_1.Op != OpArrayMake0 {
			break
		}
		mem := v.Args[2]
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	// match: (Store dst (ArrayMake1 e) mem)
	// cond:
	// result: (Store {e.Type} dst e mem)
	for {
		_ = v.Args[2]
		dst := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpArrayMake1 {
			break
		}
		e := v_1.Args[0]
		mem := v.Args[2]
		v.reset(OpStore)
		v.Aux = e.Type
		v.AddArg(dst)
		v.AddArg(e)
		v.AddArg(mem)
		return true
	}
	// match: (Store (Load (OffPtr [c] (SP)) mem) x mem)
	// cond: isConstZero(x) 	&& mem.Op == OpStaticCall 	&& isSameSym(mem.Aux, "runtime.newobject") 	&& c == config.ctxt.FixedFrameSize() + config.RegSize
	// result: mem
	for {
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpLoad {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpOffPtr {
			break
		}
		c := v_0_0.AuxInt
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpSP {
			break
		}
		mem := v_0.Args[1]
		x := v.Args[1]
		if mem != v.Args[2] {
			break
		}
		if !(isConstZero(x) && mem.Op == OpStaticCall && isSameSym(mem.Aux, "runtime.newobject") && c == config.ctxt.FixedFrameSize()+config.RegSize) {
			break
		}
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	// match: (Store (OffPtr (Load (OffPtr [c] (SP)) mem)) x mem)
	// cond: isConstZero(x) 	&& mem.Op == OpStaticCall 	&& isSameSym(mem.Aux, "runtime.newobject") 	&& c == config.ctxt.FixedFrameSize() + config.RegSize
	// result: mem
	for {
		_ = v.Args[2]
		v_0 := v.Args[0]
		if v_0.Op != OpOffPtr {
			break
		}
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpLoad {
			break
		}
		_ = v_0_0.Args[1]
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpOffPtr {
			break
		}
		c := v_0_0_0.AuxInt
		v_0_0_0_0 := v_0_0_0.Args[0]
		if v_0_0_0_0.Op != OpSP {
			break
		}
		mem := v_0_0.Args[1]
		x := v.Args[1]
		if mem != v.Args[2] {
			break
		}
		if !(isConstZero(x) && mem.Op == OpStaticCall && isSameSym(mem.Aux, "runtime.newobject") && c == config.ctxt.FixedFrameSize()+config.RegSize) {
			break
		}
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValuegeneric_OpStringLen_0(v *Value) bool {
	// match: (StringLen (StringMake _ (Const64 <t> [c])))
	// cond:
	// result: (Const64 <t> [c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpStringMake {
			break
		}
		_ = v_0.Args[1]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		t := v_0_1.Type
		c := v_0_1.AuxInt
		v.reset(OpConst64)
		v.Type = t
		v.AuxInt = c
		return true
	}
	return false
}
func rewriteValuegeneric_OpStringPtr_0(v *Value) bool {
	// match: (StringPtr (StringMake (Const64 <t> [c]) _))
	// cond:
	// result: (Const64 <t> [c])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpStringMake {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		t := v_0_0.Type
		c := v_0_0.AuxInt
		v.reset(OpConst64)
		v.Type = t
		v.AuxInt = c
		return true
	}
	return false
}
func rewriteValuegeneric_OpStructSelect_0(v *Value) bool {
	// match: (StructSelect (StructMake1 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpStructMake1 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (StructSelect [0] (StructMake2 x _))
	// cond:
	// result: x
	for {
		if v.AuxInt != 0 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpStructMake2 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (StructSelect [1] (StructMake2 _ x))
	// cond:
	// result: x
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpStructMake2 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (StructSelect [0] (StructMake3 x _ _))
	// cond:
	// result: x
	for {
		if v.AuxInt != 0 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpStructMake3 {
			break
		}
		_ = v_0.Args[2]
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (StructSelect [1] (StructMake3 _ x _))
	// cond:
	// result: x
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpStructMake3 {
			break
		}
		_ = v_0.Args[2]
		x := v_0.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (StructSelect [2] (StructMake3 _ _ x))
	// cond:
	// result: x
	for {
		if v.AuxInt != 2 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpStructMake3 {
			break
		}
		_ = v_0.Args[2]
		x := v_0.Args[2]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (StructSelect [0] (StructMake4 x _ _ _))
	// cond:
	// result: x
	for {
		if v.AuxInt != 0 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpStructMake4 {
			break
		}
		_ = v_0.Args[3]
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (StructSelect [1] (StructMake4 _ x _ _))
	// cond:
	// result: x
	for {
		if v.AuxInt != 1 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpStructMake4 {
			break
		}
		_ = v_0.Args[3]
		x := v_0.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (StructSelect [2] (StructMake4 _ _ x _))
	// cond:
	// result: x
	for {
		if v.AuxInt != 2 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpStructMake4 {
			break
		}
		_ = v_0.Args[3]
		x := v_0.Args[2]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (StructSelect [3] (StructMake4 _ _ _ x))
	// cond:
	// result: x
	for {
		if v.AuxInt != 3 {
			break
		}
		v_0 := v.Args[0]
		if v_0.Op != OpStructMake4 {
			break
		}
		_ = v_0.Args[3]
		x := v_0.Args[3]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpStructSelect_10(v *Value) bool {
	b := v.Block
	_ = b
	fe := b.Func.fe
	_ = fe
	// match: (StructSelect [i] x:(Load <t> ptr mem))
	// cond: !fe.CanSSA(t)
	// result: @x.Block (Load <v.Type> (OffPtr <v.Type.PtrTo()> [t.FieldOff(int(i))] ptr) mem)
	for {
		i := v.AuxInt
		x := v.Args[0]
		if x.Op != OpLoad {
			break
		}
		t := x.Type
		_ = x.Args[1]
		ptr := x.Args[0]
		mem := x.Args[1]
		if !(!fe.CanSSA(t)) {
			break
		}
		b = x.Block
		v0 := b.NewValue0(v.Pos, OpLoad, v.Type)
		v.reset(OpCopy)
		v.AddArg(v0)
		v1 := b.NewValue0(v.Pos, OpOffPtr, v.Type.PtrTo())
		v1.AuxInt = t.FieldOff(int(i))
		v1.AddArg(ptr)
		v0.AddArg(v1)
		v0.AddArg(mem)
		return true
	}
	// match: (StructSelect [0] x:(IData _))
	// cond:
	// result: x
	for {
		if v.AuxInt != 0 {
			break
		}
		x := v.Args[0]
		if x.Op != OpIData {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Sub16 (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (Const16 [int64(int16(c-d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c - d))
		return true
	}
	// match: (Sub16 x (Const16 <t> [c]))
	// cond: x.Op != OpConst16
	// result: (Add16 (Const16 <t> [int64(int16(-c))]) x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		t := v_1.Type
		c := v_1.AuxInt
		if !(x.Op != OpConst16) {
			break
		}
		v.reset(OpAdd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(-c))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Sub16 x x)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	// match: (Sub16 (Add16 x y) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Sub16 (Add16 y x) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Sub16 (Add16 x y) y)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if y != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Sub16 (Add16 y x) y)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd16 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if y != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Sub16 x (Sub16 i:(Const16 <t>) z))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Sub16 (Add16 <t> x z) i)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub16 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpSub16)
		v0 := b.NewValue0(v.Pos, OpAdd16, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Sub16 x (Sub16 z i:(Const16 <t>)))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Add16 i (Sub16 <t> x z))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub16 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpAdd16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub16, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Sub16 (Const16 <t> [c]) (Sub16 x (Const16 <t> [d])))
	// cond:
	// result: (Sub16 (Const16 <t> [int64(int16(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub16 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst16 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpSub16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub16_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Sub16 (Const16 <t> [c]) (Sub16 (Const16 <t> [d]) x))
	// cond:
	// result: (Add16 (Const16 <t> [int64(int16(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAdd16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Sub32 (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (Const32 [int64(int32(c-d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c - d))
		return true
	}
	// match: (Sub32 x (Const32 <t> [c]))
	// cond: x.Op != OpConst32
	// result: (Add32 (Const32 <t> [int64(int32(-c))]) x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		t := v_1.Type
		c := v_1.AuxInt
		if !(x.Op != OpConst32) {
			break
		}
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(-c))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Sub32 x x)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	// match: (Sub32 (Add32 x y) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Sub32 (Add32 y x) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Sub32 (Add32 x y) y)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if y != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Sub32 (Add32 y x) y)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd32 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if y != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Sub32 x (Sub32 i:(Const32 <t>) z))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Sub32 (Add32 <t> x z) i)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub32 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpSub32)
		v0 := b.NewValue0(v.Pos, OpAdd32, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Sub32 x (Sub32 z i:(Const32 <t>)))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Add32 i (Sub32 <t> x z))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub32 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpAdd32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub32, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Sub32 (Const32 <t> [c]) (Sub32 x (Const32 <t> [d])))
	// cond:
	// result: (Sub32 (Const32 <t> [int64(int32(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub32 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpSub32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub32_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Sub32 (Const32 <t> [c]) (Sub32 (Const32 <t> [d]) x))
	// cond:
	// result: (Add32 (Const32 <t> [int64(int32(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAdd32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub32F_0(v *Value) bool {
	// match: (Sub32F (Const32F [c]) (Const32F [d]))
	// cond:
	// result: (Const32F [f2i(float64(i2f32(c) - i2f32(d)))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32F {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32F {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32F)
		v.AuxInt = f2i(float64(i2f32(c) - i2f32(d)))
		return true
	}
	// match: (Sub32F x (Const32F [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32F {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Sub64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (Const64 [c-d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c - d
		return true
	}
	// match: (Sub64 x (Const64 <t> [c]))
	// cond: x.Op != OpConst64
	// result: (Add64 (Const64 <t> [-c]) x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		t := v_1.Type
		c := v_1.AuxInt
		if !(x.Op != OpConst64) {
			break
		}
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = -c
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Sub64 x x)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	// match: (Sub64 (Add64 x y) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Sub64 (Add64 y x) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Sub64 (Add64 x y) y)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if y != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Sub64 (Add64 y x) y)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd64 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if y != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Sub64 x (Sub64 i:(Const64 <t>) z))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Sub64 (Add64 <t> x z) i)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub64 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpSub64)
		v0 := b.NewValue0(v.Pos, OpAdd64, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Sub64 x (Sub64 z i:(Const64 <t>)))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Add64 i (Sub64 <t> x z))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub64 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpAdd64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub64, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Sub64 (Const64 <t> [c]) (Sub64 x (Const64 <t> [d])))
	// cond:
	// result: (Sub64 (Const64 <t> [c+d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub64 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpSub64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c + d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub64_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Sub64 (Const64 <t> [c]) (Sub64 (Const64 <t> [d]) x))
	// cond:
	// result: (Add64 (Const64 <t> [c-d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAdd64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c - d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub64F_0(v *Value) bool {
	// match: (Sub64F (Const64F [c]) (Const64F [d]))
	// cond:
	// result: (Const64F [f2i(i2f(c) - i2f(d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64F {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64F {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64F)
		v.AuxInt = f2i(i2f(c) - i2f(d))
		return true
	}
	// match: (Sub64F x (Const64F [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64F {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Sub8 (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (Const8 [int64(int8(c-d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c - d))
		return true
	}
	// match: (Sub8 x (Const8 <t> [c]))
	// cond: x.Op != OpConst8
	// result: (Add8  (Const8  <t> [int64(int8(-c))]) x)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		t := v_1.Type
		c := v_1.AuxInt
		if !(x.Op != OpConst8) {
			break
		}
		v.reset(OpAdd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(-c))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Sub8 x x)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	// match: (Sub8 (Add8 x y) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Sub8 (Add8 y x) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Sub8 (Add8 x y) y)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if y != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Sub8 (Add8 y x) y)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpAdd8 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if y != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Sub8 x (Sub8 i:(Const8 <t>) z))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Sub8  (Add8  <t> x z) i)
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub8 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpSub8)
		v0 := b.NewValue0(v.Pos, OpAdd8, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		v.AddArg(i)
		return true
	}
	// match: (Sub8 x (Sub8 z i:(Const8 <t>)))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Add8  i (Sub8  <t> x z))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpSub8 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpAdd8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpSub8, t)
		v0.AddArg(x)
		v0.AddArg(z)
		v.AddArg(v0)
		return true
	}
	// match: (Sub8 (Const8 <t> [c]) (Sub8 x (Const8 <t> [d])))
	// cond:
	// result: (Sub8  (Const8  <t> [int64(int8(c+d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub8 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst8 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpSub8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c + d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpSub8_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Sub8 (Const8 <t> [c]) (Sub8 (Const8 <t> [d]) x))
	// cond:
	// result: (Add8  (Const8  <t> [int64(int8(c-d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpSub8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpAdd8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c - d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpTrunc16to8_0(v *Value) bool {
	// match: (Trunc16to8 (Const16 [c]))
	// cond:
	// result: (Const8   [int64(int8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c))
		return true
	}
	// match: (Trunc16to8 (ZeroExt8to16 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to16 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Trunc16to8 (SignExt8to16 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSignExt8to16 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Trunc16to8 (And16 (Const16 [y]) x))
	// cond: y&0xFF == 0xFF
	// result: (Trunc16to8 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst16 {
			break
		}
		y := v_0_0.AuxInt
		x := v_0.Args[1]
		if !(y&0xFF == 0xFF) {
			break
		}
		v.reset(OpTrunc16to8)
		v.AddArg(x)
		return true
	}
	// match: (Trunc16to8 (And16 x (Const16 [y])))
	// cond: y&0xFF == 0xFF
	// result: (Trunc16to8 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAnd16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst16 {
			break
		}
		y := v_0_1.AuxInt
		if !(y&0xFF == 0xFF) {
			break
		}
		v.reset(OpTrunc16to8)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpTrunc32to16_0(v *Value) bool {
	// match: (Trunc32to16 (Const32 [c]))
	// cond:
	// result: (Const16  [int64(int16(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c))
		return true
	}
	// match: (Trunc32to16 (ZeroExt8to32 x))
	// cond:
	// result: (ZeroExt8to16  x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to32 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpZeroExt8to16)
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to16 (ZeroExt16to32 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt16to32 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to16 (SignExt8to32 x))
	// cond:
	// result: (SignExt8to16  x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSignExt8to32 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpSignExt8to16)
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to16 (SignExt16to32 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSignExt16to32 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to16 (And32 (Const32 [y]) x))
	// cond: y&0xFFFF == 0xFFFF
	// result: (Trunc32to16 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		y := v_0_0.AuxInt
		x := v_0.Args[1]
		if !(y&0xFFFF == 0xFFFF) {
			break
		}
		v.reset(OpTrunc32to16)
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to16 (And32 x (Const32 [y])))
	// cond: y&0xFFFF == 0xFFFF
	// result: (Trunc32to16 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		y := v_0_1.AuxInt
		if !(y&0xFFFF == 0xFFFF) {
			break
		}
		v.reset(OpTrunc32to16)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpTrunc32to8_0(v *Value) bool {
	// match: (Trunc32to8 (Const32 [c]))
	// cond:
	// result: (Const8   [int64(int8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c))
		return true
	}
	// match: (Trunc32to8 (ZeroExt8to32 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to32 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to8 (SignExt8to32 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSignExt8to32 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to8 (And32 (Const32 [y]) x))
	// cond: y&0xFF == 0xFF
	// result: (Trunc32to8 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		y := v_0_0.AuxInt
		x := v_0.Args[1]
		if !(y&0xFF == 0xFF) {
			break
		}
		v.reset(OpTrunc32to8)
		v.AddArg(x)
		return true
	}
	// match: (Trunc32to8 (And32 x (Const32 [y])))
	// cond: y&0xFF == 0xFF
	// result: (Trunc32to8 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAnd32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		y := v_0_1.AuxInt
		if !(y&0xFF == 0xFF) {
			break
		}
		v.reset(OpTrunc32to8)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpTrunc64to16_0(v *Value) bool {
	// match: (Trunc64to16 (Const64 [c]))
	// cond:
	// result: (Const16  [int64(int16(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c))
		return true
	}
	// match: (Trunc64to16 (ZeroExt8to64 x))
	// cond:
	// result: (ZeroExt8to16  x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpZeroExt8to16)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to16 (ZeroExt16to64 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt16to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to16 (SignExt8to64 x))
	// cond:
	// result: (SignExt8to16  x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSignExt8to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpSignExt8to16)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to16 (SignExt16to64 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSignExt16to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to16 (And64 (Const64 [y]) x))
	// cond: y&0xFFFF == 0xFFFF
	// result: (Trunc64to16 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		y := v_0_0.AuxInt
		x := v_0.Args[1]
		if !(y&0xFFFF == 0xFFFF) {
			break
		}
		v.reset(OpTrunc64to16)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to16 (And64 x (Const64 [y])))
	// cond: y&0xFFFF == 0xFFFF
	// result: (Trunc64to16 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		y := v_0_1.AuxInt
		if !(y&0xFFFF == 0xFFFF) {
			break
		}
		v.reset(OpTrunc64to16)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpTrunc64to32_0(v *Value) bool {
	// match: (Trunc64to32 (Const64 [c]))
	// cond:
	// result: (Const32  [int64(int32(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c))
		return true
	}
	// match: (Trunc64to32 (ZeroExt8to64 x))
	// cond:
	// result: (ZeroExt8to32  x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpZeroExt8to32)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 (ZeroExt16to64 x))
	// cond:
	// result: (ZeroExt16to32 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt16to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpZeroExt16to32)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 (ZeroExt32to64 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt32to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 (SignExt8to64 x))
	// cond:
	// result: (SignExt8to32  x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSignExt8to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpSignExt8to32)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 (SignExt16to64 x))
	// cond:
	// result: (SignExt16to32 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSignExt16to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpSignExt16to32)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 (SignExt32to64 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSignExt32to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 (And64 (Const64 [y]) x))
	// cond: y&0xFFFFFFFF == 0xFFFFFFFF
	// result: (Trunc64to32 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		y := v_0_0.AuxInt
		x := v_0.Args[1]
		if !(y&0xFFFFFFFF == 0xFFFFFFFF) {
			break
		}
		v.reset(OpTrunc64to32)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to32 (And64 x (Const64 [y])))
	// cond: y&0xFFFFFFFF == 0xFFFFFFFF
	// result: (Trunc64to32 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		y := v_0_1.AuxInt
		if !(y&0xFFFFFFFF == 0xFFFFFFFF) {
			break
		}
		v.reset(OpTrunc64to32)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpTrunc64to8_0(v *Value) bool {
	// match: (Trunc64to8 (Const64 [c]))
	// cond:
	// result: (Const8   [int64(int8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c))
		return true
	}
	// match: (Trunc64to8 (ZeroExt8to64 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpZeroExt8to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to8 (SignExt8to64 x))
	// cond:
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpSignExt8to64 {
			break
		}
		x := v_0.Args[0]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to8 (And64 (Const64 [y]) x))
	// cond: y&0xFF == 0xFF
	// result: (Trunc64to8 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		y := v_0_0.AuxInt
		x := v_0.Args[1]
		if !(y&0xFF == 0xFF) {
			break
		}
		v.reset(OpTrunc64to8)
		v.AddArg(x)
		return true
	}
	// match: (Trunc64to8 (And64 x (Const64 [y])))
	// cond: y&0xFF == 0xFF
	// result: (Trunc64to8 x)
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpAnd64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		y := v_0_1.AuxInt
		if !(y&0xFF == 0xFF) {
			break
		}
		v.reset(OpTrunc64to8)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpXor16_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Xor16 (Const16 [c]) (Const16 [d]))
	// cond:
	// result: (Const16 [int64(int16(c^d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c ^ d))
		return true
	}
	// match: (Xor16 (Const16 [d]) (Const16 [c]))
	// cond:
	// result: (Const16 [int64(int16(c^d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(int16(c ^ d))
		return true
	}
	// match: (Xor16 x x)
	// cond:
	// result: (Const16 [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConst16)
		v.AuxInt = 0
		return true
	}
	// match: (Xor16 (Const16 [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Xor16 x (Const16 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Xor16 x (Xor16 x y))
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor16 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor16 x (Xor16 y x))
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor16 {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor16 (Xor16 x y) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor16 (Xor16 y x) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor16 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor16 (Xor16 i:(Const16 <t>) z) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Xor16 i (Xor16 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor16 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpXor16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpXor16_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Xor16 (Xor16 z i:(Const16 <t>)) x)
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Xor16 i (Xor16 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor16 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpXor16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Xor16 x (Xor16 i:(Const16 <t>) z))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Xor16 i (Xor16 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor16 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpXor16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Xor16 x (Xor16 z i:(Const16 <t>)))
	// cond: (z.Op != OpConst16 && x.Op != OpConst16)
	// result: (Xor16 i (Xor16 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor16 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst16 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst16 && x.Op != OpConst16) {
			break
		}
		v.reset(OpXor16)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor16, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Xor16 (Const16 <t> [c]) (Xor16 (Const16 <t> [d]) x))
	// cond:
	// result: (Xor16 (Const16 <t> [int64(int16(c^d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpXor16 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst16 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpXor16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c ^ d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Xor16 (Const16 <t> [c]) (Xor16 x (Const16 <t> [d])))
	// cond:
	// result: (Xor16 (Const16 <t> [int64(int16(c^d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpXor16 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst16 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpXor16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c ^ d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Xor16 (Xor16 (Const16 <t> [d]) x) (Const16 <t> [c]))
	// cond:
	// result: (Xor16 (Const16 <t> [int64(int16(c^d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor16 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst16 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpXor16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c ^ d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Xor16 (Xor16 x (Const16 <t> [d])) (Const16 <t> [c]))
	// cond:
	// result: (Xor16 (Const16 <t> [int64(int16(c^d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor16 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst16 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst16 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpXor16)
		v0 := b.NewValue0(v.Pos, OpConst16, t)
		v0.AuxInt = int64(int16(c ^ d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpXor32_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Xor32 (Const32 [c]) (Const32 [d]))
	// cond:
	// result: (Const32 [int64(int32(c^d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c ^ d))
		return true
	}
	// match: (Xor32 (Const32 [d]) (Const32 [c]))
	// cond:
	// result: (Const32 [int64(int32(c^d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(int32(c ^ d))
		return true
	}
	// match: (Xor32 x x)
	// cond:
	// result: (Const32 [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConst32)
		v.AuxInt = 0
		return true
	}
	// match: (Xor32 (Const32 [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Xor32 x (Const32 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Xor32 x (Xor32 x y))
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor32 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor32 x (Xor32 y x))
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor32 {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor32 (Xor32 x y) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor32 (Xor32 y x) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor32 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor32 (Xor32 i:(Const32 <t>) z) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Xor32 i (Xor32 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor32 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpXor32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpXor32_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Xor32 (Xor32 z i:(Const32 <t>)) x)
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Xor32 i (Xor32 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor32 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpXor32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Xor32 x (Xor32 i:(Const32 <t>) z))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Xor32 i (Xor32 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor32 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpXor32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Xor32 x (Xor32 z i:(Const32 <t>)))
	// cond: (z.Op != OpConst32 && x.Op != OpConst32)
	// result: (Xor32 i (Xor32 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor32 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst32 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst32 && x.Op != OpConst32) {
			break
		}
		v.reset(OpXor32)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor32, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Xor32 (Const32 <t> [c]) (Xor32 (Const32 <t> [d]) x))
	// cond:
	// result: (Xor32 (Const32 <t> [int64(int32(c^d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpXor32 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst32 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpXor32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c ^ d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Xor32 (Const32 <t> [c]) (Xor32 x (Const32 <t> [d])))
	// cond:
	// result: (Xor32 (Const32 <t> [int64(int32(c^d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpXor32 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst32 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpXor32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c ^ d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Xor32 (Xor32 (Const32 <t> [d]) x) (Const32 <t> [c]))
	// cond:
	// result: (Xor32 (Const32 <t> [int64(int32(c^d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor32 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst32 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpXor32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c ^ d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Xor32 (Xor32 x (Const32 <t> [d])) (Const32 <t> [c]))
	// cond:
	// result: (Xor32 (Const32 <t> [int64(int32(c^d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor32 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst32 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst32 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpXor32)
		v0 := b.NewValue0(v.Pos, OpConst32, t)
		v0.AuxInt = int64(int32(c ^ d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpXor64_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Xor64 (Const64 [c]) (Const64 [d]))
	// cond:
	// result: (Const64 [c^d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c ^ d
		return true
	}
	// match: (Xor64 (Const64 [d]) (Const64 [c]))
	// cond:
	// result: (Const64 [c^d])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst64)
		v.AuxInt = c ^ d
		return true
	}
	// match: (Xor64 x x)
	// cond:
	// result: (Const64 [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConst64)
		v.AuxInt = 0
		return true
	}
	// match: (Xor64 (Const64 [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Xor64 x (Const64 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Xor64 x (Xor64 x y))
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor64 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor64 x (Xor64 y x))
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor64 {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor64 (Xor64 x y) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor64 (Xor64 y x) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor64 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor64 (Xor64 i:(Const64 <t>) z) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Xor64 i (Xor64 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor64 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpXor64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpXor64_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Xor64 (Xor64 z i:(Const64 <t>)) x)
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Xor64 i (Xor64 <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor64 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpXor64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Xor64 x (Xor64 i:(Const64 <t>) z))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Xor64 i (Xor64 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor64 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpXor64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Xor64 x (Xor64 z i:(Const64 <t>)))
	// cond: (z.Op != OpConst64 && x.Op != OpConst64)
	// result: (Xor64 i (Xor64 <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor64 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst64 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst64 && x.Op != OpConst64) {
			break
		}
		v.reset(OpXor64)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor64, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Xor64 (Const64 <t> [c]) (Xor64 (Const64 <t> [d]) x))
	// cond:
	// result: (Xor64 (Const64 <t> [c^d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpXor64 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst64 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpXor64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c ^ d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Xor64 (Const64 <t> [c]) (Xor64 x (Const64 <t> [d])))
	// cond:
	// result: (Xor64 (Const64 <t> [c^d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst64 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpXor64 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst64 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpXor64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c ^ d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Xor64 (Xor64 (Const64 <t> [d]) x) (Const64 <t> [c]))
	// cond:
	// result: (Xor64 (Const64 <t> [c^d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor64 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst64 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpXor64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c ^ d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Xor64 (Xor64 x (Const64 <t> [d])) (Const64 <t> [c]))
	// cond:
	// result: (Xor64 (Const64 <t> [c^d]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor64 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst64 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst64 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpXor64)
		v0 := b.NewValue0(v.Pos, OpConst64, t)
		v0.AuxInt = c ^ d
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpXor8_0(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Xor8 (Const8 [c]) (Const8 [d]))
	// cond:
	// result: (Const8  [int64(int8(c^d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		d := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c ^ d))
		return true
	}
	// match: (Xor8 (Const8 [d]) (Const8 [c]))
	// cond:
	// result: (Const8  [int64(int8(c^d))])
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		d := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		c := v_1.AuxInt
		v.reset(OpConst8)
		v.AuxInt = int64(int8(c ^ d))
		return true
	}
	// match: (Xor8 x x)
	// cond:
	// result: (Const8  [0])
	for {
		_ = v.Args[1]
		x := v.Args[0]
		if x != v.Args[1] {
			break
		}
		v.reset(OpConst8)
		v.AuxInt = 0
		return true
	}
	// match: (Xor8 (Const8 [0]) x)
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		if v_0.AuxInt != 0 {
			break
		}
		x := v.Args[1]
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Xor8 x (Const8 [0]))
	// cond:
	// result: x
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.AuxInt != 0 {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	// match: (Xor8 x (Xor8 x y))
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor8 {
			break
		}
		_ = v_1.Args[1]
		if x != v_1.Args[0] {
			break
		}
		y := v_1.Args[1]
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor8 x (Xor8 y x))
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor8 {
			break
		}
		_ = v_1.Args[1]
		y := v_1.Args[0]
		if x != v_1.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor8 (Xor8 x y) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		y := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor8 (Xor8 y x) x)
	// cond:
	// result: y
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor8 {
			break
		}
		_ = v_0.Args[1]
		y := v_0.Args[0]
		x := v_0.Args[1]
		if x != v.Args[1] {
			break
		}
		v.reset(OpCopy)
		v.Type = y.Type
		v.AddArg(y)
		return true
	}
	// match: (Xor8 (Xor8 i:(Const8 <t>) z) x)
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Xor8  i (Xor8  <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor8 {
			break
		}
		_ = v_0.Args[1]
		i := v_0.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_0.Args[1]
		x := v.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpXor8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	return false
}
func rewriteValuegeneric_OpXor8_10(v *Value) bool {
	b := v.Block
	_ = b
	// match: (Xor8 (Xor8 z i:(Const8 <t>)) x)
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Xor8  i (Xor8  <t> z x))
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor8 {
			break
		}
		_ = v_0.Args[1]
		z := v_0.Args[0]
		i := v_0.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		x := v.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpXor8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Xor8 x (Xor8 i:(Const8 <t>) z))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Xor8  i (Xor8  <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor8 {
			break
		}
		_ = v_1.Args[1]
		i := v_1.Args[0]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		z := v_1.Args[1]
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpXor8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Xor8 x (Xor8 z i:(Const8 <t>)))
	// cond: (z.Op != OpConst8  && x.Op != OpConst8)
	// result: (Xor8  i (Xor8  <t> z x))
	for {
		_ = v.Args[1]
		x := v.Args[0]
		v_1 := v.Args[1]
		if v_1.Op != OpXor8 {
			break
		}
		_ = v_1.Args[1]
		z := v_1.Args[0]
		i := v_1.Args[1]
		if i.Op != OpConst8 {
			break
		}
		t := i.Type
		if !(z.Op != OpConst8 && x.Op != OpConst8) {
			break
		}
		v.reset(OpXor8)
		v.AddArg(i)
		v0 := b.NewValue0(v.Pos, OpXor8, t)
		v0.AddArg(z)
		v0.AddArg(x)
		v.AddArg(v0)
		return true
	}
	// match: (Xor8 (Const8 <t> [c]) (Xor8 (Const8 <t> [d]) x))
	// cond:
	// result: (Xor8  (Const8  <t> [int64(int8(c^d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpXor8 {
			break
		}
		_ = v_1.Args[1]
		v_1_0 := v_1.Args[0]
		if v_1_0.Op != OpConst8 {
			break
		}
		if v_1_0.Type != t {
			break
		}
		d := v_1_0.AuxInt
		x := v_1.Args[1]
		v.reset(OpXor8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c ^ d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Xor8 (Const8 <t> [c]) (Xor8 x (Const8 <t> [d])))
	// cond:
	// result: (Xor8  (Const8  <t> [int64(int8(c^d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		t := v_0.Type
		c := v_0.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpXor8 {
			break
		}
		_ = v_1.Args[1]
		x := v_1.Args[0]
		v_1_1 := v_1.Args[1]
		if v_1_1.Op != OpConst8 {
			break
		}
		if v_1_1.Type != t {
			break
		}
		d := v_1_1.AuxInt
		v.reset(OpXor8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c ^ d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Xor8 (Xor8 (Const8 <t> [d]) x) (Const8 <t> [c]))
	// cond:
	// result: (Xor8  (Const8  <t> [int64(int8(c^d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor8 {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpConst8 {
			break
		}
		t := v_0_0.Type
		d := v_0_0.AuxInt
		x := v_0.Args[1]
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpXor8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c ^ d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	// match: (Xor8 (Xor8 x (Const8 <t> [d])) (Const8 <t> [c]))
	// cond:
	// result: (Xor8  (Const8  <t> [int64(int8(c^d))]) x)
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpXor8 {
			break
		}
		_ = v_0.Args[1]
		x := v_0.Args[0]
		v_0_1 := v_0.Args[1]
		if v_0_1.Op != OpConst8 {
			break
		}
		t := v_0_1.Type
		d := v_0_1.AuxInt
		v_1 := v.Args[1]
		if v_1.Op != OpConst8 {
			break
		}
		if v_1.Type != t {
			break
		}
		c := v_1.AuxInt
		v.reset(OpXor8)
		v0 := b.NewValue0(v.Pos, OpConst8, t)
		v0.AuxInt = int64(int8(c ^ d))
		v.AddArg(v0)
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZero_0(v *Value) bool {
	b := v.Block
	_ = b
	config := b.Func.Config
	_ = config
	// match: (Zero (Load (OffPtr [c] (SP)) mem) mem)
	// cond: mem.Op == OpStaticCall 	&& isSameSym(mem.Aux, "runtime.newobject") 	&& c == config.ctxt.FixedFrameSize() + config.RegSize
	// result: mem
	for {
		_ = v.Args[1]
		v_0 := v.Args[0]
		if v_0.Op != OpLoad {
			break
		}
		_ = v_0.Args[1]
		v_0_0 := v_0.Args[0]
		if v_0_0.Op != OpOffPtr {
			break
		}
		c := v_0_0.AuxInt
		v_0_0_0 := v_0_0.Args[0]
		if v_0_0_0.Op != OpSP {
			break
		}
		mem := v_0.Args[1]
		if mem != v.Args[1] {
			break
		}
		if !(mem.Op == OpStaticCall && isSameSym(mem.Aux, "runtime.newobject") && c == config.ctxt.FixedFrameSize()+config.RegSize) {
			break
		}
		v.reset(OpCopy)
		v.Type = mem.Type
		v.AddArg(mem)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZeroExt16to32_0(v *Value) bool {
	// match: (ZeroExt16to32 (Const16 [c]))
	// cond:
	// result: (Const32 [int64(uint16(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(uint16(c))
		return true
	}
	// match: (ZeroExt16to32 (Trunc32to16 x:(Rsh32Ux64 _ (Const64 [s]))))
	// cond: s >= 16
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpTrunc32to16 {
			break
		}
		x := v_0.Args[0]
		if x.Op != OpRsh32Ux64 {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != OpConst64 {
			break
		}
		s := x_1.AuxInt
		if !(s >= 16) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZeroExt16to64_0(v *Value) bool {
	// match: (ZeroExt16to64 (Const16 [c]))
	// cond:
	// result: (Const64 [int64(uint16(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst16 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst64)
		v.AuxInt = int64(uint16(c))
		return true
	}
	// match: (ZeroExt16to64 (Trunc64to16 x:(Rsh64Ux64 _ (Const64 [s]))))
	// cond: s >= 48
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpTrunc64to16 {
			break
		}
		x := v_0.Args[0]
		if x.Op != OpRsh64Ux64 {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != OpConst64 {
			break
		}
		s := x_1.AuxInt
		if !(s >= 48) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZeroExt32to64_0(v *Value) bool {
	// match: (ZeroExt32to64 (Const32 [c]))
	// cond:
	// result: (Const64 [int64(uint32(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst32 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst64)
		v.AuxInt = int64(uint32(c))
		return true
	}
	// match: (ZeroExt32to64 (Trunc64to32 x:(Rsh64Ux64 _ (Const64 [s]))))
	// cond: s >= 32
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpTrunc64to32 {
			break
		}
		x := v_0.Args[0]
		if x.Op != OpRsh64Ux64 {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != OpConst64 {
			break
		}
		s := x_1.AuxInt
		if !(s >= 32) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZeroExt8to16_0(v *Value) bool {
	// match: (ZeroExt8to16 (Const8 [c]))
	// cond:
	// result: (Const16 [int64( uint8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst16)
		v.AuxInt = int64(uint8(c))
		return true
	}
	// match: (ZeroExt8to16 (Trunc16to8 x:(Rsh16Ux64 _ (Const64 [s]))))
	// cond: s >= 8
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpTrunc16to8 {
			break
		}
		x := v_0.Args[0]
		if x.Op != OpRsh16Ux64 {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != OpConst64 {
			break
		}
		s := x_1.AuxInt
		if !(s >= 8) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZeroExt8to32_0(v *Value) bool {
	// match: (ZeroExt8to32 (Const8 [c]))
	// cond:
	// result: (Const32 [int64( uint8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst32)
		v.AuxInt = int64(uint8(c))
		return true
	}
	// match: (ZeroExt8to32 (Trunc32to8 x:(Rsh32Ux64 _ (Const64 [s]))))
	// cond: s >= 24
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpTrunc32to8 {
			break
		}
		x := v_0.Args[0]
		if x.Op != OpRsh32Ux64 {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != OpConst64 {
			break
		}
		s := x_1.AuxInt
		if !(s >= 24) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteValuegeneric_OpZeroExt8to64_0(v *Value) bool {
	// match: (ZeroExt8to64 (Const8 [c]))
	// cond:
	// result: (Const64 [int64( uint8(c))])
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpConst8 {
			break
		}
		c := v_0.AuxInt
		v.reset(OpConst64)
		v.AuxInt = int64(uint8(c))
		return true
	}
	// match: (ZeroExt8to64 (Trunc64to8 x:(Rsh64Ux64 _ (Const64 [s]))))
	// cond: s >= 56
	// result: x
	for {
		v_0 := v.Args[0]
		if v_0.Op != OpTrunc64to8 {
			break
		}
		x := v_0.Args[0]
		if x.Op != OpRsh64Ux64 {
			break
		}
		_ = x.Args[1]
		x_1 := x.Args[1]
		if x_1.Op != OpConst64 {
			break
		}
		s := x_1.AuxInt
		if !(s >= 56) {
			break
		}
		v.reset(OpCopy)
		v.Type = x.Type
		v.AddArg(x)
		return true
	}
	return false
}
func rewriteBlockgeneric(b *Block) bool {
	config := b.Func.Config
	_ = config
	fe := b.Func.fe
	_ = fe
	typ := &config.Types
	_ = typ
	switch b.Kind {
	case BlockIf:
		// match: (If (Not cond) yes no)
		// cond:
		// result: (If cond no yes)
		for {
			v := b.Control
			if v.Op != OpNot {
				break
			}
			cond := v.Args[0]
			b.Kind = BlockIf
			b.SetControl(cond)
			b.swapSuccessors()
			return true
		}
		// match: (If (ConstBool [c]) yes no)
		// cond: c == 1
		// result: (First nil yes no)
		for {
			v := b.Control
			if v.Op != OpConstBool {
				break
			}
			c := v.AuxInt
			if !(c == 1) {
				break
			}
			b.Kind = BlockFirst
			b.SetControl(nil)
			return true
		}
		// match: (If (ConstBool [c]) yes no)
		// cond: c == 0
		// result: (First nil no yes)
		for {
			v := b.Control
			if v.Op != OpConstBool {
				break
			}
			c := v.AuxInt
			if !(c == 0) {
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
